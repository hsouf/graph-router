"""Query-time routing.

The contract: given a question, return a few hundred candidate files while
*reading* as few files as possible. "Reading" here means touching a document's
posting entry at all — that count is reported as `files_touched` and is the
number the whole design optimises.

Pipeline per query::

    query -> terms -> anchor seeds
          -> frontier PPR over the term graph   (vocabulary bridging)
          -> topic selection                    (shard pruning: the big win)
          -> budgeted postings walk             (hard cap on files touched)
          -> BM25 rescore of touched docs only
          -> top-N files
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import graph as G
from .index import GraphIndex
from .text import query_terms


@dataclass
class RouteConfig:
    top_n: int = 100
    """How many files to hand back to the reader/LLM stage."""
    budget: int = 4_000
    """Hard ceiling on files touched. The router stops gathering once hit."""

    # --- graph expansion ---
    expand_terms: int = 64
    """Hard cap on how many expanded terms may contribute postings."""
    min_term_weight: float = 0.02
    """Drop expanded terms below this fraction of the strongest term's weight.
    Each surviving term costs a Python-level posting walk, and the tail of a
    PPR distribution contributes rounding noise to the ranking, so cutting it
    is close to free in quality and most of the win in latency."""
    ppr_alpha: float = 0.45
    ppr_hops: int = 3
    ppr_beam: int = 256

    # --- shard pruning ---
    use_topic_gate: bool = True
    topic_coverage: float = 0.85
    """Select topics until this fraction of PPR mass is covered."""
    max_topics: int = 12

    # --- postings walk ---
    df_ceiling_ratio: float = 0.05
    """Skip expanded terms that appear in more than this fraction of files.
    They cost budget without discriminating. Literal query terms are exempt."""
    literal_boost: float = 2.5
    """Exact query matches outrank graph-inferred neighbours by this factor."""

    # --- BM25 ---
    k1: float = 1.2
    b: float = 0.75

    # --- output safety ---
    score_floor: float = 0.35
    """Drop candidates scoring below this fraction of the best hit, returning
    fewer than `top_n` rather than padding the list with weak matches.

    This exists because the dangerous failure of a RAG router is not missing a
    file, it is handing the reader a plausible file about the wrong subject.
    The reader stage cannot tell the difference: a summary built on the wrong
    document is fluent, confident and wrong ("context poisoning"). Returning 12
    good files beats returning 12 good files and 88 bystanders. Set to 0.0 to
    always fill `top_n`.
    """
    min_results: int = 1
    """Never let the floor cut below this many results, so a query with a flat
    score distribution still returns something."""


@dataclass
class RouteResult:
    files: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    doc_ids: List[int] = field(default_factory=list)
    files_touched: int = 0
    corpus_size: int = 0
    topics: List[int] = field(default_factory=list)
    seed_terms: List[str] = field(default_factory=list)
    expanded_terms: List[Tuple[str, float]] = field(default_factory=list)
    latency_ms: float = 0.0
    budget_exhausted: bool = False
    cut_by_floor: int = 0
    """How many candidates the score floor removed from the returned list."""
    filtered_out: int = 0
    """How many candidates the document filter excluded during gathering."""

    @property
    def reduction(self) -> float:
        """How much of the corpus we avoided touching, as a factor."""
        return self.corpus_size / max(self.files_touched, 1)

    def explain(self) -> str:
        top = ", ".join(f"{t}({w:.2f})" for t, w in self.expanded_terms[:10])
        lines = [
            f"query terms: {', '.join(self.seed_terms) or '(none matched)'}",
            f"expanded:    {top}",
            f"topics:      {self.topics}",
            f"touched:     {self.files_touched:,} / {self.corpus_size:,} files "
            f"({self.reduction:.0f}x reduction)"
            f"{' [budget exhausted]' if self.budget_exhausted else ''}",
        ]
        if self.filtered_out:
            lines.append(f"filtered:    {self.filtered_out:,} files excluded by doc filter")
        if self.cut_by_floor:
            lines.append(
                f"score floor: {self.cut_by_floor} weak candidates dropped "
                f"(would have been padding)"
            )
        lines.append(
            f"returned:    {len(self.files)} files in {self.latency_ms:.1f} ms"
        )
        return "\n".join(lines)


class Router:
    def __init__(self, index: GraphIndex, config: Optional[RouteConfig] = None):
        self.index = index
        self.config = config or RouteConfig()
        # Reused across queries; only the touched slots are reset, so the cost
        # of the scratch buffer is O(files touched), not O(corpus).
        self._acc = np.zeros(index.n_docs, dtype=np.float32)
        self._seen = np.zeros(index.n_docs, dtype=bool)
        self._rejected = np.zeros(index.n_docs, dtype=bool)

    @classmethod
    def load(cls, path: str, config: Optional[RouteConfig] = None) -> "Router":
        return cls(GraphIndex.load(path), config)

    # ------------------------------------------------------------------ api

    def route(
        self,
        query: str,
        top_n: Optional[int] = None,
        doc_filter: Optional[np.ndarray] = None,
        **overrides,
    ) -> RouteResult:
        """Route `query` to candidate files.

        `doc_filter` is a boolean mask over document ids marking which files
        this caller may see — permissions, tenant, language, freshness, any
        per-request predicate. It is applied *during* gathering rather than to
        the finished ranking, which matters: filtering a top-100 after the fact
        silently returns 8 files when 92 were inadmissible, whereas filtering
        inside the walk spends the budget entirely on files the caller can
        actually read. Build the mask once per role and reuse it; see
        `GraphIndex.mask_where`.
        """
        cfg = self.config
        if overrides:
            cfg = RouteConfig(**{**cfg.__dict__, **overrides})
        if top_n is not None:
            cfg.top_n = top_n

        if doc_filter is not None:
            doc_filter = np.asarray(doc_filter)
            if doc_filter.dtype != bool or doc_filter.shape != (self.index.n_docs,):
                raise ValueError(
                    f"doc_filter must be a boolean array of shape "
                    f"({self.index.n_docs},), got {doc_filter.dtype} "
                    f"{doc_filter.shape}"
                )

        idx = self.index
        t0 = time.perf_counter()
        result = RouteResult(corpus_size=idx.n_docs)

        literal_ids = [idx.term_id[t] for t in query_terms(query) if t in idx.term_id]
        result.seed_terms = [idx.terms[t] for t in literal_ids]
        if not literal_ids:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        weights = self._expand(literal_ids, cfg)
        topics = self._select_topics(weights, cfg) if cfg.use_topic_gate else None
        result.topics = sorted(topics) if topics is not None else []
        result.expanded_terms = [
            (idx.terms[t], float(w))
            for t, w in sorted(weights.items(), key=lambda kv: -kv[1])[:32]
        ]

        touched, exhausted, rejected = self._gather(
            weights, set(literal_ids), topics, cfg, doc_filter
        )
        result.files_touched = int(touched.size)
        result.budget_exhausted = exhausted
        result.filtered_out = rejected

        if touched.size:
            scores = self._acc[touched]
            n = min(cfg.top_n, touched.size)
            part = np.argpartition(-scores, n - 1)[:n]
            order = part[np.argsort(-scores[part])]

            if cfg.score_floor > 0 and order.size:
                top_score = float(scores[order[0]])
                if top_score > 0:
                    keep = scores[order] >= top_score * cfg.score_floor
                    cut = max(int(keep.sum()), min(cfg.min_results, order.size))
                    result.cut_by_floor = int(order.size - cut)
                    order = order[:cut]

            best = touched[order]
            result.doc_ids = best.tolist()
            result.files = [idx.paths[i] for i in best]
            result.scores = scores[order].tolist()
            self._acc[touched] = 0.0
            self._seen[touched] = False

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def batch(self, queries: Sequence[str], **kw) -> List[RouteResult]:
        return [self.route(q, **kw) for q in queries]

    # -------------------------------------------------------------- stages

    def _expand(self, literal_ids: List[int], cfg: RouteConfig) -> Dict[int, float]:
        """Literal query terms + graph neighbourhood, as term_id -> weight."""
        idx = self.index
        seeds: Dict[int, float] = {}
        for tid in literal_ids:
            aid = int(idx.anchor_of[tid])
            if aid >= 0:
                seeds[aid] = seeds.get(aid, 0.0) + float(idx.idf[tid])

        weights: Dict[int, float] = {}
        if seeds:
            nodes, scores = G.frontier_ppr(
                idx.adj, seeds,
                alpha=cfg.ppr_alpha, hops=cfg.ppr_hops, beam=cfg.ppr_beam,
            )
            if nodes.size:
                self._last_ppr = (nodes, scores)
                term_ids = idx.anchor_terms[nodes]
                peak = float(scores[0]) or 1.0
                for tid, sc in zip(term_ids.tolist(), (scores / peak).tolist()):
                    weights[tid] = max(weights.get(tid, 0.0), sc)
            else:
                self._last_ppr = (np.empty(0, np.int64), np.empty(0, np.float32))
        else:
            self._last_ppr = (np.empty(0, np.int64), np.empty(0, np.float32))

        for tid in literal_ids:
            weights[tid] = weights.get(tid, 0.0) + cfg.literal_boost
        return weights

    def _select_topics(self, weights: Dict[int, float], cfg: RouteConfig):
        """Pick the smallest set of communities covering most of the mass.

        This is where millions become thousands. Everything after it is
        ordinary retrieval on a small candidate pool.
        """
        idx = self.index
        nodes, scores = getattr(self, "_last_ppr", (None, None))
        if nodes is None or nodes.size == 0:
            return None

        mass = np.zeros(idx.n_topics, dtype=np.float64)
        np.add.at(mass, idx.community[nodes], scores.astype(np.float64))
        total = mass.sum()
        if total <= 0:
            return None

        order = np.argsort(-mass)
        cum = np.cumsum(mass[order]) / total
        cutoff = int(np.searchsorted(cum, cfg.topic_coverage) + 1)
        chosen = order[: min(cutoff, cfg.max_topics)]
        return set(int(t) for t in chosen if mass[t] > 0)

    def _gather(
        self,
        weights: Dict[int, float],
        literal: set,
        topics,
        cfg: RouteConfig,
        doc_filter: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, int]:
        """Walk postings in descending term weight until the budget is spent.

        Returns (touched doc ids, budget exhausted, distinct docs rejected by
        the filter).
        """
        idx = self.index
        post = idx.postings
        df_ceiling = max(1, int(cfg.df_ceiling_ratio * idx.n_docs))

        # A term is "selective" if its posting list is small enough to be worth
        # walking. Common words fail this test, and a literal match on a common
        # word buys nothing: its idf is near zero anyway.
        selective = {t for t, _ in weights.items() if idx.df[t] <= df_ceiling}
        if not selective:
            # Degenerate query (every term is a corpus stopword). Fall back to
            # the most selective literal term rather than returning nothing.
            selective = {min(literal, key=lambda t: idx.df[t])} if literal else set()

        ranked = sorted(
            ((t, w) for t, w in weights.items() if t in selective),
            key=lambda kv: -(kv[1] * float(idx.idf[kv[0]])),
        )[: cfg.expand_terms]
        if ranked and cfg.min_term_weight > 0:
            floor = ranked[0][1] * float(idx.idf[ranked[0][0]]) * cfg.min_term_weight
            ranked = [
                (t, w) for t, w in ranked if w * float(idx.idf[t]) >= floor
            ]
        # Selective literal terms go first: exact matches must never be starved
        # by the budget, and they are exempt from the topic gate below.
        exempt = literal & selective
        ranked.sort(key=lambda kv: (kv[0] not in exempt,))

        gate = None
        if topics:
            gate = np.zeros(idx.n_topics, dtype=bool)
            gate[list(topics)] = True

        touched_blocks: List[np.ndarray] = []
        rejected_blocks: List[np.ndarray] = []
        n_touched = 0
        n_rejected = 0
        exhausted = False
        k1, b = cfg.k1, cfg.b
        avg_len = idx.avg_doc_len or 1.0

        for tid, w in ranked:
            is_literal = tid in exempt
            start, end = post.indptr[tid], post.indptr[tid + 1]
            if start == end:
                continue
            docs = post.indices[start:end]
            tf = post.data[start:end]

            # Access control comes first and applies to every term, literal
            # ones included: an exact match on a file the caller may not read
            # is still a file the caller may not read.
            if doc_filter is not None:
                admissible = doc_filter[docs]
                if not admissible.all():
                    blocked = docs[~admissible]
                    novel = blocked[~self._rejected[blocked]]
                    if novel.size:
                        self._rejected[novel] = True
                        rejected_blocks.append(novel)
                        n_rejected += int(novel.size)
                    if not admissible.any():
                        continue
                    docs, tf = docs[admissible], tf[admissible]

            if gate is not None and not is_literal:
                keep = gate[idx.doc_topic[docs]]
                if not keep.any():
                    continue
                docs, tf = docs[keep], tf[keep]

            if n_touched >= cfg.budget:
                exhausted = True
                break
            if n_touched + docs.size > cfg.budget:
                # Partial admission: keep the highest-tf docs of this term.
                room = cfg.budget - n_touched
                sel = np.argpartition(-tf, room - 1)[:room]
                docs, tf = docs[sel], tf[sel]
                exhausted = True

            length = idx.doc_len[docs]
            denom = tf + k1 * (1.0 - b + b * length / avg_len)
            contrib = (float(idx.idf[tid]) * w) * (tf * (k1 + 1.0) / denom)

            fresh = ~self._seen[docs]
            if fresh.any():
                new_docs = docs[fresh]
                self._seen[new_docs] = True
                touched_blocks.append(new_docs)
                n_touched += int(new_docs.size)
            np.add.at(self._acc, docs, contrib.astype(np.float32))

            if exhausted:
                break

        for block in rejected_blocks:
            self._rejected[block] = False

        touched = (
            np.concatenate(touched_blocks) if touched_blocks else np.empty(0, dtype=np.int64)
        )
        return touched, exhausted, n_rejected


def bm25_scores(index: GraphIndex, query: str, k1: float = 1.2, b: float = 0.75):
    """Dense BM25 score for every document. O(corpus), used as the reference."""
    ids = [index.term_id[t] for t in query_terms(query) if t in index.term_id]
    scores = np.zeros(index.n_docs, dtype=np.float32)
    post = index.postings
    avg_len = index.avg_doc_len or 1.0
    for tid in ids:
        start, end = post.indptr[tid], post.indptr[tid + 1]
        docs, tf = post.indices[start:end], post.data[start:end]
        length = index.doc_len[docs]
        denom = tf + k1 * (1.0 - b + b * length / avg_len)
        np.add.at(
            scores, docs,
            (float(index.idf[tid]) * tf * (k1 + 1.0) / denom).astype(np.float32),
        )
    return scores


def full_scan_baseline(index: GraphIndex, query: str, top_n: int = 100):
    """Reference implementation: BM25 over every posting, no pruning.

    Only here so the benchmark can report honest precision and speedup numbers.
    """
    scores = bm25_scores(index, query)
    n = min(top_n, index.n_docs)
    part = np.argpartition(-scores, n - 1)[:n]
    order = part[np.argsort(-scores[part])]
    order = order[scores[order] > 0]
    return order, scores[order]
