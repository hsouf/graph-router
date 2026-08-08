"""Index construction.

Three layers are built, cheapest first:

1. **Inverted index** — classic postings, term -> (doc, tf). Gives exact lookup.
2. **Term graph** — NPMI co-occurrence between "anchor" terms, degree-capped.
   Gives vocabulary bridging: a query word reaches words it never appears with
   literally but which live in the same neighbourhood.
3. **Communities** — label propagation over the term graph, projected onto
   documents. Gives shards: a coarse partition the router uses to refuse to
   look at most of the corpus.

None of this calls a model or a network. Index time is dominated by disk reads.
"""

from __future__ import annotations

import io
import json
import os
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from . import graph as G
from .ingest import Document, walk
from .text import count_terms


@dataclass
class IndexConfig:
    # --- vocabulary ---
    min_df: int = 2
    """Terms in fewer than this many files are dropped (typos, hapax noise)."""
    max_df_ratio: float = 0.15
    """Terms in more than this fraction of files are dropped (corpus stopwords).
    Relaxed automatically if it would leave fewer than `min_anchors` anchors."""
    min_anchors: int = 64
    """Floor on graph nodes before the df ceiling is widened. Below this there
    is not enough structure for traversal to beat a plain inverted index."""
    max_vocab: int = 2_000_000

    # --- term graph ---
    max_anchors: int = 200_000
    """Cap on graph nodes. Anchors are the terms allowed to carry traversal."""
    graph_terms_per_doc: int = 32
    """Top-k anchors per document used to form edges. Bounds edge count at
    O(n_docs * k^2) instead of O(n_docs * terms^2)."""
    graph_degree: int = 24
    """Max neighbours kept per node. Bounds per-hop branching at query time."""
    min_pair: int = 2
    """Minimum co-occurrence count for an edge to exist at all."""

    # --- communities ---
    lp_iterations: int = 10

    # --- ingest ---
    include_ext: Optional[Sequence[str]] = None
    max_file_bytes: int = 2_000_000


@dataclass
class IndexStats:
    n_docs: int = 0
    n_terms: int = 0
    n_anchors: int = 0
    n_edges: int = 0
    n_topics: int = 0
    build_seconds: float = 0.0
    topic_sizes_p50: int = 0
    topic_sizes_p95: int = 0


class GraphIndex:
    """Everything the router needs, and nothing it doesn't."""

    def __init__(
        self,
        paths: List[str],
        terms: List[str],
        postings: sp.csc_matrix,
        doc_len: np.ndarray,
        df: np.ndarray,
        anchor_of: np.ndarray,
        anchor_terms: np.ndarray,
        adj: sp.csr_matrix,
        community: np.ndarray,
        doc_topic: np.ndarray,
        config: IndexConfig,
        stats: IndexStats,
    ):
        self.paths = paths
        self.terms = terms
        self.term_id = {t: i for i, t in enumerate(terms)}
        self.postings = postings          # n_docs x n_terms, CSC (term-major)
        self.doc_len = doc_len
        self.df = df
        self.anchor_of = anchor_of        # term id -> anchor id, or -1
        self.anchor_terms = anchor_terms  # anchor id -> term id
        self.adj = adj                    # anchor x anchor, row-normalised
        self.community = community        # anchor id -> topic id
        self.doc_topic = doc_topic        # doc id -> topic id
        self.config = config
        self.stats = stats

        self.n_docs = len(paths)
        self.avg_doc_len = float(doc_len.mean()) if self.n_docs else 0.0
        self.idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        n_topics = int(doc_topic.max()) + 1 if doc_topic.size else 0
        self.n_topics = n_topics
        order = np.argsort(doc_topic, kind="stable")
        counts = np.bincount(doc_topic, minlength=n_topics) if n_topics else np.zeros(0, int)
        self.topic_doc_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.topic_doc_ids = order.astype(np.int64)

    # ------------------------------------------------------------------ io

    def save(self, path: str) -> None:
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            post_data=self.postings.data,
            post_indices=self.postings.indices,
            post_indptr=self.postings.indptr,
            post_shape=np.array(self.postings.shape),
            doc_len=self.doc_len,
            df=self.df,
            anchor_of=self.anchor_of,
            anchor_terms=self.anchor_terms,
            adj_data=self.adj.data,
            adj_indices=self.adj.indices,
            adj_indptr=self.adj.indptr,
            adj_shape=np.array(self.adj.shape),
            community=self.community,
            doc_topic=self.doc_topic,
        )
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("arrays.npz", buf.getvalue())
            zf.writestr("paths.json", json.dumps(self.paths))
            zf.writestr("terms.json", json.dumps(self.terms))
            zf.writestr(
                "meta.json",
                json.dumps({"config": asdict(self.config), "stats": asdict(self.stats)}),
            )

    @classmethod
    def load(cls, path: str) -> "GraphIndex":
        with zipfile.ZipFile(path) as zf:
            arrays = np.load(io.BytesIO(zf.read("arrays.npz")))
            paths = json.loads(zf.read("paths.json"))
            terms = json.loads(zf.read("terms.json"))
            meta = json.loads(zf.read("meta.json"))

        postings = sp.csc_matrix(
            (arrays["post_data"], arrays["post_indices"], arrays["post_indptr"]),
            shape=tuple(arrays["post_shape"]),
        )
        adj = sp.csr_matrix(
            (arrays["adj_data"], arrays["adj_indices"], arrays["adj_indptr"]),
            shape=tuple(arrays["adj_shape"]),
        )
        return cls(
            paths=paths,
            terms=terms,
            postings=postings,
            doc_len=arrays["doc_len"],
            df=arrays["df"],
            anchor_of=arrays["anchor_of"],
            anchor_terms=arrays["anchor_terms"],
            adj=adj,
            community=arrays["community"],
            doc_topic=arrays["doc_topic"],
            config=IndexConfig(**meta["config"]),
            stats=IndexStats(**meta["stats"]),
        )

    # -------------------------------------------------------------- helpers

    def docs_in_topics(self, topics: Iterable[int]) -> np.ndarray:
        chunks = [
            self.topic_doc_ids[self.topic_doc_ptr[t] : self.topic_doc_ptr[t + 1]]
            for t in topics
            if 0 <= t < self.n_topics
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    def mask_where(self, predicate) -> np.ndarray:
        """Build a `doc_filter` mask from a predicate over file paths.

        Convenience only, and O(n_docs) in Python — build a mask once per role
        or tenant and reuse it across queries rather than calling this per
        request. For anything beyond path matching, keep your own attribute
        arrays keyed by document id and build masks with NumPy directly::

            owner = np.array([...])          # one entry per doc, built at index time
            mask = (owner == user) | public  # cheap, vectorised, reusable
        """
        return np.fromiter(
            (bool(predicate(p)) for p in self.paths),
            dtype=bool,
            count=len(self.paths),
        )

    def summary(self) -> str:
        s = self.stats
        return (
            f"{s.n_docs:,} files | {s.n_terms:,} terms | {s.n_anchors:,} anchors | "
            f"{s.n_edges:,} edges | {s.n_topics:,} topics "
            f"(p50 {s.topic_sizes_p50:,} files, p95 {s.topic_sizes_p95:,}) | "
            f"built in {s.build_seconds:.1f}s"
        )


# ---------------------------------------------------------------- building


def _accumulate(docs: Iterable[Document], cfg: IndexConfig):
    """Single pass: assign term ids and collect a COO doc-term matrix."""
    term_id: Dict[str, int] = {}
    paths: List[str] = []
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    doc_len: List[int] = []

    for doc in docs:
        counts = count_terms(doc.text)
        if not counts:
            continue
        # Structural neighbours are folded in as pseudo-terms so that an import
        # graph or a wiki link graph contributes edges for free.
        for link in doc.links:
            counts["link:" + os.path.basename(link).lower()] = (
                counts.get("link:" + os.path.basename(link).lower(), 0) + 2
            )

        doc_idx = len(paths)
        paths.append(doc.path)
        ids = np.empty(len(counts), dtype=np.int64)
        tf = np.empty(len(counts), dtype=np.float32)
        for i, (term, c) in enumerate(counts.items()):
            tid = term_id.get(term)
            if tid is None:
                if len(term_id) >= cfg.max_vocab:
                    ids[i] = -1
                    tf[i] = 0
                    continue
                tid = len(term_id)
                term_id[term] = tid
            ids[i] = tid
            tf[i] = c
        keep = ids >= 0
        ids, tf = ids[keep], tf[keep]
        rows.append(np.full(ids.shape, doc_idx, dtype=np.int64))
        cols.append(ids)
        vals.append(tf)
        doc_len.append(int(tf.sum()))

    if not paths:
        raise ValueError("no readable documents found")

    terms = [""] * len(term_id)
    for t, i in term_id.items():
        terms[i] = t

    mat = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(paths), len(terms)),
    ).tocsr()
    return paths, terms, mat, np.asarray(doc_len, dtype=np.int32)


def _select_anchors(df: np.ndarray, n_docs: int, cfg: IndexConfig):
    """Pick the terms allowed to be graph nodes, relaxing the ceiling if needed.

    `max_df_ratio` is a fraction of the corpus, which quietly breaks on small
    or few-topic corpora: with four topics, a perfectly discriminative term
    appears in ~25% of files and a 15% ceiling discards it, leaving no graph at
    all. Rather than fail or force the caller to know this, widen the band until
    enough terms qualify. On a large corpus the first attempt always succeeds,
    so this costs nothing where it does not matter.

    Returns (anchor term ids, the ratio actually used).
    """
    for ratio in (cfg.max_df_ratio, 0.3, 0.5, 0.8):
        if ratio < cfg.max_df_ratio:
            continue
        hi = max(cfg.min_df, int(ratio * n_docs))
        band = np.flatnonzero((df >= cfg.min_df) & (df <= hi))
        if band.size >= cfg.min_anchors:
            break
    if band.size > cfg.max_anchors:
        # Prefer the better-connected end of the band: those terms bridge more.
        band = band[np.argsort(-df[band])[: cfg.max_anchors]]
        band.sort()
    return band, ratio


def _doc_anchor_topk(
    doc_term: sp.csr_matrix, anchors: np.ndarray, idf: np.ndarray, k: int
) -> sp.csr_matrix:
    """Binary doc x anchor matrix keeping each doc's k highest tf-idf anchors."""
    sub = doc_term[:, anchors].tocsr().astype(np.float32)
    if sub.nnz == 0:
        return sub
    weighted = sub.copy()
    weighted.data = weighted.data * idf[anchors][weighted.indices]
    top = G.prune_topk(weighted, k)
    top.data = np.ones_like(top.data, dtype=np.float32)
    return top


def build_index(
    source,
    config: Optional[IndexConfig] = None,
    verbose: bool = False,
) -> GraphIndex:
    """Build an index from a corpus root path or an iterable of Documents."""
    cfg = config or IndexConfig()
    t0 = time.time()

    docs = (
        walk(source, include_ext=cfg.include_ext, max_bytes=cfg.max_file_bytes)
        if isinstance(source, (str, os.PathLike))
        else source
    )
    paths, terms, doc_term, doc_len = _accumulate(docs, cfg)
    n_docs, n_terms = doc_term.shape
    if verbose:
        print(f"[index] read {n_docs:,} files, {n_terms:,} raw terms "
              f"({time.time() - t0:.1f}s)")

    binary = doc_term.copy()
    binary.data = np.ones_like(binary.data)
    df = np.asarray(binary.sum(axis=0)).ravel().astype(np.int32)
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    anchors, used_ratio = _select_anchors(df, n_docs, cfg)
    anchor_of = np.full(n_terms, -1, dtype=np.int32)
    anchor_of[anchors] = np.arange(anchors.size, dtype=np.int32)
    if anchors.size == 0:
        warnings.warn(
            f"no anchor terms: every term is in <{cfg.min_df} files or more "
            f"than 80% of them. The graph and topic layers will be empty and "
            f"routing falls back to the inverted index, which still works. "
            f"This usually means the corpus is too small or too homogeneous to "
            f"have structure worth traversing.",
            RuntimeWarning,
            stacklevel=2,
        )
    if verbose:
        note = "" if used_ratio == cfg.max_df_ratio else (
            f" (df ceiling relaxed {cfg.max_df_ratio:.0%} -> {used_ratio:.0%})"
        )
        print(f"[index] {anchors.size:,} anchor terms{note}")

    doc_anchor = _doc_anchor_topk(doc_term, anchors, idf, cfg.graph_terms_per_doc)
    adj = G.cooccurrence(doc_anchor, weight="npmi", min_pair=cfg.min_pair)
    adj = G.prune_topk(adj, cfg.graph_degree)
    adj = adj.maximum(adj.T)          # symmetrise after pruning
    adj_norm = G.row_normalize(adj)
    if verbose:
        print(f"[index] term graph: {adj.nnz:,} edges ({time.time() - t0:.1f}s)")

    community = G.label_propagation(adj, max_iter=cfg.lp_iterations)
    n_topics = int(community.max()) + 1 if community.size else 1

    if community.size == 0:
        # No anchors survived: every term is either too rare or corpus-wide.
        # This is the honest answer for a small or highly homogeneous corpus.
        # Degrade to a single shard so the inverted index still works rather
        # than failing the build.
        doc_topic = np.zeros(n_docs, dtype=np.int32)
    else:
        # Project topics onto documents: a doc belongs to the topic holding
        # most of its anchor mass. Anchor-less docs land in topic 0.
        onehot = sp.csr_matrix(
            (np.ones(community.size, dtype=np.float32),
             (np.arange(community.size), community)),
            shape=(community.size, n_topics),
        )
        doc_topic_mass = (doc_anchor @ onehot).tocsr()
        doc_topic = np.asarray(doc_topic_mass.argmax(axis=1)).ravel().astype(np.int32)
        doc_topic[np.diff(doc_topic_mass.indptr) == 0] = 0

    sizes = np.bincount(doc_topic, minlength=n_topics)
    stats = IndexStats(
        n_docs=n_docs,
        n_terms=n_terms,
        n_anchors=int(anchors.size),
        n_edges=int(adj.nnz),
        n_topics=n_topics,
        build_seconds=time.time() - t0,
        topic_sizes_p50=int(np.percentile(sizes, 50)) if sizes.size else 0,
        topic_sizes_p95=int(np.percentile(sizes, 95)) if sizes.size else 0,
    )

    index = GraphIndex(
        paths=paths,
        terms=terms,
        postings=doc_term.tocsc(),
        doc_len=doc_len,
        df=df,
        anchor_of=anchor_of,
        anchor_terms=anchors.astype(np.int64),
        adj=adj_norm,
        community=community,
        doc_topic=doc_topic,
        config=cfg,
        stats=stats,
    )
    if verbose:
        print("[index] " + index.summary())
    return index
