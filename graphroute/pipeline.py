"""Question in, passages out.

Everything else in this package is a stage. This is the thing you actually run::

    from graphroute import ask

    for p in ask("./books/", "what is leukemia", passages=8):
        print(p.header())
        print(p.text)

The first call over a corpus builds sections, a lexical index and (if
sentence-transformers is installed) an embedding index, and caches all three
under `~/.cache/graphroute/<hash>/`. Later calls load them. The hash covers
every input file's path, size and mtime, so editing the corpus rebuilds and
touching nothing does not.

Retrieval is two channels fused, then a within-section pass:

1. **Lexical** — the router, over section-documents. Terms, graph expansion,
   topic gate, budgeted walk, BM25.
2. **Dense** — cosine over section embeddings. Optional.
3. **Fusion** — reciprocal rank fusion. Ranks only, no score mixing and no
   weight to tune: the two channels' scores are not on comparable scales and
   any weight between them would be a number invented to fit one corpus.
4. **Passages** — split the surviving sections into paragraphs and rank those,
   by cosine to the question and by overlap with the router's expanded terms.
   The router already computed those weights, so that half is free.
"""

from __future__ import annotations

import hashlib
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .index import GraphIndex, IndexConfig, build_index
from .ingest import Document
from .router import RouteConfig, Router
from .sections import BREADCRUMB_SEP, Section, collect_sections
from .text import count_terms

RRF_K = 60
"""Reciprocal rank fusion constant. 60 is the value from the original TREC
work and is not sensitive: it only sets how quickly rank 1 stops dominating."""

CHUNK_MIN, CHUNK_MAX = 300, 900
"""Target passage size in characters. Below the minimum a paragraph is a
fragment with no context; above the maximum it stops being a passage."""

_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


# ------------------------------------------------------------------ results


@dataclass
class Passage:
    text: str
    breadcrumb: str
    book_path: str
    pages: Tuple[int, int]
    score: float
    section_index: int = 0

    def header(self) -> str:
        """`[Book › Chapter › p.1721]`, the line printed above the text."""
        start, end = self.pages
        page = f"p.{start}" if start == end else f"pp.{start}–{end}"
        return f"[{self.breadcrumb}{BREADCRUMB_SEP}{page}]"

    def __str__(self) -> str:
        return f"{self.header()}\n{self.text}\n"


@dataclass
class Corpus:
    """A built-and-cached corpus: sections, lexical index, optional vectors."""

    sections: List[Section]
    index: GraphIndex
    embeddings: object = None            # EmbeddingIndex | None
    cache_dir: str = ""
    _row_of: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self._row_of = {s.doc_path: i for i, s in enumerate(self.sections)}

    def section_of_doc(self, doc_id: int) -> Optional[Section]:
        i = self._row_of.get(self.index.paths[doc_id], -1)
        return self.sections[i] if i >= 0 else None


# -------------------------------------------------------------- build/cache


def corpus_fingerprint(source: str, include_ext: Optional[Sequence[str]] = None) -> str:
    """Hash of what the corpus *is*: every file's path, size and mtime.

    Content hashing would be honest but reads gigabytes to decide whether it
    needs to read gigabytes. Size plus mtime is what every build tool uses and
    is wrong only if a file is edited to exactly its old length within the
    filesystem's timestamp resolution.
    """
    from .sections import _iter_files

    h = hashlib.sha1()
    h.update(os.path.abspath(source).encode())
    for path in sorted(_iter_files(source, include_ext)):
        try:
            st = os.stat(path)
        except OSError:
            continue
        h.update(path.encode())
        h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def default_cache_root() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "graphroute")


def _save_sections(path: str, sections: Sequence[Section]) -> None:
    import gzip
    import json

    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump([s.to_dict() for s in sections], fh)


def _load_sections(path: str) -> List[Section]:
    import gzip
    import json

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [Section.from_dict(d) for d in json.load(fh)]


def _no_documents_hint(source: str, cfg: IndexConfig) -> str:
    """Explain an empty corpus when the cause is a missing optional dependency.

    A shelf of PDFs on a core-only install reads as "no readable documents",
    which is true and useless. Name the actual problem.
    """
    from .sections import _iter_files

    try:
        import pypdf  # noqa: F401
    except ImportError:
        if any(
            p.lower().endswith(".pdf") for p in _iter_files(source, cfg.include_ext)
        ):
            return (
                ": the corpus is PDFs and pypdf is not installed. "
                "Install it with: pip install 'graphroute[rich]'"
            )
    return ""


def build_corpus(
    source: str,
    cache_dir: Optional[str] = None,
    index_config: Optional[IndexConfig] = None,
    embed: bool = True,
    embed_fn: Optional[Callable] = None,
    model_name: Optional[str] = None,
    rebuild: bool = False,
    verbose: bool = False,
) -> Corpus:
    """Build or load the section index for `source`.

    `source` may be a directory of books or a single file. `model_name` is a
    Hub id or a local model directory; see `embed.build_embedding_index`.
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f"no such file or directory: {source}")

    cfg = index_config or IndexConfig()
    if cache_dir is None:
        cache_dir = os.path.join(
            default_cache_root(), corpus_fingerprint(source, cfg.include_ext)
        )
    sec_path = os.path.join(cache_dir, "sections.json.gz")
    idx_path = os.path.join(cache_dir, "index.grx")
    emb_path = os.path.join(cache_dir, "index.emb.npz")

    cached = not rebuild and os.path.exists(sec_path) and os.path.exists(idx_path)
    if cached:
        sections = _load_sections(sec_path)
        index = GraphIndex.load(idx_path)
        if verbose:
            print(f"[graphroute] cache hit: {cache_dir}")
    else:
        if verbose:
            print(f"[graphroute] building {source} -> {cache_dir}")
        sections = collect_sections(
            source,
            include_ext=cfg.include_ext,
            max_bytes=cfg.max_file_bytes,
            on_error=(lambda p, e: print(f"[graphroute] skipped {p}: {e}"))
            if verbose
            else None,
        )
        if not sections:
            raise ValueError(
                f"no readable documents under {source}{_no_documents_hint(source, cfg)}"
            )
        docs = (
            Document(path=s.doc_path, text=s.full_text, size=len(s.full_text))
            for s in sections
        )
        index = build_index(docs, cfg, verbose=verbose)
        os.makedirs(cache_dir, exist_ok=True)
        _save_sections(sec_path, sections)
        index.save(idx_path)

    embeddings = None
    if embed:
        from .embed import EmbeddingIndex, build_embedding_index

        if not rebuild and os.path.exists(emb_path):
            embeddings = EmbeddingIndex.load(emb_path)
        else:
            embeddings = build_embedding_index(
                sections, index=index, embed_fn=embed_fn, model_name=model_name
            )
            if embeddings is not None:
                os.makedirs(cache_dir, exist_ok=True)
                embeddings.save(emb_path)
    return Corpus(
        sections=sections, index=index, embeddings=embeddings, cache_dir=cache_dir
    )


# -------------------------------------------------------------------- fusion


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = RRF_K
) -> List[Tuple[int, float]]:
    """Fuse ranked id lists into one, `score = Σ 1/(k + rank)`.

    Rank-based on purpose. BM25 scores are unbounded sums of idf terms and
    cosines live in [-1, 1]; combining them directly requires a normalisation
    and a weight, both of which are corpus-specific numbers that look principled
    and are not. RRF needs neither and degrades gracefully when one channel is
    missing.
    """
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


# ------------------------------------------------------------------ chunking


def paragraph_chunks(
    text: str, min_chars: int = CHUNK_MIN, max_chars: int = CHUNK_MAX
) -> List[Tuple[int, str]]:
    """Split into passage-sized chunks. Returns `(offset, text)` pairs.

    Blank lines first, because a paragraph break is the only structural signal
    that survives PDF extraction. Short paragraphs are merged forward until they
    reach `min_chars`; anything still over `max_chars` is cut at a sentence
    boundary where one exists nearby, and mid-text where one does not.
    """
    chunks: List[Tuple[int, str]] = []
    pos = 0
    buf: List[str] = []
    buf_start = 0
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            joined = "\n\n".join(buf).strip()
            if joined:
                chunks.append((buf_start, joined))
        buf, buf_len = [], 0

    for para in _PARA_SPLIT_RE.split(text):
        start = text.find(para, pos)
        if start < 0:
            start = pos
        pos = start + len(para)
        para = para.strip()
        if not para:
            continue
        if not buf:
            buf_start = start
        buf.append(para)
        buf_len += len(para) + 2
        if buf_len >= min_chars:
            flush()

    flush()

    out: List[Tuple[int, str]] = []
    for offset, chunk in chunks:
        while len(chunk) > max_chars:
            cut = chunk.rfind(". ", min_chars, max_chars)
            cut = cut + 1 if cut > 0 else max_chars
            out.append((offset, chunk[:cut].strip()))
            offset += cut
            chunk = chunk[cut:].lstrip()
        if chunk.strip():
            out.append((offset, chunk.strip()))
    return out


def _lexical_chunk_scores(
    chunks: Sequence[str], term_weights: Dict[str, float]
) -> np.ndarray:
    """Overlap between each chunk and the router's expanded query terms.

    Free: the router computed these weights on its way to picking sections.
    Length-normalised by sqrt so a long chunk cannot win on volume alone.
    """
    scores = np.zeros(len(chunks), dtype=np.float32)
    if not term_weights:
        return scores
    for i, chunk in enumerate(chunks):
        counts = count_terms(chunk)
        if not counts:
            continue
        hit = sum(term_weights.get(t, 0.0) * min(c, 3) for t, c in counts.items())
        scores[i] = hit / np.sqrt(max(len(counts), 1))
    return scores


def _normalize(x: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]; a flat vector contributes nothing rather than NaN."""
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# ---------------------------------------------------------------------- ask


def ask(
    source,
    question: str,
    passages: int = 8,
    top_sections: int = 50,
    cache_dir: Optional[str] = None,
    embed: bool = True,
    embed_fn: Optional[Callable] = None,
    model_name: Optional[str] = None,
    index_config: Optional[IndexConfig] = None,
    route_config: Optional[RouteConfig] = None,
    seed_expansion: bool = False,
    rebuild: bool = False,
    verbose: bool = False,
) -> List[Passage]:
    """Answer-ready passages for `question` over the corpus at `source`.

    `source` is a directory, a single file, or an already built `Corpus`.
    """
    corpus = (
        source
        if isinstance(source, Corpus)
        else build_corpus(
            source,
            cache_dir=cache_dir,
            index_config=index_config,
            embed=embed,
            embed_fn=embed_fn,
            model_name=model_name,
            rebuild=rebuild,
            verbose=verbose,
        )
    )
    return ask_corpus(
        corpus,
        question,
        passages=passages,
        top_sections=top_sections,
        embed_fn=embed_fn,
        route_config=route_config,
        seed_expansion=seed_expansion,
    )


def ask_corpus(
    corpus: Corpus,
    question: str,
    passages: int = 8,
    top_sections: int = 50,
    embed_fn: Optional[Callable] = None,
    route_config: Optional[RouteConfig] = None,
    seed_expansion: bool = False,
) -> List[Passage]:
    """`ask` against a corpus that is already built. Reuses everything.

    `seed_expansion` additionally seeds the router's graph walk from the
    embedding hits (see `embed.embedding_seed_fn`). Off by default: it helps
    when the question shares no vocabulary with the corpus and costs a little
    latency when it does.
    """
    index = corpus.index
    # The score floor's job — never hand the reader a plausible file about the
    # wrong subject — is done at passage level here, so the fusion input is the
    # full ranking. Cutting it twice would starve RRF of the second channel's
    # disagreements, which are the only reason to fuse.
    cfg = route_config or RouteConfig(score_floor=0.0)
    router = Router(index, cfg)

    query_vec = None
    emb = corpus.embeddings
    if emb is not None and len(emb):
        from .embed import (
            CUSTOM_MODEL,
            default_embedder,
            embed_query,
            embedding_available,
        )

        # Resolve the embedder once and hand the same callable to the question,
        # the seeds and the candidate chunks below.
        if embed_fn is None and emb.model_name == CUSTOM_MODEL:
            # Cosine between vectors from two different encoders is a number
            # with no meaning. Drop the channel rather than rank on noise.
            warnings.warn(
                "this embedding index was built with a custom embed_fn; pass "
                "the same embed_fn to query it. Falling back to lexical-only.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif embed_fn is None and embedding_available():
            embed_fn = default_embedder(emb.model_name)
        if embed_fn is not None:
            query_vec = embed_query(question, emb, embed_fn=embed_fn)

    seed_fn = None
    if seed_expansion and query_vec is not None:
        from .embed import embedding_seed_fn

        seed_fn = embedding_seed_fn(emb, query_vec)

    result = router.route(
        question, top_n=max(top_sections * 2, 100), seed_fn=seed_fn
    )
    lexical_ids = list(result.doc_ids)

    dense_ids: List[int] = []
    if query_vec is not None:
        rows, _ = emb.search(query_vec, k=max(top_sections * 2, 100))
        path_to_doc = {p: i for i, p in enumerate(index.paths)}
        dense_ids = [
            path_to_doc[emb.doc_paths[r]]
            for r in rows.tolist()
            if emb.doc_paths[r] in path_to_doc
        ]

    rankings = [r for r in (lexical_ids, dense_ids) if r]
    if not rankings:
        return []
    fused = reciprocal_rank_fusion(rankings)[:top_sections]

    term_weights = {t: w for t, w in result.expanded_terms}
    return _select_passages(
        corpus, [doc_id for doc_id, _ in fused], question, term_weights,
        passages, query_vec, embed_fn,
    )


def _select_passages(
    corpus: Corpus,
    doc_ids: Sequence[int],
    question: str,
    term_weights: Dict[str, float],
    passages: int,
    query_vec,
    embed_fn,
) -> List[Passage]:
    chunks: List[Tuple[Section, int, str]] = []
    for doc_id in doc_ids:
        section = corpus.section_of_doc(doc_id)
        if section is None:
            continue
        for offset, text in paragraph_chunks(section.full_text):
            chunks.append((section, offset, text))
    if not chunks:
        return []

    texts = [c[2] for c in chunks]
    lex = _normalize(_lexical_chunk_scores(texts, term_weights))

    # Chunk vectors are computed per query rather than cached: there are ~10x
    # more chunks than sections, they are only ever needed for the handful of
    # sections that survived fusion, and embedding a few hundred short strings
    # costs less than storing and invalidating a second matrix.
    dense = np.zeros(len(chunks), dtype=np.float32)
    if query_vec is not None and embed_fn is not None:
        from .embed import l2_normalize

        vecs = l2_normalize(embed_fn(texts))
        dense = _normalize(
            (vecs @ np.asarray(query_vec, np.float32)).astype(np.float32)
        )

    total = lex + dense
    take = min(passages, total.size)
    order = np.argpartition(-total, take - 1)[:take]
    order = order[np.argsort(-total[order])]

    out: List[Passage] = []
    for i in order.tolist():
        section, offset, text = chunks[i]
        start = section.page_of(offset)
        end = section.page_of(offset + len(text) - 1)
        out.append(
            Passage(
                text=text,
                breadcrumb=section.breadcrumb,
                book_path=section.book_path,
                pages=(start, max(start, end)),
                score=float(total[i]),
                section_index=section.index,
            )
        )
    return out
