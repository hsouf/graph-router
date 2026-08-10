"""An optional dense channel over sections.

The lexical router answers "which sections use these words". It cannot answer
"which sections are about this", and a question asked in a reader's vocabulary
("what is leukemia") rarely uses a book's vocabulary ("lymphoblastic
proliferation of the marrow"). One small embedding model closes that gap.

Three deliberate limits:

* **Optional.** `sentence-transformers` lives in the `[embed]` extra. Without
  it the pipeline runs lexical-only and says so. The core install stays
  numpy/scipy.
* **Pluggable.** `build_embedding_index(sections, embed_fn=...)` takes any
  `list[str] -> np.ndarray`. The default happens to be MiniLM.
* **Brute force.** An L2-normalised float16 matrix and one matmul. Exact top-k
  over 100k sections costs a few milliseconds and a hundred megabytes; an ANN
  index would add a dependency, a build step and an approximation to save time
  nobody is spending. Below ~1M rows this is simply the right answer.

What gets embedded is where the coupling to the lexical index pays: the
breadcrumb, the opening prose, *and* the section's highest tf-idf terms taken
straight out of the inverted index. The index already knows what makes a
section different from its neighbours; handing that to the embedder beats
embedding a truncated first page.
"""

from __future__ import annotations

import io
import json
import warnings
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
"""Downloaded from the Hugging Face Hub on first use and cached by
sentence-transformers. Pass a local directory instead to stay off the network
entirely, or supply your own `embed_fn` and skip the dependency altogether."""

CUSTOM_MODEL = "custom"
"""Recorded when the caller supplied an `embed_fn` and did not name a model.

Vectors are only comparable to other vectors from the same encoder. Without a
marker, a cached index built by a custom encoder would be reloaded later and
queried with MiniLM, and the cosines would be arithmetic performed on unrelated
number lines — no error, just a silently wrong ranking.
"""

TOP_TERMS = 15
"""Salient terms appended to each section's embedding text."""

EmbedFn = Callable[[List[str]], np.ndarray]


def embedding_available() -> bool:
    """True if the default embedder can be constructed."""
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=4)
def default_embedder(model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> EmbedFn:
    """A `list[str] -> np.ndarray` backed by sentence-transformers.

    Cached on the model name: loading MiniLM takes a second or two and a single
    `ask()` needs it three times (sections at build, the question, the candidate
    chunks). Paying that once per process rather than once per call is the
    difference between a fast query and a slow one.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "embeddings need sentence-transformers. Install with: "
            "pip install 'graphroute[embed]'"
        ) from exc

    model = SentenceTransformer(model_name)

    def encode(texts: List[str]) -> np.ndarray:
        return np.asarray(
            model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )

    return encode


# ------------------------------------------------------------------ vectors


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


@dataclass
class EmbeddingIndex:
    """L2-normalised section vectors, stored at half precision.

    float16 halves the footprint and costs nothing measurable in ranking
    quality — the vectors are already normalised, so the dynamic range is
    [-1, 1] where float16 has ~3 decimal digits, well below the noise floor of
    the model itself. Search promotes to float32 for the matmul.
    """

    vectors: np.ndarray                 # (n_sections, dim) float16, unit norm
    doc_paths: List[str]                # section doc ids, row-aligned
    model_name: str = DEFAULT_MODEL
    dim: int = 0

    def __post_init__(self):
        self.dim = int(self.vectors.shape[1]) if self.vectors.size else 0
        self._row_of = {p: i for i, p in enumerate(self.doc_paths)}

    def __len__(self) -> int:
        return len(self.doc_paths)

    def row_of(self, doc_path: str) -> int:
        return self._row_of.get(doc_path, -1)

    def search(self, query_vec: np.ndarray, k: int = 50):
        """Top-k rows by cosine similarity. Returns (row ids, scores)."""
        if not len(self) or query_vec is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        q = l2_normalize(np.asarray(query_vec, dtype=np.float32))[0]
        sims = (self.vectors.astype(np.float32) @ q).astype(np.float32)
        k = min(k, sims.size)
        part = np.argpartition(-sims, k - 1)[:k]
        order = part[np.argsort(-sims[part])]
        return order.astype(np.int64), sims[order]

    def similarities(self, query_vec: np.ndarray) -> np.ndarray:
        """Cosine of every row against `query_vec`."""
        if not len(self):
            return np.empty(0, dtype=np.float32)
        q = l2_normalize(np.asarray(query_vec, dtype=np.float32))[0]
        return (self.vectors.astype(np.float32) @ q).astype(np.float32)

    # ------------------------------------------------------------------ io

    def save(self, path: str) -> None:
        buf = io.BytesIO()
        np.savez_compressed(buf, vectors=self.vectors)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("vectors.npz", buf.getvalue())
            zf.writestr(
                "meta.json",
                json.dumps({"doc_paths": self.doc_paths, "model_name": self.model_name}),
            )

    @classmethod
    def load(cls, path: str) -> "EmbeddingIndex":
        with zipfile.ZipFile(path) as zf:
            arrays = np.load(io.BytesIO(zf.read("vectors.npz")))
            meta = json.loads(zf.read("meta.json"))
        return cls(
            vectors=arrays["vectors"],
            doc_paths=list(meta["doc_paths"]),
            model_name=meta.get("model_name", DEFAULT_MODEL),
        )


# ------------------------------------------------------------- salient terms


def salient_terms(index, doc_ids: Sequence[int], k: int = TOP_TERMS) -> List[List[str]]:
    """Each document's top-k terms by tf-idf, straight from the lexical index.

    This is the coupling that makes the embedding text worth more than the
    first page of prose: the inverted index has already worked out which terms
    distinguish this section from the rest of the corpus.
    """
    csr = index.postings.tocsr()
    idf = index.idf
    out: List[List[str]] = []
    for doc_id in doc_ids:
        if doc_id < 0:
            out.append([])
            continue
        row = csr[doc_id]
        if row.nnz == 0:
            out.append([])
            continue
        weights = row.data * idf[row.indices]
        take = min(k, weights.size)
        top = np.argpartition(-weights, take - 1)[:take]
        top = top[np.argsort(-weights[top])]
        out.append(
            [
                index.terms[int(t)]
                for t in row.indices[top]
                # Pseudo-terms are structure, not vocabulary: "link:ch12.pdf"
                # tells the embedder nothing about the subject.
                if not index.terms[int(t)].startswith("link:")
            ]
        )
    return out


def embedding_text(section, terms: Sequence[str]) -> str:
    """The string actually embedded for a section."""
    parts = [section.breadcrumb]
    if section.first_para:
        parts.append(" — " + section.first_para)
    if terms:
        parts.append(" | " + " ".join(terms))
    return "".join(parts)


# ---------------------------------------------------------------- building


def build_embedding_index(
    sections: Sequence,
    index=None,
    embed_fn: Optional[EmbedFn] = None,
    top_terms: int = TOP_TERMS,
    model_name: Optional[str] = None,
) -> Optional[EmbeddingIndex]:
    """Embed every section. Returns None if no embedder is available.

    `index` is the built `GraphIndex` over the same sections; when given, each
    section's salient terms are pulled from it and appended to the embedded
    text. Passing None still works and just embeds breadcrumb + first paragraph.

    `model_name` may be a Hub id or a local directory. It is recorded in the
    saved index so the same encoder is used at query time; supplying `embed_fn`
    without naming a model records `CUSTOM_MODEL` instead.
    """
    if not sections:
        return None
    resolved = model_name or (DEFAULT_MODEL if embed_fn is None else CUSTOM_MODEL)
    if embed_fn is None:
        if not embedding_available():
            warnings.warn(
                "sentence-transformers is not installed, so the embedding "
                "channel is disabled and retrieval is lexical-only. Install it "
                "with: pip install 'graphroute[embed]'",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        embed_fn = default_embedder(resolved)

    terms_per_section: List[List[str]] = [[]] * len(sections)
    if index is not None:
        row_of = {p: i for i, p in enumerate(index.paths)}
        doc_ids = [row_of.get(s.doc_path, -1) for s in sections]
        terms_per_section = salient_terms(index, doc_ids, k=top_terms)

    texts = [embedding_text(s, t) for s, t in zip(sections, terms_per_section)]
    vectors = l2_normalize(embed_fn(texts))
    if vectors.shape[0] != len(sections):
        raise ValueError(
            f"embed_fn returned {vectors.shape[0]} vectors for "
            f"{len(sections)} sections"
        )
    return EmbeddingIndex(
        vectors=vectors.astype(np.float16),
        doc_paths=[s.doc_path for s in sections],
        model_name=resolved,
    )


def embed_query(question: str, emb: EmbeddingIndex, embed_fn: Optional[EmbedFn] = None):
    """Vector for a question, using the same model the index was built with."""
    if embed_fn is None:
        if not embedding_available():
            return None
        embed_fn = default_embedder(emb.model_name)
    return l2_normalize(embed_fn([question]))[0]


def embedding_seed_fn(
    emb: EmbeddingIndex,
    query_vec: np.ndarray,
    top_sections: int = 20,
    terms_per_section: int = 4,
    min_similarity: float = 0.0,
):
    """A `Router` `seed_fn` that seeds the graph walk from embedding hits.

    The router's walk starts at the anchors of the query's *literal* terms, so
    a question phrased in the reader's vocabulary rather than the book's starts
    nowhere. This takes the sections the dense channel liked, reads off their
    dominant anchor terms, and offers those as extra seeds weighted by cosine
    similarity — the semantic seed selection the README has always pointed at
    as the clean insertion point.

    Off by default. It is a second opinion about where to start walking, and on
    a corpus where the lexical seeds are already right it only adds latency.
    """

    def seed_fn(index, literal_ids):
        rows, sims = emb.search(query_vec, k=top_sections)
        if not rows.size:
            return {}
        path_to_doc = {p: i for i, p in enumerate(index.paths)}
        csr = index.postings.tocsr()
        seeds: Dict[int, float] = {}
        for row, sim in zip(rows.tolist(), sims.tolist()):
            if sim <= min_similarity:
                continue
            doc_id = path_to_doc.get(emb.doc_paths[row], -1)
            if doc_id < 0:
                continue
            doc = csr[doc_id]
            if doc.nnz == 0:
                continue
            weights = doc.data * index.idf[doc.indices]
            take = min(terms_per_section, weights.size)
            top = np.argpartition(-weights, take - 1)[:take]
            for term_id in doc.indices[top].tolist():
                anchor = int(index.anchor_of[term_id])
                if anchor >= 0:
                    seeds[anchor] = max(seeds.get(anchor, 0.0), float(sim))
        return seeds

    return seed_fn
