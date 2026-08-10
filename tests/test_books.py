"""Sections, embeddings, fusion, and the end-to-end `ask` pipeline.

Every PDF here is built in-test by `synthetic_pdf`, so the suite exercises the
real pypdf extraction path rather than a hand-written string that never went
through a PDF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("pypdf", reason="PDF sectioning needs the [rich] extra")

from demo.synthetic_pdf import write_pdf, write_pdf_with_outline
from graphroute import ask, build_corpus
from graphroute.embed import (
    EmbeddingIndex,
    build_embedding_index,
    embedding_text,
    l2_normalize,
    salient_terms,
)
from graphroute.pipeline import (
    Passage,
    paragraph_chunks,
    reciprocal_rank_fusion,
)
from graphroute.sections import Section, collect_sections, extract_sections


# ------------------------------------------------------------------ fixtures


def _body(topic: str, n: int = 12):
    return [
        f"{topic} is discussed here in sentence {i} with enough words to matter."
        for i in range(n)
    ]


HEMA_PAGES = [
    ["Chapter 87 Erythrocyte Production"] + _body("erythropoiesis marrow iron"),
    ["Reticulocyte kinetics"] + _body("reticulocyte maturation counting"),
    ["Chapter 88 Acute Lymphoblastic Leukemia"]
    + [
        "Leukemia is a malignant proliferation of haematopoietic precursors",
        "that accumulate in the marrow and displace normal haematopoiesis.",
    ]
    + _body("lymphoblastic leukemia blasts childhood"),
    ["Treatment of leukemia"] + _body("induction consolidation maintenance"),
    ["Chapter 89 Congestion control"] + _body("congestion window bandwidth queue"),
]

OUTLINE = [
    ("Ch 87: Erythrocyte Production", 0, 0),
    ("Reticulocyte kinetics", 1, 1),
    ("Ch 88: Acute Lymphoblastic Leukemia", 2, 0),
    ("Treatment", 3, 1),
    ("Ch 89: Congestion control", 4, 0),
]


@pytest.fixture(scope="module")
def book_with_outline(tmp_path_factory):
    path = tmp_path_factory.mktemp("books") / "williams-hematology.pdf"
    write_pdf_with_outline(
        str(path), HEMA_PAGES, OUTLINE, title="Williams Hematology"
    )
    return str(path)


@pytest.fixture(scope="module")
def book_without_outline(tmp_path_factory):
    path = tmp_path_factory.mktemp("books") / "plain-hematology.pdf"
    write_pdf(str(path), HEMA_PAGES, title="Plain Hematology")
    return str(path)


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory):
    """A small shelf: one book with an outline, one without, one off-topic."""
    root = tmp_path_factory.mktemp("shelf")
    write_pdf_with_outline(
        str(root / "williams-hematology.pdf"),
        HEMA_PAGES,
        OUTLINE,
        title="Williams Hematology",
    )
    write_pdf(
        str(root / "clinical-pharmacology.pdf"),
        [
            ["Chapter 12 Antimetabolites"]
            + _body("methotrexate dihydrofolate thymidylate"),
            ["12.1 Purine analogues"] + _body("mercaptopurine thioguanine tpmt"),
        ],
        title="Clinical Pharmacology",
    )
    write_pdf(
        str(root / "computer-networks.pdf"),
        [
            ["Chapter 4 Routing protocols"] + _body("bgp ospf autonomous prefix"),
            ["4.1 Queue management"] + _body("codel sojourn buffer marking"),
        ],
        title="Computer Networks",
    )
    return str(root)


# ------------------------------------------------------------------ sections


def test_sections_from_outline(book_with_outline):
    sections = extract_sections(book_with_outline)
    assert sections, "outline should produce sections"
    assert all(s.tier == "outline" for s in sections)

    crumbs = [s.breadcrumb for s in sections]
    assert all(c.startswith("Williams Hematology › ") for c in crumbs)
    # Nesting is preserved: a sub-section names its parent chapter.
    assert any(
        c == "Williams Hematology › Ch 88: Acute Lymphoblastic Leukemia › Treatment"
        for c in crumbs
    ), crumbs

    leukemia = next(s for s in sections if "Ch 88" in s.breadcrumb and "›" not in
                    s.breadcrumb.split("Ch 88")[1])
    assert "leukemia" in leukemia.full_text.lower()
    assert leukemia.page_start == 3
    assert leukemia.first_para and "Chapter 88" not in leukemia.first_para


def test_sections_are_contiguous_and_lose_no_pages(book_with_outline):
    """Every page of the book must land in some section."""
    sections = extract_sections(book_with_outline)
    covered = set()
    for s in sections:
        covered.update(range(s.page_start, s.page_end + 1))
    assert covered == set(range(1, len(HEMA_PAGES) + 1))


def test_sections_from_headings_when_there_is_no_outline(book_without_outline):
    sections = extract_sections(book_without_outline)
    assert sections
    assert all(s.tier == "headings" for s in sections)
    titles = [s.title for s in sections]
    assert any("Chapter 88" in t for t in titles), titles
    assert any(s.breadcrumb.startswith("Plain Hematology › ") for s in sections)
    assert any("leukemia" in s.full_text.lower() for s in sections)


def test_sections_fall_back_to_page_windows(tmp_path):
    """No outline and no headings: honest `pp.X–Y` breadcrumbs, nothing invented."""
    path = tmp_path / "featureless.pdf"
    write_pdf(
        str(path),
        [_body("continuous prose with no structure whatsoever", 16) for _ in range(20)],
        title="Featureless",
    )
    sections = extract_sections(str(path))
    assert sections
    assert all(s.tier == "windows" for s in sections)
    assert sections[0].breadcrumb == "Featureless › pp.1–8"
    assert sections[0].page_start == 1 and sections[0].page_end == 8
    assert sections[-1].page_end == 20


def test_short_sections_are_merged_into_the_previous_one(tmp_path):
    path = tmp_path / "stubby.pdf"
    write_pdf_with_outline(
        str(path),
        [
            ["Chapter 1 Real content"] + _body("substantial discussion", 14),
            ["Chapter 2 Stub", "One line."],
            ["Chapter 3 Real again"] + _body("more substantial discussion", 14),
        ],
        [("Chapter 1", 0, 0), ("Chapter 2", 1, 0), ("Chapter 3", 2, 0)],
        title="Stubby",
    )
    sections = extract_sections(str(path))
    assert all(len(s.full_text) >= 200 for s in sections[1:]), [
        len(s.full_text) for s in sections
    ]
    assert any("One line." in s.full_text for s in sections), "stub text is kept"
    assert [s.index for s in sections] == list(range(len(sections)))


def test_book_title_falls_back_to_the_filename(tmp_path):
    path = tmp_path / "some_untitled-book.pdf"
    write_pdf(str(path), HEMA_PAGES)      # no /Title
    sections = extract_sections(str(path))
    assert all(s.breadcrumb.startswith("some untitled book") for s in sections)


def test_page_of_attributes_offsets_to_the_right_page(tmp_path):
    """A passage found deep in a multi-page section reports the page it is on."""
    path = tmp_path / "long-chapter.pdf"
    write_pdf_with_outline(
        str(path),
        [
            ["Chapter 1 Opening"] + _body("first page material", 12),
            _body("second page material", 12),
            _body("third page material", 12),
            ["Chapter 2 Next"] + _body("elsewhere entirely", 12),
        ],
        [("Chapter 1", 0, 0), ("Chapter 2", 3, 0)],
        title="Long",
    )
    section = extract_sections(str(path))[0]
    assert section.page_start == 1 and section.page_end == 3
    assert len(section.page_offsets) == 3

    assert section.page_of(0) == 1
    assert section.page_of(section.page_offsets[1]) == 2
    assert section.page_of(section.page_offsets[2]) == 3
    assert section.page_of(10 ** 9) == section.page_end

    # And the attribution is right for real text, not just for the offsets.
    where = section.full_text.index("third page material")
    assert section.page_of(where) == 3


def test_collect_sections_mixes_pdfs_and_plain_files(corpus_dir, tmp_path):
    (Path(corpus_dir) / "notes.md").write_text("# Notes\n\nleukemia notes here\n")
    sections = collect_sections(corpus_dir)
    assert any(s.book_path.endswith(".md") and s.tier == "file" for s in sections)
    assert any(s.book_path.endswith(".pdf") for s in sections)
    (Path(corpus_dir) / "notes.md").unlink()


def test_doc_path_is_stable_and_unique(book_with_outline):
    sections = extract_sections(book_with_outline)
    paths = [s.doc_path for s in sections]
    assert len(set(paths)) == len(paths)
    assert paths[0] == f"{book_with_outline}#s0"


def test_walk_sections_yields_one_document_per_section(book_with_outline):
    from graphroute.ingest import walk_sections

    docs = list(walk_sections(book_with_outline))
    sections = extract_sections(book_with_outline)
    assert len(docs) == len(sections)
    assert [d.path for d in docs] == [s.doc_path for s in sections]


# ------------------------------------------------------------------- fusion


def test_rrf_rewards_agreement_between_channels():
    lexical = [10, 20, 30]
    dense = [20, 40, 10]
    fused = reciprocal_rank_fusion([lexical, dense])
    ids = [i for i, _ in fused]
    # 20 is 2nd and 1st; 10 is 1st and 3rd. Both beat anything ranked once.
    assert ids[:2] == [20, 10]
    assert set(ids) == {10, 20, 30, 40}
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_with_a_single_channel_preserves_its_order():
    fused = reciprocal_rank_fusion([[7, 8, 9]])
    assert [i for i, _ in fused] == [7, 8, 9]


def test_rrf_ignores_empty_rankings():
    assert reciprocal_rank_fusion([[], []]) == []
    assert [i for i, _ in reciprocal_rank_fusion([[1, 2], []])] == [1, 2]


# ----------------------------------------------------------------- chunking


def test_paragraph_chunks_merge_up_to_the_minimum():
    text = "\n\n".join(["short one.", "short two.", "short three."])
    chunks = paragraph_chunks(text, min_chars=20, max_chars=900)
    assert len(chunks) < 3
    assert all(len(c) <= 900 for _, c in chunks)


def test_paragraph_chunks_split_long_text_at_sentence_boundaries():
    para = " ".join(f"Sentence number {i} says something." for i in range(200))
    chunks = paragraph_chunks(para, min_chars=300, max_chars=900)
    assert len(chunks) > 1
    assert all(len(c) <= 900 for _, c in chunks)
    assert sum(len(c) for _, c in chunks) > 0.9 * len(para)


def test_paragraph_chunk_offsets_point_into_the_source():
    text = "alpha paragraph here.\n\nbeta paragraph here.\n\ngamma paragraph here."
    for offset, chunk in paragraph_chunks(text, min_chars=1, max_chars=900):
        assert text[offset:offset + 5] == chunk[:5]


# --------------------------------------------------------------- embeddings


def _stub_embedder(dim: int = 24):
    """Deterministic bag-of-words hashing embedder.

    Not a good embedding — good enough to prove the plumbing is right without
    downloading a model, which is exactly what the pluggable `embed_fn` is for.
    """

    def encode(texts):
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, hash(word) % dim] += 1.0
        return out

    return encode


def test_embedding_index_roundtrip_and_search(tmp_path, book_with_outline):
    sections = extract_sections(book_with_outline)
    emb = build_embedding_index(sections, embed_fn=_stub_embedder())
    assert emb is not None
    assert emb.vectors.dtype == np.float16
    assert len(emb) == len(sections)
    norms = np.linalg.norm(emb.vectors.astype(np.float32), axis=1)
    assert norms == pytest.approx(np.ones(len(sections)), abs=1e-2)

    path = tmp_path / "index.emb.npz"
    emb.save(str(path))
    loaded = EmbeddingIndex.load(str(path))
    assert loaded.doc_paths == emb.doc_paths
    assert (loaded.vectors == emb.vectors).all()

    rows, scores = loaded.search(emb.vectors[0].astype(np.float32), k=3)
    assert rows[0] == 0                       # a vector is nearest to itself
    assert scores[0] == pytest.approx(1.0, abs=1e-2)
    assert len(rows) == 3


def test_embedding_text_includes_breadcrumb_and_salient_terms():
    section = Section(
        book_path="/b.pdf",
        breadcrumb="Book › Ch 1",
        page_start=1,
        page_end=2,
        title="Ch 1",
        first_para="opening prose",
        full_text="opening prose and more",
    )
    text = embedding_text(section, ["leukemia", "marrow"])
    assert text.startswith("Book › Ch 1")
    assert "opening prose" in text
    assert "leukemia marrow" in text


def test_salient_terms_come_from_the_lexical_index(corpus_dir):
    corpus = build_corpus(corpus_dir, cache_dir=None, embed=False, rebuild=True)
    index = corpus.index
    target = next(s for s in corpus.sections if "leukemia" in s.full_text.lower())
    doc_id = index.paths.index(target.doc_path)
    terms = salient_terms(index, [doc_id], k=15)[0]
    assert terms
    assert "leukemia" in terms
    assert not any(t.startswith("link:") for t in terms)


def test_l2_normalize_handles_zero_vectors():
    out = l2_normalize(np.zeros((2, 4), dtype=np.float32))
    assert np.isfinite(out).all()
    assert out.sum() == 0


def test_build_embedding_index_returns_none_without_a_backend(monkeypatch):
    import graphroute.embed as E

    monkeypatch.setattr(E, "embedding_available", lambda: False)
    section = Section("/b.pdf", "Book", 1, 1, "t", "p", "full text here")
    with pytest.warns(RuntimeWarning, match="sentence-transformers"):
        assert E.build_embedding_index([section]) is None


# --------------------------------------------------------------------- ask


def test_ask_returns_passages_from_the_right_section(corpus_dir, tmp_path):
    found = ask(
        corpus_dir,
        "what is leukemia",
        passages=4,
        cache_dir=str(tmp_path / "cache"),
        embed=False,
    )
    assert found
    assert all(isinstance(p, Passage) for p in found)
    top = found[0]
    assert "leukemia" in top.text.lower()
    assert "Acute Lymphoblastic Leukemia" in top.breadcrumb
    assert top.book_path.endswith("williams-hematology.pdf")
    assert top.pages[0] >= 1
    assert top.header().startswith("[Williams Hematology › ")
    assert top.header().rstrip("]").endswith("p.3") or "pp." in top.header()


def test_ask_routes_a_different_question_to_a_different_book(corpus_dir, tmp_path):
    cache = str(tmp_path / "cache")
    networks = ask(corpus_dir, "queue management and buffer marking",
                   passages=3, cache_dir=cache, embed=False)
    assert networks
    assert networks[0].book_path.endswith("computer-networks.pdf"), (
        networks[0].breadcrumb
    )


def test_ask_respects_the_passage_count(corpus_dir, tmp_path):
    found = ask(corpus_dir, "leukemia treatment", passages=2,
                cache_dir=str(tmp_path / "cache"), embed=False)
    assert len(found) <= 2


def test_ask_uses_the_cache_on_the_second_call(corpus_dir, tmp_path):
    cache = tmp_path / "cache"
    ask(corpus_dir, "leukemia", passages=2, cache_dir=str(cache), embed=False)
    assert (cache / "sections.json.gz").exists()
    assert (cache / "index.grx").exists()

    marker = cache / "index.grx"
    stamp = marker.stat().st_mtime_ns
    ask(corpus_dir, "leukemia", passages=2, cache_dir=str(cache), embed=False)
    assert marker.stat().st_mtime_ns == stamp, "second call must not rebuild"


def test_ask_with_an_embedding_channel(corpus_dir, tmp_path):
    """Both channels on, fused. The stub embedder stands in for MiniLM."""
    found = ask(
        corpus_dir,
        "what is leukemia",
        passages=4,
        cache_dir=str(tmp_path / "cache"),
        embed=True,
        embed_fn=_stub_embedder(),
    )
    assert found
    assert any("leukemia" in p.text.lower() for p in found)
    assert (tmp_path / "cache" / "index.emb.npz").exists()


def test_custom_embedder_is_recorded_as_custom(corpus_dir, tmp_path):
    from graphroute.embed import CUSTOM_MODEL, DEFAULT_MODEL

    corpus = build_corpus(
        corpus_dir, cache_dir=str(tmp_path / "c"), embed=True,
        embed_fn=_stub_embedder(),
    )
    assert corpus.embeddings.model_name == CUSTOM_MODEL

    named = build_corpus(
        corpus_dir, cache_dir=str(tmp_path / "d"), embed=True,
        embed_fn=_stub_embedder(), model_name="my-local-model",
    )
    assert named.embeddings.model_name == "my-local-model"
    assert DEFAULT_MODEL not in (corpus.embeddings.model_name,
                                 named.embeddings.model_name)


def test_reloading_a_custom_index_without_its_embedder_refuses_to_guess(
    corpus_dir, tmp_path
):
    """Cosine across two different encoders is meaningless, not merely worse.

    The cached index must not be queried with whatever model happens to be the
    default, because that produces a confident ranking built on noise.
    """
    from graphroute.pipeline import ask_corpus

    cache = str(tmp_path / "c")
    build_corpus(corpus_dir, cache_dir=cache, embed=True, embed_fn=_stub_embedder())

    reloaded = build_corpus(corpus_dir, cache_dir=cache, embed=True)
    with pytest.warns(RuntimeWarning, match="custom embed_fn"):
        found = ask_corpus(reloaded, "what is leukemia", passages=3)
    assert found, "must still answer lexically rather than fail"
    assert any("leukemia" in p.text.lower() for p in found)

    # Handed the right encoder, it uses the dense channel without complaint.
    import warnings as W

    with W.catch_warnings():
        W.simplefilter("error")
        assert ask_corpus(
            reloaded, "what is leukemia", passages=3, embed_fn=_stub_embedder()
        )


def test_ask_degrades_to_lexical_when_embeddings_are_unavailable(
    corpus_dir, tmp_path, monkeypatch
):
    import graphroute.embed as E

    monkeypatch.setattr(E, "embedding_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="lexical-only"):
        found = ask(corpus_dir, "what is leukemia", passages=3,
                    cache_dir=str(tmp_path / "cache"), embed=True)
    assert found
    assert "leukemia" in found[0].text.lower()


def test_ask_on_a_single_file(book_with_outline, tmp_path):
    found = ask(book_with_outline, "what is leukemia", passages=2,
                cache_dir=str(tmp_path / "cache"), embed=False)
    assert found
    assert found[0].book_path == book_with_outline


def test_ask_on_a_missing_source_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file or directory"):
        ask(str(tmp_path / "nope"), "anything", cache_dir=str(tmp_path / "c"))


def test_empty_pdf_corpus_without_pypdf_says_why(tmp_path, monkeypatch):
    """"No readable documents" is true and useless when the cause is a missing
    optional dependency."""
    import builtins

    import graphroute.pipeline as P

    write_pdf(str(tmp_path / "book.pdf"), HEMA_PAGES, title="Book")
    real_import = builtins.__import__

    def no_pypdf(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("pypdf is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pypdf)
    with pytest.raises(ValueError, match=r"pypdf is not installed.*graphroute\[rich\]"):
        P.build_corpus(str(tmp_path), cache_dir=str(tmp_path / "c"), embed=False)


def test_collect_sections_skips_unreadable_files(tmp_path):
    """One corrupt download must not take the whole shelf down with it."""
    (tmp_path / "broken.pdf").write_bytes(b"this is definitely not a pdf")
    write_pdf(str(tmp_path / "good.pdf"), HEMA_PAGES, title="Good")

    errors = []
    sections = collect_sections(str(tmp_path), on_error=lambda p, e: errors.append(p))
    assert sections
    assert all(s.book_path.endswith("good.pdf") for s in sections)
    assert errors and errors[0].endswith("broken.pdf")


# ------------------------------------------------ embedding-seeded expansion


def test_seed_fn_adds_ppr_seeds_to_the_walk(corpus_dir, tmp_path):
    """A seed_fn can start the walk somewhere the literal terms never reach."""
    from graphroute.router import RouteConfig, Router

    corpus = build_corpus(corpus_dir, cache_dir=str(tmp_path / "c"), embed=False)
    index = corpus.index
    router = Router(index, RouteConfig(top_n=50, score_floor=0.0))

    question = "leukemia"
    plain = router.route(question)
    reached = dict(plain.expanded_terms)

    # Any anchor the literal walk did not reach: seeding it must pull it in.
    unreached = [
        (a, index.terms[int(t)])
        for a, t in enumerate(index.anchor_terms.tolist())
        if index.terms[int(t)] not in reached
    ]
    assert unreached, "corpus is too small for this test to mean anything"
    anchor, term = unreached[0]

    seeded = router.route(question, seed_fn=lambda idx, lit: {anchor: 5.0})
    assert term in dict(seeded.expanded_terms)
    assert term not in reached
    assert len(seeded.expanded_terms) >= len(plain.expanded_terms)


def test_seed_fn_ignores_out_of_range_anchors(corpus_dir, tmp_path):
    from graphroute.router import RouteConfig, Router

    corpus = build_corpus(corpus_dir, cache_dir=str(tmp_path / "c"), embed=False)
    router = Router(corpus.index, RouteConfig(top_n=10))
    res = router.route("leukemia", seed_fn=lambda idx, lit: {10 ** 9: 1.0, -3: 2.0})
    assert res.files, "a bad seed must not break the query"


def test_embedding_seed_fn_returns_anchor_seeds(corpus_dir, tmp_path):
    from graphroute.embed import embedding_seed_fn

    corpus = build_corpus(
        corpus_dir, cache_dir=str(tmp_path / "c"), embed=True,
        embed_fn=_stub_embedder(),
    )
    emb = corpus.embeddings
    query_vec = emb.vectors[0].astype(np.float32)
    seeds = embedding_seed_fn(emb, query_vec, top_sections=5)(corpus.index, [])
    assert seeds
    assert all(0 <= a < corpus.index.adj.shape[0] for a in seeds)
    assert all(w > 0 for w in seeds.values())


def test_ask_with_seed_expansion_still_finds_the_right_section(corpus_dir, tmp_path):
    found = ask(
        corpus_dir,
        "what is leukemia",
        passages=3,
        cache_dir=str(tmp_path / "cache"),
        embed=True,
        embed_fn=_stub_embedder(),
        seed_expansion=True,
    )
    assert found
    assert any("leukemia" in p.text.lower() for p in found)


def test_corpus_fingerprint_changes_when_a_file_changes(corpus_dir, tmp_path):
    from graphroute.pipeline import corpus_fingerprint

    before = corpus_fingerprint(corpus_dir)
    assert before == corpus_fingerprint(corpus_dir)

    extra = Path(corpus_dir) / "extra.md"
    extra.write_text("a new document appears")
    try:
        assert corpus_fingerprint(corpus_dir) != before
    finally:
        extra.unlink()
    assert corpus_fingerprint(corpus_dir) == before
