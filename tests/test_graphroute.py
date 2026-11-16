from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.corpus import make_corpus
from graphroute import build_index
from graphroute.graph import (
    cooccurrence,
    frontier_ppr,
    label_propagation,
    prune_topk,
)
from graphroute.index import GraphIndex, IndexConfig
from graphroute.ingest import Document, extract_links
from graphroute.router import RouteConfig, Router, bm25_scores, full_scan_baseline
from graphroute.text import query_terms, split_identifier


# --------------------------------------------------------------------- text


def test_split_identifier():
    assert split_identifier("refreshOAuthToken") == ["refresh", "oauth", "token"]
    assert split_identifier("retry_after_seconds") == ["retry", "after", "seconds"]
    assert split_identifier("token") == []


def test_query_terms_dedupes_and_strips_stopwords():
    terms = query_terms("How does the token refresh token work")
    assert "the" not in terms and "does" not in terms
    assert terms.count("token") == 1


def test_extract_links():
    text = "see [spec](docs/spec.md)\nimport auth.service\n"
    assert "docs/spec.md" in extract_links(text)
    assert "auth.service" in extract_links(text)


# -------------------------------------------------------------------- graph


def test_prune_topk_bounds_degree_and_keeps_largest():
    dense = np.array([[0, 5, 3, 1], [2, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
    out = prune_topk(sp.csr_matrix(dense), 2).toarray()
    assert (out[0] == [0, 5, 3, 0]).all()
    assert (out[1] == [2, 0, 0, 0]).all()
    assert out[2].sum() == 0
    assert (np.diff(sp.csr_matrix(out).indptr) <= 2).all()


def test_label_propagation_finds_disconnected_blocks():
    blocks = sp.block_diag([np.ones((4, 4)), np.ones((5, 5))], format="csr")
    labels = label_propagation(blocks.astype(np.float32))
    assert len(set(labels[:4].tolist())) == 1
    assert len(set(labels[4:].tolist())) == 1
    assert labels[0] != labels[4]


def test_frontier_ppr_prefers_near_nodes():
    # path graph 0-1-2-3-4, seeded at 0
    adj = sp.csr_matrix(np.eye(5, k=1) + np.eye(5, k=-1), dtype=np.float32)
    nodes, scores = frontier_ppr(adj, {0: 1.0}, hops=4)
    rank = {int(n): i for i, n in enumerate(nodes)}
    assert rank[0] < rank[1] < rank[2]


def test_frontier_ppr_respects_beam():
    star = sp.csr_matrix(np.ones((1, 50)), dtype=np.float32)
    star = sp.vstack([star, sp.csr_matrix((49, 50), dtype=np.float32)]).tocsr()
    nodes, _ = frontier_ppr(star, {0: 1.0}, hops=2, beam=5)
    assert len(nodes) <= 1 + 5 * 2


def test_frontier_ppr_empty_seeds():
    adj = sp.csr_matrix(np.eye(3), dtype=np.float32)
    nodes, scores = frontier_ppr(adj, {})
    assert nodes.size == 0 and scores.size == 0


# -------------------------------------------------------------------- index


@pytest.fixture(scope="module")
def small_corpus():
    return make_corpus(n_files=2_000, n_topics=20, n_queries=40, seed=3)


@pytest.fixture(scope="module")
def small_index(small_corpus):
    return build_index(iter(small_corpus.docs), IndexConfig())


def test_index_shapes(small_index):
    idx = small_index
    assert idx.n_docs == len(idx.paths)
    assert idx.postings.shape == (idx.n_docs, len(idx.terms))
    assert idx.doc_topic.shape == (idx.n_docs,)
    assert idx.adj.shape[0] == idx.anchor_terms.size
    assert idx.stats.n_edges > 0


def test_index_drops_corpus_stopwords(small_index, small_corpus):
    """Terms present in a large fraction of files must not become anchors."""
    common = [t for t in small_corpus.common_terms if t in small_index.term_id]
    assert common, "generator should produce common terms"
    anchored = [t for t in common if small_index.anchor_of[small_index.term_id[t]] >= 0]
    assert len(anchored) / len(common) < 0.2


def test_index_roundtrip(small_index, tmp_path):
    path = tmp_path / "idx.grx"
    small_index.save(str(path))
    loaded = GraphIndex.load(str(path))
    assert loaded.paths == small_index.paths
    assert loaded.terms == small_index.terms
    assert (loaded.doc_topic == small_index.doc_topic).all()
    assert (loaded.adj != small_index.adj).nnz == 0
    assert loaded.doc_topics.shape == small_index.doc_topics.shape
    assert (loaded.doc_topics != small_index.doc_topics).nnz == 0
    assert (loaded.doc_untopiced == small_index.doc_untopiced).all()


# ------------------------------------------------------- link pseudo-terms


def test_link_target_emits_its_own_pseudo_term():
    """A -> B must put the *same* pseudo-term in both documents' postings.

    Without the self-term, `link:b.md` occurs only in the documents citing
    b.md, so b.md never shares it with its own citers and the graph can learn
    co-citation but never citer -> citee.
    """
    docs = [
        Document(path="/docs/a.md", text="alpha beta gamma delta",
                 links=["/docs/b.md"]),
        Document(path="/docs/b.md", text="epsilon zeta eta theta"),
    ]
    index = build_index(iter(docs), IndexConfig(min_anchors=1))
    assert "link:b.md" in index.term_id
    tid = index.term_id["link:b.md"]
    holders = {index.paths[d] for d in index.postings[:, tid].tocoo().row}
    assert holders == {"/docs/a.md", "/docs/b.md"}


# ------------------------------------------------------ npmi support discount


def _always_together(n_docs: int, joint: int) -> sp.csr_matrix:
    """Two terms appearing in `joint` documents, always together, plus filler.

    NPMI is exactly 1.0 for this shape regardless of `joint`, which is the
    whole problem: support has to come from somewhere else.
    """
    rows, cols = [], []
    for d in range(joint):
        rows += [d, d]
        cols += [0, 1]
    for d in range(joint, n_docs):        # filler so df/n_docs stays fixed
        rows.append(d)
        cols.append(2)
    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_docs, 3)
    )


def test_npmi_support_discount_is_monotone_in_joint_count():
    # Identical probabilities in both corpora, hence identical raw NPMI (1.0):
    # only the amount of support differs, and more support must never weigh
    # less than less support.
    raw_weak = cooccurrence(_always_together(100, 2), support_saturation=0)[0, 1]
    raw_strong = cooccurrence(_always_together(400, 8), support_saturation=0)[0, 1]
    assert raw_weak == pytest.approx(raw_strong, rel=1e-5) == pytest.approx(1.0)

    weak = cooccurrence(_always_together(100, 2))[0, 1]
    strong = cooccurrence(_always_together(400, 8))[0, 1]
    assert strong >= weak
    assert strong > weak, "the discount must actually separate them"
    assert strong == pytest.approx(1.0, rel=1e-5)
    assert weak == pytest.approx(np.log1p(2) / np.log1p(8), rel=1e-5)


def test_npmi_support_discount_saturates_at_one():
    """Well-supported pairs keep their full NPMI: the factor only ever cuts."""
    w = cooccurrence(_always_together(2000, 500))[0, 1]
    assert 0 < w <= 1.0 + 1e-6


# ------------------------------------------------------- soft topic membership


def test_soft_topics_admit_multi_topic_documents():
    """A document straddling two topics must pass the gate for either one."""
    from graphroute.index import _soft_doc_topics

    mass = sp.csr_matrix(np.array([
        [10.0, 9.0, 0.5],     # two real topics, one incidental
        [10.0, 0.0, 0.0],     # single topic
    ], dtype=np.float32))
    membership = _soft_doc_topics(mass, max_topics=3, min_share=0.15)
    assert set(membership[0].indices.tolist()) == {0, 1}   # 0.5 is below 15%
    assert set(membership[1].indices.tolist()) == {0}


def test_soft_topics_cap_at_max_doc_topics():
    from graphroute.index import _soft_doc_topics

    mass = sp.csr_matrix(np.array([[5.0, 4.0, 3.0, 2.0]], dtype=np.float32))
    membership = _soft_doc_topics(mass, max_topics=3, min_share=0.0)
    assert membership.nnz == 3
    assert set(membership[0].indices.tolist()) == {0, 1, 2}


def test_multi_topic_doc_passes_the_gate_for_both_of_its_topics():
    """End to end: a document mixing two vocabularies is reachable from both.

    The argmax projection made it a member of exactly one community, so half
    the queries that should have found it were gated out before scoring.
    """
    docs = _mini_corpus()
    blend = " ".join(
        ["oauth token refresh bearer session"] * 12
        + ["invoice ledger charge subscription proration"] * 12
    )
    docs.append(Document(path="/mixed/auth-billing.md", text=blend))
    index = build_index(iter(docs))

    doc_id = index.paths.index("/mixed/auth-billing.md")
    assert index.doc_topics[doc_id].nnz >= 2, "should belong to both topics"

    router = Router(index, RouteConfig(top_n=200))
    for query in ("bearer token session refresh", "invoice ledger proration charge"):
        res = router.route(query)
        assert res.topics, "the gate must actually be active for this to mean anything"
        gate = index.topic_gate_mask(res.topics)
        assert gate[doc_id], (query, index.doc_topics[doc_id].indices, res.topics)


def test_anchorless_documents_always_pass_the_gate():
    """Docs with no anchors have no topics, and must not be gated out.

    Parking them in topic 0 made them reachable only by queries that happened
    to select topic 0 — a silent black hole in the corpus.
    """
    docs = _mini_corpus()
    docs.append(Document(path="/odd/hapax.md", text="zzqq wibble frobnicate " * 3))
    index = build_index(iter(docs))

    doc_id = index.paths.index("/odd/hapax.md")
    assert index.doc_untopiced[doc_id], "hapax-only doc should have no topics"
    for topics in ([0], [1], [0, 2], list(range(index.n_topics))):
        assert index.topic_gate_mask(topics)[doc_id]

    res = Router(index, RouteConfig(top_n=50)).route("wibble frobnicate")
    assert "/odd/hapax.md" in res.files


def test_communities_align_with_ground_truth(small_index, small_corpus):
    """Discovered topics should be near-pure w.r.t. generated topics."""
    gold = {d.path: t for d, t in zip(small_corpus.docs, small_corpus.doc_topic)}
    truth = np.array([gold[p] for p in small_index.paths])
    purity = 0
    for topic in range(small_index.n_topics):
        members = truth[small_index.doc_topic == topic]
        if members.size:
            purity += np.bincount(members).max()
    assert purity / truth.size > 0.85


# ------------------------------------------------------------------- router


def test_route_returns_files_and_respects_top_n(small_index):
    router = Router(small_index, RouteConfig(top_n=10))
    res = router.route("nonsense-term-that-does-not-exist")
    assert res.files == []

    query = small_index.terms[int(small_index.anchor_terms[0])]
    res = router.route(query)
    assert len(res.files) <= 10
    assert res.files_touched <= router.config.budget
    assert all(f in small_index.paths for f in res.files)


def test_top_n_override_does_not_mutate_the_shared_config(small_index):
    """`route(top_n=5)` must not cap every later query on the same Router.

    With no `**overrides`, `cfg` *was* `self.config`, so assigning `top_n` onto
    it permanently rewrote the router's configuration.
    """
    router = Router(small_index, RouteConfig(top_n=25))
    query = " ".join(small_index.terms[int(t)] for t in small_index.anchor_terms[:4])

    baseline = router.route(query)
    assert len(baseline.files) > 5, "need headroom for the assertion to bite"

    narrowed = router.route(query, top_n=5)
    assert len(narrowed.files) <= 5
    assert router.config.top_n == 25

    after = router.route(query)
    assert len(after.files) == len(baseline.files)
    assert after.files == baseline.files


def test_overrides_do_not_leak_into_the_shared_config(small_index):
    router = Router(small_index, RouteConfig(top_n=25, budget=4000))
    router.route("anything at all", budget=10, top_n=2)
    assert router.config.budget == 4000
    assert router.config.top_n == 25


def test_route_respects_budget(small_index, small_corpus):
    router = Router(small_index, RouteConfig(top_n=10, budget=50))
    for query, _ in small_corpus.queries[:10]:
        assert router.route(query).files_touched <= 50


@pytest.mark.filterwarnings("ignore:no anchor terms")
def test_budget_counts_only_fresh_documents():
    """Re-scoring an already-touched doc is free and must not spend budget.

    Every document here contains both query terms, so the second term's posting
    list is entirely docs the first term already touched. Counting them again
    declared the budget exhausted after touching 30 of 30 files.
    """
    docs = [
        Document(path=f"/d{i:03d}.md", text="alpha beta gamma delta epsilon")
        for i in range(30)
    ]
    index = build_index(iter(docs), IndexConfig(min_anchors=1, max_df_ratio=0.9))
    res = Router(index, RouteConfig(top_n=50, budget=30)).route("alpha beta")
    assert res.files_touched == 30
    assert not res.budget_exhausted


@pytest.mark.filterwarnings("ignore:no anchor terms")
def test_budget_still_binds_on_genuinely_fresh_documents():
    docs = [
        Document(path=f"/d{i:03d}.md", text=f"alpha term{i} gamma")
        for i in range(50)
    ]
    index = build_index(iter(docs), IndexConfig(min_anchors=1, max_df_ratio=0.9))
    res = Router(index, RouteConfig(top_n=50, budget=20, df_ceiling_ratio=1.0)).route(
        "alpha"
    )
    assert res.files_touched == 20
    assert res.budget_exhausted


def _common_term_corpus():
    """1,000 files of filler. 'leukemia' is in files 0-99 (10%, above the 5%
    df ceiling); 'imatinib' is in files 90-109 (2%). Files 90-99 have both."""
    rng = np.random.default_rng(0)
    filler = [f"filler{i}" for i in range(400)]
    docs = []
    for i in range(1000):
        words = list(rng.choice(filler, 80))
        if i < 100:
            words += ["leukemia"] * 3
        if 90 <= i < 110:
            words += ["imatinib"] * 2
        docs.append(Document(path=f"/d/{i:04d}.txt", text=" ".join(words)))
    return build_index(iter(docs), IndexConfig())


def test_query_terms_above_the_df_ceiling_still_score():
    """A common query word must still count for the files that were touched.

    'leukemia' is too common to walk, so only 'imatinib' brings files in. Before
    the rescore step 'leukemia' then contributed nothing, and the ten files
    matching *both* words ranked below the ten matching only 'imatinib' — they
    are longer, and BM25's length normalisation charged them for it.
    """
    index = _common_term_corpus()
    res = Router(index, RouteConfig(score_floor=0.0)).route("leukemia imatinib", top_n=10)
    both = {f"/d/{i:04d}.txt" for i in range(90, 100)}
    assert set(res.files) == both
    # And the ranking agrees with exhaustive BM25 on the same query.
    ids, _ = full_scan_baseline(index, "leukemia imatinib", top_n=10)
    assert set(res.doc_ids) == set(ids.tolist())


def test_rescore_does_not_admit_new_files():
    """Scoring the common term over touched files must not widen the walk."""
    index = _common_term_corpus()
    res = Router(index, RouteConfig(score_floor=0.0)).route("leukemia imatinib", top_n=50)
    assert res.files_touched == 20
    assert {int(p[3:7]) for p in res.files} <= set(range(90, 110))


def test_router_state_is_clean_between_queries(small_index, small_corpus):
    """The scratch buffers must be fully reset, or scores leak across queries."""
    router = Router(small_index, RouteConfig(top_n=20))
    first = router.route(small_corpus.queries[0][0])
    for query, _ in small_corpus.queries[1:6]:
        router.route(query)
    again = router.route(small_corpus.queries[0][0])
    assert again.files == first.files
    assert again.scores == pytest.approx(first.scores)
    assert not router._seen.any()
    assert not router._acc.any()
    assert not router._rejected.any()


def test_routing_touches_far_fewer_files_than_full_scan(small_index, small_corpus):
    router = Router(small_index, RouteConfig(top_n=50))
    touched = [router.route(q).files_touched for q, _ in small_corpus.queries[:40]]
    assert np.mean(touched) < 0.35 * small_index.n_docs


def test_routing_precision_beats_chance(small_index, small_corpus):
    gold = {d.path: t for d, t in zip(small_corpus.docs, small_corpus.doc_topic)}
    truth = np.array([gold[p] for p in small_index.paths])
    router = Router(small_index, RouteConfig(top_n=25))

    hits = []
    for query, gold_topic in small_corpus.queries:
        res = router.route(query)
        if res.doc_ids:
            hits.append(float((truth[res.doc_ids] == gold_topic).mean()))
    assert np.mean(hits) > 0.8


def test_graph_expansion_adds_terms_beyond_the_query(small_index, small_corpus):
    router = Router(small_index, RouteConfig(top_n=10))
    res = router.route(small_corpus.queries[0][0])
    assert len(res.expanded_terms) > len(res.seed_terms)


def test_full_scan_baseline_agrees_on_easy_query(small_index):
    """With the gate off and no budget, routing must not lose the true top hits.

    Compared on score rather than on document identity: the synthetic corpus
    produces heavy ties, so which of several equally-scoring documents surfaces
    is arbitrary and not a property worth asserting.
    """
    term = small_index.terms[int(small_index.anchor_terms[5])]
    router = Router(
        small_index,
        RouteConfig(top_n=10, budget=10**9, use_topic_gate=False,
                    expand_terms=1, literal_boost=1.0, df_ceiling_ratio=1.0),
    )
    routed = router.route(term)
    reference = bm25_scores(small_index, term)
    ids, _ = full_scan_baseline(small_index, term, top_n=10)

    assert len(routed.doc_ids) == 10
    assert reference[routed.doc_ids] == pytest.approx(reference[ids], rel=1e-5)


def test_empty_corpus_raises():
    with pytest.raises(ValueError):
        build_index(iter([Document(path="a", text="")]))


# ---------------------------------------------------- degenerate corpora
# Regression tests for bugs found by running the CLI on a real directory
# rather than on the generator, which is friendlier than reality.


def _mini_corpus(n_per_topic=40, n_topics=4, seed=11):
    import random

    rng = random.Random(seed)
    vocabs = {
        "auth": "oauth token refresh bearer session cookie login credential scope grant".split(),
        "billing": "invoice ledger charge tenant subscription proration refund coupon tax receipt".split(),
        "search": "shard replica cursor ranking planner recall posting segment merge analyzer".split(),
        "storage": "bucket object replication checksum compaction snapshot tier manifest block volume".split(),
    }
    common = "the this that with from will should when where system value config service".split()
    docs = []
    for name, vocab in list(vocabs.items())[:n_topics]:
        for i in range(n_per_topic):
            words = [
                rng.choice(vocab) if rng.random() < 0.5 else rng.choice(common)
                for _ in range(120)
            ]
            docs.append(Document(path=f"/{name}/{name}{i:03d}.md", text=" ".join(words)))
    return docs


def test_small_corpus_still_builds_a_graph():
    """With few topics a discriminative term legitimately sits in ~25% of files.

    A fixed 15% df ceiling discards exactly those terms and leaves no graph, so
    the ceiling has to relax on small corpora.
    """
    index = build_index(iter(_mini_corpus()))
    assert index.stats.n_anchors > 0
    assert index.stats.n_edges > 0
    assert index.n_topics >= 3


def test_small_corpus_routes_to_the_right_topic():
    index = build_index(iter(_mini_corpus()))
    router = Router(index, RouteConfig(top_n=5))
    for query, expected in [
        ("how does the bearer token get rotated", "/auth/"),
        ("refund a customer subscription charge", "/billing/"),
        ("shard replica ranking planner", "/search/"),
    ]:
        files = router.route(query).files
        assert files, query
        assert all(expected in f for f in files), (query, files[:3])


# ------------------------------------------- score floor / context poisoning


POISON_QUERY = "acore0 acore1 acore2 acore3 sharedterm"


def _poison_corpus(seed=5):
    """40 genuinely relevant files, 15 that share exactly one query term.

    The B files are the shape of a context-poisoning candidate: they match the
    query on one real, rare, high-idf term and nothing else. They are relevant
    enough to be retrieved and wrong enough to mislead whatever reads them.
    Measured score ratio is ~0.24 for B against ~0.77-1.00 for A, so a floor
    anywhere in (0.24, 0.77) separates them cleanly.
    """
    import random

    rng = random.Random(seed)
    filler = [f"filler{i}" for i in range(400)]
    docs = []
    for i in range(40):
        words = ["acore0", "acore1", "acore2", "acore3"] + [
            rng.choice(filler) for _ in range(80)
        ]
        docs.append(Document(path=f"/a/a{i:03d}.md", text=" ".join(words)))
    for i in range(15):
        words = (
            ["sharedterm"]
            + [f"beta{rng.randrange(50)}" for _ in range(20)]
            + [rng.choice(filler) for _ in range(60)]
        )
        docs.append(Document(path=f"/b/b{i:03d}.md", text=" ".join(words)))
    for i in range(945):
        docs.append(
            Document(path=f"/c/c{i:03d}.md",
                     text=" ".join(rng.choice(filler) for _ in range(80)))
        )
    # The shared term appears in a few A files too, so it is genuinely shared
    # rather than a B-only marker.
    for i in range(5):
        docs[i].text += " sharedterm"
    return docs


@pytest.fixture(scope="module")
def poison_index():
    return build_index(iter(_poison_corpus()))


def test_score_floor_drops_weak_off_topic_matches(poison_index):
    unfiltered = Router(poison_index, RouteConfig(top_n=100, score_floor=0.0)).route(
        POISON_QUERY
    )
    floored = Router(poison_index, RouteConfig(top_n=100, score_floor=0.35)).route(
        POISON_QUERY
    )

    def wrong(res):
        return [f for f in res.files if f.startswith("/b/")]

    assert wrong(unfiltered), "corpus should produce poisoning candidates"
    assert wrong(floored) == []
    assert len(floored.files) == 40
    assert floored.cut_by_floor == len(wrong(unfiltered))
    # The floor is a ranking cut, not a retrieval cut: same work, fewer results.
    assert floored.files_touched == unfiltered.files_touched


def test_score_floor_never_returns_nothing(poison_index):
    router = Router(poison_index, RouteConfig(top_n=50, score_floor=0.99, min_results=3))
    assert len(router.route(POISON_QUERY).files) >= 3


def test_score_floor_disabled_returns_full_top_n(poison_index):
    router = Router(poison_index, RouteConfig(top_n=20, score_floor=0.0))
    assert len(router.route(POISON_QUERY).files) == 20


# ------------------------------------------------- document filter / access


def test_doc_filter_excludes_forbidden_files(small_index):
    allowed = np.zeros(small_index.n_docs, dtype=bool)
    allowed[::3] = True
    router = Router(small_index, RouteConfig(top_n=25))
    res = router.route(small_index.terms[int(small_index.anchor_terms[0])],
                       doc_filter=allowed)
    assert res.doc_ids
    assert all(allowed[d] for d in res.doc_ids)
    assert res.filtered_out > 0


def test_doc_filter_beats_post_hoc_filtering_under_budget(small_index):
    """Filtering during the walk spends the budget on readable files only.

    Filtering the finished ranking instead throws away most of the work: the
    budget was spent on files the caller was never allowed to see.
    """
    rng = np.random.default_rng(0)
    allowed = np.zeros(small_index.n_docs, dtype=bool)
    allowed[rng.choice(small_index.n_docs, size=60, replace=False)] = True
    # A deliberately wide query spanning many topics, so the budget is the
    # binding constraint rather than the size of the posting lists — that is
    # the only regime in which the two strategies differ.
    query = " ".join(
        small_index.terms[int(t)] for t in small_index.anchor_terms[::37][:12]
    )

    router = Router(small_index, RouteConfig(top_n=50, budget=100))
    during = router.route(query, doc_filter=allowed)
    unfiltered = router.route(query)
    post_hoc = [d for d in unfiltered.doc_ids if allowed[d]]

    assert unfiltered.budget_exhausted, "test premise: the budget must bind"
    assert all(allowed[d] for d in during.doc_ids)
    assert len(during.doc_ids) > len(post_hoc)


def test_doc_filter_validates_shape(small_index):
    router = Router(small_index)
    with pytest.raises(ValueError, match="boolean array"):
        router.route("anything", doc_filter=np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="boolean array"):
        router.route("anything", doc_filter=np.zeros(small_index.n_docs, dtype=int))


def test_doc_filter_all_false_returns_nothing(small_index):
    router = Router(small_index, RouteConfig(top_n=10))
    blocked = np.zeros(small_index.n_docs, dtype=bool)
    res = router.route(small_index.terms[int(small_index.anchor_terms[0])],
                       doc_filter=blocked)
    assert res.files == []
    assert res.files_touched == 0


def test_mask_where_builds_a_usable_filter(small_index):
    mask = small_index.mask_where(lambda p: p.endswith("0.md"))
    assert mask.dtype == bool and mask.shape == (small_index.n_docs,)
    assert mask.sum() > 0
    res = Router(small_index, RouteConfig(top_n=10)).route(
        small_index.terms[int(small_index.anchor_terms[0])], doc_filter=mask
    )
    assert all(f.endswith("0.md") for f in res.files)


def test_zero_anchor_corpus_degrades_instead_of_crashing():
    """Every term corpus-wide => no graph. Must still build and still route."""
    docs = [
        Document(path=f"/d{i}.md", text="alpha beta gamma delta alpha beta")
        for i in range(30)
    ]
    with pytest.warns(RuntimeWarning, match="no anchor terms"):
        index = build_index(iter(docs), IndexConfig(max_df_ratio=0.01, min_anchors=1))
    assert index.stats.n_anchors == 0
    assert index.n_topics == 1
    result = Router(index, RouteConfig(top_n=5)).route("alpha gamma")
    assert result.files  # inverted index alone still answers
    assert result.files_touched == 30
