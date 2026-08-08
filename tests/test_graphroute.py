from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.corpus import make_corpus
from graphroute import build_index
from graphroute.graph import frontier_ppr, label_propagation, prune_topk
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


def test_route_respects_budget(small_index, small_corpus):
    router = Router(small_index, RouteConfig(top_n=10, budget=50))
    for query, _ in small_corpus.queries[:10]:
        assert router.route(query).files_touched <= 50


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
    query = " ".join(
        small_index.terms[int(t)] for t in small_index.anchor_terms[:3]
    )

    router = Router(small_index, RouteConfig(top_n=50, budget=150))
    during = router.route(query, doc_filter=allowed)
    post_hoc = [d for d in router.route(query).doc_ids if allowed[d]]

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
