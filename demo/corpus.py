"""Synthetic corpus with known ground truth.

Real benchmarks need labels. This generator builds N files across T topics,
where each topic owns a private vocabulary, shares a "bridge" vocabulary with
two neighbouring topics, and draws filler from a global noise vocabulary.

The bridge vocabulary is the interesting part: it is what makes naive lexical
retrieval leak across topics, and what the graph is supposed to resolve
correctly. Queries are built from a topic's terms but deliberately include
bridge terms, so a query is genuinely ambiguous at the surface level.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List, Tuple

from graphroute.ingest import Document

WORD_PARTS = [
    "auth", "token", "cache", "queue", "shard", "vector", "ledger", "policy",
    "tenant", "billing", "invoice", "retry", "webhook", "cursor", "schema",
    "replica", "quota", "digest", "cluster", "session", "payload", "encoder",
    "router", "throttle", "manifest", "snapshot", "lease", "compact", "gossip",
    "bucket", "codec", "beacon", "harness", "probe", "fabric", "mesh", "spool",
]
SUFFIXES = ["ion", "er", "ing", "ate", "ity", "ance", "ment", "ology", "ics"]


def _vocab(rng: random.Random, n: int, tag: str) -> List[str]:
    out = set()
    while len(out) < n:
        w = rng.choice(WORD_PARTS) + tag + rng.choice(SUFFIXES) + str(rng.randrange(90))
        out.add(w.lower())
    return sorted(out)


@dataclass
class Corpus:
    docs: List[Document]
    doc_topic: List[int]
    queries: List[Tuple[str, int]]  # (query text, gold topic)
    common_terms: List[str]         # the "how / does / work" of this corpus

    @property
    def gold_files(self):
        by_topic = {}
        for i, t in enumerate(self.doc_topic):
            by_topic.setdefault(t, set()).add(i)
        return by_topic


def make_corpus(
    n_files: int = 20_000,
    n_topics: int = 60,
    words_per_file: int = 220,
    n_queries: int = 200,
    seed: int = 7,
) -> Corpus:
    rng = random.Random(seed)

    topic_vocabs = [_vocab(rng, 60, f"t{t}") for t in range(n_topics)]
    bridge_vocabs = [_vocab(rng, 20, f"b{t}") for t in range(n_topics)]
    # Small noise vocabulary => each noise word lands in a large fraction of
    # files. These stand in for the common words real questions are full of,
    # and they are what makes a naive posting walk touch the whole corpus.
    noise = _vocab(rng, 120, "n")

    docs: List[Document] = []
    doc_topic: List[int] = []
    per_topic = max(1, n_files // n_topics)

    for topic in range(n_topics):
        left = bridge_vocabs[(topic - 1) % n_topics]
        right = bridge_vocabs[topic]
        for j in range(per_topic):
            words: List[str] = []
            for _ in range(words_per_file):
                r = rng.random()
                if r < 0.55:
                    words.append(rng.choice(topic_vocabs[topic]))
                elif r < 0.70:
                    words.append(rng.choice(left + right))
                else:
                    words.append(rng.choice(noise))
            text = " ".join(words)
            docs.append(
                Document(path=f"/corpus/topic{topic:03d}/doc{j:05d}.md", text=text, size=len(text))
            )
            doc_topic.append(topic)

    order = list(range(len(docs)))
    rng.shuffle(order)
    docs = [docs[i] for i in order]
    doc_topic = [doc_topic[i] for i in order]

    queries: List[Tuple[str, int]] = []
    for _ in range(n_queries):
        topic = rng.randrange(n_topics)
        # 2 topic terms + 2 bridge terms + 3 common terms. Ambiguous on the
        # surface, resolvable once you know which neighbourhood the terms live
        # in — and padded with common words the way real questions are.
        terms = (
            rng.sample(topic_vocabs[topic], 2)
            + rng.sample(bridge_vocabs[topic] + bridge_vocabs[(topic - 1) % n_topics], 2)
            + rng.sample(noise, 3)
        )
        rng.shuffle(terms)
        queries.append((" ".join(terms), topic))

    return Corpus(docs=docs, doc_topic=doc_topic, queries=queries, common_terms=noise)


def iter_documents(corpus: Corpus) -> Iterator[Document]:
    yield from corpus.docs
