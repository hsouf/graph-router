# graphroute

A file **router** for RAG. Given a question and a corpus of millions of files,
return the few hundred files worth reading — while touching as few files as
possible on the way there.

This is not a retriever and not an answer engine. It is the stage *before*
retrieval: the thing that decides which slice of the corpus the expensive
components are allowed to look at. No models, no network, no vector database.
NumPy and SciPy only.

```python
from graphroute import build_index, Router

index = build_index("/path/to/corpus")
index.save("index.grx")

router = Router.load("index.grx")
result = router.route("how does token refresh work", top_n=100)

result.files           # 100 paths
result.files_touched   # e.g. 747 out of 19,980
print(result.explain())
```

## The idea

Lexical retrieval fails on large corpora for a boring reason: a question is
mostly common words, and common words have enormous posting lists. Scoring
"how does token refresh work" means walking every document containing "work".
Vector search fixes the vocabulary problem but replaces a cheap scan with a
scan over embeddings — still linear in corpus size, now with a heavier constant.

graphroute attacks the corpus size directly, in three layers, cheapest first.

**1. Inverted index.** Standard postings, `term -> (doc, tf)`. Exact lookup.

**2. Term graph.** Anchor terms connected by NPMI co-occurrence, degree-capped
at 24 neighbours. This is the vocabulary bridge: a query word reaches words it
never literally co-occurs with but which sit in the same neighbourhood. NPMI
rather than raw counts matters here — raw counts connect everything to whatever
is popular, which produces a graph where every walk ends up in the same place.

**3. Communities.** Label propagation over the term graph, projected onto
documents. Every file gets a topic. This is the layer that makes the whole
thing work: at query time the router picks a handful of topics and refuses to
consider anything outside them.

At query time:

```
query
  -> terms                              exact vocabulary hits
  -> anchor seeds                       terms that are graph nodes
  -> frontier PPR over the term graph   vocabulary bridging, beam-limited
  -> topic selection                    shard pruning -- the big win
  -> budgeted postings walk             hard cap on files touched
  -> BM25 rescore of touched docs only
  -> top-N files
```

The personalised PageRank is beam-limited: each hop keeps only the 256 heaviest
residual nodes, so the walk costs `O(hops x beam x degree)` regardless of how
large the graph is. Topic selection then takes the smallest set of communities
covering 85% of the PPR mass. Everything after that is ordinary retrieval on a
small candidate pool.

`files_touched` — the number of documents whose posting entry was read at all —
is the metric the design optimises, and every result reports it.

## Two things that are not about speed

**The score floor.** `route()` returns *fewer* than `top_n` when the score tail
collapses, rather than padding the list with weak matches. The dangerous
failure of a RAG router is not a missing file, it is handing the reader a
plausible file about the wrong subject — the reader summarises it fluently and
confidently, and the output is not factually wrong so much as built on the
wrong document. Returning 12 good files beats returning 12 good files and 88
bystanders. Set `score_floor=0.0` to always fill `top_n`.

**Document filtering happens during the walk, not after.** Pass a boolean mask
of files this caller may see:

```python
allowed = index.mask_where(lambda p: p.startswith("/public/"))
result = router.route("...", doc_filter=allowed)
```

Filtering the finished ranking instead would silently return 8 files when 92 of
the top 100 were inadmissible. Filtering inside the walk spends the entire
budget on files the caller can actually read. The same hook covers tenant
isolation, language, and freshness — anything you can express as a mask over
document ids. Build the mask once per role and reuse it.

## Honest benchmark

`demo/bench.py` builds a labelled synthetic corpus (each topic owns a private
vocabulary, shares a bridge vocabulary with its neighbours, and draws filler
from a small pool of corpus-wide common words that stand in for "how / does /
work"). Queries mix topic terms, bridge terms and common terms, so they are
genuinely ambiguous at the surface.

```
python demo/bench.py --files 20000 --topics 60
python demo/bench.py --scale 10000 40000 160000    # scaling sweep
python demo/bench.py --ablate                      # which layer earns its keep
```

Measured, with the baseline being BM25 over every posting vectorised in NumPy:

| files | files touched | graph ms | full-scan ms | P@100 graph | P@100 base |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 515 | 1.04 | 0.15 | 0.974 | 0.958 |
| 40,000 | 523 | 1.09 | 0.40 | 0.974 | 0.962 |
| 160,000 | 524 | 1.43 | 1.52 | 0.976 | 0.963 |
| 400,000 | 449 | 1.97 | 2.46 | 0.944 | 0.925 |

Files touched stays flat at ~500 while the corpus grows 40x. That is the whole
claim.

**Below ~100k files the baseline wins outright**, and the benchmark says so:
the router's fixed cost — graph walk, topic selection, a Python-level loop over
expanded terms — exceeds the cost of simply scoring 10,000 documents. Crossover
on this corpus is around 110k–160k files. The reason to build the graph is what
happens above 10^6, where a flat line and a linear line have separated by orders
of magnitude.

The ablation in `BENCHMARK.md` is worth reading before you tune anything: on
this corpus the *df ceiling* — the least clever component — does most of the
pruning, and the topic gate only earns its place under wide query expansion.
Run `--ablate` on your own data before assuming the graph is what is helping.

## Tuning

Everything below is a field on `RouteConfig` or `IndexConfig`.

| knob | default | effect |
| --- | --- | --- |
| `budget` | 4000 | hard ceiling on files touched. The main speed/recall dial |
| `top_n` | 100 | max files handed to the reader stage |
| `score_floor` | 0.35 | drop hits below this fraction of the best score; 0 disables |
| `min_results` | 1 | floor never cuts below this many results |
| `max_topics` / `topic_coverage` | 12 / 0.85 | how many shards may be opened |
| `use_topic_gate` | True | off = no shard pruning, useful for A/B |
| `df_ceiling_ratio` | 0.05 | terms in >5% of files never contribute postings |
| `expand_terms` / `min_term_weight` | 64 / 0.02 | size of the expanded query |
| `ppr_hops` / `ppr_beam` | 3 / 256 | how far and how wide the graph walk goes |
| `graph_degree` | 24 | neighbours kept per node — bounds branching |
| `graph_terms_per_doc` | 32 | anchors per doc — bounds edge count at index time |
| `max_df_ratio` | 0.15 | corpus stopword cutoff; auto-relaxes on small corpora |

Recall too low? Raise `budget`, `max_topics`, `ppr_hops`. Too slow? Lower
`budget` and `expand_terms` first — they dominate.

## What this deliberately does not do

- **No LLM at index time.** Microsoft-GraphRAG-style entity extraction produces
  better edges, at one model call per chunk. That is not affordable over
  millions of files, and this design assumes you cannot afford it.
- **No embeddings.** If you want semantic seed selection, the clean insertion
  point is `Router._expand`: replace the literal-term seeds with the anchor
  terms nearest the query embedding. Everything downstream is unchanged.
- **No incremental reindex.** `build_index` is a full rebuild. Adding it means
  persisting the term graph and re-running label propagation over the touched
  neighbourhood only.
- **No chunking.** Routing is at file granularity by design. Chunk *after*
  routing, on the few hundred files that survive.

## Layout

```
graphroute/
  text.py      tokenisation, identifier splitting, term extraction
  ingest.py    corpus walking, per-format text extraction, structural links
  graph.py     sparse primitives: top-k pruning, NPMI, label propagation, PPR
  index.py     index construction and serialisation
  router.py    query-time routing + the full-scan BM25 reference
  cli.py       graphroute index / query / stats
demo/
  corpus.py    labelled synthetic corpus generator
  bench.py     routing vs full-scan, single run or scaling sweep
tests/
```

## CLI

```
graphroute index ./corpus -o index.grx --ext .md .py .txt
graphroute query "how does token refresh work" -i index.grx --explain
graphroute stats -i index.grx
```

## Install

```
pip install -e ".[dev]"     # numpy, scipy, pytest
pip install -e ".[rich]"    # + pypdf, python-docx for PDF/DOCX ingestion
pytest tests -q
```

Real corpora: point `build_index` at a directory. PDF and DOCX support is
optional and lazy-imported; unsupported formats are skipped rather than
crashing the build.
