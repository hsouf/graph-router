# graphroute

A file **router** for RAG. Given a question and a corpus of millions of files,
return the few hundred files worth reading — while touching as few files as
possible on the way there.

The core is not a retriever and not an answer engine. It is the stage *before*
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

On top of that core there is one batteries-included pipeline, `ask`, that takes
a shelf of PDFs and a question and hands back quotable passages. Jump to
[PDF books](#pdf-books) for it.

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
documents. Every file gets up to three topics, not one — documents are not
single-subject, and forcing an argmax makes the gate reject a file for half the
queries it should answer. Files with no anchor terms get no topics and always
pass the gate rather than being parked in topic 0 where nothing can find them.
This is the layer that makes the whole thing work: at query time the router
picks a handful of topics and refuses to consider anything outside them.

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
| 10,000 | 485 | 0.85 | 0.15 | 0.967 | 0.958 |
| 40,000 | 492 | 0.73 | 0.39 | 0.967 | 0.962 |
| 160,000 | 497 | 1.04 | 1.34 | 0.967 | 0.963 |
| 400,000 | 399 | 1.62 | 2.39 | 0.944 | 0.925 |

Files touched stays flat at ~500 while the corpus grows 40x. That is the whole
claim.

**Below ~100k files the baseline wins outright**, and the benchmark says so:
the router's fixed cost — graph walk, topic selection, a Python-level loop over
expanded terms — exceeds the cost of simply scoring 10,000 documents. Crossover
on this corpus is around 100k–160k files. The reason to build the graph is what
happens above 10^6, where a flat line and a linear line have separated by orders
of magnitude.

The ablation in `BENCHMARK.md` is worth reading before you tune anything: on
this corpus the *df ceiling* — the least clever component — does most of the
pruning, and the topic gate only earns its place under wide query expansion.
Run `--ablate` on your own data before assuming the graph is what is helping.

## PDF books

Routing at file granularity is right for a source tree and wrong for a
2,000-page reference book, which is one file containing four hundred unrelated
subjects. Route to it and you have narrowed the corpus to "the book", which the
reader already knew.

`graphroute ask` splits each book into sections first, then runs the whole
stack — index, graph, communities, router — at section granularity, and returns
paragraph-level passages with a breadcrumb saying where each came from.

```
graphroute ask ./books/ "what is leukemia" --passages 8
graphroute ask ./hematology.pdf "how is it treated"
```

```
[Williams Hematology › Ch 88: Acute Lymphoblastic Leukemia › p.3]
Chapter 88 Acute Lymphoblastic Leukemia
Leukemia is a malignant proliferation of haematopoietic precursor cells
that accumulate in the bone marrow and displace normal haematopoiesis.
Acute lymphoblastic leukemia arises from lymphoid progenitors and is the
commonest malignancy of childhood, with a peak incidence between two and
five years of age. Patients present with fatigue, pallor, bruising and
recurrent infection, all consequences of marrow replacement by blasts.

[Williams Hematology › Ch 89: Chronic Myeloid Leukemia › p.6]
Chapter 89 Chronic Myeloid Leukemia
Chronic myeloid leukemia is defined by the Philadelphia chromosome, a
translocation between chromosomes nine and twenty two producing the BCR
ABL fusion kinase. Tyrosine kinase inhibitors transformed this disease
from uniformly fatal to a chronic condition managed with a daily tablet.

[Williams Hematology › Ch 88: Acute Lymphoblastic Leukemia › Diagnosis › p.4]
Diagnosis of acute lymphoblastic leukemia
Diagnosis requires bone marrow aspiration demonstrating at least twenty
percent lymphoblasts. Immunophenotyping by flow cytometry distinguishes
B lineage from T lineage disease, and cytogenetic analysis identifies the
translocations that determine risk group and therefore treatment intensity.
```

That is a real run of `graphroute ask . "what is leukemia" --passages 3` over
three small generated books; `demo/books.py` writes them if you want to
reproduce it.

Output is ready to paste into an LLM context. Same thing from Python:

```python
from graphroute import ask

for p in ask("./books/", "what is leukemia", passages=8):
    print(p.header())   # [Williams Hematology › Ch 88: ... › p.1721]
    print(p.text)

    p.breadcrumb        # 'Williams Hematology › Ch 88: Acute Lymphoblastic Leukemia'
    p.book_path         # '/books/williams-hematology.pdf'
    p.pages             # (1721, 1721) — 1-based, inclusive
    p.score             # fused chunk score, comparable within one call only
```

To use section granularity with your own index rather than through `ask`,
`walk_sections` is a drop-in replacement for `walk` — one `Document` per
section, path `book.pdf#s7`, and the rest of the stack is unchanged:

```python
from graphroute import build_index
from graphroute.ingest import walk_sections

index = build_index(walk_sections("./books/"))
```

The first run builds and caches; later runs load. The cache lives in
`~/.cache/graphroute/<hash>/` and the hash covers every input file's path, size
and mtime, so editing a book rebuilds and touching nothing does not. Pass
`cache_dir=` to put it somewhere else, `rebuild=True` to force.

### How a section is found

**Sectioning** (`sections.py`) tries three tiers per PDF and stops at the first
that works:

1. **Outline/bookmarks.** The publisher already did the work.
2. **Heading detection.** `Chapter N`, numbered headings, short ALL-CAPS lines.
3. **Fixed 8-page windows**, with breadcrumbs that degrade honestly to
   `Book › pp.120–128` rather than inventing a title.

Sections under 200 characters are merged into the previous one, and every page
of the book lands in exactly one section — including the preamble between a
chapter heading and its first sub-heading, which taking outline leaves alone
would have dropped.

*Exactly* one, which takes some care. A bookmark destination resolves to a
page, so a heading and the two sub-headings printed below it are three entries
pointing at the same place and indistinguishable from one another. Giving each
of them that page made three documents holding the same text: on a real 600-page
book, 978 sections of which only 470 were distinct. Duplicates are not free —
each copy adds to every one of its terms' document frequency, which is the
number idf and the router's selectivity test are computed from, and the reader
gets the same paragraph three times under three breadcrumbs. Entries sharing a
page are folded into one section named after the first of them, with the rest
kept in `also_titles` so the dense channel can still match on them.

**Retrieval** is two channels fused:

```
question
  -> lexical channel    the router, over section-documents
  -> dense channel      cosine over section embeddings   (optional)
  -> reciprocal rank fusion
  -> top 50 sections -> paragraph chunks -> rank chunks -> top 8 passages
```

Fusion is rank-based (`Σ 1/(60 + rank)`), with no weight to tune. BM25 scores
are unbounded sums of idf terms and cosines live in [-1, 1]; any weight
combining them directly is a number invented to fit one corpus. RRF needs none
and degrades gracefully when the dense channel is absent.

The last step also refuses to hand back the same passage twice. Sectioning
keeps a book from repeating itself, but a corpus can repeat itself in ways the
splitter never sees — the same document filed under two paths, a chapter
reprinted as an appendix — and the copies need not be byte-identical: chunked
from a different offset, the same prose comes out shifted. Passages are
compared as 5-word shingles and suppressed above 0.8 *containment* (overlap
over the shorter of the two, so a passage wholly inside a longer one counts as
a repeat, which Jaccard scores low exactly when the redundancy is total). A
suppressed passage frees its slot for the next distinct one rather than
shortening the answer. Repetition is worth spending code on because of how it
reads downstream: three copies of a claim look like three sources agreeing.

### Embeddings are optional

```
pip install -e ".[embed]"     # + sentence-transformers (all-MiniLM-L6-v2)
```

Without it, `ask` runs lexical-only and says so with a warning. With it, the
question "why do children get blood cancer" finds the acute lymphoblastic
leukemia chapter, which shares almost no vocabulary with the question.

What gets embedded per section is where the coupling to the lexical index pays:

```
breadcrumb + " — " + first paragraph + " | " + top-15 tf-idf terms
```

Those terms come out of the inverted index, which has already worked out what
distinguishes this section from its neighbours. Storage is an L2-normalised
float16 matrix beside the `.grx`; search is one matmul and an `argpartition`.
No vector database and no ANN index — brute force is exact and fast below about
a million rows, and an approximation would be saving time nobody is spending.

The embedder is pluggable, which is also how the tests avoid downloading a
model:

```python
ask("./books/", "...", embed_fn=my_encoder)   # list[str] -> np.ndarray
```

### Running offline

The core install never touches the network. The `[embed]` extra does, and
caching the weights is *not* enough to stop it: given a Hub id,
`huggingface_hub` still makes an unauthenticated revision check on every load,
which is where the `You are sending unauthenticated requests to the HF Hub`
line on every run comes from.

Setting `HF_HUB_OFFLINE=1` suppresses the check once the weights are cached.
To never contact the Hub at all, export the model once on a connected machine:

```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").save("./minilm")
```

then point `--model` at the directory:

```
graphroute ask ./books/ "what is leukemia" --model ./minilm
```

Verified with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and an empty
`HF_HOME`. `--no-embed` needs no model whatsoever.

The model name is recorded inside the saved embedding index and reused for the
question, because **cosine between vectors from two different encoders is a
number with no meaning** — not a worse ranking, an arbitrary one. If you build
with your own `embed_fn` and later query without it, graphroute refuses to
substitute the default model, warns, and falls back to lexical-only rather than
ranking on noise. Pass `model_name=` alongside `embed_fn=` to record a name for
your own encoder.

### Seeding the walk semantically

The router's graph walk starts at the anchors of the query's *literal* terms,
so a question phrased in the reader's vocabulary rather than the book's starts
nowhere. `Router.route(seed_fn=...)` takes extra PPR seeds, and
`embed.embedding_seed_fn` supplies them from the dense channel's top hits,
weighted by cosine. Off by default — on a corpus where the lexical seeds are
already right it only adds latency:

```
graphroute ask ./books/ "..." --seed-expansion
```

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
| `ppr_hops` / `ppr_beam` | 2 / 256 | how far and how wide the graph walk goes |
| `graph_degree` | 24 | neighbours kept per node — bounds branching |
| `graph_terms_per_doc` | 32 | anchors per doc — bounds edge count at index time |
| `max_df_ratio` | 0.15 | corpus stopword cutoff; auto-relaxes on small corpora |
| `max_doc_topics` / `min_topic_share` | 3 / 0.15 | how many communities a file may belong to |
| `pdf_max_pages` | None | stop reading each PDF after N pages; None reads all |

Recall too low? Raise `budget`, `max_topics`, `ppr_hops`. Too slow? Lower
`budget` and `expand_terms` first — they dominate.

## What this deliberately does not do

- **No LLM at index time.** Microsoft-GraphRAG-style entity extraction produces
  better edges, at one model call per chunk. That is not affordable over
  millions of files, and this design assumes you cannot afford it.
- **No embeddings in the core.** The router itself never loads a model, and
  `pip install graphroute` pulls numpy and scipy and nothing else. The dense
  channel is an optional extra used by `ask`, and it is a second channel fused
  by rank — it does not replace anything.
- **No vector database, no ANN index.** One float16 matrix and a matmul. See
  [PDF books](#pdf-books).
- **No incremental reindex.** `build_index` is a full rebuild. Adding it means
  persisting the term graph and re-running label propagation over the touched
  neighbourhood only.
- **No chunking in the router.** Routing is at document granularity by design.
  For books, the document *is* a section (`walk_sections`), and paragraph
  chunking happens after routing, on the few dozen sections that survive.

## Layout

```
graphroute/
  text.py      tokenisation, identifier splitting, term extraction
  ingest.py    corpus walking, per-format text extraction, structural links
  graph.py     sparse primitives: top-k pruning, NPMI, label propagation, PPR
  index.py     index construction and serialisation
  router.py    query-time routing + the full-scan BM25 reference
  sections.py  PDF -> sections, three tiers (outline / headings / windows)
  embed.py     optional dense channel over sections
  pipeline.py  ask(): build-or-load, fuse, chunk, rank passages
  cli.py       graphroute index / query / stats / ask
demo/
  corpus.py        labelled synthetic corpus generator
  bench.py         routing vs full-scan, single run or scaling sweep
  synthetic_pdf.py minimal PDF writer, so books can be generated not shipped
  books.py         writes the three PDFs used by the ask demo
tests/
```

## CLI

```
graphroute index ./corpus -o index.grx --ext .md .py .txt
graphroute query "how does token refresh work" -i index.grx --explain
graphroute stats -i index.grx
graphroute ask ./books/ "what is leukemia" --passages 8
```

`ask` flags worth knowing:

| flag | effect |
| --- | --- |
| `--passages N` | how many passages to print (default 8) |
| `--top-sections N` | how many fused sections to draw passages from (default 50) |
| `--cache-dir DIR` | build here instead of `~/.cache/graphroute/<hash>/` |
| `--rebuild` | ignore any cached index and rebuild |
| `--no-embed` | lexical only, no model loaded |
| `--model PATH_OR_ID` | embedding model; a local directory keeps it off the network |
| `--seed-expansion` | also seed the graph walk from the embedding hits |
| `-v` | show sectioning and index build progress |

The cache is derived data — deleting `~/.cache/graphroute` is always safe, and
the next `ask` rebuilds. It is keyed on file paths, sizes and mtimes, so an
edited corpus rebuilds by itself; `--rebuild` is for when the inputs are
unchanged but the config is not, which the key does not cover.

## Install

```
pip install -e .            # numpy, scipy — the router, nothing else
pip install -e ".[dev]"     # + pytest, pypdf (the test suite builds PDFs)
pip install -e ".[rich]"    # + pypdf, python-docx for PDF/DOCX ingestion
pip install -e ".[embed]"   # + sentence-transformers for the dense channel
pytest tests -q
```

Real corpora: point `build_index` at a directory. PDF and DOCX support is
optional and lazy-imported; unsupported formats are skipped rather than
crashing the build. `ask` needs `[rich]` for PDFs and runs lexical-only without
`[embed]`.

## Threading

A `GraphIndex` is read-only and safe to share. A `Router` is **not**: its
scratch buffers are allocated once and reused across queries, which is what
keeps per-query allocation proportional to files touched rather than to the
corpus. Use one `Router` per thread over one shared index.
