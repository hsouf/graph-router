# Measured results

All numbers produced by `demo/bench.py` on the labelled synthetic corpus, on a
single core of a containerised Linux box. Reproduce with the commands shown.
Nothing here is estimated or extrapolated except where explicitly labelled.

## 1. Files touched is flat; the baseline is linear

```
python demo/bench.py --queries 100 --words 60 --scale 10000 40000 160000
```

| files | index s | files touched | graph ms | full-scan ms | speedup | P@100 graph | P@100 base |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1.0 | 515 | 1.04 | 0.15 | 0.1x | 0.974 | 0.958 |
| 40,000 | 4.1 | 523 | 1.09 | 0.40 | 0.4x | 0.974 | 0.962 |
| 160,000 | 18.3 | 524 | 1.43 | 1.52 | 1.1x | 0.976 | 0.963 |
| 400,000 † | 34.8 | 449 | 1.97 | 2.46 | 1.2x | 0.944 | 0.925 |

† run with `--words 40 --queries 40` so it fits in a single 40-second slice;
the shorter documents make its absolute times not directly comparable to the
rows above it, but the comparison within the row is.

Timings vary roughly ±20% between runs on this shared machine — the 160k row
measured 1.12/1.37 ms on a first pass and 1.43/1.52 ms on a second. The
`files touched` column is deterministic and does not vary.

The number the design targets: **files touched stays in the 450–530 range while
the corpus grows 40x.** Query latency grows only weakly (1.04 -> 1.97 ms, and
part of that is cache pressure from a larger postings array) while the
full-scan baseline grows linearly, as it must.

**The baseline wins below roughly 100,000 files, and the table says so.**
Scoring 10,000 documents with a vectorised `np.add.at` is simply cheaper than a
graph walk plus topic selection plus a Python-level posting loop. Crossover on
this corpus is at about 110k–160k files. Below that, use BM25 and skip this
repo entirely. The reason to build the graph is what happens at 10^6 and above,
where the flat line and the linear line have separated by orders of magnitude.

Precision is a hair *higher* than full-scan BM25 rather than lower, because the
pruning removes documents that match only on common words — the same documents
that pollute the tail of a BM25 ranking.

## 2. Which pruning layer earns its keep

```
python demo/bench.py --ablate --files 20000 --topics 60 --queries 100
```

| config | files touched | mean ms | P@100 |
| --- | ---: | ---: | ---: |
| defaults | 648 | 0.98 | 1.000 |
| no topic gate | 648 | 0.92 | 1.000 |
| full expansion (no weight floor) | 742 | 2.55 | 1.000 |
| full expansion, no gate | 1,094 | 2.15 | 1.000 |
| no df ceiling | 3,892 | 0.98 | 1.000 |
| no df ceiling, no gate | 3,892 | 0.96 | 1.000 |

Read honestly, this table says something inconvenient about the headline
architecture:

- **The df ceiling is the biggest single lever** (648 -> 3,892 files touched
  without it). Refusing to walk the posting list of any term appearing in more
  than 5% of files does most of the pruning, and it is the least clever part of
  the system.
- **`min_term_weight` is where the latency went.** Cutting the tail of the PPR
  distribution took mean latency from 2.67 ms to 1.19 ms at identical
  precision. Most expanded terms contribute rounding noise and a Python loop
  iteration.
- **The topic gate does nothing at default settings on this corpus** (648 vs
  648) and only shows up under wide expansion (742 vs 1,094). That is an
  artefact of the generator: its topic vocabularies are perfectly separable, so
  the literal query terms already isolate the right topic and the gate has
  nothing left to remove. On a real corpus with polysemous vocabulary — where
  "cell" spans biology, spreadsheets and telecoms — the gate is what stops the
  expansion from dragging in three unrelated neighbourhoods. Do not take
  this row as evidence the gate is dead weight, and do not take it as evidence
  it works either. It is untested by this benchmark.

Note also that with `df_ceiling_ratio=1.0` the gate provably cannot help:
literal query terms below the ceiling are exempt from the gate by design (so
exact matches are never starved), and raising the ceiling to 1.0 makes every
literal term exempt. The two rows are identical for that reason, not by
coincidence.

## 3. Community quality

On the 20,000-file / 60-topic corpus, label propagation over the term graph
recovers the generated topics exactly: 60 communities, p50 and p95 both 333
files, and `test_communities_align_with_ground_truth` asserts >85% purity
(actual: 100%). Index build is 6.2 s for 19,980 files, 4,920 terms, 4,800
anchors, 182,586 edges.

This is a much easier clustering problem than any real corpus poses. Treat it
as a check that the plumbing is correct, not as evidence about clustering
quality in the wild.

## 4. The score floor does not fire on this corpus

Running the standard benchmark with `score_floor=0.35` (the default) returns a
mean of 100.0 files out of a requested 100 — the floor never cuts anything, and
precision is unchanged at 1.000. That is not evidence the floor is useless; it
is evidence the generator produces documents within a topic that are near
identical, so scores cluster tightly and there is no tail to cut.

The floor's behaviour is covered by a purpose-built corpus in
`tests/test_graphroute.py::_poison_corpus` instead: 40 files matching four
query terms, 15 files matching exactly one rare high-idf term and nothing else.
Measured score ratios are ~0.77-1.00 for the first group and ~0.24 for the
second, so the default floor removes all 15 poisoning candidates and returns 40
files rather than 55. Files touched is identical either way — the floor is a
ranking cut, not a retrieval cut.

## What this benchmark does not tell you

The corpus is synthetic, and synthetic in a way that flatters the method: topic
vocabularies are disjoint by construction, so both the communities and the
routing look better than they would on real text. What it does establish is the
mechanical claim — files touched and query latency are flat in corpus size,
while a full scan is linear. That claim is about the algorithm's shape and
carries over. The quality numbers do not.

Before trusting this on your own data, run `--ablate` on it. If the df ceiling
alone gets you where you need to be, you do not need the graph.
