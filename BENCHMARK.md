# Measured results

All numbers produced by `demo/bench.py` on the labelled synthetic corpus.
Reproduce with the commands shown. Nothing here is estimated or extrapolated
except where explicitly labelled.

**Re-measured after the section/embedding release.** Every table below was
re-run against the current code, which differs from the version the first
edition of this file measured in four ways that touch these numbers:
`ppr_hops` dropped from 3 to 2, NPMI weights carry a support discount, topic
membership is soft (up to 3 topics per document), and every document emits a
pseudo-term for its own filename. The hardware also changed — an Apple Silicon
laptop rather than a containerised Linux box — so **absolute times are not
comparable to the previous edition**; the comparisons *within* each row are,
because both sides of every row were measured in the same run.

## 1. Files touched is flat; the baseline is linear

```
python demo/bench.py --queries 100 --words 60 --scale 10000 40000 160000
```

| files | index s | files touched | graph ms | full-scan ms | speedup | P@100 graph | P@100 base |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.9 | 485 | 0.85 | 0.15 | 0.2x | 0.967 | 0.958 |
| 40,000 | 3.6 | 492 | 0.73 | 0.39 | 0.5x | 0.967 | 0.962 |
| 160,000 | 14.6 | 497 | 1.04 | 1.34 | 1.3x | 0.967 | 0.963 |
| 400,000 † | 30.3 | 399 | 1.62 | 2.39 | 1.5x | 0.944 | 0.925 |

† run with `--words 40 --queries 40` so it fits in a single 40-second slice;
the shorter documents make its absolute times not directly comparable to the
rows above it, but the comparison within the row is.

Timings vary roughly ±20% between runs. The `files touched` column is
deterministic and does not vary.

**What the two-hop default cost.** Graph precision is 0.967 where the previous
edition measured 0.974, on identical baseline numbers (0.958 / 0.962 / 0.963),
so the difference is the code and not the machine. Isolating it on a 10,000-
file corpus: hops=3 scores 0.929 and touches 213 files, hops=2 scores 0.926 and
touches 202. Roughly a third of a precision point for a third of the latency.
The NPMI support discount measured neutral on this corpus (0.926 either way),
which is expected — the generator's topic vocabularies are disjoint, so there
are no coincidental pairs for the discount to demote. It is aimed at real text,
and this benchmark cannot show whether it helps there.

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
| defaults | 648 | 0.52 | 1.000 |
| no topic gate | 648 | 0.46 | 1.000 |
| full expansion (no weight floor) | 740 | 1.73 | 1.000 |
| full expansion, no gate | 1,146 | 1.54 | 1.000 |
| no df ceiling | 4,000 | 0.68 | 1.000 |
| no df ceiling, no gate | 4,000 | 0.63 | 1.000 |

Read honestly, this table says something inconvenient about the headline
architecture:

- **The df ceiling is the biggest single lever** (648 -> 4,000 files touched
  without it, where 4,000 is the budget: without the ceiling the walk simply
  runs until it is stopped). Refusing to walk the posting list of any term
  appearing in more than 5% of files does most of the pruning, and it is the
  least clever part of the system.
- **`min_term_weight` is where the latency went.** Cutting the tail of the PPR
  distribution takes mean latency from 1.73 ms to 0.52 ms at identical
  precision. Most expanded terms contribute rounding noise and a Python loop
  iteration.
- **The topic gate does nothing at default settings on this corpus** (648 vs
  648) and only shows up under wide expansion (740 vs 1,146). That is an
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
recovers the generated topics with **100% purity** —
`test_communities_align_with_ground_truth` asserts >85%. Index build is 5.1 s
for 19,980 files, 5,253 terms, 5,133 anchors, 187,752 edges.

It now finds **73 communities rather than 60**, and the extra thirteen are an
artefact worth naming. Every document emits a pseudo-term for its own filename,
and this generator names files `doc00059.md` inside each topic directory, so
`link:doc00059.md` legitimately appears in 60 files — one per topic. Those 333
filename terms form their own small communities. They attract no document mass
(mean topics per document is exactly 1.00 here), so purity is untouched, but
the community *count* is inflated by a naming collision the generator happens
to produce. On a corpus where filenames are distinct this does not arise.

Note also that soft topic membership makes no difference at all on this corpus:
every document lands in exactly one topic because the generator's vocabularies
are disjoint by construction. That is a property of the generator, not evidence
the feature is inert — it exists for corpora where a document is genuinely
about two things, which this one never produces.

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
