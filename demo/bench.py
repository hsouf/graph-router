"""Benchmark: graph routing vs. a full-scan BM25 baseline.

Reports the two numbers the design is judged on — how many files were touched,
and how long the query took — alongside precision, so a speedup that comes from
returning garbage is visible rather than flattering.

Run:  python demo/bench.py --files 20000 --topics 60
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from demo.corpus import make_corpus
from graphroute.index import IndexConfig, build_index
from graphroute.router import RouteConfig, Router, full_scan_baseline


def precision_at(doc_ids, gold: set) -> float:
    if not len(doc_ids):
        return 0.0
    return sum(1 for d in doc_ids if d in gold) / len(doc_ids)


def run_one(args, verbose: bool = True):
    if verbose:
        print(f"building corpus: {args.files:,} files / {args.topics} topics ...")
    t0 = time.time()
    corpus = make_corpus(
        n_files=args.files,
        n_topics=args.topics,
        n_queries=args.queries,
        words_per_file=args.words,
    )
    if verbose:
        print(f"  {len(corpus.docs):,} docs generated in {time.time() - t0:.1f}s")

    index = build_index(iter(corpus.docs), IndexConfig(), verbose=verbose)

    # Ground truth lives on generated docs; map back through the index's order.
    path_to_topic = {d.path: t for d, t in zip(corpus.docs, corpus.doc_topic)}
    doc_topic_gold = np.array([path_to_topic[p] for p in index.paths])
    gold_by_topic = {
        t: set(np.flatnonzero(doc_topic_gold == t).tolist()) for t in range(args.topics)
    }

    router = Router(index, RouteConfig(top_n=args.top_n, budget=args.budget))

    rows = []
    for mode in ("graph", "graph-nogate", "baseline"):
        touched, latency, precision, returned = [], [], [], []
        for query, gold_topic in corpus.queries:
            gold = gold_by_topic[gold_topic]
            if mode == "baseline":
                t = time.perf_counter()
                ids, _ = full_scan_baseline(index, query, top_n=args.top_n)
                latency.append((time.perf_counter() - t) * 1000)
                touched.append(index.n_docs)
                precision.append(precision_at(ids.tolist(), gold))
                returned.append(len(ids))
            else:
                res = router.route(query, use_topic_gate=(mode == "graph"))
                latency.append(res.latency_ms)
                touched.append(res.files_touched)
                precision.append(precision_at(res.doc_ids, gold))
                returned.append(len(res.doc_ids))
        rows.append(
            (
                mode,
                statistics.mean(touched),
                statistics.mean(latency),
                float(np.percentile(latency, 95)),
                statistics.mean(precision),
                statistics.mean(returned),
            )
        )

    if verbose:
        print()
        print(f"{'mode':<14}{'files touched':>15}{'mean ms':>10}{'p95 ms':>9}"
              f"{'P@' + str(args.top_n):>9}{'returned':>10}")
        print("-" * 67)
        for mode, tch, ms, p95, prec, ret in rows:
            print(f"{mode:<14}{tch:>15,.0f}{ms:>10.2f}{p95:>9.2f}"
                  f"{prec:>9.3f}{ret:>10.1f}")

        graph_row, base_row = rows[0], rows[-1]
        print()
        print(
            f"graph routing touches {base_row[1] / max(graph_row[1], 1):.0f}x fewer "
            f"files and runs {base_row[2] / max(graph_row[2], 1e-9):.1f}x faster "
            f"at {graph_row[4]:.3f} vs {base_row[4]:.3f} precision"
        )
        print("\n--- explain (first query) ---")
        print(router.route(corpus.queries[0][0]).explain())

    return index, rows


def run_scaling(args) -> None:
    """The claim worth testing is that query cost is flat in corpus size.

    One corpus size cannot show that. At small N the router loses outright:
    its fixed cost (graph walk, topic selection, Python-level posting loop)
    is larger than simply scoring every document with vectorised numpy. The
    curve is the point, not any single row.
    """
    if not args.no_header:
        print(f"{'files':>10}{'index s':>10}{'touched':>10}{'graph ms':>10}"
              f"{'base ms':>10}{'speedup':>9}{'P@n graph':>11}{'P@n base':>10}")
        print("-" * 80)
    for n_files in args.scale:
        args.files = n_files
        args.topics = max(20, n_files // 400)
        index, rows = run_one(args, verbose=False)
        graph, base = rows[0], rows[-1]
        print(
            f"{index.n_docs:>10,}{index.stats.build_seconds:>10.1f}"
            f"{graph[1]:>10,.0f}{graph[2]:>10.2f}{base[2]:>10.2f}"
            f"{base[2] / max(graph[2], 1e-9):>8.1f}x{graph[4]:>11.3f}{base[4]:>10.3f}"
        )


ABLATIONS = [
    ("defaults", {}),
    ("no topic gate", {"use_topic_gate": False}),
    ("full expansion", {"min_term_weight": 0.0, "expand_terms": 128}),
    ("full expansion, no gate",
     {"min_term_weight": 0.0, "expand_terms": 128, "use_topic_gate": False}),
    ("no df ceiling", {"df_ceiling_ratio": 1.0}),
    ("no df ceiling, no gate", {"df_ceiling_ratio": 1.0, "use_topic_gate": False}),
]


def run_ablation(args) -> None:
    """Which pruning layer is actually doing the work?

    Worth running on your own corpus before tuning anything: the answer is
    corpus-dependent and not always the layer you expect.
    """
    corpus = make_corpus(
        n_files=args.files, n_topics=args.topics,
        n_queries=args.queries, words_per_file=args.words,
    )
    index = build_index(iter(corpus.docs), IndexConfig(), verbose=False)
    gold = {d.path: t for d, t in zip(corpus.docs, corpus.doc_topic)}
    truth = np.array([gold[p] for p in index.paths])

    print(f"{'config':<26}{'touched':>10}{'mean ms':>10}{'P@' + str(args.top_n):>9}")
    print("-" * 55)
    for name, kw in ABLATIONS:
        router = Router(index, RouteConfig(top_n=args.top_n, budget=args.budget))
        touched, latency, precision = [], [], []
        for query, gold_topic in corpus.queries:
            res = router.route(query, **kw)
            touched.append(res.files_touched)
            latency.append(res.latency_ms)
            if res.doc_ids:
                precision.append(float((truth[res.doc_ids] == gold_topic).mean()))
        print(f"{name:<26}{statistics.mean(touched):>10,.0f}"
              f"{statistics.mean(latency):>10.2f}{statistics.mean(precision):>9.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=20_000)
    ap.add_argument("--topics", type=int, default=60)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--budget", type=int, default=4_000)
    ap.add_argument("--words", type=int, default=220, help="words per file")
    ap.add_argument(
        "--scale", type=int, nargs="*", default=None,
        help="run a scaling sweep over these corpus sizes instead of one run",
    )
    ap.add_argument("--no-header", action="store_true",
                    help="omit the table header (for appending slices)")
    ap.add_argument("--ablate", action="store_true",
                    help="report which pruning layer earns its keep")
    args = ap.parse_args()

    if args.ablate:
        run_ablation(args)
    elif args.scale:
        run_scaling(args)
    else:
        run_one(args, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
