"""Command line interface: `graphroute index` / `graphroute query`."""

from __future__ import annotations

import argparse
import sys

from .index import GraphIndex, IndexConfig, build_index
from .router import RouteConfig, Router


def _cmd_index(args) -> int:
    cfg = IndexConfig(
        max_df_ratio=args.max_df_ratio,
        graph_degree=args.degree,
        graph_terms_per_doc=args.terms_per_doc,
        include_ext=args.ext,
    )
    index = build_index(args.corpus, cfg, verbose=True)
    index.save(args.out)
    print(f"saved -> {args.out}")
    return 0


def _cmd_query(args) -> int:
    index = GraphIndex.load(args.index)
    router = Router(
        index,
        RouteConfig(
            top_n=args.top_n,
            budget=args.budget,
            use_topic_gate=not args.no_gate,
            score_floor=args.score_floor,
        ),
    )
    doc_filter = None
    if args.path_prefix:
        doc_filter = index.mask_where(lambda p: p.startswith(tuple(args.path_prefix)))
    result = router.route(args.query, doc_filter=doc_filter)
    if args.explain:
        print(result.explain(), file=sys.stderr)
        print("-" * 60, file=sys.stderr)
    for path, score in zip(result.files, result.scores):
        print(f"{score:8.3f}  {path}")
    return 0


def _cmd_stats(args) -> int:
    print(GraphIndex.load(args.index).summary())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="graphroute")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="build an index over a corpus directory")
    p.add_argument("corpus")
    p.add_argument("-o", "--out", default="index.grx")
    p.add_argument("--max-df-ratio", type=float, default=0.15)
    p.add_argument("--degree", type=int, default=24)
    p.add_argument("--terms-per-doc", type=int, default=32)
    p.add_argument("--ext", nargs="*", default=None, help="restrict to extensions")
    p.set_defaults(func=_cmd_index)

    p = sub.add_parser("query", help="route a question to candidate files")
    p.add_argument("query")
    p.add_argument("-i", "--index", default="index.grx")
    p.add_argument("-n", "--top-n", type=int, default=50)
    p.add_argument("--budget", type=int, default=4000)
    p.add_argument("--no-gate", action="store_true", help="disable topic pruning")
    p.add_argument("--score-floor", type=float, default=0.35,
                   help="drop hits below this fraction of the best score (0 = off)")
    p.add_argument("--path-prefix", nargs="*", default=None,
                   help="only consider files under these path prefixes")
    p.add_argument("--explain", action="store_true")
    p.set_defaults(func=_cmd_query)

    p = sub.add_parser("stats", help="print index statistics")
    p.add_argument("-i", "--index", default="index.grx")
    p.set_defaults(func=_cmd_stats)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
