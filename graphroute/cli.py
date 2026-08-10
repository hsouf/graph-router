"""Command line interface: `graphroute index` / `graphroute query`."""

from __future__ import annotations

import argparse
import sys

from .embed import DEFAULT_MODEL as EMBED_MODEL
from .index import GraphIndex, IndexConfig, build_index
from .router import RouteConfig, Router

# Argparse defaults come from the dataclasses so there is exactly one place a
# default lives. A CLI that hardcodes 0.35 next to a config that says 0.4 is a
# bug waiting for someone to tune the wrong one.
IDX = IndexConfig()
ROUTE = RouteConfig()


def _cmd_index(args) -> int:
    cfg = IndexConfig(
        max_df_ratio=args.max_df_ratio,
        graph_degree=args.degree,
        graph_terms_per_doc=args.terms_per_doc,
        pdf_max_pages=args.pdf_max_pages,
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


def _cmd_ask(args) -> int:
    from .pipeline import ask

    try:
        found = ask(
            args.source,
            args.question,
            passages=args.passages,
            top_sections=args.top_sections,
            cache_dir=args.cache_dir,
            embed=not args.no_embed,
            model_name=args.model,
            seed_expansion=args.seed_expansion,
            rebuild=args.rebuild,
            verbose=args.verbose,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        # These are all "you gave me something I cannot read" and the message
        # says which. A traceback would bury it.
        print(f"graphroute: {exc}", file=sys.stderr)
        return 1
    if not found:
        print("no passages matched.", file=sys.stderr)
        return 1
    for passage in found:
        print(passage.header())
        print(passage.text)
        print()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="graphroute")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="build an index over a corpus directory")
    p.add_argument("corpus")
    p.add_argument("-o", "--out", default="index.grx")
    p.add_argument("--max-df-ratio", type=float, default=IDX.max_df_ratio)
    p.add_argument("--degree", type=int, default=IDX.graph_degree)
    p.add_argument("--terms-per-doc", type=int, default=IDX.graph_terms_per_doc)
    p.add_argument("--pdf-max-pages", type=int, default=IDX.pdf_max_pages,
                   help="stop reading each PDF after N pages (default: all)")
    p.add_argument("--ext", nargs="*", default=None, help="restrict to extensions")
    p.set_defaults(func=_cmd_index)

    p = sub.add_parser("query", help="route a question to candidate files")
    p.add_argument("query")
    p.add_argument("-i", "--index", default="index.grx")
    p.add_argument("-n", "--top-n", type=int, default=ROUTE.top_n)
    p.add_argument("--budget", type=int, default=ROUTE.budget)
    p.add_argument("--no-gate", action="store_true", help="disable topic pruning")
    p.add_argument("--score-floor", type=float, default=ROUTE.score_floor,
                   help="drop hits below this fraction of the best score (0 = off)")
    p.add_argument("--path-prefix", nargs="*", default=None,
                   help="only consider files under these path prefixes")
    p.add_argument("--explain", action="store_true")
    p.set_defaults(func=_cmd_query)

    p = sub.add_parser("stats", help="print index statistics")
    p.add_argument("-i", "--index", default="index.grx")
    p.set_defaults(func=_cmd_stats)

    p = sub.add_parser(
        "ask",
        help="question -> passages over a directory of books (builds and caches "
             "an index on first run)",
    )
    p.add_argument("source", help="a directory of documents, or a single file")
    p.add_argument("question")
    p.add_argument("-p", "--passages", type=int, default=8)
    p.add_argument("--top-sections", type=int, default=50,
                   help="how many fused sections to draw passages from")
    p.add_argument("--cache-dir", default=None,
                   help="override the cache location (default: "
                        "~/.cache/graphroute/<corpus hash>)")
    p.add_argument("--no-embed", action="store_true",
                   help="lexical only, even if sentence-transformers is installed")
    p.add_argument("--model", default=None,
                   help="embedding model: a Hugging Face id, or a local "
                        "directory to stay off the network entirely "
                        f"(default: {EMBED_MODEL})")
    p.add_argument("--seed-expansion", action="store_true",
                   help="also seed the graph walk from the embedding hits")
    p.add_argument("--rebuild", action="store_true", help="ignore any cached index")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=_cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
