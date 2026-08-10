"""graphroute — narrow a huge file corpus to a handful of candidates using a term graph.

Routing, the original job::

    from graphroute import build_index, Router

    idx = build_index("/path/to/corpus")
    idx.save("index.grx")

    router = Router.load("index.grx")
    result = router.route("how does token refresh work", top_n=100)
    print(result.files, result.files_touched)

Or the whole pipeline, question to quotable passages, over a shelf of PDFs::

    from graphroute import ask

    for p in ask("./books/", "what is leukemia", passages=8):
        print(p.header())
        print(p.text)
"""

from .index import IndexConfig, GraphIndex, build_index
from .pipeline import Corpus, Passage, ask, ask_corpus, build_corpus
from .router import Router, RouteConfig, RouteResult
from .sections import Section, extract_sections

__all__ = [
    "IndexConfig",
    "GraphIndex",
    "build_index",
    "Router",
    "RouteConfig",
    "RouteResult",
    "ask",
    "ask_corpus",
    "build_corpus",
    "Corpus",
    "Passage",
    "Section",
    "extract_sections",
]
__version__ = "0.1.0"
