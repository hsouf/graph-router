"""graphroute — narrow a huge file corpus to a handful of candidates using a term graph.

Typical use::

    from graphroute import build_index, Router

    idx = build_index("/path/to/corpus")
    idx.save("index.grx")

    router = Router.load("index.grx")
    result = router.route("how does token refresh work", top_n=100)
    print(result.files, result.files_touched)
"""

from .index import IndexConfig, GraphIndex, build_index
from .router import Router, RouteConfig, RouteResult

__all__ = [
    "IndexConfig",
    "GraphIndex",
    "build_index",
    "Router",
    "RouteConfig",
    "RouteResult",
]
__version__ = "0.1.0"
