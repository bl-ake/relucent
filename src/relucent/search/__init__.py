"""Complex exploration: BFS/DFS, boundary discovery, and worker pools."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import (
        ALL_GEOMETRY_PROPERTIES,
        SEARCH_REQUIRED_GEOMETRY_PROPERTIES,
        CubeMode,
        greedy_path,
        hamming_astar,
        parallel_add,
        parallel_compute_geometric_properties,
        searcher,
    )
    from .exploration import SearchResult

__all__ = [
    "ALL_GEOMETRY_PROPERTIES",
    "SEARCH_REQUIRED_GEOMETRY_PROPERTIES",
    "CubeMode",
    "SearchResult",
    "greedy_path",
    "hamming_astar",
    "parallel_add",
    "parallel_compute_geometric_properties",
    "searcher",
]

_MODULE_OF: dict[str, str] = {name: ".engine" for name in __all__} | {"SearchResult": ".exploration"}


def __getattr__(name: str) -> Any:
    if name in _MODULE_OF:
        return getattr(import_module(_MODULE_OF[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
