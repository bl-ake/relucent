"""Complex exploration: BFS/DFS, boundary discovery, and worker pools."""

from importlib import import_module
from typing import Any

__all__ = [
    "ALL_GEOMETRY_PROPERTIES",
    "SEARCH_REQUIRED_GEOMETRY_PROPERTIES",
    "astar_calculations",
    "blocking_bad_shi_computations",
    "get_ip",
    "greedy_path",
    "hamming_astar",
    "parallel_compute_geometric_properties",
    "parallel_add",
    "retain_geometry_caches",
    "search_calculations",
    "searcher",
    "true_phantom_neighbor_error",
]


def __getattr__(name: str) -> Any:
    return getattr(import_module(".engine", __name__), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(import_module(".engine", __name__))))
