"""Betti numbers, filtrations, and persistent homology."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .betti import (
        C_BACKEND_AVAILABLE,
        ChainComplexInconsistent,
        ConnectedComponentsMismatch,
        get_betti_numbers,
        gf2_matmul_packed_stacked_rows,
        gf2_rank_boundary,
        gf2_rank_packed,
        gf2_rank_sparse_rowsets,
    )

__all__ = [
    "ChainComplexInconsistent",
    "ConnectedComponentsMismatch",
    "C_BACKEND_AVAILABLE",
    "get_betti_numbers",
    "gf2_matmul_packed_stacked_rows",
    "gf2_rank_boundary",
    "gf2_rank_packed",
    "gf2_rank_sparse_rowsets",
]


def __getattr__(name: str) -> Any:
    return getattr(import_module(".betti", __name__), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(import_module(".betti", __name__))))
