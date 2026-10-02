"""Betti numbers, filtrations, and persistent homology."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .betti import (
        C_BACKEND_AVAILABLE,
        ChainComplexInconsistent,
        Compactify,
        ConnectedComponentsMismatch,
        betti_numbers,
        gf2_matmul_packed_stacked_rows,
        gf2_rank_boundary,
        gf2_rank_packed,
        gf2_rank_sparse_rowsets,
    )
    from .filtration import (
        AffineOutputFiltration,
        ConstantFiltration,
        Filtration,
        LogitSublevelFiltration,
        NeuronActivationFiltration,
        TrainingDistanceFiltration,
    )
    from .persistence import (
        PersistenceDiagram,
        PersistencePair,
        betti_at_filtration_end,
        betti_curve,
        compute_persistent_homology,
    )

_MODULE_OF: dict[str, str] = {
    **dict.fromkeys(
        (
            "C_BACKEND_AVAILABLE",
            "ChainComplexInconsistent",
            "Compactify",
            "ConnectedComponentsMismatch",
            "betti_numbers",
            "gf2_matmul_packed_stacked_rows",
            "gf2_rank_boundary",
            "gf2_rank_packed",
            "gf2_rank_sparse_rowsets",
        ),
        ".betti",
    ),
    **dict.fromkeys(
        (
            "AffineOutputFiltration",
            "ConstantFiltration",
            "Filtration",
            "LogitSublevelFiltration",
            "NeuronActivationFiltration",
            "TrainingDistanceFiltration",
        ),
        ".filtration",
    ),
    **dict.fromkeys(
        (
            "PersistenceDiagram",
            "PersistencePair",
            "betti_at_filtration_end",
            "betti_curve",
            "compute_persistent_homology",
        ),
        ".persistence",
    ),
}

__all__ = [
    "AffineOutputFiltration",
    "C_BACKEND_AVAILABLE",
    "ChainComplexInconsistent",
    "Compactify",
    "ConnectedComponentsMismatch",
    "ConstantFiltration",
    "Filtration",
    "LogitSublevelFiltration",
    "NeuronActivationFiltration",
    "PersistenceDiagram",
    "PersistencePair",
    "TrainingDistanceFiltration",
    "betti_at_filtration_end",
    "betti_curve",
    "compute_persistent_homology",
    "betti_numbers",
    "gf2_matmul_packed_stacked_rows",
    "gf2_rank_boundary",
    "gf2_rank_packed",
    "gf2_rank_sparse_rowsets",
]
assert set(__all__) == set(_MODULE_OF)


def __getattr__(name: str) -> Any:
    if name in _MODULE_OF:
        return getattr(import_module(_MODULE_OF[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
