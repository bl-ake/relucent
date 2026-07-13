"""Core polyhedral complex data model."""

from importlib import import_module
from typing import Any

__all__ = [
    "Complex",
    "Polyhedron",
    "SSManager",
    "ComplexNotCompleteError",
    "ComplexNotVerifiedError",
    "CubicalAmbiguityError",
    "CubicalConsistencyError",
    "DualGraphAsymmetricEdgeError",
    "IncompleteDualGraphError",
    "NonGenericArrangementError",
    "ShiFlipInvariantError",
    "ShiProofError",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Complex": (".complex", "Complex"),
    "Polyhedron": (".poly", "Polyhedron"),
    "SSManager": (".ss", "SSManager"),
    "ComplexNotCompleteError": (".errors", "ComplexNotCompleteError"),
    "ComplexNotVerifiedError": (".errors", "ComplexNotVerifiedError"),
    "CubicalAmbiguityError": (".errors", "CubicalAmbiguityError"),
    "CubicalConsistencyError": (".errors", "CubicalConsistencyError"),
    "DualGraphAsymmetricEdgeError": (".errors", "DualGraphAsymmetricEdgeError"),
    "IncompleteDualGraphError": (".errors", "IncompleteDualGraphError"),
    "NonGenericArrangementError": (".errors", "NonGenericArrangementError"),
    "ShiFlipInvariantError": (".errors", "ShiFlipInvariantError"),
    "ShiProofError": (".errors", "ShiProofError"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
