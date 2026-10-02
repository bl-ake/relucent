"""Relucent: polyhedral complexes of ReLU networks.

The public API is exported lazily from this module; see https://bl-ake.github.io/relucent/.
"""

import tomllib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from . import config
from .config import update_settings


def _read_version() -> str:
    try:
        return version("relucent")
    except PackageNotFoundError:
        _pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with _pyproject.open("rb") as _fp:
            return tomllib.load(_fp)["project"]["version"]


__version__ = _read_version()

if TYPE_CHECKING:
    from .core.complex import Complex
    from .core.errors import (
        AmbiguousGeometryError,
        ComplexNotCompleteError,
        ComplexNotVerifiedError,
        IncompleteDualGraphError,
        NonGenericArrangementError,
    )
    from .core.poly import Polyhedron
    from .model.convert_model import convert
    from .search.exploration import SearchResult, explore_for_topology, generic_topology_start
    from .utils import add_output_relu, mlp, set_seeds, split_sequential
    from .verify.certify import CertifyLevel
    from .vis import get_colors, plot_complex, plot_polyhedron

__all__ = [
    "__version__",
    "AmbiguousGeometryError",
    "CertifyLevel",
    "Complex",
    "ComplexNotCompleteError",
    "ComplexNotVerifiedError",
    "IncompleteDualGraphError",
    "NonGenericArrangementError",
    "Polyhedron",
    "SearchResult",
    "config",
    "update_settings",
    "convert",
    "get_colors",
    "add_output_relu",
    "explore_for_topology",
    "generic_topology_start",
    "mlp",
    "plot_complex",
    "plot_polyhedron",
    "set_seeds",
    "split_sequential",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Complex": ("relucent.core.complex", "Complex"),
    "AmbiguousGeometryError": ("relucent.core.errors", "AmbiguousGeometryError"),
    "CertifyLevel": ("relucent.verify.certify", "CertifyLevel"),
    "ComplexNotCompleteError": ("relucent.core.errors", "ComplexNotCompleteError"),
    "ComplexNotVerifiedError": ("relucent.core.errors", "ComplexNotVerifiedError"),
    "IncompleteDualGraphError": ("relucent.core.errors", "IncompleteDualGraphError"),
    "NonGenericArrangementError": ("relucent.core.errors", "NonGenericArrangementError"),
    "Polyhedron": ("relucent.core.poly", "Polyhedron"),
    "SearchResult": ("relucent.search.exploration", "SearchResult"),
    "convert": ("relucent.model.convert_model", "convert"),
    "get_colors": ("relucent.vis", "get_colors"),
    "add_output_relu": ("relucent.utils", "add_output_relu"),
    "explore_for_topology": ("relucent.search.exploration", "explore_for_topology"),
    "generic_topology_start": ("relucent.search.exploration", "generic_topology_start"),
    "mlp": ("relucent.utils", "mlp"),
    "plot_complex": ("relucent.vis", "plot_complex"),
    "plot_polyhedron": ("relucent.vis", "plot_polyhedron"),
    "set_seeds": ("relucent.utils", "set_seeds"),
    "split_sequential": ("relucent.utils", "split_sequential"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
