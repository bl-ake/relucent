import os
import tomllib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from . import config
from .config import update_settings
from .numeric_tolerances import apply_tolerances


def _bootstrap_tolerances() -> None:
    if os.getenv("RELUCENT_SKIP_NUMERIC_BOOTSTRAP", "0") == "1":
        return
    apply_tolerances()


_bootstrap_tolerances()


def _read_version() -> str:
    try:
        return version("relucent")
    except PackageNotFoundError:
        _pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with _pyproject.open("rb") as _fp:
            return tomllib.load(_fp)["project"]["version"]


__version__ = _read_version()

if TYPE_CHECKING:
    from .complex import Complex
    from .convert_model import convert
    from .meta_graph import NonGenericArrangementError
    from .poly import Polyhedron
    from .ss import SSManager
    from .utils import add_output_relu, get_env, mlp, set_seeds, split_sequential
    from .vis import get_colors, plot_complex, plot_polyhedron

__all__ = [
    "__version__",
    "Complex",
    "NonGenericArrangementError",
    "Polyhedron",
    "SSManager",
    "config",
    "update_settings",
    "convert",
    "get_colors",
    "get_env",
    "add_output_relu",
    "mlp",
    "plot_complex",
    "plot_polyhedron",
    "set_seeds",
    "split_sequential",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Complex": ("relucent.complex", "Complex"),
    "NonGenericArrangementError": ("relucent.meta_graph", "NonGenericArrangementError"),
    "Polyhedron": ("relucent.poly", "Polyhedron"),
    "SSManager": ("relucent.ss", "SSManager"),
    "convert": ("relucent.convert_model", "convert"),
    "get_colors": ("relucent.vis", "get_colors"),
    "get_env": ("relucent.utils", "get_env"),
    "add_output_relu": ("relucent.utils", "add_output_relu"),
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
