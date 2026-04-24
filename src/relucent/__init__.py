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
    from .complex import Complex
    from .convert_model import convert
    from .poly import Polyhedron
    from .ss import SSManager
    from .utils import get_env, mlp, set_seeds, split_sequential
    from .vis import get_colors, plot_complex, plot_polyhedron

__all__ = [
    "__version__",
    "Complex",
    "Polyhedron",
    "SSManager",
    "config",
    "update_settings",
    "convert",
    "get_colors",
    "get_env",
    "mlp",
    "plot_complex",
    "plot_polyhedron",
    "set_seeds",
    "split_sequential",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Complex": ("relucent.complex", "Complex"),
    "Polyhedron": ("relucent.poly", "Polyhedron"),
    "SSManager": ("relucent.ss", "SSManager"),
    "convert": ("relucent.convert_model", "convert"),
    "get_colors": ("relucent.vis", "get_colors"),
    "get_env": ("relucent.utils", "get_env"),
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
