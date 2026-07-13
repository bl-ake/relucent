"""Betti numbers, filtrations, and persistent homology."""

from importlib import import_module
from typing import Any

from .betti import __all__ as _betti_all

__all__ = list(_betti_all)


def __getattr__(name: str) -> Any:
    return getattr(import_module(".betti", __name__), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(import_module(".betti", __name__))))
