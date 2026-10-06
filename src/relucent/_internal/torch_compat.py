"""Torch compatibility layer for optional dependency support.

torch is imported on first use, not when relucent is: ``torch`` and ``nn`` stand in for the modules and
import them when an attribute is first read. Importing torch takes seconds, and each worker of a pool
started with ``spawn`` (Windows, macOS) would otherwise pay that even when it only handles numpy arrays.
Code that may receive numpy input checks for tensors with :func:`is_torch_tensor` and disables gradients
with :func:`no_grad`; neither imports torch.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from typing_extensions import TypeIs

__all__ = ["TORCH_AVAILABLE", "is_torch_tensor", "nn", "no_grad", "torch"]

F = TypeVar("F", bound=Callable[..., Any])


def _missing_torch(*_args: Any, **_kwargs: Any) -> Any:
    raise ImportError('This relucent feature requires PyTorch. Install it with `pip install "relucent[torch]"`.')


def _torch_installed() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


TORCH_AVAILABLE: bool = _torch_installed()


def is_torch_tensor(x: object) -> TypeIs[torch.Tensor]:
    """Whether ``x`` is a torch tensor. Never imports torch: a tensor can only exist once torch is loaded."""
    torch_module = sys.modules.get("torch")
    return torch_module is not None and isinstance(x, torch_module.Tensor)


def no_grad(func: F) -> F:
    """``@torch.no_grad()`` that imports nothing: gradients are disabled when torch is loaded at call time.

    A tensor that tracks gradients can only exist once torch is loaded, so if it is not, there is nothing
    to disable.
    """
    wrapped: Callable[..., Any] | None = None

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal wrapped
        target = wrapped
        if target is None:
            torch_module = sys.modules.get("torch")
            if torch_module is None:
                return func(*args, **kwargs)
            target = wrapped = torch_module.no_grad()(func)
        return target(*args, **kwargs)

    return cast(F, wrapper)


class _LazyModule:
    """Stands in for a module and imports it when an attribute is first read."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._module: Any = None

    def __getattr__(self, attr: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return getattr(self._module, attr)

    def __repr__(self) -> str:
        return f"<lazily imported module {self._name!r}>"


if TYPE_CHECKING:
    import torch
    import torch.nn as nn
elif TORCH_AVAILABLE:
    torch = _LazyModule("torch")
    nn = _LazyModule("torch.nn")
else:

    class _MissingModule:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _missing_torch()

    class _NNStub:
        Module = _MissingModule
        Linear = _MissingModule
        ReLU = _MissingModule
        Flatten = _MissingModule
        Sequential = _MissingModule
        Conv2d = _MissingModule
        AvgPool2d = _MissingModule
        MaxPool2d = _MissingModule
        Dropout = _MissingModule

    class _LinalgStub:
        def __getattr__(self, _name: str) -> Callable[..., Any]:
            return _missing_torch

    def _no_grad_stub(func: Callable[..., Any] | None = None) -> Any:
        if func is None:

            def _decorator(f: Callable[..., Any]) -> Callable[..., Any]:
                return f

            return _decorator
        return func

    class _TorchStub:
        Tensor = _MissingModule
        float64 = None
        linalg = _LinalgStub()
        no_grad = staticmethod(_no_grad_stub)

        def __getattr__(self, _name: str) -> Callable[..., Any]:
            return _missing_torch

    torch = _TorchStub()
    nn = _NNStub()
