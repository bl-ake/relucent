"""Torch compatibility layer for optional dependency support."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["TORCH_AVAILABLE", "nn", "torch"]


def _missing_torch(*_args: Any, **_kwargs: Any) -> Any:
    raise ImportError('This relucent feature requires PyTorch. Install it with `pip install "relucent[torch]"`.')


try:
    import torch  # type: ignore[assignment]
    import torch.nn as nn  # type: ignore[assignment]

    _torch_available = True
except ImportError:
    _torch_available = False

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

    torch = _TorchStub()  # type: ignore[assignment]
    nn = _NNStub()  # type: ignore[assignment]

TORCH_AVAILABLE: bool = _torch_available
