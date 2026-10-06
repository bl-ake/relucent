"""Canonical ReLU network representation and model conversion.

The exports are loaded lazily: ``builders`` and ``convert_model`` import torch, and a worker process that
only unpickles a :class:`ReLUNetwork` (from ``model``, which needs only numpy) should not pay for that.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .builders import add_output_relu, mlp, normalize_weights, set_seeds, split_sequential, torch_mlp
    from .convert_model import convert
    from .model import FlattenLayer, Layer, LinearLayer, ReLULayer, ReLUNetwork

__all__ = [
    "FlattenLayer",
    "Layer",
    "LinearLayer",
    "ReLULayer",
    "ReLUNetwork",
    "add_output_relu",
    "convert",
    "mlp",
    "normalize_weights",
    "set_seeds",
    "split_sequential",
    "torch_mlp",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "FlattenLayer": ("relucent.model.model", "FlattenLayer"),
    "Layer": ("relucent.model.model", "Layer"),
    "LinearLayer": ("relucent.model.model", "LinearLayer"),
    "ReLULayer": ("relucent.model.model", "ReLULayer"),
    "ReLUNetwork": ("relucent.model.model", "ReLUNetwork"),
    "add_output_relu": ("relucent.model.builders", "add_output_relu"),
    "convert": ("relucent.model.convert_model", "convert"),
    "mlp": ("relucent.model.builders", "mlp"),
    "normalize_weights": ("relucent.model.builders", "normalize_weights"),
    "set_seeds": ("relucent.model.builders", "set_seeds"),
    "split_sequential": ("relucent.model.builders", "split_sequential"),
    "torch_mlp": ("relucent.model.builders", "torch_mlp"),
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
