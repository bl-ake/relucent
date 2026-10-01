"""Canonical feedforward ReLU network representation.

:class:`ReLUNetwork` wraps an ordered sequence of :class:`LinearLayer`,
:class:`ReLULayer`, and :class:`FlattenLayer` objects and provides NumPy/PyTorch
agnostic forward passes.  All PyTorch models are converted to this format by
:func:`relucent.model.convert_model.convert` before use.
"""

from collections import OrderedDict
from collections.abc import Container, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from relucent._internal.logging import logger

__all__ = ["ReLUNetwork", "LinearLayer", "ReLULayer", "FlattenLayer"]


@dataclass
class LinearLayer:
    """Affine map ``x -> x @ weight.T + bias``; ``weight`` has shape ``(out, in)``."""

    weight: np.ndarray
    bias: np.ndarray


@dataclass
class ReLULayer:
    """Elementwise ``max(x, 0)``. Each unit of the preceding linear layer is one hyperplane."""

    pass


@dataclass
class FlattenLayer:
    """Reshape each sample to a vector, keeping the batch dimension."""

    pass


Layer = LinearLayer | ReLULayer | FlattenLayer


class ReLUNetwork:
    """Canonical feedforward network: affine + ReLU (+ optional input flatten).

    Args:
        layers: Layers in order, either an iterable (auto-named ``layer0``, ``layer1``, ...) or a
            mapping from layer name to layer.
        input_shape: Shape of one input sample. If omitted, it is inferred as
            ``(in_features,)`` from the first :class:`LinearLayer`; a ``ValueError`` is raised if the
            network has no linear layer.

    After construction, ``layers`` (ordered name -> layer mapping), ``input_shape``, and ``trained_on``
    (optional training data used by distance-to-training filtrations; ``None`` by default) are available.
    """

    def __init__(
        self,
        layers: Iterable[Layer] | Mapping[str, Layer],
        input_shape: tuple[int, ...] | None = None,
    ) -> None:
        if isinstance(layers, Mapping):
            self.layers: OrderedDict[str, Layer] = OrderedDict((str(name), layer) for name, layer in layers.items())
        else:
            self.layers = OrderedDict((f"layer{i}", layer) for i, layer in enumerate(layers))
        if input_shape is None:
            first_linear = next((layer for layer in self.layers.values() if isinstance(layer, LinearLayer)), None)
            if first_linear is None:
                raise ValueError("Input shape must be provided")
            input_shape = (int(first_linear.weight.shape[1]),)
        self.input_shape = input_shape
        self.trained_on = None

    @property
    def num_relus(self) -> int:
        """Number of :class:`ReLULayer` layers (not neurons)."""
        return sum(isinstance(layer, ReLULayer) for layer in self.layers.values())

    def __call__(self, data: np.ndarray | Any) -> np.ndarray | Any:
        return self.forward(data)

    def forward(self, data: np.ndarray | Any) -> np.ndarray | Any:
        """Run the network on a NumPy array or torch tensor.

        ``data`` is reshaped to ``(-1, *input_shape)``; the output has the same array type as the input.
        """
        x = data.reshape((-1,) + self.input_shape)
        for layer in self.layers.values():
            x = self._apply_layer(layer, x)
        return x

    @staticmethod
    def _apply_layer(layer: Layer, x: np.ndarray | Any) -> np.ndarray | Any:
        """Apply a single layer. Supports both NumPy arrays and PyTorch tensors."""
        if isinstance(layer, LinearLayer):
            w = layer.weight
            b = layer.bias
            if isinstance(x, np.ndarray):
                return (x @ w.T) + b
            return (x @ x.new_tensor(w).T) + x.new_tensor(b)
        if isinstance(layer, ReLULayer):
            if isinstance(x, np.ndarray):
                return np.maximum(x, 0)
            return x.clamp_min(0)
        if isinstance(layer, FlattenLayer):
            return x.reshape(x.shape[0], -1)
        raise ValueError(f"Unsupported canonical layer: {type(layer)}")

    def get_all_layer_outputs(
        self, data: np.ndarray | Any, layers: Container[str] | None = None, verbose: bool = False
    ) -> OrderedDict[str, np.ndarray | Any]:
        """Return the output of each layer for ``data``.

        Unlike :meth:`forward`, ``data`` is not reshaped first. ``layers`` restricts the result to the named
        layers (default: all); ``verbose`` logs each layer as it runs.
        """
        outputs: list[tuple[str, np.ndarray | Any]] = []
        x = data
        for name, layer in self.layers.items():
            if verbose:
                logger.info("Layer %s: %s", name, layer)
            x = self._apply_layer(layer, x)
            if layers is None or name in layers:
                outputs.append((name, x))
        return OrderedDict(outputs)

    def shi2weights(self, shi: int, return_idx: bool = False) -> np.ndarray | tuple[str, int]:
        """Locate the neuron behind hyperplane index ``shi``.

        Hyperplane indices count the rows of all linear layers in order. Returns that neuron's weight row,
        or ``(layer_name, row_within_layer)`` if ``return_idx`` is true. Raises ``ValueError`` if ``shi``
        is out of range.
        """
        remaining_rows = shi
        for name, layer in self.layers.items():
            if isinstance(layer, LinearLayer):
                if layer.weight.shape[0] > remaining_rows:
                    if return_idx:
                        return (cast(str, name), remaining_rows)
                    return layer.weight[remaining_rows]
                remaining_rows -= layer.weight.shape[0]
        raise ValueError(f"Invalid Neuron Index: {shi}")
