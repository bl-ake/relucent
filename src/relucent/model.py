"""Feedforward ReLU model representation used by relucent."""

from collections import OrderedDict
from collections.abc import Container, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

__all__ = ["ReLUNetwork", "LinearLayer", "ReLULayer", "FlattenLayer"]


@dataclass
class LinearLayer:
    weight: np.ndarray
    bias: np.ndarray
    kind: str = "linear"


@dataclass
class ReLULayer:
    kind: str = "relu"


@dataclass
class FlattenLayer:
    kind: str = "flatten"


Layer = LinearLayer | ReLULayer | FlattenLayer


class ReLUNetwork:
    """Canonical feedforward network: affine + ReLU (+ optional input flatten)."""

    def __init__(
        self,
        layers: Iterable[Layer] | Mapping[str, Layer],
        input_shape: tuple[int, ...] | None = None,
        **_: Any,
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
        self._device = "cpu"
        self._dtype = np.float64

    @property
    def num_relus(self) -> int:
        return sum(isinstance(layer, ReLULayer) for layer in self.layers.values())

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> Any:
        try:
            import torch  # type: ignore

            return torch.float64
        except Exception:
            return self._dtype

    def save_numpy_weights(self) -> None:
        return

    def __call__(self, data: np.ndarray | Any) -> np.ndarray | Any:
        return self.forward(data)

    def forward(self, data: np.ndarray | Any) -> np.ndarray | Any:
        x = data.reshape((-1,) + self.input_shape)
        for layer in self.layers.values():
            x = self._apply_layer(layer, x)
        return x

    @staticmethod
    def _apply_layer(layer: Layer, x: np.ndarray | Any) -> np.ndarray | Any:
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
        outputs: list[tuple[str, np.ndarray | Any]] = []
        x = data
        for name, layer in self.layers.items():
            if verbose:
                print(f"Layer {name}: {layer}")
            x = self._apply_layer(layer, x)
            if layers is None or name in layers:
                outputs.append((name, x))
        return OrderedDict(outputs)

    def shi2weights(self, shi: int, return_idx: bool = False) -> np.ndarray | tuple[str, int]:
        remaining_rows = shi
        for name, layer in self.layers.items():
            if isinstance(layer, LinearLayer):
                if layer.weight.shape[0] > remaining_rows:
                    if return_idx:
                        return (cast(str, name), remaining_rows)
                    return layer.weight[remaining_rows]
                remaining_rows -= layer.weight.shape[0]
        raise ValueError(f"Invalid Neuron Index: {shi}")

