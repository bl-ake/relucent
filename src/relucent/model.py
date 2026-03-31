"""Neural network wrapper for use with the relucent polyhedral complex tools.

Provides the :class:`NN` class, a thin :class:`torch.nn.Module` subclass that
stores layer weights in a named :class:`~torch.nn.ModuleDict` and exposes
helpers for grid generation, layer-output inspection, and weight access by
neuron index.  Also provides :func:`get_mlp_model` for constructing standard
ReLU MLPs.
"""

from collections import OrderedDict
from collections.abc import Container, Iterable

import numpy as np
import torch
import torch.nn as nn

from relucent.config import DEFAULT_GRID_BOUNDS, DEFAULT_GRID_RES

__all__ = ["NN", "get_mlp_model"]


class NN(nn.Module):
    """Neural network class that interfaces with the rest of the package"""

    def __init__(
        self,
        layers: OrderedDict[str, nn.Module] | None = None,
        input_shape: tuple[int, ...] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize a neural network.

        Args:
            layers: Dictionary of layers (nn.ModuleDict or dict-like). If None,
                creates an empty network. Defaults to None.
            input_shape: Shape of the input data (excluding batch dimension).
                If None, infers from the first Linear layer. Defaults to None.
            device: PyTorch device to use. If None, uses the device of the first
                parameter. Defaults to None.
            dtype: PyTorch dtype to use. If None, uses the dtype of the first
                parameter. Defaults to None.

        Raises:
            ValueError: If input_shape cannot be determined.
        """
        super().__init__()

        self.layers = nn.ModuleDict(layers) if layers is not None else nn.ModuleDict()

        if input_shape is not None:
            resolved_shape: tuple[int, ...] = input_shape
        elif isinstance(fl := next(iter(self.layers.values())), nn.Linear):
            resolved_shape = (fl.in_features,)
        else:
            raise ValueError("Input shape must be provided")

        self.input_shape: tuple[int, ...] = resolved_shape
        self.trained_on = None

        self.to(device or self.device, dtype or self.dtype)

    def save_numpy_weights(self) -> None:
        """Save NumPy weights and biases for all Linear layers.

        This method saves the weights and biases of all Linear layers to the
        `weight_cpu` and `bias_cpu` attributes of their respective layer objects.
        """
        for layer in self.layers.values():
            if isinstance(layer, nn.Linear):
                object.__setattr__(layer, "weight_cpu", layer.weight.detach().cpu().numpy())
                object.__setattr__(layer, "bias_cpu", layer.bias.detach().cpu().numpy().reshape(1, -1))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def num_relus(self) -> int:
        return len([layer for layer in self.layers.values() if isinstance(layer, nn.ReLU)])

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = data.reshape((-1,) + self.input_shape)
        for layer in self.layers.values():
            x = layer(x)
        return x

    def get_all_layer_outputs(
        self, data: torch.Tensor, layers: Container[str] | None = None, verbose: bool = False
    ) -> OrderedDict[str, torch.Tensor]:
        """Get outputs from specified layers.

        Args:
            data: Input tensor to the network.
            layers: List of layer names to include. If None, includes all layers.
                Defaults to None.
            verbose: If True, prints layer information. Defaults to False.

        Returns:
            OrderedDict: Dictionary mapping layer names to their outputs.
        """
        outputs = []
        x = data
        for name, layer in self.layers.items():
            if verbose:
                print(f"Layer {name}: {layer}")
            x = layer(x)
            if verbose:
                print(f"    Output shape: {x.shape}")
            if layers is None or name in layers:
                outputs.append((name, x))
        return OrderedDict(outputs)

    def get_grid(
        self, bounds: float | None = None, res: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a 2D grid of input points.

        Creates a regular grid of points in 2D space. Only works for 2D input spaces.

        Args:
            bounds: Half-width of the grid (grid spans [-bounds, bounds]).
                Defaults to config.DEFAULT_GRID_BOUNDS.
            res: Resolution (number of points per dimension). Defaults to config.DEFAULT_GRID_RES.

        Returns:
            tuple: (x_coords, y_coords, input_points) where input_points is an
                array of shape (res*res, 2).
        """
        bounds = bounds if bounds is not None else DEFAULT_GRID_BOUNDS
        res = res if res is not None else DEFAULT_GRID_RES
        x = np.linspace(-bounds, bounds, res)
        y = np.copy(x)

        X, Y = np.meshgrid(x, y)

        X = np.reshape(X, -1)
        Y = np.reshape(Y, -1)

        input_val = np.vstack((X, Y)).T
        return x, y, input_val

    def output_grid(
        self, bounds: float | None = None, res: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, OrderedDict[str, torch.Tensor]]:
        """Generate a grid and compute network outputs for all points.

        Args:
            bounds: Half-width of the grid. Defaults to config.DEFAULT_GRID_BOUNDS.
            res: Resolution (number of points per dimension). Defaults to config.DEFAULT_GRID_RES.

        Returns:
            tuple: (x_coords, y_coords, layer_outputs) where layer_outputs is
                an OrderedDict mapping layer names to outputs.
        """
        x, y, input_val = self.get_grid(bounds, res)

        outs = self.get_all_layer_outputs(torch.Tensor(input_val).to(self.device, self.dtype))

        return x, y, outs

    def shi2weights(self, shi: int, return_idx: bool = False) -> torch.Tensor | tuple[str, int]:
        """Get weights corresponding to a neuron index.


        Args:
            shi: Index of the neuron (supporting hyperplane index).
            return_idx: If True, returns (layer_name, neuron_index_in_layer).
                If False, returns a pointer to the weight tensor. Defaults to False.

        Returns:
            If return_idx is False: torch.Tensor weight vector.
            If return_idx is True: (layer_name, neuron_index) tuple.

        Raises:
            ValueError: If the neuron index is invalid.
        """
        remaining_rows = shi
        for name, layer in self.layers.items():
            if remaining_rows < 0:
                raise ValueError("Invalid Neuron Index")
            if isinstance(layer, nn.Linear):
                if layer.weight.shape[0] > remaining_rows:
                    return (name, remaining_rows) if return_idx else layer.weight.data[remaining_rows]
                else:
                    remaining_rows -= layer.weight.shape[0]
        raise ValueError(f"Invalid Neuron Index: {shi}")


def get_mlp_model(widths: Iterable[int], add_last_relu: bool = False) -> NN:
    """Create an NN object for a multi-layer perceptron (MLP).

    Constructs a fully connected neural network with the specified layer widths.
    Each layer (except optionally the last) is followed by a ReLU activation.

    Args:
        widths: List of integers specifying the number of neurons in each layer,
            including the input layer. For example, [2, 10, 5, 1] creates a
            network with input dimension 2, two hidden layers with 10 and 5 neurons,
            and output dimension 1.
        add_last_relu: If True, adds a ReLU after the last layer. Defaults to False.

    Returns:
        NN: A configured neural network object.
    """
    widths = list(widths)
    layers = []
    for i in range(len(widths) - 1):
        layers.append((f"fc{i}", nn.Linear(widths[i], widths[i + 1])))
        if i < len(widths) - 2 or add_last_relu:
            layers.append((f"relu{i}", nn.ReLU()))
    net = NN(layers=OrderedDict(layers))
    object.__setattr__(net, "widths", widths)
    return net
