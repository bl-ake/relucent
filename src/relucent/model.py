from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class NN(nn.Module):
    """Neural network class that interfaces with the rest of the package"""

    def __init__(self, layers=None, input_shape=None, device=None, dtype=None):
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
        super(NN, self).__init__()

        self.layers = nn.ModuleDict(layers) if layers is not None else nn.ModuleDict()

        self.to(device or self.device, dtype or self.dtype)
        if input_shape is not None:
            self.input_shape = input_shape
        elif isinstance(fl := next(iter(self.layers.values())), nn.Linear):
            self.input_shape = (fl.in_features,)
        else:
            raise ValueError("Input shape must be provided")

        self.to(device or self.device, dtype or self.dtype)

        self.trained_on = None

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def num_relus(self):
        return len([layer for layer in self.layers.values() if isinstance(layer, nn.ReLU)])

    def forward(self, data):
        x = data.reshape((-1,) + self.input_shape)
        for layer in self.layers.values():
            x = layer(x)
        return x

    def get_all_layer_outputs(self, data, layers=None, verbose=False):
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

    def get_grid(self, bounds=2, res=100):
        """Generate a 2D grid of input points.

        Creates a regular grid of points in 2D space. Only works for 2D input spaces.

        Args:
            bounds: Half-width of the grid (grid spans [-bounds, bounds]).
                Defaults to 2.
            res: Resolution (number of points per dimension). Defaults to 100.

        Returns:
            tuple: (x_coords, y_coords, input_points) where input_points is an
                array of shape (res*res, 2).
        """
        x = np.linspace(-bounds, bounds, res)
        y = np.copy(x)

        X, Y = np.meshgrid(x, y)

        X = np.reshape(X, -1)
        Y = np.reshape(Y, -1)

        inputVal = np.vstack((X, Y)).T
        return x, y, inputVal

    def output_grid(self, bounds=2, res=100):
        """Generate a grid and compute network outputs for all points.

        Args:
            bounds: Half-width of the grid. Defaults to 2.
            res: Resolution (number of points per dimension). Defaults to 100.

        Returns:
            tuple: (x_coords, y_coords, layer_outputs) where layer_outputs is
                an OrderedDict mapping layer names to outputs.
        """
        x, y, inputVal = self.get_grid(bounds, res)

        outs = self.get_all_layer_outputs(torch.Tensor(inputVal).to(self.device, self.dtype))

        return x, y, outs

    def shi2weights(self, shi, return_idx=False):
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
        raise ValueError("Invalid Neuron Index")


def get_mlp_model(widths, add_last_relu=False):
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
    layers = []
    for i in range(len(widths) - 1):
        layers.append((f"fc{i}", nn.Linear(widths[i], widths[i + 1])))
        if i < len(widths) - 2 or add_last_relu:
            layers.append((f"relu{i}", nn.ReLU()))
    net = NN(layers=OrderedDict(layers))
    net.widths = widths
    return net
