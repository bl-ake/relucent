"""Tests for relucent.model and relucent.model.builders."""

from typing import Any, cast

import numpy as np
import pytest
import torch
import torch.nn as nn

from relucent.model import LinearLayer, ReLULayer, ReLUNetwork
from relucent.model.builders import MLP_INIT_METHODS, mlp, set_seeds, torch_mlp


class TestNumpyMlp:
    """Tests for mlp (NumPy weights, no PyTorch needed)."""

    def test_returns_relu_network_with_expected_layers(self):
        net = mlp([3, 5, 2], add_last_relu=True)
        assert isinstance(net, ReLUNetwork)
        assert list(net.layers) == ["fc0", "relu0", "fc1", "relu1"]
        assert net.input_shape == (3,)
        fc0, fc1 = net.layers["fc0"], net.layers["fc1"]
        assert isinstance(fc0, LinearLayer) and isinstance(fc1, LinearLayer)
        assert fc0.weight.shape == (5, 3) and fc0.bias.shape == (1, 5)
        assert fc1.weight.shape == (2, 5) and fc1.bias.shape == (1, 2)
        assert fc0.weight.dtype == np.float64
        assert list(mlp([2, 4, 1]).layers) == ["fc0", "relu0", "fc1"]

    def test_seeded_and_reproducible(self):
        set_seeds(7)
        a = mlp([2, 6, 1])
        set_seeds(7)
        b = mlp([2, 6, 1])
        for name in ("fc0", "fc1"):
            la, lb = a.layers[name], b.layers[name]
            assert isinstance(la, LinearLayer) and isinstance(lb, LinearLayer)
            assert np.array_equal(la.weight, lb.weight) and np.array_equal(la.bias, lb.bias)

    @pytest.mark.parametrize("init", sorted(MLP_INIT_METHODS))
    def test_init_methods(self, init: str):
        set_seeds(0)
        fan_in, fan_out = 40, 30
        layer = mlp([fan_in, fan_out], init=init).layers["fc0"]
        assert isinstance(layer, LinearLayer)
        w, b = layer.weight, layer.bias
        if init == "uniform":
            bound = 1 / np.sqrt(fan_in)
            assert np.abs(w).max() <= bound and np.abs(b).max() <= bound and np.abs(b).max() > 0
        else:
            assert np.all(b == 0)
        if init == "xavier_uniform":
            assert np.abs(w).max() <= np.sqrt(6 / (fan_in + fan_out))
        if init == "kaiming_uniform":
            assert np.abs(w).max() <= np.sqrt(2) * np.sqrt(3 / fan_in)
        if init == "kaiming_normal":
            assert abs(w.std() - np.sqrt(2 / fan_in)) < 0.1 * np.sqrt(2 / fan_in)
        if init == "orthogonal":
            assert np.allclose(w @ w.T, np.eye(fan_out))

    def test_unknown_init_raises(self):
        with pytest.raises(ValueError, match="Unknown init"):
            mlp([2, 3], init="zeros")


class TestTorchMlp:
    """Tests for torch_mlp."""

    def test_widths_and_add_last_relu(self, seeded):
        assert seeded is not None
        net = cast(Any, torch_mlp(widths=[3, 5, 2], add_last_relu=True))
        if isinstance(net, nn.Sequential):
            layers = list(net.children())
            assert len([lyr for lyr in layers if isinstance(lyr, nn.Linear)]) == 2
            assert len([lyr for lyr in layers if isinstance(lyr, nn.ReLU)]) == 2
        else:
            assert net.input_shape == (3,)
            assert len([lyr for lyr in net.layers.values() if isinstance(lyr, LinearLayer)]) == 2
            assert len([lyr for lyr in net.layers.values() if isinstance(lyr, ReLULayer)]) == 2
        assert net.widths == [3, 5, 2]

    def test_no_last_relu(self, seeded):
        assert seeded is not None
        net = cast(Any, torch_mlp(widths=[2, 4, 1], add_last_relu=False))
        if isinstance(net, nn.Sequential):
            assert len([lyr for lyr in net.children() if isinstance(lyr, nn.ReLU)]) == 1
        else:
            assert net.input_shape == (2,)
            assert len([lyr for lyr in net.layers.values() if isinstance(lyr, ReLULayer)]) == 1

    def test_single_hidden(self, seeded):
        assert seeded is not None
        net = cast(Any, torch_mlp(widths=[4, 8], add_last_relu=True))
        if isinstance(net, nn.Sequential):
            assert len([lyr for lyr in net.children() if isinstance(lyr, nn.ReLU)]) == 1
        else:
            assert net.input_shape == (4,)
            assert net.num_relus == 1

    def test_forward_shape(self, seeded):
        assert seeded is not None
        net = cast(Any, torch_mlp(widths=[5, 10, 3], add_last_relu=False))
        x = torch.randn(2, 5, device=net.device, dtype=net.dtype)
        y = net(x)
        assert y.shape == (2, 3)


class TestReLUNetwork:
    """Tests for ReLUNetwork class."""

    def test_network_type(self, seeded):
        assert seeded is not None
        net = cast(Any, torch_mlp(widths=[2, 4, 1]))
        assert isinstance(net, (ReLUNetwork, nn.Sequential))

    def test_device_dtype(self, seeded):
        assert seeded is not None
        net = cast(Any, torch_mlp(widths=[2, 4, 1]))
        assert net.device == "cpu"
        assert net.dtype == torch.float64

    def test_num_relus(self, seeded):
        assert seeded is not None
        net = cast(Any, torch_mlp(widths=[2, 6, 4, 1], add_last_relu=True))
        if isinstance(net, nn.Sequential):
            assert len([lyr for lyr in net.children() if isinstance(lyr, nn.ReLU)]) == 3
        else:
            assert net.num_relus == 3

    def test_get_all_layer_outputs(self, seeded):
        assert seeded is not None
        net = mlp(widths=[3, 5, 2])
        x = np.random.randn(4, 3)
        outs = net.get_all_layer_outputs(x)
        assert list(outs) == list(net.layers)
        assert all(isinstance(t, np.ndarray) and t.shape[0] == 4 for t in outs.values())
        assert np.allclose(outs["fc1"], net.forward(x))

    def test_shi2weights_return_tensor(self, seeded):
        assert seeded is not None
        net = mlp(widths=[4, 8, 2])
        w = net.shi2weights(0, return_idx=False)
        assert not isinstance(w, tuple)
        assert w.shape == (4,)

    def test_shi2weights_return_idx(self, seeded):
        assert seeded is not None
        net = mlp(widths=[4, 8, 2])
        name, idx = net.shi2weights(3, return_idx=True)
        assert isinstance(name, str)
        assert isinstance(idx, int)
        assert 0 <= idx < 8

    def test_shi2weights_invalid_raises(self, seeded):
        assert seeded is not None
        net = mlp(widths=[4, 8, 2])
        with pytest.raises(ValueError, match="Invalid Neuron Index"):
            net.shi2weights(1000, return_idx=False)
