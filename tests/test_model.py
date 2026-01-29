"""Tests for relucent.model (NN, get_mlp_model)."""

import pytest
import torch
import torch.nn as nn

from relucent.model import get_mlp_model


class TestGetMlpModel:
    """Tests for get_mlp_model."""

    def test_widths_and_add_last_relu(self, seeded):
        net = get_mlp_model(widths=[3, 5, 2], add_last_relu=True)
        assert net.input_shape == (3,)
        assert len([lyr for lyr in net.layers.values() if isinstance(lyr, nn.Linear)]) == 2
        assert len([lyr for lyr in net.layers.values() if isinstance(lyr, nn.ReLU)]) == 2
        assert net.widths == [3, 5, 2]

    def test_no_last_relu(self, seeded):
        net = get_mlp_model(widths=[2, 4, 1], add_last_relu=False)
        assert net.input_shape == (2,)
        assert len([lyr for lyr in net.layers.values() if isinstance(lyr, nn.ReLU)]) == 1

    def test_single_hidden(self, seeded):
        net = get_mlp_model(widths=[4, 8], add_last_relu=True)
        assert net.input_shape == (4,)
        assert net.num_relus == 1

    def test_forward_shape(self, seeded):
        net = get_mlp_model(widths=[5, 10, 3], add_last_relu=False)
        x = torch.randn(2, 5, device=net.device, dtype=net.dtype)
        y = net(x)
        assert y.shape == (2, 3)


class TestNN:
    """Tests for NN class."""

    def test_save_numpy_weights(self, seeded):
        net = get_mlp_model(widths=[2, 4, 1])
        net.save_numpy_weights()
        for layer in net.layers.values():
            if isinstance(layer, nn.Linear):
                assert hasattr(layer, "weight_cpu") and hasattr(layer, "bias_cpu")
                assert layer.weight_cpu.shape == layer.weight.shape
                assert layer.bias_cpu.shape == (1, layer.bias.numel())

    def test_device_dtype(self, seeded):
        net = get_mlp_model(widths=[2, 4, 1])
        assert net.device == next(net.parameters()).device
        assert net.dtype == next(net.parameters()).dtype

    def test_num_relus(self, seeded):
        net = get_mlp_model(widths=[2, 6, 4, 1], add_last_relu=True)
        assert net.num_relus == 3

    def test_get_all_layer_outputs(self, seeded):
        net = get_mlp_model(widths=[3, 5, 2])
        x = torch.randn(4, 3, device=net.device, dtype=net.dtype)
        outs = net.get_all_layer_outputs(x)
        assert isinstance(outs, dict)
        names = list(outs.keys())
        assert len(names) == len(net.layers)
        for n, t in outs.items():
            assert isinstance(t, torch.Tensor)

    def test_get_grid(self, seeded):
        net = get_mlp_model(widths=[2, 4, 1])
        x, y, pts = net.get_grid(bounds=2, res=10)
        assert x.shape == (10,)
        assert y.shape == (10,)
        assert pts.shape == (100, 2)

    def test_output_grid(self, seeded):
        net = get_mlp_model(widths=[2, 4, 1])
        x, y, outs = net.output_grid(bounds=2, res=5)
        assert len(outs) == len(net.layers)
        last_name = list(outs.keys())[-1]
        assert outs[last_name].shape == (25, 1)

    def test_shi2weights_return_tensor(self, seeded):
        net = get_mlp_model(widths=[4, 8, 2])
        net.save_numpy_weights()
        w = net.shi2weights(0, return_idx=False)
        assert isinstance(w, torch.Tensor)
        assert w.shape == (4,)

    def test_shi2weights_return_idx(self, seeded):
        net = get_mlp_model(widths=[4, 8, 2])
        name, idx = net.shi2weights(3, return_idx=True)
        assert isinstance(name, str)
        assert isinstance(idx, int)
        assert 0 <= idx < 8

    def test_shi2weights_invalid_raises(self, seeded):
        net = get_mlp_model(widths=[4, 8, 2])
        with pytest.raises(ValueError, match="Invalid Neuron Index"):
            net.shi2weights(1000, return_idx=False)
