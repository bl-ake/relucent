"""Tests for relucent.model and relucent.utils.mlp."""

from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from relucent.model import LinearLayer, ReLULayer, ReLUNetwork
from relucent.utils import mlp


class TestMlp:
    """Tests for mlp."""

    def test_widths_and_add_last_relu(self, seeded):
        assert seeded is not None
        net = cast(Any, mlp(widths=[3, 5, 2], add_last_relu=True))
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
        net = cast(Any, mlp(widths=[2, 4, 1], add_last_relu=False))
        if isinstance(net, nn.Sequential):
            assert len([lyr for lyr in net.children() if isinstance(lyr, nn.ReLU)]) == 1
        else:
            assert net.input_shape == (2,)
            assert len([lyr for lyr in net.layers.values() if isinstance(lyr, ReLULayer)]) == 1

    def test_single_hidden(self, seeded):
        assert seeded is not None
        net = cast(Any, mlp(widths=[4, 8], add_last_relu=True))
        if isinstance(net, nn.Sequential):
            assert len([lyr for lyr in net.children() if isinstance(lyr, nn.ReLU)]) == 1
        else:
            assert net.input_shape == (4,)
            assert net.num_relus == 1

    def test_forward_shape(self, seeded):
        assert seeded is not None
        net = cast(Any, mlp(widths=[5, 10, 3], add_last_relu=False))
        x = torch.randn(2, 5, device=net.device, dtype=net.dtype)
        y = net(x)
        assert y.shape == (2, 3)


class TestReLUNetwork:
    """Tests for ReLUNetwork class."""

    def test_network_type(self, seeded):
        assert seeded is not None
        net = cast(Any, mlp(widths=[2, 4, 1]))
        assert isinstance(net, (ReLUNetwork, nn.Sequential))

    def test_device_dtype(self, seeded):
        assert seeded is not None
        net = cast(Any, mlp(widths=[2, 4, 1]))
        assert net.device == "cpu"
        assert net.dtype == torch.float64

    def test_num_relus(self, seeded):
        assert seeded is not None
        net = cast(Any, mlp(widths=[2, 6, 4, 1], add_last_relu=True))
        if isinstance(net, nn.Sequential):
            assert len([lyr for lyr in net.children() if isinstance(lyr, nn.ReLU)]) == 3
        else:
            assert net.num_relus == 3

    def test_get_all_layer_outputs(self, seeded):
        assert seeded is not None
        net = cast(Any, mlp(widths=[3, 5, 2]))
        if isinstance(net, nn.Sequential):
            pytest.skip("get_all_layer_outputs is specific to canonical wrapper")
        x = torch.randn(4, 3, device=net.device, dtype=net.dtype)
        outs = net.get_all_layer_outputs(x)
        assert isinstance(outs, dict)
        names = list(outs.keys())
        assert len(names) == len(net.layers)
        for _n, t in outs.items():
            assert isinstance(t, torch.Tensor)

    def test_shi2weights_return_tensor(self, seeded):
        assert seeded is not None
        net = cast(Any, mlp(widths=[4, 8, 2]))
        if isinstance(net, nn.Sequential):
            pytest.skip("shi2weights is specific to canonical wrapper")
        w = net.shi2weights(0, return_idx=False)
        assert not isinstance(w, tuple)
        assert w.shape == (4,)

    def test_shi2weights_return_idx(self, seeded):
        assert seeded is not None
        net = cast(Any, mlp(widths=[4, 8, 2]))
        if isinstance(net, nn.Sequential):
            pytest.skip("shi2weights is specific to canonical wrapper")
        name, idx = net.shi2weights(3, return_idx=True)  # type: ignore[misc]
        assert isinstance(name, str)
        assert isinstance(idx, int)
        assert 0 <= idx < 8

    def test_shi2weights_invalid_raises(self, seeded):
        assert seeded is not None
        net = cast(Any, mlp(widths=[4, 8, 2]))
        if isinstance(net, nn.Sequential):
            pytest.skip("shi2weights is specific to canonical wrapper")
        with pytest.raises(ValueError, match="Invalid Neuron Index"):
            net.shi2weights(1000, return_idx=False)
