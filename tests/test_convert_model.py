"""Tests for relucent.convert_model."""

from collections import OrderedDict

import torch
import torch.nn as nn

from relucent.convert_model import combine_linear_layers, convert, flatten_to_affine
from relucent.model import NN, get_mlp_model


class TestFlattenToAffine:
    def test_shape(self):
        out = flatten_to_affine((3, 4, 5))
        assert isinstance(out, nn.Linear)
        assert out.in_features == out.out_features == 60


class TestCombineLinearLayers:
    def test_two_linears_combined(self, seeded):
        a = nn.Linear(4, 6)
        b = nn.Linear(6, 2)
        layers = OrderedDict([("a", a), ("b", b)])
        merged = combine_linear_layers(layers)
        assert len(merged) == 1
        name = next(iter(merged))
        assert "+" in name
        layer = merged[name]
        assert layer.in_features == 4 and layer.out_features == 2

    def test_linear_relu_linear_not_merged(self, seeded):
        layers = OrderedDict(
            [
                ("fc0", nn.Linear(3, 5)),
                ("relu0", nn.ReLU()),
                ("fc1", nn.Linear(5, 2)),
            ]
        )
        merged = combine_linear_layers(layers)
        assert len(merged) == 3

    def test_forward_preserved(self, seeded):
        a = nn.Linear(4, 6)
        b = nn.Linear(6, 2)
        layers = OrderedDict([("a", a), ("b", b)])
        merged = combine_linear_layers(layers)
        m = next(iter(merged.values()))
        x = torch.randn(2, 4)
        y_orig = b(a(x))
        y_merged = m(x)
        assert torch.allclose(y_orig, y_merged)


class TestConvert:
    def test_mlp_roundtrip(self, seeded):
        net = get_mlp_model(widths=[4, 8, 3])
        canonical = convert(net)
        assert isinstance(canonical, NN)
        assert canonical.input_shape == (4,)
        x = torch.randn(2, 4, device=net.device, dtype=net.dtype)
        y_orig = net(x)
        y_can = canonical(x)
        assert torch.allclose(y_orig, y_can, atol=1e-5)
