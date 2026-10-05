"""Tests for relucent.model.convert_model."""

from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn

from relucent.model import FlattenLayer, LinearLayer, ReLULayer, ReLUNetwork
from relucent.model.builders import TorchMLP, torch_mlp
from relucent.model.convert_model import avgpool2d_to_affine, combine_linear_layers, convert, torch_conv_layer_to_affine


class TestCombineLinearLayers:
    def test_two_linears_combined(self, seeded):
        assert seeded is not None
        a = nn.Linear(4, 6)
        b = nn.Linear(6, 2)
        layers: OrderedDict[str, nn.Module] = OrderedDict([("a", a), ("b", b)])
        merged = combine_linear_layers(layers)
        assert len(merged) == 1
        name = next(iter(merged))
        assert "+" in name
        layer = merged[name]
        assert layer.in_features == 4 and layer.out_features == 2

    def test_linear_relu_linear_not_merged(self, seeded):
        assert seeded is not None
        layers: OrderedDict[str, nn.Module] = OrderedDict(
            [
                ("fc0", nn.Linear(3, 5)),
                ("relu0", nn.ReLU()),
                ("fc1", nn.Linear(5, 2)),
            ]
        )
        merged = combine_linear_layers(layers)
        assert len(merged) == 3

    def test_forward_preserved(self, seeded):
        assert seeded is not None
        a = nn.Linear(4, 6)
        b = nn.Linear(6, 2)
        layers: OrderedDict[str, nn.Module] = OrderedDict([("a", a), ("b", b)])
        merged = combine_linear_layers(layers)
        m = next(iter(merged.values()))
        x = torch.randn(2, 4)
        y_orig = b(a(x))
        y_merged = m(x)
        assert torch.allclose(y_orig, y_merged)

    def test_three_linears_combined(self, seeded):
        assert seeded is not None
        a = nn.Linear(4, 6)
        b = nn.Linear(6, 5)
        c = nn.Linear(5, 2)
        layers: OrderedDict[str, nn.Module] = OrderedDict([("a", a), ("b", b), ("c", c)])
        merged = combine_linear_layers(layers)
        assert len(merged) == 1
        m = next(iter(merged.values()))
        assert m.in_features == 4 and m.out_features == 2
        x = torch.randn(3, 4)
        assert torch.allclose(c(b(a(x))), m(x), atol=1e-5)

    def test_single_linear_passes_through(self, seeded):
        assert seeded is not None
        a = nn.Linear(4, 2)
        layers: OrderedDict[str, nn.Module] = OrderedDict([("a", a)])
        merged = combine_linear_layers(layers)
        assert len(merged) == 1
        m = next(iter(merged.values()))
        assert m.in_features == 4 and m.out_features == 2

    def test_non_linear_only_preserved(self, seeded):
        assert seeded is not None
        layers: OrderedDict[str, nn.Module] = OrderedDict([("relu0", nn.ReLU())])
        merged = combine_linear_layers(layers)
        assert list(merged.keys()) == ["relu0"]

    def test_combined_name_uses_plus(self, seeded):
        assert seeded is not None
        a = nn.Linear(4, 6)
        b = nn.Linear(6, 2)
        layers: OrderedDict[str, nn.Module] = OrderedDict([("fc0", a), ("fc1", b)])
        merged = combine_linear_layers(layers)
        assert next(iter(merged)) == "fc0+fc1"

    def test_linear_relu_ordering_preserved(self, seeded):
        assert seeded is not None
        layers: OrderedDict[str, nn.Module] = OrderedDict(
            [
                ("fc0", nn.Linear(3, 5)),
                ("relu0", nn.ReLU()),
                ("fc1", nn.Linear(5, 4)),
                ("relu1", nn.ReLU()),
                ("fc2", nn.Linear(4, 2)),
            ]
        )
        merged = combine_linear_layers(layers)
        assert list(merged.keys()) == ["fc0", "relu0", "fc1", "relu1", "fc2"]


class TestTorchConvLayerToAffine:
    def test_returns_linear(self, seeded):
        assert seeded is not None
        conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        fc = torch_conv_layer_to_affine(conv, (1, 8, 8))
        assert isinstance(fc, nn.Linear)

    def test_feature_counts_with_same_padding(self, seeded):
        assert seeded is not None
        conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        fc = torch_conv_layer_to_affine(conv, (1, 8, 8))
        assert fc.in_features == 1 * 8 * 8
        assert fc.out_features == 4 * 8 * 8

    def test_feature_counts_without_padding(self, seeded):
        assert seeded is not None
        conv = nn.Conv2d(1, 2, kernel_size=3, padding=0)
        fc = torch_conv_layer_to_affine(conv, (1, 8, 8))
        assert fc.in_features == 1 * 8 * 8
        assert fc.out_features == 2 * 6 * 6

    def test_numerical_equivalence_with_padding(self, seeded):
        assert seeded is not None
        conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
        input_size = (2, 6, 6)
        fc = torch_conv_layer_to_affine(conv, input_size)
        x = torch.randn(1, *input_size)
        assert torch.allclose(conv(x).reshape(-1), fc(x.reshape(-1)), atol=1e-5)

    def test_numerical_equivalence_no_padding(self, seeded):
        assert seeded is not None
        conv = nn.Conv2d(1, 2, kernel_size=3, padding=0)
        input_size = (1, 8, 8)
        fc = torch_conv_layer_to_affine(conv, input_size)
        x = torch.randn(1, *input_size)
        assert torch.allclose(conv(x).reshape(-1), fc(x.reshape(-1)), atol=1e-5)

    def test_numerical_equivalence_with_stride(self, seeded):
        assert seeded is not None
        conv = nn.Conv2d(1, 2, kernel_size=2, stride=2, padding=0)
        input_size = (1, 8, 8)
        fc = torch_conv_layer_to_affine(conv, input_size)
        x = torch.randn(1, *input_size)
        assert torch.allclose(conv(x).reshape(-1), fc(x.reshape(-1)), atol=1e-5)

    def test_numerical_equivalence_multichannel(self, seeded):
        assert seeded is not None
        conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        input_size = (3, 6, 6)
        fc = torch_conv_layer_to_affine(conv, input_size)
        x = torch.randn(1, *input_size)
        assert torch.allclose(conv(x).reshape(-1), fc(x.reshape(-1)), atol=1e-5)


class TestAvgPool2dToAffine:
    def test_returns_linear(self, seeded):
        assert seeded is not None
        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        fc = avgpool2d_to_affine(pool, (1, 8, 8))
        assert isinstance(fc, nn.Linear)

    def test_feature_counts(self, seeded):
        assert seeded is not None
        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        fc = avgpool2d_to_affine(pool, (1, 8, 8))
        assert fc.in_features == 1 * 8 * 8
        assert fc.out_features == 1 * 4 * 4

    def test_numerical_equivalence_int_kernel(self, seeded):
        assert seeded is not None
        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        input_size = (2, 8, 8)
        fc = avgpool2d_to_affine(pool, input_size)
        x = torch.randn(1, *input_size)
        assert torch.allclose(pool(x).reshape(-1), fc(x.reshape(-1)), atol=1e-5)

    def test_numerical_equivalence_tuple_kernel(self, seeded):
        assert seeded is not None
        pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        input_size = (1, 8, 8)
        fc = avgpool2d_to_affine(pool, input_size)
        x = torch.randn(1, *input_size)
        assert torch.allclose(pool(x).reshape(-1), fc(x.reshape(-1)), atol=1e-5)

    def test_numerical_equivalence_multichannel(self, seeded):
        assert seeded is not None
        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        input_size = (4, 8, 8)
        fc = avgpool2d_to_affine(pool, input_size)
        x = torch.randn(1, *input_size)
        assert torch.allclose(pool(x).reshape(-1), fc(x.reshape(-1)), atol=1e-5)


class TestConvert:
    def test_affine_array_like_pairs_are_accepted(self, seeded):
        assert seeded is not None
        w0 = np.random.randn(6, 4).tolist()
        b0 = np.random.randn(6).tolist()
        w1 = np.random.randn(3, 6).tolist()
        b1 = np.random.randn(3).tolist()
        canonical = convert([[w0, b0], [w1, b1]])
        assert isinstance(canonical, ReLUNetwork)
        assert canonical.input_shape == (4,)
        linear_layers = [layer for layer in canonical.layers.values() if isinstance(layer, LinearLayer)]
        assert len(linear_layers) == 2
        assert linear_layers[0].weight.shape == (6, 4)
        assert linear_layers[1].weight.shape == (3, 6)

    def test_mlp_roundtrip(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[4, 8, 3])
        assert isinstance(net, TorchMLP)
        canonical = convert(net)
        assert isinstance(canonical, ReLUNetwork)
        assert canonical.input_shape == (4,)
        x = torch.randn(2, 4, dtype=torch.float64)
        y_orig = net(x)
        y_can = canonical(x)
        assert torch.allclose(y_orig, torch.as_tensor(y_can, dtype=y_orig.dtype), atol=1e-5)

    def test_conv2d_model_roundtrip(self, seeded):
        assert seeded is not None
        C, H, W = 1, 8, 8
        conv = nn.Conv2d(C, 4, kernel_size=3, padding=1)
        relu = nn.ReLU()
        flatten = nn.Flatten()
        fc = nn.Linear(4 * H * W, 2)
        layers: OrderedDict[str, nn.Module] = OrderedDict([("conv", conv), ("relu", relu), ("flatten", flatten), ("fc", fc)])
        canonical = convert(layers, input_shape=(C, H, W))
        assert isinstance(canonical, ReLUNetwork)
        x = torch.randn(1, C, H, W)
        y_orig = fc(flatten(relu(conv(x))))
        y_can = canonical(x.reshape(1, -1))
        assert torch.allclose(y_orig, torch.as_tensor(y_can, dtype=y_orig.dtype), atol=1e-4)

    def test_avgpool2d_model_roundtrip(self, seeded):
        assert seeded is not None
        C, H, W = 1, 8, 8
        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        flatten = nn.Flatten()
        fc = nn.Linear(C * (H // 2) * (W // 2), 2)
        layers: OrderedDict[str, nn.Module] = OrderedDict([("pool", pool), ("flatten", flatten), ("fc", fc)])
        canonical = convert(layers, input_shape=(C, H, W))
        assert isinstance(canonical, ReLUNetwork)
        x = torch.randn(1, C, H, W)
        y_orig = fc(flatten(pool(x)))
        y_can = canonical(x.reshape(1, -1))
        assert torch.allclose(y_orig, torch.as_tensor(y_can, dtype=y_orig.dtype), atol=1e-4)

    def test_dropout_is_stripped(self, seeded):
        assert seeded is not None
        layers: OrderedDict[str, nn.Module] = OrderedDict(
            [
                ("fc0", nn.Linear(4, 8)),
                ("dropout", nn.Dropout(0.5)),
                ("relu0", nn.ReLU()),
                ("fc1", nn.Linear(8, 2)),
            ]
        )
        canonical = convert(layers, input_shape=(4,))
        assert not any(isinstance(layer, nn.Dropout) for layer in canonical.layers.values())

    def test_logsoftmax_stops_conversion(self, seeded):
        assert seeded is not None
        layers: OrderedDict[str, nn.Module] = OrderedDict(
            [
                ("fc0", nn.Linear(4, 8)),
                ("relu0", nn.ReLU()),
                ("fc1", nn.Linear(8, 3)),
                ("logsoftmax", nn.LogSoftmax(dim=1)),
            ]
        )
        canonical = convert(layers, input_shape=(4,))
        assert not any(isinstance(layer, nn.LogSoftmax) for layer in canonical.layers.values())
        assert any(isinstance(layer, LinearLayer) and layer.weight.shape[0] == 3 for layer in canonical.layers.values())

    def test_unsupported_layer_raises(self, seeded):
        assert seeded is not None
        layers: OrderedDict[str, nn.Module] = OrderedDict(
            [
                ("fc0", nn.Linear(4, 4)),
                ("sigmoid", nn.Sigmoid()),
            ]
        )
        with pytest.raises(ValueError, match="not supported"):
            convert(layers, input_shape=(4,))

    def test_output_is_linear_relu_only(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[4, 8, 3])
        canonical = convert(net)
        for layer in canonical.layers.values():
            assert isinstance(layer, (LinearLayer, ReLULayer, FlattenLayer))

    def test_consecutive_linears_are_merged(self, seeded):
        assert seeded is not None
        layers: OrderedDict[str, nn.Module] = OrderedDict(
            [
                ("fc0", nn.Linear(4, 8)),
                ("fc1", nn.Linear(8, 3)),
            ]
        )
        canonical = convert(layers, input_shape=(4,))
        linear_layers = [layer for layer in canonical.layers.values() if isinstance(layer, LinearLayer)]
        assert len(linear_layers) == 1
        assert linear_layers[0].weight.shape[1] == 4
        assert linear_layers[0].weight.shape[0] == 3


class _Residual(nn.Module):
    """``forward`` adds a skip connection, so it is not the composition of its children."""

    def __init__(self) -> None:
        super().__init__()
        self.fc0 = nn.Linear(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.relu(self.fc0(x)) + x)


class TestConvertChecksSource:
    @pytest.mark.parametrize(("h", "w", "stride", "padding"), [(6, 6, 1, 0), (6, 5, 1, 1), (7, 7, 2, 0), (6, 6, 1, "valid")])
    def test_conv_sequential_converts(self, seeded, h, w, stride, padding):
        """An ``nn.Sequential`` conv net (with a Flatten) converts, checked against its own forward pass."""
        assert seeded is not None
        conv = nn.Conv2d(1, 2, kernel_size=3, stride=stride, padding=padding)
        x = torch.randn(4, 1, h, w)
        n_out = conv(x).shape[1:].numel()
        model = nn.Sequential(conv, nn.ReLU(), nn.Flatten(), nn.Linear(n_out, 1))
        canonical = convert(model, input_shape=(1, h, w))
        y = torch.as_tensor(canonical(x.reshape(4, -1)))
        assert torch.allclose(model(x), y.to(torch.float32), atol=1e-5)

    def test_skip_connection_raises(self, seeded):
        assert seeded is not None
        with pytest.raises(ValueError, match="composes the model's child modules in order"):
            convert(_Residual())

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"dilation": 2}, "dilation"),
            ({"groups": 2}, "groups"),
            ({"padding": "same"}, "padding="),
            ({"padding": 1, "padding_mode": "reflect"}, "padding_mode"),
        ],
    )
    def test_unsupported_conv_options_raise(self, seeded, kwargs, match):
        assert seeded is not None
        conv = nn.Conv2d(2, 2, kernel_size=3, **kwargs)
        model = nn.Sequential(conv, nn.ReLU(), nn.Flatten(), nn.Linear(conv(torch.zeros(1, 2, 7, 7)).shape[1:].numel(), 1))
        with pytest.raises(ValueError, match=match):
            convert(model, input_shape=(2, 7, 7))

    @pytest.mark.parametrize(
        ("kwargs", "match"), [({"ceil_mode": True}, "ceil_mode"), ({"divisor_override": 3}, "divisor_override")]
    )
    def test_unsupported_avgpool_options_raise(self, seeded, kwargs, match):
        assert seeded is not None
        pool = nn.AvgPool2d(kernel_size=2, stride=2, **kwargs)
        with pytest.raises(ValueError, match=match):
            convert(nn.Sequential(pool, nn.Flatten(), nn.Linear(16, 1)), input_shape=(1, 8, 8))

    def test_global_rng_is_untouched(self, seeded):
        assert seeded is not None
        model = torch_mlp(widths=[3, 5, 1])
        state = torch.random.get_rng_state()
        convert(model)
        assert torch.equal(torch.random.get_rng_state(), state)

    def test_relu_network_input_is_not_mutated(self, seeded):
        assert seeded is not None
        net = convert([[np.eye(4), np.zeros(4)], [np.ones((1, 4)), np.zeros(1)]])
        out = convert(net, input_shape=(2, 2))
        assert net.input_shape == (4,)
        assert out.input_shape == (2, 2)
