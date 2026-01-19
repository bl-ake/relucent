from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .model import NN


# https://gist.github.com/vvolhejn/e265665c65d3df37e381316bf57b8421
@torch.no_grad()
def torch_conv_layer_to_affine(conv: torch.nn.Conv2d, input_size: Tuple[int, int, int]) -> torch.nn.Linear:
    def range2d(to_a, to_b):
        for a in range(to_a):
            for b in range(to_b):
                yield a, b

    def enc_tuple(tup: Tuple, shape: Tuple) -> int:
        res = 0
        coef = 1
        for i in reversed(range(len(shape))):
            assert tup[i] < shape[i]
            res += coef * tup[i]
            coef *= shape[i]

        return res

    def dec_tuple(x: int, shape: Tuple) -> Tuple:
        res = []
        for i in reversed(range(len(shape))):
            res.append(x % shape[i])
            x //= shape[i]

        return tuple(reversed(res))

    nfeatures, w, h = input_size

    # Formula from the Torch docs:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    output_size = [
        (input_size[i + 1] + 2 * conv.padding[i] - conv.kernel_size[i]) // conv.stride[i] + 1 for i in [0, 1]
    ]

    in_shape = (conv.in_channels, w, h)
    out_shape = (conv.out_channels, output_size[0], output_size[1])

    fc = nn.Linear(in_features=np.product(in_shape), out_features=np.product(out_shape), device=conv.weight.device)
    fc.weight.data.fill_(0.0)

    # Output coordinates
    for xo, yo in tqdm(
        range2d(output_size[0], output_size[1]),
        desc="Converting Conv2d to Linear",
        total=output_size[0] * output_size[1],
        leave=False,
    ):
        # The upper-left corner of the filter in the input tensor
        xi0 = -conv.padding[0] + conv.stride[0] * xo
        yi0 = -conv.padding[1] + conv.stride[1] * yo

        # Position within the filter
        for xd, yd in range2d(conv.kernel_size[0], conv.kernel_size[1]):
            # Output channel
            for co in range(conv.out_channels):
                fc.bias[enc_tuple((co, xo, yo), out_shape)] = conv.bias[co]
                for ci in range(conv.in_channels):
                    # Make sure we are within the input image (and not in the padding)
                    if 0 <= xi0 + xd < w and 0 <= yi0 + yd < h:
                        cw = conv.weight[co, ci, xd, yd]
                        # Flatten the weight position to 1d in "canonical ordering",
                        # i.e. guaranteeing that:
                        # FC(img.reshape(-1)) == Conv(img).reshape(-1)
                        fc.weight[
                            enc_tuple((co, xo, yo), out_shape),
                            enc_tuple((ci, xi0 + xd, yi0 + yd), in_shape),
                        ] = cw

    return fc


@torch.no_grad()
def avgpool2d_to_affine(avgpool: torch.nn.AvgPool2d, input_size: Tuple[int, int, int]) -> torch.nn.Linear:
    # https://www.researchgate.net/figure/The-mean-pooling-is-described-with-the-matrix-multiplication-of-the-reshaped-feature-map_fig2_357833254
    conv2d = nn.Conv2d(
        in_channels=input_size[0],
        out_channels=input_size[0],
        kernel_size=avgpool.kernel_size,
        stride=avgpool.stride,
        padding=avgpool.padding,
        bias=True,
    )
    conv2d.weight.data.fill_(0.0)
    for i in range(input_size[0]):
        conv2d.weight.data[i, i, :, :].fill_(1.0 / (avgpool.kernel_size**2))
    conv2d.bias.data.fill_(0.0)
    return torch_conv_layer_to_affine(conv2d, input_size)


@torch.no_grad()
def flatten_to_affine(input_size: Tuple[int, int, int]) -> torch.nn.Linear:
    nn.Linear(in_features=np.product(input_size), out_features=np.product(input_size))


def combine_linear_layers(old_layers):
    new_layers = OrderedDict()
    current_linear = None
    current_name = ""
    for name, layer in old_layers.items():
        if isinstance(layer, nn.Linear):
            if current_linear is None:
                current_linear = layer
                current_name = name
            else:
                # Combine current linear with the next linear layer
                new_weight = layer.weight @ current_linear.weight
                if current_linear.bias is None:
                    new_bias = layer.bias
                else:
                    new_bias = layer.weight @ current_linear.bias + (layer.bias if layer.bias is not None else 0)
                current_linear = nn.Linear(current_linear.in_features, layer.out_features)
                current_linear.weight.data = new_weight
                current_linear.bias.data = new_bias
                current_name = f"{current_name}+{name}"
        else:
            if current_linear is not None:
                new_layers[current_name] = current_linear
                current_linear = None
                current_name = ""
            new_layers[name] = layer
    if current_linear is not None:
        new_layers[current_name] = current_linear
    return new_layers


@torch.no_grad()
def convert(model: nn.Module) -> nn.Module:
    ## TODO: Write a test
    x = torch.zeros((1, *model.input_shape), dtype=next(model.parameters()).dtype, device=model.device)
    layers = OrderedDict()
    assert "Flatten Input" not in model.layers
    layers["Flatten Input"] = nn.Flatten()
    print("\nConverting model to canonical format")
    for name, module in list(model.layers.items()):
        print("    Layer:", name)
        if isinstance(module, (nn.Linear, nn.ReLU)):
            layers[name] = module
        elif isinstance(module, nn.Dropout):
            pass
        elif isinstance(module, nn.Flatten):
            pass
        elif isinstance(module, nn.LogSoftmax):
            break
        elif isinstance(module, nn.Conv2d):
            new_layer = torch_conv_layer_to_affine(module, x.shape[1:]).to(model.device, model.dtype)
            layers[name] = new_layer
        elif isinstance(module, nn.AvgPool2d) and module.kernel_size == module.stride:
            new_layer = avgpool2d_to_affine(module, x.shape[1:]).to(model.device, model.dtype)
            layers[name] = new_layer
        else:
            raise ValueError(f"Module {name} is not supported: {module}")
        x = module(x)
        module.to(model.device)
    layers = combine_linear_layers(layers)
    new_model = NN(layers=layers, input_shape=(np.prod(model.input_shape, dtype=int),)).to(model.device, model.dtype)
    return new_model
