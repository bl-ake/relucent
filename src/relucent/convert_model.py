"""Convert PyTorch models to the canonical NN format.

This module provides utilities to convert various PyTorch model architectures
(including Conv2d, AvgPool2d, etc.) into the canonical format used by relucent,
which consists of Linear and ReLU layers only.
"""

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from relucent._torch_compat import nn, torch
from relucent.model import FlattenLayer, LinearLayer, ReLULayer, ReLUNetwork

__all__ = ["convert"]

AffineLayerTuple = tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]


def _canonicalize_layer(layer: object) -> LinearLayer | ReLULayer | FlattenLayer:
    if isinstance(layer, (LinearLayer, ReLULayer, FlattenLayer)):
        return layer
    kind = type(layer).__name__.lower()
    if isinstance(layer, nn.Linear):
        weight = np.asarray(layer.weight.detach().cpu().numpy(), dtype=np.float64)
        if layer.bias is None:
            bias = np.zeros((1, weight.shape[0]), dtype=weight.dtype)
        else:
            bias = np.asarray(layer.bias.detach().cpu().numpy(), dtype=weight.dtype).reshape(1, -1)
        return LinearLayer(weight=weight, bias=bias)
    if isinstance(layer, nn.ReLU) or "relu" in kind:
        return ReLULayer()
    if isinstance(layer, nn.Flatten) or "flatten" in kind:
        return FlattenLayer()
    raise ValueError(f"Unsupported layer type: {type(layer)}")


def _canonicalize_named_layers(layers: Mapping[str, object]) -> OrderedDict[str, LinearLayer | ReLULayer | FlattenLayer]:
    return OrderedDict((name, _canonicalize_layer(layer)) for name, layer in layers.items())


def _is_affine_tuple_layer(item: object) -> bool:
    return isinstance(item, tuple) and len(item) == 2


def _canonical_from_affine_tuples(
    layers: Sequence[AffineLayerTuple],
) -> OrderedDict[str, LinearLayer | ReLULayer]:
    canonical: OrderedDict[str, LinearLayer | ReLULayer] = OrderedDict()
    for i, (w_raw, b_raw) in enumerate(layers):
        w = np.asarray(w_raw, dtype=np.float64)
        b = np.asarray(b_raw, dtype=np.float64)
        if w.ndim != 2:
            raise ValueError(f"Layer {i}: weight matrix must be 2D, got shape {w.shape}")
        if b.ndim == 1:
            b = b.reshape(1, -1)
        if b.shape != (1, w.shape[0]):
            raise ValueError(f"Layer {i}: bias must have shape ({1}, {w.shape[0]}) or ({w.shape[0]},), got {b.shape}")
        canonical[f"fc{i}"] = LinearLayer(weight=w, bias=b)
        if i < len(layers) - 1:
            canonical[f"relu{i}"] = ReLULayer()
    return canonical


# https://gist.github.com/vvolhejn/e265665c65d3df37e381316bf57b8421
@torch.no_grad()
def torch_conv_layer_to_affine(conv: nn.Conv2d, input_size: tuple[int, int, int]) -> nn.Linear:
    """Convert a Conv2d layer to an equivalent Linear layer.

    Args:
        conv: The Conv2d layer to convert.
        input_size: Input size as (channels, height, width) tuple.

    Returns:
        nn.Linear: A Linear layer that performs the equivalent operation.

    Reference:
        Based on: https://gist.github.com/vvolhejn/e265665c65d3df37e381316bf57b8421
    """

    def range2d(to_a: int, to_b: int) -> Iterable[tuple[int, int]]:
        for a in range(to_a):
            for b in range(to_b):
                yield a, b

    def enc_tuple(tup: tuple[int, int, int], shape: tuple[int, int, int]) -> int:
        res = 0
        coef = 1
        for i in reversed(range(len(shape))):
            assert tup[i] < shape[i]
            res += coef * tup[i]
            coef *= shape[i]

        return res

    def dec_tuple(x: int, shape: tuple[int, int, int]) -> tuple[int, int, int]:
        res: list[int] = []
        for i in reversed(range(len(shape))):
            res.append(x % shape[i])
            x //= shape[i]

        rev = list(reversed(res))
        return rev[0], rev[1], rev[2]

    _, w, h = input_size

    # Formula from the Torch docs:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    output_size = [(input_size[i + 1] + 2 * int(conv.padding[i]) - conv.kernel_size[i]) // conv.stride[i] + 1 for i in [0, 1]]

    in_shape = (conv.in_channels, w, h)
    out_shape = (conv.out_channels, output_size[0], output_size[1])

    bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)

    fc = nn.Linear(in_features=np.prod(in_shape).item(), out_features=np.prod(out_shape).item(), device=conv.weight.device)
    fc.weight.data.fill_(0.0)

    # Output coordinates
    for xo, yo in tqdm(
        range2d(output_size[0], output_size[1]),
        desc="Converting Conv2d to Linear",
        total=output_size[0] * output_size[1],
        leave=False,
    ):
        # The upper-left corner of the filter in the input tensor
        xi0 = -int(conv.padding[0]) + int(conv.stride[0]) * xo
        yi0 = -int(conv.padding[1]) + int(conv.stride[1]) * yo

        # Position within the filter
        for xd, yd in range2d(conv.kernel_size[0], conv.kernel_size[1]):
            # Output channel
            for co in range(conv.out_channels):
                fc.bias[enc_tuple((co, xo, yo), out_shape)] = bias[co]
                for ci in range(conv.in_channels):
                    # Make sure we are within the input image (and not in the padding)
                    if 0 <= xi0 + xd < w and 0 <= yi0 + yd < h:
                        cw = conv.weight[co, ci, xd, yd]
                        fc.weight[
                            enc_tuple((co, xo, yo), out_shape),
                            enc_tuple((ci, xi0 + xd, yi0 + yd), in_shape),
                        ] = cw

    return fc


@torch.no_grad()
def avgpool2d_to_affine(avgpool: nn.AvgPool2d, input_size: tuple[int, int, int]) -> nn.Linear:
    """Convert an AvgPool2d layer to an equivalent Linear layer.

    Converts average pooling into a fully connected layer by representing it
    as a convolution with uniform weights, then converting that to a Linear layer.

    Args:
        avgpool: The AvgPool2d layer to convert.
        input_size: Input size as (channels, height, width) tuple.

    Returns:
        nn.Linear: A Linear layer that performs the equivalent operation.

    Reference:
        Based on: https://www.researchgate.net/figure/The-mean-pooling-is-described-with-the-matrix-multiplication-of-the-reshaped-feature-map_fig2_357833254
    """
    # https://www.researchgate.net/figure/The-mean-pooling-is-described-with-the-matrix-multiplication-of-the-reshaped-feature-map_fig2_357833254
    conv2d = nn.Conv2d(
        in_channels=input_size[0],
        out_channels=input_size[0],
        kernel_size=avgpool.kernel_size,
        stride=avgpool.stride,
        padding=avgpool.padding,
        bias=True,  # always create with bias so we can zero-fill it
    )
    conv2d.weight.data.fill_(0.0)
    kernel_size = avgpool.kernel_size
    if isinstance(kernel_size, tuple):
        kernel_area = int(kernel_size[0]) * int(kernel_size[1])
    else:
        kernel_area = int(kernel_size) * int(kernel_size)
    scale = 1.0 / float(kernel_area)
    for i in range(input_size[0]):
        conv2d.weight.data[i, i, :, :].fill_(scale)
    assert conv2d.bias is not None
    conv2d.bias.data.fill_(0.0)
    return torch_conv_layer_to_affine(conv2d, input_size)


def combine_linear_layers(old_layers: OrderedDict[str, nn.Module]) -> OrderedDict[str, nn.Module]:
    """Combine consecutive Linear layers into a single layer.

    Since the composition of two linear transformations is itself linear,
    multiple consecutive Linear layers can be combined into one for efficiency.

    Args:
        old_layers: OrderedDict of layers to process.

    Returns:
        OrderedDict: New dictionary with consecutive Linear layers combined.
            Layer names are concatenated with '+' for combined layers.
    """
    new_layers: OrderedDict[str, nn.Module] = OrderedDict([])
    current_linear: nn.Linear | None = None
    current_name: str = ""
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
def convert(
    model: nn.Module | Iterable[nn.Module] | Mapping[str, nn.Module] | Sequence[AffineLayerTuple],
    input_shape: tuple[int, ...] | None = None,
) -> ReLUNetwork:
    """Convert a PyTorch model to canonical NN format.

    Converts various PyTorch layer types (Conv2d, AvgPool2d, etc.) into the
    canonical format consisting only of Linear and ReLU layers.

    Supported layer types:
        - ``Linear``, ``ReLU``: passed through unchanged.
        - ``Conv2d``: converted to ``Linear``.
        - ``AvgPool2d``: converted to ``Linear`` (requires ``kernel_size == stride``).
        - ``Flatten``, ``Dropout``: dropped (identity / inference-only).
        - ``LogSoftmax``: halts conversion at the output layer.

    Args:
        model: A ``torch.nn.Module``, a ``ModuleList``, a ``ModuleDict``, or an
            iterable of affine layer tuples ``(W_i, b_i)``.
        input_shape: Shape of the input data (excluding batch dimension).
            Inferred from the first ``Linear`` layer when ``None``.

    Returns:
        A :class:`~relucent.model.ReLUNetwork` in canonical format.

    Raises:
        ValueError: If an unsupported layer type is encountered or ``input_shape``
            cannot be inferred.
    """
    if isinstance(model, ReLUNetwork):
        if input_shape is not None:
            model.input_shape = input_shape
        return model

    if isinstance(model, Sequence) and len(model) > 0 and all(_is_affine_tuple_layer(layer) for layer in model):
        affine_layers = [layer for layer in model if isinstance(layer, tuple)]
        canonical_layers = _canonical_from_affine_tuples(affine_layers)
        if input_shape is None:
            first_linear = next((m for m in canonical_layers.values() if isinstance(m, LinearLayer)), None)
            if first_linear is None:
                raise ValueError("Model must define input_shape for conversion.")
            input_shape = (int(first_linear.weight.shape[1]),)
        return ReLUNetwork(layers=canonical_layers, input_shape=input_shape)

    if isinstance(model, nn.Module):
        source_layers: OrderedDict[str, object] = OrderedDict((name, module) for name, module in model.named_children())
        params = next(model.parameters(), None)
        dtype = params.dtype if params is not None else torch.float32
        device = params.device if params is not None else torch.device("cpu")
        if input_shape is None:
            first_linear = next((m for m in source_layers.values() if isinstance(m, nn.Linear)), None)
            if first_linear is None:
                raise ValueError("Model must define input_shape for conversion.")
            input_shape = (int(first_linear.in_features),)
    elif isinstance(model, Mapping):
        source_layers = OrderedDict((str(k), v) for k, v in model.items())
        dtype = torch.float32
        device = torch.device("cpu")
        if input_shape is None:
            first_linear = next((m for m in source_layers.values() if isinstance(m, nn.Linear)), None)
            if first_linear is None:
                raise ValueError("Model must define input_shape for conversion.")
            input_shape = (int(first_linear.in_features),)
    elif isinstance(model, Iterable):
        source_layers = OrderedDict((f"layer{i}", module) for i, module in enumerate(model))
        dtype = torch.float32
        device = torch.device("cpu")
        if input_shape is None:
            first_linear = next((m for m in source_layers.values() if isinstance(m, nn.Linear)), None)
            if first_linear is None:
                raise ValueError("Model must define input_shape for conversion.")
            input_shape = (int(first_linear.in_features),)
    else:
        raise ValueError(
            f"Unsupported input type: {type(input)}."
            + "Must be a canonical relu network, an Iterable/Mapping of nn.Module objects, or an iterable of (W, b)."
        )

    if input_shape is None:
        raise ValueError("Model must define input_shape for conversion.")
    x = torch.zeros((1,) + input_shape, dtype=dtype, device=device)
    layers = OrderedDict()
    assert "Flatten Input" not in source_layers
    layers["Flatten Input"] = nn.Flatten()
    for name, module in list(source_layers.items()):
        if isinstance(module, (nn.Linear, nn.ReLU)):
            layers[name] = module
        elif isinstance(module, (nn.Dropout, nn.Flatten)):
            pass
        elif isinstance(module, nn.LogSoftmax):
            break
        elif isinstance(module, nn.Conv2d):
            shape = tuple(int(dim) for dim in x.shape[1:])
            assert isinstance(shape, tuple) and len(shape) == 3
            new_layer = torch_conv_layer_to_affine(module, shape).to(device=device, dtype=dtype)
            layers[name] = new_layer
        elif isinstance(module, nn.AvgPool2d) and module.kernel_size == module.stride:
            shape = tuple(int(dim) for dim in x.shape[1:])
            assert isinstance(shape, tuple) and len(shape) == 3
            new_layer = avgpool2d_to_affine(module, shape).to(device=device, dtype=dtype)
            layers[name] = new_layer
        else:
            raise ValueError(f"Module {name} is not supported: {module}")
        x = module(x)
        module.to(device=device)
    layers = combine_linear_layers(layers)
    canonical_layers = _canonicalize_named_layers(layers)
    new_model = ReLUNetwork(layers=canonical_layers, input_shape=(np.prod(input_shape, dtype=int),))

    has_logsoftmax = any(isinstance(m, nn.LogSoftmax) for m in source_layers.values())
    if not has_logsoftmax and isinstance(model, nn.Module) and not isinstance(model, nn.ModuleList):
        was_training = model.training
        try:
            model.eval()
            x = torch.randn(input_shape, dtype=dtype, device=device)
            old_y = model(x)
            new_y = torch.as_tensor(new_model(x))
            assert torch.allclose(old_y, new_y, atol=1e-5, rtol=1e-5)
        except Exception as e:
            raise ValueError(f"Conversion failed: {e}") from e
        finally:
            model.train(was_training)
    return new_model
