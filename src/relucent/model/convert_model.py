"""Convert PyTorch models to the canonical NN format.

This module provides utilities to convert various PyTorch model architectures
(including Conv2d, AvgPool2d, etc.) into the canonical format used by relucent,
which consists of Linear and ReLU layers only.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard

import numpy as np

from relucent._internal.logging import progress
from relucent._internal.torch_compat import is_torch_tensor, nn, no_grad, torch
from relucent.model.model import FlattenLayer, LinearLayer, ReLULayer, ReLUNetwork

if TYPE_CHECKING:
    import numpy.typing as npt

    # Annotation-only: evaluating ``torch.Tensor`` at import would import torch.
    AffineArrayLike: TypeAlias = npt.ArrayLike | torch.Tensor
    AffineLayerPair: TypeAlias = Sequence[AffineArrayLike]

__all__ = ["convert"]


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


def _is_affine_pair_layer(item: object) -> bool:
    return isinstance(item, Sequence) and not isinstance(item, str | bytes) and len(item) == 2


def _is_affine_pair_sequence(model: object) -> TypeGuard[Sequence[AffineLayerPair]]:
    return isinstance(model, Sequence) and len(model) > 0 and all(_is_affine_pair_layer(layer) for layer in model)


def _canonical_from_affine_tuples(
    layers: Sequence[AffineLayerPair],
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


def _conv_padding(conv: nn.Conv2d) -> tuple[int, int]:
    """``conv``'s padding, after checking :func:`torch_conv_layer_to_affine` reproduces ``conv`` exactly.

    Raises:
        ValueError: For dilation, groups, a padding mode other than zeros, or ``padding="same"``.
    """
    unsupported = []
    if tuple(int(d) for d in conv.dilation) != (1, 1):
        unsupported.append(f"dilation={tuple(conv.dilation)}")
    if conv.groups != 1:
        unsupported.append(f"groups={conv.groups}")
    if conv.padding_mode != "zeros":
        unsupported.append(f"padding_mode={conv.padding_mode!r}")
    padding: Any = conv.padding
    if isinstance(padding, str):
        if padding == "valid":
            padding = (0, 0)
        else:
            unsupported.append(f"padding={padding!r}")
    if unsupported:
        raise ValueError(f"Conv2d with {', '.join(unsupported)} is not supported: {conv}")
    return int(padding[0]), int(padding[1])


def _check_avgpool_supported(pool: nn.AvgPool2d) -> None:
    """Raise unless :func:`avgpool2d_to_affine` reproduces ``pool`` exactly."""
    padding = pool.padding if isinstance(pool.padding, tuple) else (pool.padding, pool.padding)
    unsupported = []
    if pool.ceil_mode:
        unsupported.append("ceil_mode=True")
    if pool.divisor_override is not None:
        unsupported.append(f"divisor_override={pool.divisor_override}")
    if not pool.count_include_pad and any(int(p) for p in padding):
        unsupported.append("count_include_pad=False with padding")
    if unsupported:
        raise ValueError(f"AvgPool2d with {', '.join(unsupported)} is not supported: {pool}")


# https://gist.github.com/vvolhejn/e265665c65d3df37e381316bf57b8421
@no_grad
def torch_conv_layer_to_affine(conv: nn.Conv2d, input_size: tuple[int, int, int]) -> nn.Linear:
    """Convert a Conv2d layer to an equivalent Linear layer.

    Args:
        conv: The Conv2d layer to convert.
        input_size: Input size as (channels, height, width) tuple.

    Returns:
        nn.Linear: A Linear layer that performs the equivalent operation.

    Raises:
        ValueError: For dilation, groups, a padding mode other than zeros, or ``padding="same"``.

    Reference:
        Based on: https://gist.github.com/vvolhejn/e265665c65d3df37e381316bf57b8421
    """
    padding = _conv_padding(conv)

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

    _, w, h = input_size

    # Formula from the Torch docs:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    output_size = [(input_size[i + 1] + 2 * padding[i] - conv.kernel_size[i]) // conv.stride[i] + 1 for i in [0, 1]]

    in_shape = (conv.in_channels, w, h)
    out_shape = (conv.out_channels, output_size[0], output_size[1])

    bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)

    fc = nn.Linear(in_features=np.prod(in_shape).item(), out_features=np.prod(out_shape).item(), device=conv.weight.device)
    fc.weight.data.fill_(0.0)

    # Output coordinates
    for xo, yo in progress(
        range2d(output_size[0], output_size[1]),
        desc="Converting Conv2d to Linear",
        total=output_size[0] * output_size[1],
        leave=False,
    ):
        # The upper-left corner of the filter in the input tensor
        xi0 = -padding[0] + int(conv.stride[0]) * xo
        yi0 = -padding[1] + int(conv.stride[1]) * yo

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


@no_grad
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
    _check_avgpool_supported(avgpool)
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


def _as_float64(value: Any) -> np.ndarray:
    if is_torch_tensor(value):
        return value.detach().double().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _conversion_error_bound(layers: Mapping[str, object], x: torch.Tensor, dtype: torch.dtype) -> np.ndarray:
    """Bound on |source output - converted output| at ``x`` from rounding alone.

    Both models evaluate the same affine maps and ReLUs in ``dtype``, in a different order.
    Each output's rounding is at most ``gamma(K)`` times the network evaluated with ``|W|``,
    ``|b|`` and ``|x|`` and every unit on (an upper bound for any activation pattern), computed
    over the layers *before* consecutive linear maps are merged, since a merged product's ``|W|``
    can be far smaller than the product of the ``|W|``'s the source model rounds through.
    """
    from relucent._internal import rounding

    v = np.abs(x.detach().double().cpu().numpy()).reshape(-1)
    n_terms = 1
    for layer in layers.values():
        weight = getattr(layer, "weight", None)
        if weight is None:
            continue  # ReLU / Flatten / pooling-free reshape: nonnegative input passes through
        w = np.abs(_as_float64(weight))
        bias = getattr(layer, "bias", None)
        b = 0.0 if bias is None else np.abs(_as_float64(bias)).reshape(-1)
        v = w.reshape(w.shape[0], -1) @ v + b
        n_terms += int(w.reshape(w.shape[0], -1).shape[1]) + 1
    u = float(torch.finfo(dtype).eps) / 2.0 if dtype.is_floating_point else rounding.EPS
    g = n_terms * u / (1.0 - n_terms * u)
    # Two evaluations, each off by at most g times the abs network; 4 ulps for the bound itself.
    return 2.0 * g * v * (1.0 + 4.0 * rounding.EPS)


def _check_matches_source(
    model: nn.Module,
    new_model: ReLUNetwork,
    layers: Mapping[str, object],
    input_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    n_samples: int = 8,
) -> None:
    """Raise unless ``new_model`` reproduces ``model``'s forward pass on random inputs, to within rounding.

    :func:`convert` composes ``model``'s child modules in order, so a ``forward`` that does
    anything else (a skip connection, a reused or reordered module) converts to a different
    function. The inputs come from a private generator, so the global torch RNG is untouched.
    """
    generator = torch.Generator().manual_seed(0)
    x = torch.randn((n_samples, *input_shape), generator=generator, dtype=torch.float64).to(device=device, dtype=dtype)
    was_training = model.training
    try:
        model.eval()
        source = model(x)
    except Exception as e:
        raise ValueError(f"Conversion failed: the source model's forward pass raised: {e}") from e
    finally:
        model.train(was_training)
    old = torch.as_tensor(source).detach().double().cpu().reshape(n_samples, -1)
    new = torch.as_tensor(new_model(x)).detach().double().cpu().reshape(n_samples, -1)
    if old.shape != new.shape:
        raise ValueError(
            f"Conversion failed: the source model outputs shape {tuple(old.shape[1:])}, "
            + f"the converted one {tuple(new.shape[1:])}"
        )
    for i in range(n_samples):
        bound = torch.as_tensor(_conversion_error_bound(layers, x[i], dtype), dtype=torch.float64)
        diff = (old[i] - new[i]).abs()
        if not bool((diff <= bound).all()):
            raise ValueError(
                f"Conversion failed: the converted model differs from the source by {float(diff.max()):.3e}, "
                + f"beyond its rounding bound {float(bound.max()):.3e}. convert() composes the model's child "
                + "modules in order; a forward() that does anything else (skip connections, reused or "
                + "reordered modules) is not supported."
            )


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


@no_grad
def convert(
    model: ReLUNetwork | nn.Module | Iterable[nn.Module] | Mapping[str, nn.Module] | Sequence[AffineLayerPair],
    input_shape: tuple[int, ...] | None = None,
) -> ReLUNetwork:
    """Convert a PyTorch model to canonical NN format.

    Converts various PyTorch layer types (Conv2d, AvgPool2d, etc.) into the
    canonical format consisting only of Linear and ReLU layers.

    Supported layer types:
        - ``Linear``, ``ReLU``: passed through unchanged.
        - ``Conv2d``: converted to ``Linear`` (zero padding, no dilation or groups).
        - ``AvgPool2d``: converted to ``Linear`` (requires ``kernel_size == stride``, no
          ``ceil_mode`` or ``divisor_override``).
        - ``Flatten``, ``Dropout``: dropped (identity / inference-only).
        - ``LogSoftmax``: halts conversion at the output layer.

    The model is read as the composition of its child modules, in order. For a
    ``torch.nn.Module`` (other than a ``ModuleList`` / ``ModuleDict``, which have no forward
    pass), the result is checked against the model's own forward pass on random inputs, so a
    ``forward`` that does something else raises instead of converting to a different function.

    Args:
        model: A ``torch.nn.Module``, a ``ModuleList``, a ``ModuleDict``, or an
            iterable of affine layer pairs ``(W_i, b_i)``. Each pair element may be
            any NumPy array-like object or a torch tensor.
        input_shape: Shape of the input data (excluding batch dimension).
            Inferred from the first ``Linear`` layer when ``None``.

    Returns:
        A :class:`~relucent.model.model.ReLUNetwork` in canonical format.

    Raises:
        ValueError: If an unsupported layer type is encountered, ``input_shape``
            cannot be inferred, or the result does not reproduce the model's forward pass.
    """
    if isinstance(model, ReLUNetwork):
        if input_shape is not None and tuple(input_shape) != tuple(model.input_shape):
            model = copy.copy(model)  # leave the caller's network as it was
            model.input_shape = tuple(input_shape)
        return model

    if _is_affine_pair_sequence(model):
        affine_layers = [(layer[0], layer[1]) for layer in model]
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
            f"Unsupported input type: {type(model)}. "
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
            _check_avgpool_supported(module)
            shape = tuple(int(dim) for dim in x.shape[1:])
            assert isinstance(shape, tuple) and len(shape) == 3
            new_layer = avgpool2d_to_affine(module, shape).to(device=device, dtype=dtype)
            layers[name] = new_layer
        else:
            raise ValueError(f"Module {name} is not supported: {module}")
        x = module(x)
        module.to(device=device)
    uncombined_layers = dict(layers)
    layers = combine_linear_layers(layers)
    canonical_layers = _canonicalize_named_layers(layers)
    new_model = ReLUNetwork(layers=canonical_layers, input_shape=(np.prod(input_shape, dtype=int),))

    has_logsoftmax = any(isinstance(m, nn.LogSoftmax) for m in source_layers.values())
    if not has_logsoftmax and isinstance(model, nn.Module) and not isinstance(model, (nn.ModuleList, nn.ModuleDict)):
        _check_matches_source(model, new_model, uncombined_layers, tuple(input_shape), dtype, device)
    return new_model
