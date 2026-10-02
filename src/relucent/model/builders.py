"""Build small ReLU networks (random MLPs), split or extend them, and seed RNGs."""

import random
from collections import OrderedDict
from collections.abc import Iterable
from math import sqrt
from typing import Any, cast

import numpy as np

from relucent._internal.torch_compat import TORCH_AVAILABLE, nn, torch
from relucent.model.model import FlattenLayer, LinearLayer, ReLULayer, ReLUNetwork

__all__ = [
    "TorchMLP",
    "MLP_INIT_METHODS",
    "initialize_linear",
    "mlp",
    "set_seeds",
    "torch_mlp",
    "split_sequential",
    "add_output_relu",
    "normalize_weights",
]


class TorchMLP(nn.Sequential):
    """Sequential MLP with compatibility helpers used across relucent."""

    def __init__(self, layers: OrderedDict[str, nn.Module], widths: list[int]) -> None:
        super().__init__(layers)
        self.widths = widths
        self.input_shape = (widths[0],)

    @property
    def layers(self) -> OrderedDict[str, nn.Module]:
        """Named submodules, in order."""
        return cast(OrderedDict[str, nn.Module], self._modules)

    @property
    def device(self) -> str:
        """Device of the first parameter as a string (``"cpu"`` if there are none)."""
        try:
            return str(next(self.parameters()).device)
        except StopIteration:
            return "cpu"

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the first parameter (``torch.float64`` if there are none)."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float64


MLP_INIT_METHODS = frozenset(
    {
        "uniform",
        "xavier_uniform",
        "xavier_normal",
        "kaiming_uniform",
        "kaiming_normal",
        "orthogonal",
    }
)


def initialize_linear(linear: nn.Linear, method: str = "uniform") -> None:
    """Initialize a :class:`torch.nn.Linear` layer in place."""
    if method not in MLP_INIT_METHODS:
        raise ValueError(f"Unknown init method {method!r}; expected one of {sorted(MLP_INIT_METHODS)}")

    fan_in = linear.in_features
    with torch.no_grad():
        if method == "uniform":
            bound = 1.0 / sqrt(fan_in) if fan_in > 0 else 0.0
            linear.weight.uniform_(-bound, bound)
            linear.bias.uniform_(-bound, bound)
        elif method == "xavier_uniform":
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        elif method == "xavier_normal":
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
        elif method == "kaiming_uniform":
            nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
        elif method == "kaiming_normal":
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
        elif method == "orthogonal":
            nn.init.orthogonal_(linear.weight)
            nn.init.zeros_(linear.bias)


def _init_weights(fan_out: int, fan_in: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Weight ``(fan_out, fan_in)`` and bias ``(1, fan_out)`` drawn like PyTorch's initializer of the same name."""
    rng = np.random  # the global NumPy RNG, which set_seeds seeds
    zeros = np.zeros((1, fan_out))
    if method == "uniform":  # nn.Linear's default
        bound = 1.0 / sqrt(fan_in) if fan_in > 0 else 0.0
        return rng.uniform(-bound, bound, (fan_out, fan_in)), rng.uniform(-bound, bound, (1, fan_out))
    if method == "xavier_uniform":
        bound = sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-bound, bound, (fan_out, fan_in)), zeros
    if method == "xavier_normal":
        return rng.normal(0.0, sqrt(2.0 / (fan_in + fan_out)), (fan_out, fan_in)), zeros
    if method == "kaiming_uniform":  # ReLU gain sqrt(2), fan-in mode
        bound = sqrt(2.0) * sqrt(3.0 / fan_in)
        return rng.uniform(-bound, bound, (fan_out, fan_in)), zeros
    if method == "kaiming_normal":
        return rng.normal(0.0, sqrt(2.0) / sqrt(fan_in), (fan_out, fan_in)), zeros
    if method == "orthogonal":
        flat = rng.normal(0.0, 1.0, (fan_out, fan_in))
        transpose = fan_out < fan_in
        q, r = np.linalg.qr(flat.T if transpose else flat)
        q *= np.sign(np.diag(r))
        return (q.T if transpose else q), zeros
    raise ValueError(f"Unknown init method {method!r}; expected one of {sorted(MLP_INIT_METHODS)}")


def mlp(
    widths: Iterable[int],
    *,
    add_last_relu: bool = False,
    init: str = "uniform",
) -> ReLUNetwork:
    """Create a fully connected ReLU network with random weights, as a :class:`~relucent.model.model.ReLUNetwork`.

    Weights come from NumPy's global RNG (seed it with :func:`set_seeds`), so the same seed
    gives the same network whether or not PyTorch is installed. For a PyTorch module, use
    :func:`torch_mlp`.

    Args:
        widths: Layer widths including input and output widths, e.g. ``[2, 10, 5, 1]``.
        add_last_relu: If ``True``, append a ReLU after the final linear layer.
        init: Weight initialization, one of :data:`MLP_INIT_METHODS` (default ``"uniform"``,
            PyTorch's ``nn.Linear`` default). Each follows the PyTorch initializer of the same
            name; the ``kaiming_*`` ones use the ReLU gain.

    Returns:
        A network with layers ``fc0, relu0, fc1, ...``.
    """
    if init not in MLP_INIT_METHODS:
        raise ValueError(f"Unknown init {init!r}; expected one of {sorted(MLP_INIT_METHODS)}")
    widths = [int(w) for w in widths]
    layers: OrderedDict[str, LinearLayer | ReLULayer] = OrderedDict()
    for i in range(len(widths) - 1):
        weight, bias = _init_weights(widths[i + 1], widths[i], init)
        layers[f"fc{i}"] = LinearLayer(weight=weight.astype(np.float64), bias=bias.astype(np.float64))
        if i < len(widths) - 2 or add_last_relu:
            layers[f"relu{i}"] = ReLULayer()
    return ReLUNetwork(layers, input_shape=(widths[0],))


def torch_mlp(
    widths: Iterable[int],
    *,
    add_last_relu: bool = False,
    init: str = "uniform",
) -> TorchMLP:
    """Like :func:`mlp`, but as a float64 PyTorch module (needs PyTorch; seeded by :func:`set_seeds`).

    Its weights come from PyTorch's RNG, so they differ from :func:`mlp`'s for the same seed.
    """
    if init not in MLP_INIT_METHODS:
        raise ValueError(f"Unknown init {init!r}; expected one of {sorted(MLP_INIT_METHODS)}")
    widths = [int(w) for w in widths]
    layers: list[tuple[str, nn.Module]] = []
    for i in range(len(widths) - 1):
        fc = nn.Linear(widths[i], widths[i + 1], dtype=torch.float64)
        initialize_linear(fc, init)
        layers.append((f"fc{i}", fc))
        if i < len(widths) - 2 or add_last_relu:
            layers.append((f"relu{i}", nn.ReLU()))
    return TorchMLP(OrderedDict(layers), widths)


def set_seeds(seed: int) -> None:
    """Set all RNG seeds to a given value.

    Args:
        seed: Integer seed value.
    """
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def split_sequential(
    model: Any,
    split_layer: str,
) -> tuple[Any, Any]:
    """Split a neural network into two sequential parts.

    Creates two separate canonical network objects by splitting the model at a specified layer.
    The first network contains layers up to and including split_layer, and the
    second contains all subsequent layers.

    Args:
        model: The canonical network object to split.
        split_layer: Name of the layer at which to split (this layer goes to
            the first network).

    Returns:
        tuple: (nn1, nn2) where nn1 contains layers up to split_layer and
            nn2 contains the remaining layers.
    """
    if isinstance(model, TorchMLP):
        layers1: OrderedDict[str, nn.Module] = OrderedDict()
        layers2: OrderedDict[str, nn.Module] = OrderedDict()
        current_layers = layers1
        for name, layer in model.named_children():
            current_layers[name] = layer
            if name == split_layer:
                current_layers = layers2
        widths1 = _infer_torch_widths_from_layers(layers1, fallback_input_width=int(model.widths[0]))
        widths2 = _infer_torch_widths_from_layers(layers2, fallback_input_width=int(widths1[-1]))
        return TorchMLP(layers1, widths1), TorchMLP(layers2, widths2)

    layers1_c: OrderedDict[str, LinearLayer | ReLULayer | FlattenLayer] = OrderedDict()
    layers2_c: OrderedDict[str, LinearLayer | ReLULayer | FlattenLayer] = OrderedDict()
    current_layers_c = layers1_c
    for name, layer in model.layers.items():
        current_layers_c[name] = layer
        if name == split_layer:
            current_layers_c = layers2_c
    nn1 = ReLUNetwork(layers1_c, input_shape=model.input_shape)
    nn2 = ReLUNetwork(
        layers2_c,
        input_shape=tuple(int(v) for v in nn1(np.zeros((1,) + model.input_shape, dtype=np.float64)).squeeze().shape),
    )
    return nn1, nn2


def _infer_torch_widths_from_layers(
    layers: OrderedDict[str, nn.Module],
    *,
    fallback_input_width: int,
) -> list[int]:
    widths: list[int] = []
    for layer in layers.values():
        if isinstance(layer, nn.Linear):
            if not widths:
                widths.append(int(layer.in_features))
            widths.append(int(layer.out_features))
    if widths:
        return widths
    return [int(fallback_input_width)]


def add_output_relu(model: Any) -> Any:
    """Append a ReLU activation after the final linear layer.

    Classifiers are often trained without a ReLU on the output neuron. For
    topology analysis of the decision boundary, treating that boundary as the
    zero level set of the last ReLU pre-activation requires appending a final
    ReLU while keeping all earlier weights unchanged.

    Args:
        model: A :class:`TorchMLP`, plain :class:`torch.nn.Sequential`, or
            :class:`~relucent.model.model.ReLUNetwork`.

    Returns:
        A new network of the same type with one additional ReLU layer after the
        final linear layer.

    Raises:
        ValueError: If the network is empty, already ends with ReLU, or its
            final layer is not linear.
    """
    if isinstance(model, TorchMLP):
        items = list(model.named_children())
        if not items:
            raise ValueError("Network has no layers")
        last_name, last_layer = items[-1]
        if isinstance(last_layer, nn.ReLU):
            raise ValueError("Network already ends with a ReLU layer")
        if not isinstance(last_layer, nn.Linear):
            raise ValueError(f"Expected final layer to be Linear, got {type(last_layer)}")
        new_layers = OrderedDict(items)
        new_layers[f"{last_name}_relu"] = nn.ReLU()
        return TorchMLP(new_layers, list(model.widths))

    if isinstance(model, nn.Sequential):
        items = list(model.named_children())
        if not items:
            raise ValueError("Network has no layers")
        last_name, last_layer = items[-1]
        if isinstance(last_layer, nn.ReLU):
            raise ValueError("Network already ends with a ReLU layer")
        if not isinstance(last_layer, nn.Linear):
            raise ValueError(f"Expected final layer to be Linear, got {type(last_layer)}")
        new_layers = OrderedDict(items)
        new_layers[f"{last_name}_relu"] = nn.ReLU()
        fallback_width = int(getattr(last_layer, "in_features", 0))
        return TorchMLP(new_layers, _infer_torch_widths_from_layers(new_layers, fallback_input_width=fallback_width))

    items = list(model.layers.items())
    if not items:
        raise ValueError("Network has no layers")
    last_name, last_layer = items[-1]
    if isinstance(last_layer, ReLULayer):
        raise ValueError("Network already ends with a ReLU layer")
    if not isinstance(last_layer, LinearLayer):
        raise ValueError(f"Expected final layer to be Linear, got {type(last_layer)}")
    new_layers = OrderedDict(items)
    new_layers[f"{last_name}_relu"] = ReLULayer()
    out = ReLUNetwork(new_layers, input_shape=model.input_shape)
    out.trained_on = model.trained_on
    return out


def normalize_weights(model: Any) -> Any:
    """Normalize hidden neuron weights to unit norm without changing the network function.

    The incoming weights (and biases) of each Linear layer except the last one are rescaled so that each
    neuron's weight vector has unit ℓ2 norm.

    Args:
        model: The canonical network object whose weights should be normalized in-place.

    Returns:
        The same canonical network object with normalized hidden-layer weights.

    Raises:
        ValueError: If the network contains layers other than Linear or ReLU.
    """
    if isinstance(model, TorchMLP):
        layers_torch = list(model.children())
        for layer in layers_torch:
            if not isinstance(layer, (nn.Linear, nn.ReLU)):
                raise ValueError(f"Unsupported layer type: {type(layer)}")

        linear_indices_torch = [i for i, layer in enumerate(layers_torch) if isinstance(layer, nn.Linear)]
        with torch.no_grad():
            for idx, lin_idx in enumerate(linear_indices_torch):
                layer = layers_torch[lin_idx]
                assert isinstance(layer, nn.Linear)
                if idx == len(linear_indices_torch) - 1:
                    continue
                w = layer.weight.data
                norms = w.norm(dim=1, keepdim=True)
                safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
                layer.weight.data = w / safe_norms
                layer.bias.data = layer.bias.data / safe_norms.squeeze(1)

                next_linear = layers_torch[linear_indices_torch[idx + 1]]
                assert isinstance(next_linear, nn.Linear)
                next_linear.weight.data = next_linear.weight.data * safe_norms.squeeze(1)
        return model

    layers = list(model.layers.values())

    # Ensure only supported layer types are present.
    for layer in layers:
        if not isinstance(layer, (LinearLayer, ReLULayer)):
            raise ValueError(f"Unsupported layer type: {type(layer)}")

    # Indices of all Linear layers in order.
    linear_indices = [i for i, layer in enumerate(layers) if isinstance(layer, LinearLayer)]

    for idx, lin_idx in enumerate(linear_indices):
        layer = layers[lin_idx]
        assert isinstance(layer, LinearLayer)

        # Do not modify the final Linear layer
        is_last_linear = idx == len(linear_indices) - 1
        if is_last_linear:
            continue

        w = layer.weight
        norms = np.linalg.norm(w, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0, norms, np.ones_like(norms))

        layer.weight = w / safe_norms
        # Bias is stored row-wise as shape (1, out_features), so divide by the
        # transposed norms to avoid broadcasting to (out_features, out_features).
        layer.bias = layer.bias / safe_norms.T

        next_linear = layers[linear_indices[idx + 1]]
        assert isinstance(next_linear, LinearLayer)
        next_w = next_linear.weight
        scale = safe_norms.squeeze(1)
        next_linear.weight = next_w * scale

    return model
