"""PL Morse gradients and critical points for scalar ReLU networks.

Implements Brooks & Masden (arXiv:2412.18005) Lemma 9 (cell Jacobian / gradient),
Lemma 10 (edge direction), Theorem 4 (partial-derivative sign along edges), and the
vertex criticality criterion from Section 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import relucent.config as cfg
from relucent.model import LinearLayer, ReLULayer, ReLUNetwork

if TYPE_CHECKING:
    from relucent.complex import Complex
    from relucent.poly import Polyhedron

__all__ = [
    "CriticalPoint",
    "LayerJacobians",
    "assert_scalar_output",
    "coface_sign_sequence",
    "get_layer_jacobians",
    "gradient_on_cell",
    "is_pl_critical_vertex",
    "partial_derivative_sign",
    "partial_derivative_value",
    "partial_derivative_on_1cell",
    "shi_to_relu_neuron",
]


@dataclass(frozen=True)
class LayerJacobians:
    """Jacobians of layer compositions on a fixed sign sequence."""

    by_relu_layer: list[np.ndarray]  # F'_{(i)}|_C after each ReLU block
    full: np.ndarray  # product through all hidden layers
    gradient: np.ndarray  # ∇F|_C in input coordinates
    W_out: np.ndarray  # final linear map G


@dataclass(frozen=True)
class CriticalPoint:
    """A PL Morse critical vertex of the network on the complex."""

    polyhedron: Polyhedron
    tag: bytes
    ss: np.ndarray
    point: np.ndarray | None
    index: int  # Morse index; -1 marks flat/degenerate cases
    is_critical: bool = True


def assert_scalar_output(net: ReLUNetwork) -> None:
    """Raise if the network does not have a single scalar output."""
    last_linear: LinearLayer | None = None
    for layer in net.layers.values():
        if isinstance(layer, LinearLayer):
            last_linear = layer
    if last_linear is None or last_linear.weight.shape[0] != 1:
        raise ValueError(
            "Morse gradient and critical-point routines require a scalar output network "
            + f"(final linear layer width 1, got {None if last_linear is None else last_linear.weight.shape[0]})"
        )


def _coerce_ss(ss: np.ndarray) -> np.ndarray:
    arr = np.asarray(ss, dtype=np.int8)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _output_weight(net: ReLUNetwork) -> np.ndarray:
    last_linear: LinearLayer | None = None
    for layer in net.layers.values():
        if isinstance(layer, LinearLayer):
            last_linear = layer
    assert last_linear is not None
    return last_linear.weight


def get_layer_jacobians(net: ReLUNetwork, ss: np.ndarray) -> LayerJacobians:
    """Lemma 9: Jacobians ``F'_{(i)}|_C`` after each ReLU block and the input gradient."""
    ss_row = _coerce_ss(ss)
    n_in = int(np.prod(net.input_shape))
    current = np.eye(n_in, dtype=np.float64)
    by_relu: list[np.ndarray] = []
    mask_index = 0

    # Same layer walk as get_hs, but we only keep the active-neuron mask (s == +1).
    layers = list(net.layers.values())
    i = 0
    while i < len(layers):
        layer = layers[i]
        if isinstance(layer, LinearLayer) and i + 1 < len(layers) and isinstance(layers[i + 1], ReLULayer):
            current = current @ layer.weight.T
            width = layer.weight.shape[0]
            mask = ss_row[0, mask_index : mask_index + width]
            relu = (mask == 1).astype(np.float64)
            current = current * relu[np.newaxis, :]
            by_relu.append(current.copy())
            mask_index += width
            i += 2
            continue
        if isinstance(layer, LinearLayer):
            # Final output layer G; gradient uses W_out separately below.
            pass
        i += 1

    w_out = _output_weight(net)
    gradient = (current @ w_out.T).reshape(-1)
    return LayerJacobians(by_relu_layer=by_relu, full=current, gradient=gradient, W_out=w_out)


def gradient_on_cell(net: ReLUNetwork, ss: np.ndarray) -> np.ndarray:
    """Lemma 9: ``∇F|_C`` as a length-``n_in`` vector (scalar output)."""
    assert_scalar_output(net)
    return get_layer_jacobians(net, ss).gradient


def coface_sign_sequence(edge_ss: np.ndarray) -> np.ndarray:
    """Lemma 10: replace zeros with ``+1`` to obtain a top-cell sign sequence."""
    ss = _coerce_ss(edge_ss).copy()
    # Any positive extension works; +1 picks the coface used in the paper's Lemma 10 proof.
    ss[ss == 0] = 1
    return ss


def shi_to_relu_neuron(
    shi: int,
    ssi2maski: list[tuple[int, tuple[int, int]]],
    ss_layers: list[int],
) -> tuple[int, int]:
    """Map a global SHI to ``(relu_layer_index, neuron_index)``."""
    linear_idx, (_, neuron_j) = ssi2maski[int(shi)]
    try:
        relu_idx = ss_layers.index(linear_idx)
    except ValueError as exc:
        raise ValueError(f"SHI {shi} does not belong to a ReLU block") from exc
    return relu_idx, int(neuron_j)


def _diff_shi(vertex_ss: np.ndarray, edge_ss: np.ndarray) -> int:
    v = _coerce_ss(vertex_ss).ravel()
    e = _coerce_ss(edge_ss).ravel()
    if v.shape != e.shape:
        raise ValueError("vertex and edge sign sequences must have the same shape")
    diff = np.flatnonzero(v != e)
    if diff.size != 1:
        raise ValueError(f"expected exactly one differing sign entry, got {diff.size}")
    return int(diff[0])


def _sign_with_tol(value: float) -> int:
    if abs(value) <= cfg.TOL_VERIFY_AB_ATOL:
        return 0
    return 1 if value > 0 else -1


def partial_derivative_sign(
    vertex_ss: np.ndarray,
    edge_ss: np.ndarray,
    net: ReLUNetwork,
    *,
    ssi2maski: list[tuple[int, tuple[int, int]]],
    ss_layers: list[int],
) -> int:
    """Theorem 4 / Corollary 3: sign of ``∂_{vE} F`` for scalar output."""
    val = partial_derivative_value(
        vertex_ss,
        edge_ss,
        net,
        ssi2maski=ssi2maski,
        ss_layers=ss_layers,
    )
    return _sign_with_tol(val)


def _vertex_edge_direction(
    vertex_ss: np.ndarray,
    edge_ss: np.ndarray,
    jac: LayerJacobians,
    *,
    ssi2maski: list[tuple[int, tuple[int, int]]],
    ss_layers: list[int],
) -> np.ndarray:
    """Lemma 10: vector in the direction ``v→E`` in input space."""
    shi = _diff_shi(vertex_ss, edge_ss)
    n_in = jac.full.shape[0]
    zeros = np.flatnonzero(_coerce_ss(vertex_ss).ravel() == 0)
    if zeros.size != n_in:
        raise ValueError(f"vertex sign sequence must have {n_in} zero entries (got {zeros.size}) for Lemma 10 edge directions")

    # W(v, C): row k is the j_k-th row of F'_{(i_k)}|_C for each zero entry at v.
    rows: list[np.ndarray] = []
    for z in zeros:
        r_idx, n_idx = shi_to_relu_neuron(int(z), ssi2maski, ss_layers)
        rows.append(jac.by_relu_layer[r_idx][:, n_idx])
    w_mat = np.vstack(rows)

    row_pos = int(np.where(zeros == shi)[0][0])
    e_row = np.zeros(n_in, dtype=np.float64)
    e_row[row_pos] = 1.0
    scale = float(_coerce_ss(edge_ss).ravel()[shi])
    try:
        return scale * np.linalg.solve(w_mat, e_row)
    except np.linalg.LinAlgError:
        # Flat / degenerate directions can make W(v, C) singular.
        return scale * np.linalg.lstsq(w_mat, e_row, rcond=None)[0]


def partial_derivative_value(
    vertex_ss: np.ndarray,
    edge_ss: np.ndarray,
    net: ReLUNetwork,
    *,
    ssi2maski: list[tuple[int, tuple[int, int]]],
    ss_layers: list[int],
) -> float:
    """Corollary 3 / Lemma 10: numeric ``∂_{vE} F`` along the edge direction."""
    assert_scalar_output(net)
    v = _coerce_ss(vertex_ss).ravel()
    e = _coerce_ss(edge_ss).ravel()
    if not np.all((e != 0) | (v == 0)):
        raise ValueError("vertex must be a face of the edge (edge zeros are also zeros at the vertex)")

    # Gradient is evaluated on a top cell containing the edge (Lemma 10 coface).
    coface = coface_sign_sequence(edge_ss)
    jac = get_layer_jacobians(net, coface)
    direction = _vertex_edge_direction(
        vertex_ss,
        edge_ss,
        jac,
        ssi2maski=ssi2maski,
        ss_layers=ss_layers,
    )
    # Corollary 3: ∂_{vE} F = ∇F|_C · (v→E).
    return float(jac.gradient @ direction)


def _edge_ss_from_vertex(vertex_ss: np.ndarray, shi: int, sign: int) -> np.ndarray:
    ss = _coerce_ss(vertex_ss).copy()
    ss.ravel()[int(shi)] = int(sign)
    return ss


def _is_collapsed_edge(
    net: ReLUNetwork,
    edge_ss: np.ndarray,
) -> bool:
    """True when the network output is constant along the edge (flat / degenerate)."""
    coface = coface_sign_sequence(edge_ss)
    grad = gradient_on_cell(net, coface)
    return bool(np.linalg.norm(grad) <= cfg.TOL_VERIFY_AB_ATOL)


def is_pl_critical_vertex(
    vertex_ss: np.ndarray,
    net: ReLUNetwork,
    *,
    ssi2maski: list[tuple[int, tuple[int, int]]],
    ss_layers: list[int],
) -> tuple[bool, int | None]:
    """Section 3.2: PL criticality and Morse index at a vertex."""
    assert_scalar_output(net)
    v = _coerce_ss(vertex_ss).ravel()
    zeros = np.flatnonzero(v == 0)
    if zeros.size == 0:
        return False, None

    # At a vertex, each zero SHI gives a coordinate direction with edges s=±1.
    towards = 0
    for shi in zeros:
        edge_plus = _edge_ss_from_vertex(vertex_ss, int(shi), 1)
        edge_minus = _edge_ss_from_vertex(vertex_ss, int(shi), -1)
        if _is_collapsed_edge(net, edge_plus) or _is_collapsed_edge(net, edge_minus):
            return True, -1

        sign_plus = partial_derivative_sign(vertex_ss, edge_plus, net, ssi2maski=ssi2maski, ss_layers=ss_layers)
        sign_minus = partial_derivative_sign(vertex_ss, edge_minus, net, ssi2maski=ssi2maski, ss_layers=ss_layers)
        if sign_plus == 0 or sign_minus == 0:
            return True, -1
        if sign_plus == sign_minus:
            return False, None
        # Negative ∂_{vE} F means the edge is oriented towards v (F decreases along v→E).
        if sign_plus < 0 or sign_minus < 0:
            towards += 1

    return True, towards


def partial_derivative_on_1cell(
    one_cell: Polyhedron,
    from_vertex: Polyhedron,
    complex: Complex,
    *,
    value: bool = False,
) -> int | float:
    """Partial derivative of ``F`` along a 1-cell, evaluated from ``from_vertex``."""
    if one_cell.dim != 1:
        raise ValueError(f"expected a 1-cell, got dim={one_cell.dim}")
    if from_vertex.dim != 0:
        raise ValueError(f"expected a vertex, got dim={from_vertex.dim}")
    if not from_vertex.is_face_of(one_cell):
        raise ValueError("from_vertex must be a face of one_cell")

    if value:
        return partial_derivative_value(
            from_vertex.ss_np,
            one_cell.ss_np,
            complex._net,
            ssi2maski=complex.ssi2maski,
            ss_layers=complex.ss_layers,
        )
    return partial_derivative_sign(
        from_vertex.ss_np,
        one_cell.ss_np,
        complex._net,
        ssi2maski=complex.ssi2maski,
        ss_layers=complex.ss_layers,
    )
