"""PL Morse gradients and critical points for scalar ReLU networks.

Implements Brooks & Masden (arXiv:2412.18005) Lemma 9 (cell Jacobian / gradient),
Lemma 10 (edge direction), Theorem 4 (partial-derivative sign along edges), and the
vertex criticality criterion from Section 3.2.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from relucent._internal.logging import with_verbosity
from relucent.model.model import LinearLayer, ReLULayer, ReLUNetwork

if TYPE_CHECKING:
    from relucent.core.complex import Complex
    from relucent.core.poly import Polyhedron

__all__ = [
    "CriticalPoint",
    "LayerJacobians",
    "assert_scalar_output",
    "coface_sign_sequence",
    "critical_flags_for_vertices",
    "critical_points",
    "layer_jacobians",
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

    # Per ReLU block: columns are ∇(pre-activation) before that block's ReLU mask.
    # Lemma 10 needs these unmasked normals; post-mask rows vanish on inactive sides.
    by_relu_layer: list[np.ndarray]
    full: np.ndarray  # hidden-layer product after ReLU masks (input → last hidden)
    gradient: np.ndarray  # ∇F|_C in input coordinates (logit if trailing output ReLU)
    W_out: np.ndarray  # final linear map G
    # The same products with |W| (rounding scale of each entry; see relucent._internal.rounding).
    by_relu_layer_abs: list[np.ndarray] | None = None
    gradient_abs: np.ndarray | None = None
    n_terms: int = 1  # inner-product terms accumulated along the longest product


@dataclass(frozen=True)
class CriticalPoint:
    """A PL Morse critical vertex of the network on the complex."""

    polyhedron: Polyhedron
    tag: bytes
    ss: np.ndarray
    point: np.ndarray | None
    index: int  # Morse index in {0,...,d}; -1 marks flat/degenerate cases
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


# Opt-in, thread-local memoization for `layer_jacobians`.
# `is_pl_critical_vertex` needs the same coface Jacobian twice per edge
# (`_is_collapsed_edge`, then `partial_derivative_sign`); `_JacobianCacheScope` lets
# the second call reuse the first. Off by default so other callers stay cache-free;
# thread-local so parallel workers never share a cache.
_jacobian_cache_state = threading.local()


class _JacobianCacheScope:
    """Scope in which ``layer_jacobians`` memoizes by ``(id(net), ss bytes)``.

    Intended to wrap a single vertex's criticality check (see
    ``is_pl_critical_vertex``): small, short-lived cache, reset on exit so it can
    never grow unbounded across a whole complex's vertices.
    """

    def __init__(self) -> None:
        self._previous: dict[tuple[int, bytes], LayerJacobians] | None = None

    def __enter__(self) -> None:
        self._previous = getattr(_jacobian_cache_state, "cache", None)
        _jacobian_cache_state.cache = {}

    def __exit__(self, *exc_info: object) -> None:
        _jacobian_cache_state.cache = self._previous


def layer_jacobians(net: ReLUNetwork, ss: np.ndarray) -> LayerJacobians:
    """Lemma 9: cell Jacobians and input gradient; pre-activation rows for Lemma 10.

    Hidden ``Linear→ReLU`` blocks update a running map. For each block we store the
    Jacobian **before** that block's ReLU mask (bent-hyperplane normals). Masks are
    then applied so later layers and ``full`` see the correct inactive columns.

    If the network ends with an output ReLU (decision-boundary topology via
    ``add_output_relu``), Morse analyzes the **logit** (pre-output-ReLU scalar): the
    output mask is not applied to ``gradient``, and ``by_relu_layer`` records the
    logit Jacobian so Lemma 10 can use the output supporting hyperplane.

    Memoized within an active ``_JacobianCacheScope`` (see there); a plain cache miss
    (or no active scope) falls through to computing it fresh.
    """
    cache = getattr(_jacobian_cache_state, "cache", None)
    cache_key = (id(net), _coerce_ss(ss).tobytes()) if cache is not None else None
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    result = _compute_layer_jacobians(net, ss)
    if cache is not None:
        cache[cache_key] = result
    return result


def _compute_layer_jacobians(net: ReLUNetwork, ss: np.ndarray) -> LayerJacobians:
    ss_row = _coerce_ss(ss)
    n_in = int(np.prod(net.input_shape))
    current = np.eye(n_in, dtype=np.float64)
    current_abs = np.eye(n_in, dtype=np.float64)
    by_relu: list[np.ndarray] = []
    by_relu_abs: list[np.ndarray] = []
    n_terms = 1
    mask_index = 0

    layers = list(net.layers.values())
    last_linear_i = max(
        (i for i, layer in enumerate(layers) if isinstance(layer, LinearLayer)),
        default=-1,
    )
    if last_linear_i < 0:
        raise ValueError("network has no linear layers")

    # Hidden Linear→ReLU blocks only (stop before the final linear ``G``).
    i = 0
    while i < last_linear_i:
        layer = layers[i]
        if isinstance(layer, LinearLayer) and isinstance(layers[i + 1], ReLULayer):
            current = current @ layer.weight.T
            current_abs = current_abs @ np.abs(layer.weight).T
            n_terms += int(layer.weight.shape[1])
            width = layer.weight.shape[0]
            # Pre-mask columns = ∇preact_j (Lemma 10 / bent hyperplane normals).
            by_relu.append(current.copy())
            by_relu_abs.append(current_abs.copy())
            mask = ss_row[0, mask_index : mask_index + width]
            relu = (mask == 1).astype(np.float64)
            current = current * relu[np.newaxis, :]
            current_abs = current_abs * relu[np.newaxis, :]
            mask_index += width
            i += 2
            continue
        i += 1

    last_linear_layer = layers[last_linear_i]
    if not isinstance(last_linear_layer, LinearLayer):
        raise ValueError(f"expected a LinearLayer at index {last_linear_i}, got {type(last_linear_layer).__name__}")
    w_out = last_linear_layer.weight
    full = current.copy()
    current = current @ w_out.T
    current_abs = current_abs @ np.abs(w_out).T
    n_terms += int(w_out.shape[1])
    # Trailing output ReLU: keep logit gradient; still record logit map for Lemma 10.
    if last_linear_i + 1 < len(layers) and isinstance(layers[last_linear_i + 1], ReLULayer):
        by_relu.append(current.copy())
        by_relu_abs.append(current_abs.copy())
        mask_index += int(w_out.shape[0])

    if current.ndim != 2 or current.shape[1] != 1:
        raise ValueError("Morse Jacobians require a scalar network output " + f"(got trailing map shape {current.shape})")
    gradient = current.reshape(-1)
    return LayerJacobians(
        by_relu_layer=by_relu,
        full=full,
        gradient=gradient,
        W_out=w_out,
        by_relu_layer_abs=by_relu_abs,
        gradient_abs=current_abs.reshape(-1),
        n_terms=n_terms,
    )


def gradient_on_cell(net: ReLUNetwork, ss: np.ndarray) -> np.ndarray:
    """Lemma 9: ``∇F|_C`` as a length-``n_in`` vector (scalar output)."""
    assert_scalar_output(net)
    return layer_jacobians(net, ss).gradient


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


def _sign_with_bound(value: float, bound: float) -> int:
    """Sign of ``value``, or 0 when it is zero or within its float64 error ``bound`` of zero.

    0 is the explicit "degenerate or undecidable" answer that the Morse classification counts
    separately (``morse_count_deg``); it is never guessed to be +1 or -1.
    """
    if abs(value) <= bound:
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
    val, bound = partial_derivative_value_with_error(
        vertex_ss,
        edge_ss,
        net,
        ssi2maski=ssi2maski,
        ss_layers=ss_layers,
    )
    return _sign_with_bound(val, bound)


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

    # W(v, C): row k = ∇preact of zero-entry (i_k, j_k) on coface C (unmasked at that neuron).
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
    jac = layer_jacobians(net, coface)
    direction = _vertex_edge_direction(
        vertex_ss,
        edge_ss,
        jac,
        ssi2maski=ssi2maski,
        ss_layers=ss_layers,
    )
    # Corollary 3: ∂_{vE} F = ∇F|_C · (v→E).
    return float(jac.gradient @ direction)


def partial_derivative_value_with_error(
    vertex_ss: np.ndarray,
    edge_ss: np.ndarray,
    net: ReLUNetwork,
    *,
    ssi2maski: list[tuple[int, tuple[int, int]]],
    ss_layers: list[int],
) -> tuple[float, float]:
    """``∂_{vE} F`` and a bound on its float64 error (``inf`` when the direction is undecidable).

    The gradient and the vertex's hyperplane normals are products of the network's weights;
    each entry's error is at most ``2 gamma(K)`` times the same product with ``|W|``. The
    direction solves ``W(v, C) d = e``; its error follows from the perturbation of ``W(v, C)``
    and the solve residual. When ``W(v, C)`` is singular within its error the direction itself
    cannot be computed, and the bound is ``inf``.
    """
    from relucent._internal import rounding

    assert_scalar_output(net)
    v = _coerce_ss(vertex_ss).ravel()
    e = _coerce_ss(edge_ss).ravel()
    if not np.all((e != 0) | (v == 0)):
        raise ValueError("vertex must be a face of the edge (edge zeros are also zeros at the vertex)")
    coface = coface_sign_sequence(edge_ss)
    jac = layer_jacobians(net, coface)
    assert jac.by_relu_layer_abs is not None and jac.gradient_abs is not None
    g_rel = 2.0 * rounding.gamma(jac.n_terms)

    shi = _diff_shi(vertex_ss, edge_ss)
    n_in = jac.full.shape[0]
    zeros = np.flatnonzero(v == 0)
    if zeros.size != n_in:
        raise ValueError(f"vertex sign sequence must have {n_in} zero entries (got {zeros.size}) for Lemma 10 edge directions")
    rows, rows_abs = [], []
    for z in zeros:
        r_idx, n_idx = shi_to_relu_neuron(int(z), ssi2maski, ss_layers)
        rows.append(jac.by_relu_layer[r_idx][:, n_idx])
        rows_abs.append(jac.by_relu_layer_abs[r_idx][:, n_idx])
    w_mat = np.vstack(rows)
    dw = float(np.linalg.norm(g_rel * np.vstack(rows_abs)))
    smin = float(np.linalg.svd(w_mat, compute_uv=False)[-1])
    if smin <= 2.0 * dw:
        return 0.0, float("inf")
    e_row = np.zeros(n_in, dtype=np.float64)
    e_row[int(np.where(zeros == shi)[0][0])] = 1.0
    scale = float(e[shi])
    d = np.linalg.solve(w_mat, e_row)
    resid = np.abs(w_mat @ d - e_row) + rounding.gamma(n_in) * (np.abs(w_mat) @ np.abs(d))
    d_err = (float(np.linalg.norm(resid)) + dw * float(np.linalg.norm(d))) / (smin - dw)
    direction = scale * d
    grad = jac.gradient
    grad_err = g_rel * jac.gradient_abs
    value = float(grad @ direction)
    bound = (
        float(grad_err @ np.abs(direction))
        + (float(np.linalg.norm(grad)) + float(np.linalg.norm(grad_err))) * d_err
        + rounding.gamma(n_in) * float(np.abs(grad) @ np.abs(direction))
    )
    return value, bound * (1.0 + 8.0 * rounding.EPS)


def _edge_ss_from_vertex(vertex_ss: np.ndarray, shi: int, sign: int) -> np.ndarray:
    ss = _coerce_ss(vertex_ss).copy()
    ss.ravel()[int(shi)] = int(sign)
    return ss


def _is_collapsed_edge(
    net: ReLUNetwork,
    edge_ss: np.ndarray,
) -> bool:
    """True when the gradient on the coface is zero (F constant on the whole coface).

    This is a sufficient condition for the edge to be flat (zero directional
    derivative). The complementary case — non-zero gradient orthogonal to the
    edge direction — is caught downstream by the ``sign == 0`` check.
    """
    from relucent._internal import rounding

    coface = coface_sign_sequence(edge_ss)
    assert_scalar_output(net)
    jac = layer_jacobians(net, coface)
    assert jac.gradient_abs is not None
    # Zero, or zero to within every entry's own float64 error: F cannot be shown to vary.
    return bool(np.all(np.abs(jac.gradient) <= 2.0 * rounding.gamma(jac.n_terms) * jac.gradient_abs))


def is_pl_critical_vertex(
    vertex_ss: np.ndarray,
    net: ReLUNetwork,
    *,
    ssi2maski: list[tuple[int, tuple[int, int]]],
    ss_layers: list[int],
) -> tuple[bool, int | None]:
    """PL Morse criticality and index (Brooks–Masden Def. 5, Section 3.2 / [19] Thm 3.7.3).

    At a vertex of C(F) the sign sequence has exactly ``n_in`` zero entries, one
    per bent-hyperplane axis, with edges ``s=±1`` on each axis. Along each axis:

    * opposite ``∂_{vE} F`` signs ⇒ F is monotonic through the vertex (PL regular);
    * both negative ⇒ both edges oriented toward the vertex (1D local max);
    * both positive ⇒ both edges oriented away (1D local min).

    The vertex is PL critical when every axis is a 1D local max or min. The Morse
    index is the number of axes oriented toward the vertex (so index ``0`` is a
    local min of ``F`` and index ``d`` is a local max in ambient dimension ``d``).
    Flat / singular directions are reported as degenerate (index ``-1``).

    Args:
        vertex_ss: Sign sequence of a 0-dimensional cell (vertex) of C(F). Must
            have exactly ``n_in`` zero entries. Top-dimensional cells (no zeros)
            return ``(False, None)``; intermediate-dimensional cells raise.
    """
    assert_scalar_output(net)
    v = _coerce_ss(vertex_ss).ravel()
    zeros = np.flatnonzero(v == 0)
    if zeros.size == 0:
        return False, None
    n_in = int(np.prod(net.input_shape))
    if zeros.size != n_in:
        raise ValueError(
            f"vertex sign sequence must have exactly {n_in} zero entries "
            + f"(got {zeros.size}); is_pl_critical_vertex requires a 0-cell of C(F)"
        )

    # Count bent-hyperplane axes whose ± edges both point toward v. The cache scope
    # saves recomputing each edge's Jacobian (used by both `_is_collapsed_edge` and
    # `partial_derivative_sign`).
    towards = 0
    with _JacobianCacheScope():
        for shi in zeros:
            edge_plus = _edge_ss_from_vertex(vertex_ss, int(shi), 1)
            edge_minus = _edge_ss_from_vertex(vertex_ss, int(shi), -1)
            if _is_collapsed_edge(net, edge_plus) or _is_collapsed_edge(net, edge_minus):
                return True, -1

            sign_plus = partial_derivative_sign(vertex_ss, edge_plus, net, ssi2maski=ssi2maski, ss_layers=ss_layers)
            sign_minus = partial_derivative_sign(vertex_ss, edge_minus, net, ssi2maski=ssi2maski, ss_layers=ss_layers)
            if sign_plus == 0 or sign_minus == 0:
                return True, -1
            if sign_plus != sign_minus:
                # Monotonic along this axis: not a PL Morse critical vertex.
                return False, None
            # Negative ∂_{vE} F: F decreases along v→E, so the edge is oriented toward v.
            if sign_plus < 0:
                towards += 1

    return True, towards


# Each worker needs at least this many vertices to be worth its startup cost (same
# tradeoff as `graph.vertex_star.MIN_CANDIDATES_PER_WORKER`). Not calibrated yet:
# `is_pl_critical_vertex` does several Jacobian products per vertex, so this is
# deliberately conservative (low) until measured.
MIN_VERTICES_PER_WORKER = 256
PARALLEL_CRITICAL_MIN_VERTICES = 2 * MIN_VERTICES_PER_WORKER


def _is_pl_critical_vertex_chunk(
    ss_chunk: list[np.ndarray],
    net: ReLUNetwork,
    ssi2maski: list[tuple[int, tuple[int, int]]],
    ss_layers: list[int],
) -> list[tuple[bool, int | None]]:
    """Worker: criticality + Morse index for one chunk of vertex sign sequences."""
    return [is_pl_critical_vertex(ss, net, ssi2maski=ssi2maski, ss_layers=ss_layers) for ss in ss_chunk]


def critical_flags_for_vertices(
    vertex_ss_list: list[np.ndarray],
    net: ReLUNetwork,
    *,
    ssi2maski: list[tuple[int, tuple[int, int]]],
    ss_layers: list[int],
    nworkers: int = 1,
) -> list[tuple[bool, int | None]]:
    """``is_pl_critical_vertex`` for every vertex, sequential or Pool-parallel by size.

    Each vertex's criticality check is independent of every other's -- the same shape
    of embarrassingly-parallel work as candidate-vertex verification in
    ``graph.vertex_star.find_vertices``, which already farms out across a worker pool
    once there's enough of it to justify Pool startup cost. Below the size gate, or
    with a single worker, this is exactly the plain sequential loop (no Pool overhead
    for small complexes), and results are returned in the same order as the input list
    either way.
    """
    n = len(vertex_ss_list)
    if n == 0:
        return []

    if nworkers <= 1 or n < PARALLEL_CRITICAL_MIN_VERTICES:
        return [is_pl_critical_vertex(ss, net, ssi2maski=ssi2maski, ss_layers=ss_layers) for ss in vertex_ss_list]

    from relucent._internal.parallel import worker_pool

    effective_workers = min(nworkers, max(1, n // MIN_VERTICES_PER_WORKER))
    chunk_size = max(n // (effective_workers * 4), 1)
    chunks = [vertex_ss_list[i : i + chunk_size] for i in range(0, n, chunk_size)]

    results: list[tuple[bool, int | None]] = []
    with worker_pool(effective_workers) as pool:
        for chunk_results in pool.starmap(
            _is_pl_critical_vertex_chunk,
            [(chunk, net, ssi2maski, ss_layers) for chunk in chunks],
        ):
            results.extend(chunk_results)
    return results


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


@with_verbosity
def critical_points(
    cplx: Complex,
    *,
    require_complete: bool = False,
    include_degenerate: bool = False,
    verbose: int | None = None,
) -> list[CriticalPoint]:
    """Return PL Morse critical vertices and their indices in the discovered complex.

    Uses combinatorial edge data (Brooks & Masden, arXiv:2412.18005) and requires a
    scalar-output network.

    Args:
        require_complete: If True, require every combinatorial 1-cell incident to
            each tested vertex to appear in the complex.
        include_degenerate: If True, include flat / degenerate critical vertices
            (index ``-1``).
        verbose: Output level: ``0`` quiet, ``1`` progress bars and summaries, ``2`` debug
            detail. ``None`` uses :data:`relucent.config.VERBOSE`.

    Returns:
        List of :class:`~relucent.topology.morse.CriticalPoint` records.
    """
    from relucent._internal.parallel import process_aware_cpu_count
    from relucent.core.ss import encode_ss
    from relucent.graph import incidence
    from relucent.graph.vertex_star import verified_vertices, vertex_polyhedron

    assert_scalar_output(cplx._net)
    cplx.assert_topology_ready()
    if len(cplx) == 0:
        return []
    # Criticality needs only each vertex's sign sequence, so find the verified vertices
    # (the 0-cells of chain_complex) without building the rest of the chain complex.
    _, found = verified_vertices(cplx)
    if not found:
        return []
    vertices = [found[tag] for tag in sorted(found)]

    flags: list[tuple[bool, int | None]]
    if require_complete:
        meta = cplx.meta_graph(verbose=verbose)
        one_cell_tags = {tag for tag, attrs in meta.nodes(data=True) if int(attrs.get("dim", -1)) == 1}

        # The completeness check reads `cplx`/`meta` per vertex, so this path stays
        # sequential; only the common `require_complete=False` case below (every
        # caller in this codebase) is farmed out across a worker pool.
        from relucent.core.ss import encode_ss

        flags = []
        for vertex in vertices:
            # Incident edges are inferred combinatorially; this checks they were discovered.
            v_ss = vertex.ss.ravel()
            for shi in np.flatnonzero(v_ss == 0):
                for sign in (-1, 1):
                    edge_ss = v_ss.copy()
                    edge_ss[int(shi)] = sign
                    tag = encode_ss(edge_ss.reshape(1, -1))
                    if tag not in one_cell_tags:
                        raise ValueError(
                            f"combinatorial 1-cell {tag!r} incident to vertex {vertex.tag!r} "
                            + "is missing from the discovered complex"
                        )
            flags.append(
                is_pl_critical_vertex(
                    vertex.ss,
                    cplx._net,
                    ssi2maski=cplx.ssi2maski,
                    ss_layers=cplx.ss_layers,
                )
            )
    else:
        # Each vertex's criticality check is independent (like candidate verification in
        # chain_complex), so use a worker pool once there's enough work.
        nworkers = process_aware_cpu_count() or 1
        flags = critical_flags_for_vertices(
            [vertex.ss for vertex in vertices],
            cplx._net,
            ssi2maski=cplx.ssi2maski,
            ss_layers=cplx.ss_layers,
            nworkers=nworkers,
        )

    vertex_complex = cplx._empty_like()
    results: list[CriticalPoint] = []
    for vertex, (is_critical, index) in zip(vertices, flags, strict=True):
        if not is_critical:
            continue
        if index is None or (index < 0 and not include_degenerate):
            continue
        results.append(
            CriticalPoint(
                polyhedron=vertex_polyhedron(cplx, vertex_complex, vertex),
                tag=vertex.tag,
                ss=np.asarray(vertex.ss, dtype=np.int8).copy(),
                point=np.asarray(vertex.point, dtype=np.float64).reshape(-1),
                index=int(index),
            )
        )
    if len(vertex_complex):
        incidence.set_contracted_shis(vertex_complex)
    return results
