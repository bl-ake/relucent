"""Float64 error bounds for decisions made on a cell's halfspace rows.

A cell's rows ``a_j . x + b_j <= 0`` are composed from the network's weights, so each
row carries rounding error, and so does anything computed from it. Every numerical
decision (is this row a facet? is this point inside? are these vertices the same?)
must be right for the exact rows, or raise. This module supplies the bounds those
decisions compare against.

Row ``j``'s bound is stored in an error-scale matrix ``E`` shaped like the halfspace
matrix. The float64 error of row ``j`` at ``x`` is at most

    E[j, :-1] @ |x| + E[j, -1].

For rows composed from a network, ``E = 2 * gamma(K) * [|A| | |b|]``, where ``|A|, |b|``
are the rows rebuilt with ``|W|`` through the same pattern (cancellation can make this
much bigger than the row itself) and ``K`` counts the inner-product terms in the
composition plus the evaluation. Transforms of rows (slicing, appending box rows)
update ``E`` with their own rounding, so the bound travels with the rows.

Don't swap these for a network-wide constant: row magnitudes on one network span five
or more orders of magnitude.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from relucent.core.errors import AmbiguousGeometryError

if TYPE_CHECKING:
    from relucent.model.model import ReLUNetwork

EPS = float(np.finfo(np.float64).eps)

__all__ = [
    "EPS",
    "box_rows_error",
    "classify_rows",
    "composition_terms",
    "exact_rows_error",
    "gamma",
    "halfspaces_error_for_ss",
    "row_errors",
    "slice_error",
    "solve_equalities",
]


def gamma(n: int) -> float:
    """Higham's ``gamma_n = n u / (1 - n u)``: relative error bound for an ``n``-term inner product."""
    n = max(int(n), 1)
    denom = 1.0 - n * EPS
    if denom <= 0.0:
        raise ValueError(f"gamma({n}) is undefined in float64")
    return n * EPS / denom


def composition_terms(net: ReLUNetwork) -> int:
    """Inner-product terms accumulated by composing ``net``'s rows and evaluating one at a point."""
    from relucent.model.model import LinearLayer

    layers = [layer for layer in net.layers.values() if isinstance(layer, LinearLayer)]
    input_dim = int(np.prod(net.input_shape))
    # Each composition step sums over the previous layer's width (plus the bias); evaluating
    # a row at x adds input_dim + 1 more.
    return input_dim + sum(int(np.asarray(layer.weight).shape[1]) + 1 for layer in layers) + 1


def halfspaces_error_for_ss(net: ReLUNetwork, ss: np.ndarray) -> np.ndarray:
    """Error-scale matrix for the rows :func:`relucent.geometry.calculations.get_hs` builds for ``ss``.

    Mirrors ``_get_hs_numpy`` exactly, with ``|W|`` and ``|b|`` in place of ``W`` and ``b``:
    a sign-sequence entry of 0 keeps the unit's row but switches it off downstream, as there.
    """
    from relucent.model.model import FlattenLayer, LinearLayer, ReLULayer

    row = np.asarray(ss).ravel()
    cur_a: np.ndarray | None = None
    cur_b: np.ndarray | None = None
    cols_a: list[np.ndarray] = []
    cols_b: list[np.ndarray] = []
    idx = 0
    for layer in net.layers.values():
        if isinstance(layer, LinearLayer):
            w = np.abs(np.asarray(layer.weight, dtype=np.float64))
            b = np.abs(np.asarray(layer.bias, dtype=np.float64)).reshape(1, -1)
            if cur_a is None or cur_b is None:
                cur_a = np.eye(w.shape[1])
                cur_b = np.zeros((1, w.shape[1]))
            cur_a = cur_a @ w.T
            cur_b = cur_b @ w.T + b
        elif isinstance(layer, ReLULayer):
            if cur_a is None or cur_b is None:
                raise ValueError("ReLU layer must follow a linear layer")
            n = cur_a.shape[1]
            mask = row[idx : idx + n]
            cols_a.append(cur_a.copy())
            cols_b.append(cur_b.copy())
            on = (mask == 1).astype(np.float64)
            cur_a = cur_a * on
            cur_b = cur_b * on
            idx += n
        elif isinstance(layer, FlattenLayer):
            if cur_a is not None:
                raise NotImplementedError("Intermediate flatten layer not supported")
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")
    if not cols_a:
        raise ValueError("Network has no ReLU layers")
    abs_rows = np.hstack((np.hstack(cols_a).T, np.hstack(cols_b).reshape(-1, 1)))
    return 2.0 * gamma(composition_terms(net)) * abs_rows


def exact_rows_error(halfspaces: np.ndarray) -> np.ndarray:
    """Error scale for rows given as exact data: evaluation rounding only."""
    h = np.abs(np.asarray(halfspaces, dtype=np.float64))
    return 2.0 * gamma(h.shape[1] + 1) * h


def box_rows_error(dim: int, bound: float) -> np.ndarray:
    """Error scale for the ``2 * dim`` axis box rows ``+-x_k - bound <= 0`` (exact data)."""
    rows = np.hstack((np.vstack((np.eye(dim), np.eye(dim))), np.full((2 * dim, 1), float(bound))))
    return exact_rows_error(rows)


def slice_error(halfspaces: np.ndarray, error: np.ndarray, basis: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Error scale of rows restricted to ``{x0 + basis @ t}``: ``[A basis | A x0 + b]``.

    Carries the rows' own error through the substitution and adds the substitution's rounding.
    """
    h = np.asarray(halfspaces, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    v = np.abs(np.asarray(basis, dtype=np.float64))
    p = np.abs(np.asarray(x0, dtype=np.float64)).reshape(-1)
    a_abs, b_abs = np.abs(h[:, :-1]), np.abs(h[:, -1])
    g = 2.0 * gamma(h.shape[1] + 1)
    new_a = e[:, :-1] @ v + g * (a_abs @ v)
    new_b = e[:, :-1] @ p + e[:, -1] + g * (a_abs @ p + b_abs)
    return np.hstack((new_a, new_b.reshape(-1, 1)))


def row_errors(error: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-row float64 error bound of ``halfspaces @ [x, 1]`` at ``x``."""
    e = np.asarray(error, dtype=np.float64)
    ax = np.abs(np.asarray(x, dtype=np.float64)).reshape(-1)
    # The bound's own evaluation is itself rounded; 4 ulps of slack covers it.
    return (e[:, :-1] @ ax + e[:, -1]) * (1.0 + 4.0 * EPS)


def solve_equalities(halfspaces: np.ndarray, error: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Point where ``rows`` hold with equality, and a bound on its distance to the exact point.

    Returns ``None`` when the exact system is inconsistent beyond doubt. Raises
    :class:`AmbiguousGeometryError` when the equality rows are dependent within their error
    (the exact system may be singular) or consistency cannot be decided.
    """
    h = np.asarray(halfspaces, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    idx = np.asarray(rows, dtype=np.intp)
    a = h[idx, :-1]
    rhs = -h[idx, -1]
    dim = a.shape[1]
    if a.shape[0] < dim:
        raise ValueError(f"{a.shape[0]} equality rows cannot fix a point in dimension {dim}")
    # Row errors bound a perturbation dA of the exact matrix: ||dA||_2 <= ||E_A||_F.
    da = float(np.linalg.norm(e[idx, :-1]))
    s = np.linalg.svd(a, compute_uv=False)
    smin = float(s[dim - 1]) if s.size >= dim else 0.0
    if smin <= 2.0 * da:
        raise AmbiguousGeometryError(
            f"equality rows {idx.tolist()} are dependent within their float64 error "
            + f"(smallest singular value {smin:.3e}, row error {da:.3e})"
        )
    x, *_ = np.linalg.lstsq(a, rhs, rcond=None)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    resid = np.abs(a @ x - rhs)
    err_rows = row_errors(e[idx], x) + row_errors(np.abs(h[idx]) * 2.0 * gamma(dim + 1), x)
    if a.shape[0] > dim:
        # More equality rows than dimensions: generically inconsistent. A residual within the
        # rows' error means the extra hyperplanes pass through the point to float64 precision.
        if np.any(resid > 4.0 * err_rows):
            return None
        raise AmbiguousGeometryError(
            f"{a.shape[0]} equality rows {idx.tolist()} meet at one point in dimension {dim} to within float64 error"
        )
    # ||x - x*|| <= ||A^-1|| (||residual + row error at x||), with A^-1 perturbed by dA.
    x_err = float(np.linalg.norm(resid + err_rows)) / (smin - da)
    return x, x_err


def classify_rows(halfspaces: np.ndarray, error: np.ndarray, x: np.ndarray, x_err: float, rows: Any = None) -> np.ndarray:
    """Side of each row at the exact point near ``x``: -1 strictly inside, +1 outside, 0 undecidable.

    ``x_err`` bounds the distance from ``x`` to the exact point (0 for a point given exactly).
    """
    h = np.asarray(halfspaces, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    if rows is not None:
        idx = np.asarray(rows, dtype=np.intp)
        h, e = h[idx], e[idx]
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    s = h[:, :-1] @ xv + h[:, -1]
    bound = row_errors(e, xv) + float(x_err) * (np.linalg.norm(h[:, :-1], axis=1) + np.linalg.norm(e[:, :-1], axis=1))
    out = np.zeros(s.shape[0], dtype=np.int8)
    out[s < -bound] = -1
    out[s > bound] = 1
    return out
