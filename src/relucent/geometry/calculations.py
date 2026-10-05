"""Polyhedral computation helpers (halfspaces, SHIs, qhull interior / SHI models).

Functions here take a :class:`~relucent.core.poly.Polyhedron` instance; the class lives in
``poly.py`` to avoid import cycles.
"""

import warnings
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from gurobipy import GRB, Env, Model
from scipy.spatial import ConvexHull, HalfspaceIntersection
from tqdm.auto import tqdm

import relucent.config as cfg
from relucent._internal.gurobi import get_env
from relucent._internal.torch_compat import TORCH_AVAILABLE, torch
from relucent.core.ss import flip_ss_at_shi
from relucent.model.model import FlattenLayer, LinearLayer, ReLULayer

if TYPE_CHECKING:
    from relucent.core.poly import Polyhedron

__all__ = [
    "adjacent_polyhedra",
    "certified_bounded",
    "compute_properties",
    "get_hs",
    "get_shis",
    "solve_radius",
]


def _rounding() -> Any:
    from relucent._internal import rounding

    return rounding


class DegenerateHalfspaceInfeasibility(ValueError):
    """Near-zero normal with positive bias: :math:`a^\\top x + b \\le 0` is empty when ``||a||≈0`` and ``b>0``."""


def _shi_variable_bounds_to_try(bound: float) -> list[float]:
    """Ordered variable-box radii for the SHI LP; escalate when the first box is too small."""
    inf = float(GRB.INFINITY)
    if bound == inf or not np.isfinite(bound):
        return [inf]
    candidates = [float(bound)]
    for radius in cfg.advanced.INTERIOR_POINT_RADIUS_SEQUENCE:
        radius_f = float(radius)
        if radius_f > bound:
            candidates.append(radius_f)
    candidates.append(inf)
    seen: set[float] = set()
    ordered: list[float] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _degenerate_rows(halfspaces: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """Rows whose normal is exactly zero: constant constraints ``b <= 0``, to be dropped.

    Such a row always holds when ``b`` is below minus its float64 error and makes the region empty
    when ``b`` is above its error (:class:`DegenerateHalfspaceInfeasibility`); a constant within its
    error of zero cannot be decided and raises :class:`~relucent.core.errors.AmbiguousGeometryError`
    (an exact ``0 <= 0``, both value and error zero, holds). A normal that is tiny but not exactly
    zero is a real row -- often exactly parallel to another row, as when one upstream unit drives a
    deeper one -- and is kept: every decision evaluates it at a point against its own error bound.
    """
    from relucent.core.errors import AmbiguousGeometryError

    a = halfspaces[:, :-1]
    b = halfspaces[:, -1]
    eb = errors[:, -1]
    zero = np.all(a == 0.0, axis=1)
    undecided = zero & (np.abs(b) <= eb) & ~((b == 0.0) & (eb == 0.0))
    if np.any(undecided):
        raise AmbiguousGeometryError(
            f"constant halfspace rows {np.flatnonzero(undecided).tolist()} have constants within their float64 error of zero"
        )
    if np.any(zero & (b > eb)):
        bad = np.flatnonzero(zero & (b > eb)).tolist()
        raise DegenerateHalfspaceInfeasibility(
            f"Degenerate halfspace(s) imply infeasibility: constant rows {bad} are positive, so the region is empty"
        )
    return zero


def _near_duplicate_rows(halfspaces: np.ndarray, errors: np.ndarray) -> list[tuple[int, int]]:
    """Pairs of rows that may be the same halfspace (equal up to a positive scale) within their error.

    Rows whose unit-normalised ``[a | b]`` differ by at most the sum of their relative float64
    errors (twice, for slack) -- every exactly coincident pair, plus near misses that float64
    cannot tell apart from one. Rows with a zero normal are skipped.
    """
    norms = np.linalg.norm(halfspaces, axis=1)
    live = np.flatnonzero(np.any(halfspaces[:, :-1] != 0.0, axis=1) & (norms > 0.0))
    if live.size < 2:
        return []
    u = halfspaces[live] / norms[live, None]  # normalise the full row (a AND b)
    rel = np.linalg.norm(errors[live], axis=1) / norms[live]
    allow = 2.0 * (rel[:, None] + rel[None, :])
    # ||u - v||^2 = 2 - 2 u.v screens candidate pairs in O(n^2) memory; the Gram entries are off by
    # at most ~gamma(d+1) (plus the normalisation), so widen by that. Candidates are then judged by
    # direct differences, since 2 - 2 u.v cancels to 0 for rows closer than ~1e-8.
    gram = u @ u.T
    screen = 4.0 * _rounding().gamma(u.shape[1] + 2)
    cand = np.argwhere(np.triu(gram >= 1.0 - 0.5 * allow**2 - screen, k=1))
    return [(int(live[r]), int(live[c])) for r, c in cand if float(np.linalg.norm(u[r] - u[c])) <= allow[r, c]]


def _drop_degenerate_halfspaces_tracked(
    halfspaces: np.ndarray,
    *,
    errors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove halfspaces whose normal is zero, and return an old-row → new-row map.

    A row ``a^T x + b <= 0`` with ``a = 0`` either always holds (``b < 0``) or makes the region
    empty (``b > 0``). Keeping such rows can trigger Qhull errors (e.g. QH6023) and destabilize
    interior-point solves. See :func:`_degenerate_rows` for how rows are judged against their
    float64 error.

    ``old_to_new[i]`` is the row index in the returned array corresponding to
    original row ``i``, or ``-1`` if that row was dropped. ``errors`` is the rows' float64 error
    scale (:mod:`relucent._internal.rounding`); without it the rows are treated as exact data.
    """
    from relucent._internal import rounding

    n_old = halfspaces.shape[0]
    if halfspaces.size == 0:
        return halfspaces, np.arange(0, dtype=np.intp)
    if errors is None:
        errors = rounding.exact_rows_error(halfspaces)

    deg = _degenerate_rows(halfspaces, errors)

    kept = np.flatnonzero(~deg)
    old_to_new = np.full(n_old, -1, dtype=np.intp)
    old_to_new[kept] = np.arange(kept.size, dtype=np.intp)

    # Rows whose normal is within its float64 error of zero are constants, not hyperplanes
    # (typically rows constant on a face, written in that face's coordinates); comparing
    # them with each other says nothing about coincident hyperplanes.
    hyperplane = np.any(np.abs(halfspaces[:, :-1]) > errors[:, :-1], axis=1)
    kept_planes = kept[hyperplane[kept]]
    if cfg.CAREFUL_MODE and kept_planes.size > 1:
        # Look for fully identical rows (the same halfspace from two neurons). Same
        # direction with a different bias is fine: get_shis keeps the tighter constraint.
        # Only rows equal up to a positive scale, within float64 error (unit-normalised
        # rows closer than the sum of their relative errors), silently break SHI detection.
        pairs = [
            (int(kept_planes[r]), int(kept_planes[c]))
            for r, c in _near_duplicate_rows(halfspaces[kept_planes], errors[kept_planes])
        ]
        if pairs:
            from relucent.core.errors import NonGenericArrangementError

            raise NonGenericArrangementError(
                f"Network fails genericity: halfspace rows at original indices {pairs} "
                + "are identical to within float64 error (same direction and bias). Typically two "
                + "neurons have parallel weight vectors, or a unit with an exactly zero bias has one "
                + "active input."
            )

    if not np.any(deg):
        return halfspaces, np.arange(n_old, dtype=np.intp)
    return halfspaces[~deg], old_to_new


def _remap_zero_indices(zero_indices: np.ndarray | None, old_to_new: np.ndarray) -> np.ndarray | None:
    """Map sign-sequence indices through a degenerate-row removal map."""
    if zero_indices is None or len(zero_indices) == 0:
        return None
    zm = old_to_new[np.asarray(zero_indices, dtype=np.intp)]
    zm = zm[zm >= 0]
    if zm.size == 0:
        return None
    return zm


def _affine_null_basis(
    halfspaces: np.ndarray,
    zero_indices: np.ndarray,
    errors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Null-space basis for the equality sub-system of a halfspace array.

    Splits the equality rows (``zero_indices``) from the inequalities and returns what
    you need to project points or halfspace systems into the affine subspace they define.
    Uses an SVD of the equality normals, which is more stable than forming ``A A^T``.

    Args:
        halfspaces: Shape ``(n, d+1)`` halfspace array; last column is bias.
        zero_indices: Row indices to treat as equalities (the ``zero_indices``
            attribute of a non-maximal :class:`~relucent.core.poly.Polyhedron`).
            Must be non-empty.

    Returns:
        ``(x0, null_basis, ineq_mask)`` where

        * ``x0`` – shape ``(d, 1)`` particular solution satisfying
          ``eq_A @ x0 ≈ −eq_b``.
        * ``null_basis`` – shape ``(d, k)`` orthonormal columns spanning the
          null space of ``eq_A``; ``k == 0`` means the equalities pin a single
          point and no inequalities can further restrict it.
        * ``ineq_mask`` – boolean array of length ``n`` that is ``True`` for
          inequality rows (everything not in ``zero_indices``).

    Usage patterns::

        # Project a halfspace system to reduced coordinates z  (x = N z + x0)
        ineqs = halfspaces[ineq_mask]
        A_red = ineqs[:, :-1] @ null_basis
        b_red = ineqs[:, :-1] @ x0 + ineqs[:, -1:]
        reduced_halfspaces = np.hstack((A_red, b_red))

        # Project a point to reduced coordinates
        z = np.linalg.lstsq(null_basis, point.reshape(-1, 1) - x0, rcond=None)[0].reshape(-1)

        # Unproject vertices back to ambient coordinates
        ambient_verts = (null_basis @ reduced_verts.T + x0).T

        # Effective norm of inequality normals in the affine subspace (for Chebyshev LP)
        norm_vec = np.linalg.norm(ineqs[:, :-1] @ null_basis, axis=1, keepdims=True)

    The equality rows must be independent beyond their float64 error (``errors``; exact
    when omitted). Otherwise the intersection isn't transversal and
    :class:`~relucent.core.errors.AmbiguousGeometryError` is raised.
    """
    from relucent._internal import rounding
    from relucent.core.errors import AmbiguousGeometryError

    if errors is None:
        errors = rounding.exact_rows_error(halfspaces)
    equalities = halfspaces[zero_indices]
    ineq_mask = ~np.isin(np.arange(halfspaces.shape[0]), zero_indices)
    eq_A, eq_b = equalities[:, :-1], equalities[:, -1:]
    x0 = np.asarray(np.linalg.lstsq(eq_A, -eq_b, rcond=None)[0]).reshape(-1, 1)
    _, s, vh = np.linalg.svd(eq_A, full_matrices=True)
    da = float(np.linalg.norm(errors[zero_indices, :-1]))
    rank = int(np.sum(s > 2.0 * da))
    if rank < min(eq_A.shape):
        raise AmbiguousGeometryError(
            f"equality rows {np.asarray(zero_indices).tolist()} are dependent within their float64 error "
            + f"(singular values {s.tolist()}, row error {da:.3e})"
        )
    null_basis = vh[rank:, :].T  # shape (d, k) – orthonormal columns
    return x0, null_basis, ineq_mask


def _reduced_rows_error(
    halfspaces: np.ndarray,
    errors: np.ndarray,
    zero_indices: np.ndarray,
    ineq_mask: np.ndarray,
    x0: np.ndarray,
    null_basis: np.ndarray,
) -> np.ndarray:
    """Error scale of the inequality rows written in the face coordinates ``x = null_basis @ z + x0``.

    Two sources: the rows' own error carried through the substitution, and the distance from
    ``null_basis @ z + x0`` to the exact affine hull (``x0`` and ``null_basis`` are float64
    approximations), which moves each row's value by at most ``||a_j||`` times that distance.
    """
    from relucent._internal import rounding

    ineq = halfspaces[ineq_mask]
    err = rounding.slice_error(ineq, errors[ineq_mask], null_basis, x0)
    eq = halfspaces[zero_indices]
    eq_err = errors[zero_indices]
    a_eq = eq[:, :-1]
    n_abs = np.abs(null_basis)
    x0v = x0.reshape(-1)
    g = rounding.gamma(a_eq.shape[1] + 1)
    # Per-row bound on |exact equality row| at N z + x0: coefficients on |z| and a constant.
    off_z = np.abs(a_eq @ null_basis) + (eq_err[:, :-1] + g * np.abs(a_eq)) @ n_abs
    off_0 = (
        np.abs(a_eq @ x0v + eq[:, -1])
        + eq_err[:, :-1] @ np.abs(x0v)
        + eq_err[:, -1]
        + g * (np.abs(a_eq) @ np.abs(x0v) + np.abs(eq[:, -1]))
    )
    smin = float(np.linalg.svd(a_eq, compute_uv=False)[-1]) - float(np.linalg.norm(eq_err[:, :-1]))
    if smin <= 0.0:
        raise ValueError("equality rows are dependent within their error; _affine_null_basis should have raised")
    dist = np.concatenate((np.linalg.norm(off_z, axis=0), [np.linalg.norm(off_0)])) / smin
    row_norm = np.linalg.norm(ineq[:, :-1], axis=1) + np.linalg.norm(errors[ineq_mask][:, :-1], axis=1)
    return err + row_norm[:, None] * dist[None, :]


def _hull_distance_bound(halfspaces: np.ndarray, errors: np.ndarray, zero_indices: np.ndarray, x: np.ndarray) -> float:
    """Bound on the distance from ``x`` to the exact affine hull of the ``zero_indices`` rows."""
    from relucent._internal import rounding

    eq = halfspaces[zero_indices]
    a_eq = eq[:, :-1]
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    resid = np.abs(a_eq @ xv + eq[:, -1]) + rounding.row_errors(errors[zero_indices], xv)
    resid = resid + rounding.row_errors(rounding.exact_rows_error(eq), xv)
    smin = float(np.linalg.svd(a_eq, compute_uv=False)[-1]) - float(np.linalg.norm(errors[zero_indices, :-1]))
    if smin <= 0.0:
        raise ValueError("equality rows are dependent within their error; _affine_null_basis should have raised")
    return float(np.linalg.norm(resid)) / smin


def verify_interior_point(
    halfspaces: np.ndarray,
    errors: np.ndarray,
    x: np.ndarray,
    zero_indices: np.ndarray | None = None,
) -> bool:
    """True iff the exact point nearest ``x`` on the cell's affine hull is strictly inside every inequality row.

    Each row is judged against its own float64 error at ``x`` (:mod:`relucent._internal.rounding`),
    widened by the distance from ``x`` to the exact affine hull when ``zero_indices`` is given.
    """
    from relucent._internal import rounding

    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    x_err = 0.0
    rows = np.arange(halfspaces.shape[0])
    if zero_indices is not None and len(zero_indices) > 0:
        x_err = _hull_distance_bound(halfspaces, errors, np.asarray(zero_indices, dtype=np.intp), xv)
        rows = rows[~np.isin(rows, zero_indices)]
    if rows.size == 0:
        return True
    side = rounding.classify_rows(halfspaces, errors, xv, x_err, rows=rows)
    return bool(np.all(side == -1))


def _polish_chebyshev(
    inequalities: np.ndarray,
    norm_vector: np.ndarray,
    equalities: np.ndarray | None,
    cbasis: np.ndarray | None,
    ybasis: int | None,
    max_radius: float,
) -> tuple[np.ndarray, float] | None:
    """Re-solve the Chebyshev LP's tight system (from its optimal basis) directly in float64."""
    if cbasis is None or ybasis is None:
        return None
    dim = inequalities.shape[1] - 1
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for i in np.flatnonzero(cbasis != 0):  # nonbasic inequality: tight
        rows.append(np.concatenate((inequalities[i, :-1], norm_vector[i])))
        rhs.append(-float(inequalities[i, -1]))
    if equalities is not None:
        for e in equalities:
            rows.append(np.concatenate((e[:-1], [0.0])))
            rhs.append(-float(e[-1]))
    if ybasis == -2 and np.isfinite(max_radius):  # radius at its cap
        rows.append(np.concatenate((np.zeros(dim), [1.0])))
        rhs.append(float(max_radius))
    if len(rows) != dim + 1:
        return None
    try:
        sol = np.linalg.solve(np.vstack(rows), np.asarray(rhs))
    except np.linalg.LinAlgError:
        return None
    return sol[:-1], float(sol[-1])


def _certify_empty(
    halfspaces: np.ndarray,
    errors: np.ndarray,
    is_ineq: np.ndarray,
    multipliers: np.ndarray,
    x: np.ndarray,
) -> bool:
    """True when the LP's dual multipliers yield a verified proof that the rows have no common point.

    Rows are ``a_r . x + b_r <= 0`` where ``is_ineq`` and ``= 0`` elsewhere. A proof is a row ``t``
    (an inequality) and rows ``R`` with ``a_t + sum_R mu_r a_r = 0`` exactly, ``mu_r >= 0`` on the
    inequalities, and ``b_t + sum_R mu_r b_r > 0``: the combination is then a positive constant,
    yet at most 0 on any point of the cell. The multipliers only choose ``t`` and ``R`` (the rows
    they weight most); ``mu`` is re-solved and verified here, because a float combination never
    cancels the normals exactly. The system must be square (``R`` as large as the dimension), and
    its exact solution is bounded by perturbation; parallel rows cannot be certified this way.
    """
    from relucent._internal import rounding

    lam = np.asarray(multipliers, dtype=np.float64).reshape(-1)
    d = halfspaces.shape[1] - 1
    scale = float(np.max(np.abs(lam), initial=0.0))
    if scale == 0.0 or not np.all(np.isfinite(lam)):
        return False
    support = np.flatnonzero((np.abs(lam) > 1e-9 * scale) & (~is_ineq | (lam > 0.0)))
    ineq_support = support[is_ineq[support]]
    if ineq_support.size == 0:
        return False
    t = int(ineq_support[np.argmax(lam[ineq_support])])
    rest = support[support != t]
    rest = rest[np.argsort(-np.abs(lam[rest]), kind="stable")][:d]
    if rest.size != d:
        return False
    m_mat = halfspaces[rest, :-1]
    dm = float(np.linalg.norm(errors[rest, :-1]))
    smin = float(np.linalg.svd(m_mat, compute_uv=False)[-1])
    if smin <= 2.0 * dm:
        return False
    a_t = halfspaces[t, :-1]
    try:
        mu = np.linalg.solve(m_mat.T, -a_t)
    except np.linalg.LinAlgError:
        return False
    resid = np.abs(m_mat.T @ mu + a_t) + rounding.gamma(d + 1) * (np.abs(m_mat.T) @ np.abs(mu) + np.abs(a_t))
    delta = (float(np.linalg.norm(resid)) + float(np.linalg.norm(errors[t, :-1])) + dm * float(np.linalg.norm(mu))) / (
        smin - dm
    )
    delta *= 1.0 + 8.0 * rounding.EPS
    if np.any(mu[is_ineq[rest]] < delta):
        return False
    # The exact combination is one constant everywhere; bound it from below at the LP point.
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    s = halfspaces[:, :-1] @ xv + halfspaces[:, -1]
    e = rounding.row_errors(errors, xv)
    lower = s[t] + float(mu @ s[rest])
    slack = e[t] + float(np.sum((np.abs(mu) + delta) * e[rest])) + delta * float(np.sum(np.abs(s[rest])))
    return bool(lower - slack * (1.0 + 8.0 * rounding.EPS) > 0.0)


def solve_radius(
    env: Env,
    halfspaces: np.ndarray | torch.Tensor,
    max_radius: float = GRB.INFINITY,
    zero_indices: np.ndarray | None = None,
    sense: int = GRB.MAXIMIZE,
    errors: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float | None]:
    """Solve for the Chebyshev center or interior point of a polyhedron.

    Only works if all polyhedron vertices are within 2*max_radius of each other.

    Every nonempty answer is checked in float64: the returned center must lie strictly inside
    every inequality row beyond that row's own error (``errors``, the rows' error scale from
    :mod:`relucent._internal.rounding`; exact data when omitted). A cell the LP calls nonempty
    but whose center fails that check, or whose largest inscribed ball has radius 0 (a closed
    cell with empty interior, which an activation region cannot be), raises
    :class:`~relucent.core.errors.AmbiguousGeometryError` instead of being called empty or
    nonempty on a guess.

    Args:
        env: Gurobi environment for optimization.
        halfspaces: Halfspace representation of the polyhedron as an array with
            shape (n_constraints, n_dim+1), where the last column contains bias terms.
        max_radius: Maximum radius constraint for the polyhedron. Defaults to infinity.
        zero_indices: Indices of sign sequence elements that are zero (for
            lower-dimensional polyhedra). Defaults to None.
        sense: Optimization sense, should typically be GRB.MAXIMIZE. Defaults to GRB.MAXIMIZE.
        errors: Float64 error scale of ``halfspaces``, row for row.

    Returns:
        tuple: ``(center_point, radius)``. Radius is the largest ball in the **affine
        hull** of the feasible region (relative Chebyshev inradius), except that
        ``max_radius`` caps the LP when finite. For ``max_radius`` infinite:
        ``(None, None)`` if infeasible; ``(None, inf)`` if the region is all of
        :math:`\\mathbb{R}^d` with no constraints; otherwise ``(x, r)`` with
        ``r = inf`` when the hull has positive dimension and no inequalities cut it,
        ``r = 0`` when the hull is a **single point**, and ``r > 0`` when bounded
        with nonempty relative interior in the usual LP.

    Raises:
        ValueError: If the optimization fails.
        AmbiguousGeometryError: If emptiness cannot be decided in float64.
    """
    from relucent._internal import rounding
    from relucent.core.errors import AmbiguousGeometryError

    if isinstance(halfspaces, torch.Tensor):
        halfspaces = halfspaces.detach().cpu().numpy()

    if not np.isfinite(halfspaces).all():
        raise ValueError("Halfspaces contain NaN or Inf coefficients")
    if errors is None:
        errors = rounding.exact_rows_error(halfspaces)

    # Remove constant rows before building the model (see _degenerate_rows).
    # This prevents pathologies like 0*x + 0*y <= -b and makes results more stable across platforms.
    try:
        kept_hs, old_to_new = _drop_degenerate_halfspaces_tracked(halfspaces, errors=errors)
    except DegenerateHalfspaceInfeasibility:
        # Same conclusion as an infeasible Chebyshev LP: empty intersection.
        return None, None
    errors = errors[old_to_new >= 0] if kept_hs.shape[0] != halfspaces.shape[0] else errors
    halfspaces = kept_hs
    zero_indices_eff = _remap_zero_indices(zero_indices, old_to_new)
    ineq_mask = np.ones(halfspaces.shape[0], dtype=bool)

    if zero_indices_eff is not None and len(zero_indices_eff) > 0:
        # warnings.warn("Working with k<d polyhedron.", stacklevel=2)
        _x0, null_basis, ineq_mask = _affine_null_basis(halfspaces, zero_indices_eff, errors=errors)
        if null_basis.shape[1] == 0 and np.any(ineq_mask):
            # The equalities pin a single point: it is the cell iff it satisfies every other row.
            x_err = _hull_distance_bound(halfspaces, errors, zero_indices_eff, _x0)
            side = rounding.classify_rows(halfspaces, errors, _x0, x_err, rows=np.flatnonzero(ineq_mask))
            if np.any(side == 1):
                return None, None
            if np.all(side == -1):
                return _x0, 0.0
            raise AmbiguousGeometryError(
                "a vertex lies on another hyperplane to within float64 error: cannot decide whether it exists"
            )
        equalities = halfspaces[~ineq_mask]
        inequalities = halfspaces[ineq_mask]
        # Effective norm of each inequality normal *within* the affine subspace.
        # Equivalent to ||a_i P|| where P = N N^T is the orthogonal projector onto
        # the null space of eq_A, but avoids squaring the condition number via A A^T.
        norm_vector = np.linalg.norm(inequalities[:, :-1] @ null_basis, axis=1, keepdims=True)
    else:
        inequalities = halfspaces
        equalities = None
        norm_vector = np.linalg.norm(inequalities[:, :-1], axis=1, keepdims=True)

    if not np.isfinite(norm_vector).all():
        raise ValueError("Norm vector contains NaN or Inf coefficients")

    # No inequality rows: the Chebyshev LP has no constraints linking x and y.
    # Building ``norm_vector * y`` with shape (0, 1) hits a gurobipy matrix-API
    # AssertionError on some platforms (e.g. certain Python/gurobipy combos).
    if inequalities.shape[0] == 0:
        dim = halfspaces.shape[1] - 1
        if equalities is not None and equalities.shape[0] > 0:
            A_eq = equalities[:, :-1]
            b_vec = -equalities[:, -1:].ravel()
            x_feas, *_ = np.linalg.lstsq(A_eq, b_vec, rcond=None)
            x_feas = np.asarray(x_feas, dtype=np.float64).reshape(dim, 1)
            # _affine_null_basis checked the equality rows are independent beyond their
            # error, so the system is consistent and x_feas lies in it. No strict
            # inequalities: the feasible set is the affine subspace {x : A_eq x = b}, and
            # its relative inradius is infinite unless it's 0-dimensional.
            rnk = int(np.linalg.matrix_rank(A_eq))
            aff_dim = dim - rnk
            if aff_dim <= 0:
                # Unique feasible point (affine hull is a single point).
                return x_feas, 0.0
            if max_radius == GRB.INFINITY:
                return x_feas, float("inf")
            return x_feas, float(max_radius)
        if max_radius == GRB.INFINITY:
            return None, float("inf")
        return np.zeros((dim, 1), dtype=np.float64), float(max_radius)

    model = Model("Interior Point", env)
    x = model.addMVar((halfspaces.shape[1] - 1, 1), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    y = model.addMVar((1,), ub=max_radius, vtype=GRB.CONTINUOUS, name="y")
    eq_constr: Any = None
    try:
        ineq_constr = model.addConstr(inequalities[:, :-1] @ x + norm_vector * y <= -inequalities[:, -1:])
        if equalities is not None:
            eq_constr = model.addConstr(equalities[:, :-1] @ x == -equalities[:, -1:])
        model.setObjective(y, sense)
    except Exception as e:
        raise ValueError(f"Failed to build Gurobi model: {e}") from e
    model.optimize()
    status = model.status

    if status == GRB.INF_OR_UNBD:
        model.setParam(GRB.Param.DualReductions, 0)
        model.reset()
        model.optimize()
        status = model.status

    # Rarely, the Chebyshev LP can end with NUMERIC due to ill-conditioning.
    # A second attempt with NumericFocus can recover; avoid doing this for INFEASIBLE
    # since infeasibility is common during local search and the retry is expensive.
    if status == GRB.NUMERIC:
        model.setParam("NumericFocus", 1)
        model.optimize()
        status = model.status

    if status == GRB.OPTIMAL:
        objVal = model.objVal
        try:
            cbasis = np.asarray(ineq_constr.CBasis).reshape(-1)
            ybasis = int(np.asarray(y.VBasis).reshape(-1)[0])
        except Exception:  # noqa: BLE001 - no basis: no polish
            cbasis, ybasis = None, None
        duals: np.ndarray | None = None
        if objVal < 0:
            try:
                parts = [np.asarray(ineq_constr.Pi, dtype=np.float64).reshape(-1)]
                if eq_constr is not None:
                    parts.append(np.asarray(eq_constr.Pi, dtype=np.float64).reshape(-1))
                duals = np.concatenate(parts)
            except Exception:  # noqa: BLE001 - no duals: no emptiness proof
                duals = None
        x, y = x.X, float(np.squeeze(y.X))
        model.close()
        assert isinstance(x, np.ndarray)
        if objVal < 0:
            # Gurobi lets y dip below its bound 0 by its feasibility tolerance: the cell is empty
            # to Gurobi's accuracy. Its duals are a Farkas combination proving that; verify it.
            if duals is not None:
                if equalities is None:
                    rows, row_err = inequalities, errors
                    is_ineq = np.ones(rows.shape[0], dtype=bool)
                else:
                    rows = np.vstack((inequalities, equalities))
                    row_err = np.vstack((errors[ineq_mask], errors[~ineq_mask]))
                    is_ineq = np.arange(rows.shape[0]) < inequalities.shape[0]
                if _certify_empty(rows, row_err, is_ineq, duals, x):
                    return None, None
            raise AmbiguousGeometryError(
                f"Chebyshev LP radius {objVal:.4e} < 0 (empty to Gurobi's tolerance), but emptiness "
                + "could not be verified against the rows' float64 error"
            )
        # A closed cell with an empty interior (radius 0) is the closure of an empty activation
        # region, or of one too thin to resolve; a positive radius must come with a center that
        # verifies strictly inside every row. Anything else cannot be decided in float64.
        if objVal > 0 and verify_interior_point(halfspaces, errors, x, zero_indices_eff):
            return x, y
        # Gurobi's answer carries its absolute 1e-6 feasibility error, which swamps cells thinner
        # than that. The optimal basis names the tight rows; solving them directly gives the center
        # to float64 accuracy, which is then verified like any other.
        polished = _polish_chebyshev(inequalities, norm_vector, equalities, cbasis, ybasis, max_radius)
        if polished is not None:
            xp, yp = polished
            if yp > 0 and verify_interior_point(halfspaces, errors, xp, zero_indices_eff):
                return xp.reshape(-1, 1), yp
        raise AmbiguousGeometryError(
            f"Chebyshev LP radius {objVal:.4e} but its center is not strictly inside every row beyond "
            + "the rows' float64 error: cannot decide whether the cell is empty"
        )
    elif status == GRB.INTERRUPTED:
        model.close()
        raise KeyboardInterrupt
    else:
        model.close()
        if max_radius == GRB.INFINITY:
            # The Chebyshev LP can be INFEASIBLE even when the system is feasible
            # (e.g. intrinsic inradius 0); treat both as "no interior point found".
            if status == GRB.INFEASIBLE:
                return None, None
            if status == GRB.INF_OR_UNBD:
                raise ValueError("Unable to disambiguate INF_OR_UNBD status.")
            if status == GRB.UNBOUNDED:
                return None, float("inf")
            raise ValueError(f"Chebyshev LP ended with unexpected status: {status}")
        else:
            # Finite max_radius: same Chebyshev outcomes as above (intrinsic inradius 0, etc.).
            if status == GRB.INFEASIBLE:
                return None, None
            if status == GRB.NUMERIC:
                return None, None
            if status == GRB.INF_OR_UNBD:
                raise ValueError("Unable to disambiguate INF_OR_UNBD status.")
            if status == GRB.UNBOUNDED:
                return None, None
            raise ValueError(f"Chebyshev LP ended with unexpected status: {status}")


@torch.no_grad()
def adjacent_polyhedra(
    poly: "Polyhedron",
    ss2poly: Callable[..., "Polyhedron"],
) -> "set[Polyhedron]":
    """Polyhedra adjacent to ``poly`` across one bounding hyperplane (one SHI flip).

    Also works on lower-dimensional polyhedra. ``ss2poly`` maps a sign sequence
    array to the corresponding :class:`Polyhedron` (e.g. ``Complex.ss2poly``).
    """
    ps: set[Polyhedron] = set()
    for shi in poly.shis:
        if poly.ss_np[shi] == 0:
            continue
        ps.add(ss2poly(flip_ss_at_shi(poly.ss_np, shi)))
    return ps


@overload
def get_hs(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: Literal[False] = False,
    force_numpy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def get_hs(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: Literal[True],
    force_numpy: bool = False,
) -> list[dict[str, object]]: ...


def get_hs(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: bool = False,
    force_numpy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray, np.ndarray] | list[dict[str, object]]:
    """Halfspace representation of ``poly`` from all neurons in the network.

    Includes constraints from every neuron, not only supporting hyperplanes.

    Args:
        poly: "Polyhedron" whose sign sequence defines the region.
        data: Optional network input for verifying intermediate affine maps.
        get_all_Ab: If True, return per-layer ``A``, ``b`` instead of final halfspaces.
        force_numpy: If True, use the NumPy path even when ``ss`` is a tensor.

    Returns:
        If ``get_all_Ab`` is False: ``(halfspaces, W, b)``.
        If True: list of dicts with ``A``, ``b``, and ``layer`` keys.
    """
    if TORCH_AVAILABLE and isinstance(poly._ss, torch.Tensor) and not force_numpy:
        return _get_hs_torch(poly, data, get_all_Ab=get_all_Ab)
    return _get_hs_numpy(poly, data, get_all_Ab=get_all_Ab)


@overload
def _get_hs_torch(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: Literal[False] = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def _get_hs_torch(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: Literal[True],
) -> list[dict[str, object]]: ...


@torch.no_grad()
def _get_hs_torch(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[dict[str, object]]:
    assert isinstance(poly._ss, torch.Tensor)
    constr_A, constr_b = None, None
    current_A, current_b = None, None
    layer_W, layer_b = None, None
    abs_A: Any = None
    abs_b: Any = None
    if data is not None:
        assert poly._net is not None
        outs: Mapping[str, object] | None = poly._net.get_all_layer_outputs(data)
    else:
        outs = None
    all_Ab = []
    current_mask_index = 0
    assert poly._net is not None
    for name, layer in poly._net.layers.items():
        if isinstance(layer, LinearLayer):
            layer_W = torch.as_tensor(layer.weight, device=poly._ss.device, dtype=torch.float64)
            layer_b = torch.as_tensor(layer.bias, device=poly._ss.device, dtype=torch.float64)
            if current_A is None or current_b is None:
                constr_A = torch.empty((layer_W.shape[1], 0), device=poly._ss.device, dtype=torch.float64)
                constr_b = torch.empty((1, 0), device=poly._ss.device, dtype=torch.float64)
                current_A = torch.eye(layer_W.shape[1], device=poly._ss.device, dtype=torch.float64)
                current_b = torch.zeros((1, layer_W.shape[1]), device=poly._ss.device, dtype=torch.float64)

            current_A = current_A @ layer_W.T
            current_b = current_b @ layer_W.T + layer_b
            if data is not None and cfg.CAREFUL_MODE:
                if abs_A is None or abs_b is None:
                    abs_A = torch.eye(layer_W.shape[1], device=poly._ss.device, dtype=torch.float64)
                    abs_b = torch.zeros((1, layer_W.shape[1]), device=poly._ss.device, dtype=torch.float64)
                abs_A = abs_A @ layer_W.abs().T
                abs_b = abs_b @ layer_W.abs().T + layer_b.abs()
        elif isinstance(layer, ReLULayer):
            assert current_A is not None
            assert current_b is not None

            mask = poly._ss[current_mask_index : current_mask_index + current_A.shape[1]]

            # Treat mask 0s as 1 so their halfspaces are included as equalities.
            nonzero_mask = torch.where(mask == 0, torch.ones_like(mask), mask)

            new_constr_A = current_A * nonzero_mask
            new_constr_b = current_b * nonzero_mask

            assert isinstance(constr_A, torch.Tensor)
            assert isinstance(constr_b, torch.Tensor)
            assert isinstance(current_A, torch.Tensor)
            assert isinstance(current_b, torch.Tensor)
            assert isinstance(mask, torch.Tensor)

            constr_A = torch.cat((constr_A, new_constr_A), dim=1)
            constr_b = torch.cat((constr_b, new_constr_b), dim=1)

            current_A = current_A * (mask == 1)
            current_b = current_b * (mask == 1)
            current_mask_index += current_A.shape[1]
        elif isinstance(layer, FlattenLayer):
            if current_A is None:
                pass
            else:
                raise NotImplementedError("Intermediate flatten layer not supported")
        else:
            raise ValueError(f"Error while processing layer {name} - Unsupported layer type: {type(layer)} ({layer})")
        if data is not None:
            assert isinstance(current_A, torch.Tensor)
            assert isinstance(current_b, torch.Tensor)
            assert outs is not None
            if cfg.CAREFUL_MODE and abs_A is not None and abs_b is not None:
                # Both sides round; |W| composed with every unit on bounds either one's error.
                bound = 4.0 * _rounding().gamma(_rounding().composition_terms(poly._net)) * (data.abs() @ abs_A + abs_b)
                assert bool((torch.as_tensor(outs[name]) - ((data @ current_A) + current_b)).abs().le(bound).all())
        if get_all_Ab:
            assert current_A is not None
            assert current_b is not None

            all_Ab.append({"A": current_A.clone(), "b": current_b.clone(), "layer": layer})

    assert constr_A is not None
    assert constr_b is not None

    halfspaces = torch.hstack((-constr_A.T, -constr_b.reshape(-1, 1)))

    if get_all_Ab:
        return all_Ab

    assert isinstance(halfspaces, torch.Tensor)
    assert isinstance(current_A, torch.Tensor)
    assert isinstance(current_b, torch.Tensor)

    assert halfspaces.shape[0] == poly._ss.shape[0]
    return halfspaces, current_A, current_b


@overload
def _get_hs_numpy(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def _get_hs_numpy(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: Literal[True],
) -> list[dict[str, object]]: ...


@torch.no_grad()
def _get_hs_numpy(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | list[dict[str, object]]:
    constr_A, constr_b = None, None
    current_A, current_b = None, None
    layer_W, layer_b = None, None
    abs_A: Any = None
    abs_b: Any = None
    if data is not None:
        assert poly._net is not None
        outs: Mapping[str, object] | None = poly._net.get_all_layer_outputs(data)
    else:
        outs = None
    all_Ab = []
    current_mask_index = 0
    assert poly._net is not None
    for name, layer in poly._net.layers.items():
        if isinstance(layer, LinearLayer):
            layer_W = layer.weight
            layer_b = layer.bias

            if current_A is None or current_b is None:
                constr_A = np.empty((layer_W.shape[1], 0))
                constr_b = np.empty((1, 0))
                current_A = np.eye(layer_W.shape[1])
                current_b = np.zeros((1, layer_W.shape[1]))

            current_A = current_A @ layer_W.T
            current_b = current_b @ layer_W.T + layer_b
            if data is not None and cfg.CAREFUL_MODE:
                if abs_A is None or abs_b is None:
                    abs_A = np.eye(layer_W.shape[1])
                    abs_b = np.zeros((1, layer_W.shape[1]))
                abs_A = abs_A @ np.abs(layer_W).T
                abs_b = abs_b @ np.abs(layer_W).T + np.abs(layer_b)
        elif isinstance(layer, ReLULayer):
            if current_A is None:
                raise ValueError("ReLU layer must follow a linear layer")
            mask = poly.ss_np[current_mask_index : current_mask_index + current_A.shape[1]]

            nonzero_mask = np.where(mask == 0, 1, mask)

            new_constr_A = current_A * nonzero_mask
            new_constr_b = current_b * nonzero_mask

            assert constr_A is not None
            assert constr_b is not None
            assert current_A is not None
            assert current_b is not None

            constr_A = np.concatenate((constr_A, new_constr_A), axis=1)
            constr_b = np.concatenate((constr_b, new_constr_b), axis=1)

            current_A = current_A * (mask == 1)
            current_b = current_b * (mask == 1)
            current_mask_index += current_A.shape[1]
        elif isinstance(layer, FlattenLayer):
            if current_A is None:
                pass
            else:
                raise NotImplementedError("Intermediate flatten layer not supported")
        else:
            raise ValueError(f"Error while processing layer {name} - Unsupported layer type: {type(layer)} ({layer})")
        if data is not None:
            assert isinstance(current_A, np.ndarray)
            assert isinstance(current_b, np.ndarray)
            assert outs is not None
            expected = outs[name]
            if isinstance(expected, torch.Tensor):
                expected = expected.detach().cpu().numpy()
            expected = np.asarray(expected)
            if cfg.CAREFUL_MODE and abs_A is not None and abs_b is not None:
                # Both sides round; |W| composed with every unit on bounds either one's error.
                data_np = np.asarray(data)
                bound = 4.0 * _rounding().gamma(_rounding().composition_terms(poly._net)) * (np.abs(data_np) @ abs_A + abs_b)
                assert bool(np.all(np.abs(expected - ((data_np @ current_A) + current_b)) <= bound))
        if get_all_Ab:
            assert current_A is not None
            assert current_b is not None

            all_Ab.append({"A": current_A.copy(), "b": current_b.copy(), "layer": layer})

    assert constr_A is not None
    assert constr_b is not None

    halfspaces = np.hstack((-constr_A.T, -constr_b.reshape(-1, 1)))
    if get_all_Ab:
        return all_Ab
    assert isinstance(halfspaces, np.ndarray)
    assert isinstance(current_A, np.ndarray)
    assert isinstance(current_b, np.ndarray)

    assert halfspaces.shape[0] == poly.ss_np.shape[0]
    return halfspaces, current_A, current_b


def shis_are_certified(shis_kwargs: Mapping[str, Any]) -> bool:
    """Whether ``get_shis(poly, **shis_kwargs)`` returns ``poly``'s complete, certified facet list.

    Every facet decision is certified, so the answer is the cell's facet list (and recomputing it
    is redundant) unless ``subset`` restricts the candidates or ``escalate_bound=False`` makes
    the box a constraint. Recorded as ``poly._shis_strict``.
    """
    return shis_kwargs.get("subset") is None and bool(shis_kwargs.get("escalate_bound", True))


@overload
def get_shis(
    poly: "Polyhedron",
    collect_info: Literal[False] = False,
    bound: float = GRB.INFINITY,
    subset: Iterable[int] | None = None,
    env: Env | None = None,
    shi_pbar: bool = False,
    push_size: float = 1.0,
    escalate_bound: bool = True,
) -> list[int]: ...


@overload
def get_shis(
    poly: "Polyhedron",
    collect_info: Literal[True] | Literal["All"],
    bound: float = GRB.INFINITY,
    subset: Iterable[int] | None = None,
    env: Env | None = None,
    shi_pbar: bool = False,
    push_size: float = 1.0,
    escalate_bound: bool = True,
) -> tuple[list[int], list[dict[str, object]]]: ...


def get_shis(
    poly: "Polyhedron",
    collect_info: bool | str = False,
    bound: float = GRB.INFINITY,
    subset: Iterable[int] | None = None,
    env: Env | None = None,
    shi_pbar: bool = False,
    push_size: float = 1.0,
    escalate_bound: bool = True,
) -> list[int] | tuple[list[int], list[dict[str, object]]]:
    """Supporting halfspace indices (SHIs) for ``poly``.

    Indices of non-redundant halfspaces on the boundary (neurons whose BHs are
    actually faces of the polyhedron).

    Rows with an exactly zero normal are dropped (same as :func:`solve_radius`) before
    building the LP; returned SHI indices refer to the original ``halfspaces_np`` rows.

    Candidate row ``i`` is relaxed by ``push_size`` and maximised. The LP only proposes; each
    answer is certified in float64 against the rows' own error (``poly.halfspaces_err_np``,
    :mod:`relucent._internal.rounding`) by :func:`_certify_facet`: a facet needs a witness point
    where row ``i`` is strictly positive and every other row strictly negative beyond their errors,
    and a non-facet needs row ``i`` to be an exact nonnegative combination of the LP's tight rows
    with a negative constant, verified by bounding the exact multipliers (so it holds on the whole
    cell, bounded or not). Undecided answers are decided in exact arithmetic when the rows can be
    rebuilt exactly; otherwise :class:`~relucent.core.errors.AmbiguousGeometryError` is raised.

    Args:
        poly: "Polyhedron" to analyze.
        collect_info: If true, also return debug info; ``"All"`` adds more detail.
        bound: Hypercube bound for the Gurobi variable box.
        subset: Halfspace indices to consider; default is all.
        env: Gurobi environment; default uses :func:`~relucent._internal.gurobi.get_env`.
        shi_pbar: Show a progress bar.
        push_size: RHS relaxation size when testing a candidate SHI.
        escalate_bound: If False, use only the requested ``bound`` (no automatic box-radius
            escalation when clipping is suspected).

    Returns:
        List of SHI indices, or ``(shis, info)`` if ``collect_info`` is truthy.

    Raises:
        ValueError: If the initial Gurobi solve fails.
        :class:`~relucent.core.errors.AmbiguousGeometryError`: If a facet decision cannot be certified.
    """

    hs_np = poly.halfspaces_np
    err_np = poly.halfspaces_err_np
    n_orig = int(hs_np.shape[0])
    A_orig = hs_np[:, :-1]
    b_orig = hs_np[:, -1:]
    amb_d = int(A_orig.shape[1])
    env = env or get_env()

    try:
        hs_work, old_to_new = _drop_degenerate_halfspaces_tracked(hs_np, errors=err_np)
    except DegenerateHalfspaceInfeasibility:
        if collect_info:
            return [], []
        return []
    err_work = err_np[old_to_new >= 0] if hs_work.shape[0] != n_orig else err_np

    n_work = int(hs_work.shape[0])
    if n_work == 0:
        if collect_info:
            return [], []
        return []
    work_to_orig = np.full(n_work, -1, dtype=np.intp)
    for old_i in range(n_orig):
        ni = int(old_to_new[old_i])
        if ni >= 0:
            work_to_orig[ni] = old_i

    A_work = hs_work[:, :-1]
    b_work = hs_work[:, -1:]
    zero_eff = _remap_zero_indices(poly.zero_indices, old_to_new)

    # Work in intrinsic coordinates z with x = N z + x0 so every active constraint is a
    # genuine inequality in z-space (no separate Gurobi ``EQUAL`` senses for zero-sign
    # rows).  Maximal cells: N = I, x0 = 0.
    if zero_eff is not None and zero_eff.size > 0:
        x0, null_basis, ineq_mask = _affine_null_basis(hs_work, zero_eff, errors=err_work)
        k = int(null_basis.shape[1])
        if k == 0:
            if collect_info:
                return [], []
            return []
        ineq_rows = np.flatnonzero(ineq_mask)
        ineq_half = hs_work[ineq_mask]
        a_red = ineq_half[:, :-1] @ null_basis
        b_red = ineq_half[:, :-1] @ x0 + ineq_half[:, -1:]
        err_red = _reduced_rows_error(hs_work, err_work, zero_eff, ineq_mask, x0, null_basis)
        row_lp = {int(r): j for j, r in enumerate(ineq_rows.tolist())}
    else:
        x0 = np.zeros((amb_d, 1), dtype=np.float64)
        null_basis = np.eye(amb_d, dtype=np.float64)
        ineq_rows = np.arange(n_work, dtype=np.intp)
        a_red = A_work
        b_red = b_work
        err_red = err_work
        row_lp = {int(r): r for r in range(n_work)}
        k = amb_d
    h_red = np.hstack((a_red, b_red))
    # A verified interior point of the cell (face coordinates), computed on first need.
    interior: list[np.ndarray] = []

    def interior_point() -> np.ndarray:
        if not interior:
            from relucent.core.errors import AmbiguousGeometryError

            zc = None
            if poly._center is not None:
                # The cell's own center, already verified: reuse it when it verifies in z too.
                z_try = null_basis.T @ (np.asarray(poly._center, dtype=np.float64).reshape(-1, 1) - x0)
                if verify_interior_point(h_red, err_red, z_try):
                    zc = z_try
            if zc is None:
                zc, _ = solve_radius(env, h_red, max_radius=float(cfg.MAX_RADIUS), errors=err_red)
            if zc is None:
                raise AmbiguousGeometryError(f"cell {poly!r} is LP-feasible but has no interior point")
            interior.append(np.asarray(zc, dtype=np.float64).reshape(-1))
        return interior[0]

    def decide_without_lp(i: int, j: int) -> bool | None:
        """Row i's facet verdict in exact arithmetic alone, for when every LP re-solve failed.

        None when it cannot decide: no exact rows (too large a network), a fixed box (the exact
        question is over the unbounded cell), or a near-duplicate partner, whose coincidence
        check needs the LP.
        """
        if not escalate_bound or j in dup_partners:
            return None
        rows = poly._exact_rows()
        if rows is None:
            return None
        from relucent._internal import exact

        start = null_basis @ interior_point().reshape(-1, 1) + x0
        return exact.exact_facet_by_simplex(rows, i, eq_orig_idx, start)

    def record_without_lp(i: int, j: int, verdict: bool) -> None:
        """Record an exact verdict for row i (LP row j) and restore its RHS; the LP has no solution."""
        if verdict and zero_units.size:
            _raise_if_coincident_across(poly, i, zero_units, eq_orig_idx)
        if verdict:
            shis.append(i)
        if poly_info is not None:
            poly_info.append({"Status": "exact (LP failed)", "Verdict": verdict})
        lp_constrs[j].RHS = -b_red[j, 0]

    candidate_subset = subset or set(range(n_orig)) - set(poly.zero_indices)
    candidate_subset = set(candidate_subset)

    pbar = tqdm(total=len(candidate_subset), desc="Calculating SHIs", leave=False, delay=3, disable=not shi_pbar)
    shis: list[int] = []
    poly_info: list[dict[str, object]] | None = [] if collect_info else None
    bounds_to_try = _shi_variable_bounds_to_try(bound)
    if not escalate_bound:
        bounds_to_try = bounds_to_try[:1]
    last_status: int | None = None
    model: Model | None = None
    # Coincident hyperplanes make the arrangement non-simple, which relucent doesn't
    # handle (neighbors across such a facet differ in more than one sign). Two identical
    # rows each look redundant, so their facet would silently vanish, and a unit that is
    # identically zero can coincide with a facet on its other side. Both are caught below
    # and raise NonGenericArrangementError; rows coincident within float64 error count.
    dup_partners: dict[int, list[int]] = {}
    for r, c in _near_duplicate_rows(h_red, err_red):
        dup_partners.setdefault(r, []).append(c)
        dup_partners.setdefault(c, []).append(r)
    zero_units = np.flatnonzero(np.all(hs_np == 0.0, axis=1))
    eq_orig_idx = [] if zero_eff is None else [int(work_to_orig[int(w)]) for w in zero_eff]
    # Row i's objective in z: maximise obj_all[i] @ z + const_all[i] (row i at x = N z + x0).
    # Set per row as attributes: building an objective expression costs ~20x more.
    obj_all = A_orig @ null_basis
    const_all = (A_orig @ x0 + b_orig).ravel()

    for bound_index, attempt_bound in enumerate(bounds_to_try):
        if model is not None:
            model.close()
        model = Model("SHIS", env)
        model.params.OptimalityTol = cfg.advanced.GUROBI_SHI_OPTIMALITY_TOL
        model.params.ScaleFlag = cfg.advanced.GUROBI_SHI_SCALE_FLAG
        model.params.BestBdStop = GRB.INFINITY
        model.ModelSense = GRB.MAXIMIZE  # objective is zero until the first row sets it
        z = model.addMVar((k, 1), lb=-attempt_bound, ub=attempt_bound, vtype=GRB.CONTINUOUS, name="z")
        constrs = model.addConstr(a_red @ z <= -b_red, name="hyperplanes")
        model.optimize()
        lp_constrs = model.getConstrs()  # constraint handles in LP-row order: cheap RHS updates
        if model.status != GRB.OPTIMAL and bound_index + 1 == len(bounds_to_try):
            # The last box: infeasibility here is final, so rule out a solver failure first.
            _cold_retry(model)
        last_status = model.status
        if model.status != GRB.OPTIMAL:
            continue

        shis = []
        if collect_info:
            poly_info = []
        subset = set(candidate_subset)
        box_clipping_suspected = False
        while subset:
            i = subset.pop()
            if i >= poly.ss_np.shape[0] or poly.ss_np[i] == 0:
                continue
            wi = int(old_to_new[i])
            if wi < 0:
                continue
            if (A_orig[i] == 0).all():
                continue
            j = row_lp.get(wi)
            if j is None:
                continue
            pbar.set_postfix_str(f"#shis: {len(shis)}")

            # Relax halfspace i (row j in the LP over z); i is the original halfspace index.
            lp_constrs[j].RHS = -b_red[j, 0] + push_size

            z.Obj = obj_all[i].reshape(k, 1)
            model.ObjCon = float(const_all[i])
            # Stopping once the objective is well above zero only saves time: the point is
            # certified below, and an uncertified stop is re-solved to optimality.
            model.params.BestObjStop = 0.5 * push_size
            model.optimize()

            if model.status == GRB.INTERRUPTED:
                model.close()
                raise KeyboardInterrupt
            if model.status not in (GRB.OPTIMAL, GRB.USER_OBJ_LIMIT):
                # This LP relaxes one row of a cell just solved as feasible, and its objective is
                # capped by that row's relaxed bound, so anything but optimal is a solver failure.
                exact_verdict = _recover_relaxed_shi_lp(model, poly, i, partial(decide_without_lp, int(i), j))
                if exact_verdict is not None:
                    record_without_lp(int(i), j, exact_verdict)
                    continue
            if model.status == GRB.OPTIMAL or model.status == GRB.USER_OBJ_LIMIT:
                zv, cbasis = _read_lp_solution(z, constrs)
                verdict = _certify_facet_from_model(model.status, zv, cbasis, j, h_red, err_red, interior_point)
                if verdict is None and model.status == GRB.USER_OBJ_LIMIT:
                    model.params.BestObjStop = GRB.INFINITY
                    model.optimize()
                    if model.status != GRB.OPTIMAL:
                        exact_verdict = _recover_relaxed_shi_lp(model, poly, i, partial(decide_without_lp, int(i), j))
                        if exact_verdict is not None:
                            record_without_lp(int(i), j, exact_verdict)
                            continue
                    zv, cbasis = _read_lp_solution(z, constrs)
                    verdict = _certify_facet_from_model(model.status, zv, cbasis, j, h_red, err_red, interior_point)
                if verdict is None and escalate_bound:
                    # Float64 can't decide (usually a redundant row parallel to a facet, with
                    # multiplier exactly zero, or hyperplanes concurrent at the LP vertex).
                    # Decide exactly. The box is only a numerical aid, so the exact question is
                    # over the unbounded cell.
                    exact_rows = poly._exact_rows()
                    if exact_rows is not None:
                        from relucent._internal import exact

                        if model.status == GRB.OPTIMAL and cbasis is not None:
                            # First the LP's own dual, exactly: one small rational solve on its
                            # tight rows. It settles the usual case, a row whose normal is exactly
                            # a nonnegative combination of tight rows with a zero multiplier.
                            tight_orig = [int(work_to_orig[int(ineq_rows[r])]) for r in np.flatnonzero(cbasis != 0) if r != j]
                            bound_c = exact.exact_dual_bound(exact_rows, tight_orig, eq_orig_idx, int(i))
                            if bound_c is not None and bound_c < 0:
                                verdict = False  # row i is below a negative constant on the cell
                        if verdict is None:
                            # Otherwise the facet question itself, from the cell's verified interior point.
                            start = null_basis @ interior_point().reshape(-1, 1) + x0
                            verdict = exact.exact_facet_by_simplex(exact_rows, int(i), eq_orig_idx, start)
                if verdict is False and j in dup_partners:
                    _raise_if_coincident_facet(
                        poly,
                        model,
                        z,
                        lp_constrs,
                        j,
                        int(i),
                        dup_partners[j],
                        h_red,
                        err_red,
                        b_red,
                        push_size,
                        ineq_rows,
                        work_to_orig,
                        interior_point,
                    )
                elif verdict is True and zero_units.size:
                    _raise_if_coincident_across(poly, int(i), zero_units, eq_orig_idx)
                x_proof = null_basis @ zv.reshape(-1, 1) + x0
                x_norm = float(np.linalg.norm(x_proof))
                if verdict is True:
                    shis.append(i)
                elif verdict is None:
                    if np.isfinite(attempt_bound) and escalate_bound and bound_index + 1 < len(bounds_to_try):
                        # Undecided with a finite box: the box may be what keeps row i down.
                        box_clipping_suspected = True
                    else:
                        from relucent.core.errors import AmbiguousGeometryError

                        raise AmbiguousGeometryError(
                            f"cannot certify whether halfspace {i} is a facet of {poly!r} in float64 "
                            + f"(LP objective {model.objVal:.4e})"
                        )

                basis_indices = cbasis != 0 if cbasis is not None else None
            else:
                raise ValueError(f"Model status: {model.status}")

            if collect_info:
                assert poly_info is not None
                poly_info.append(
                    {
                        "Objective Value": model.objVal,
                        "Min Non-Basis Slack": np.min(
                            constrs.Slack if basis_indices is None else constrs.Slack[~basis_indices]
                        ),
                        "Status": model.status,
                    }
                )
                if hasattr(model, "objVal"):
                    poly_info[-1]["Objective Value"] = model.objVal
                if hasattr(model, "objBound"):
                    poly_info[-1]["Objective Bound"] = model.objBound
                poly_info[-1]["x Norm"] = x_norm
                if collect_info == "All":
                    poly_info[-1] |= {"Slacks": constrs.Slack, "-b[i]": -b_orig[i], "Status": model.status}

                    poly_info[-1]["Proof"] = x_proof

            # Restore halfspace i
            lp_constrs[j].RHS = -b_red[j, 0]

            pbar.update(n_orig - len(subset) - pbar.n)

        if box_clipping_suspected and bound_index + 1 < len(bounds_to_try):
            continue
        break
    else:
        if model is not None:
            model.close()
        # Gurobi found no feasible point under any box. Only a verified interior point inside the
        # largest box tried can contradict that (a cell may simply not reach a finite box); without
        # one, keep the solver's (unverified) verdict.
        zc, _ = solve_radius(env, h_red, max_radius=float(cfg.MAX_RADIUS), errors=err_red)
        last_box = float(bounds_to_try[-1])
        if zc is not None and float(np.max(np.abs(np.asarray(zc, dtype=np.float64)))) < last_box:
            from relucent.core.errors import AmbiguousGeometryError

            raise AmbiguousGeometryError(
                f"the SHI LP for {poly!r} was reported infeasible (status {last_status}), but the cell has a "
                + "verified interior point: the LP solver is wrong here, so its facets cannot be computed"
            )
        raise ValueError(f"Initial Solve Failed: Model status: {last_status}")

    assert model is not None
    model.close()
    if collect_info:
        assert poly_info is not None
        return shis, poly_info
    return shis


# ScaleFlag values tried in turn, each from scratch, when a SHI LP fails. In a census of real cells,
# re-solving with no scaling (0) recovered every failure that any ScaleFlag recovered; the rest
# need the exact decision in get_shis (docs/search_shi_and_graphs.rst, "LP solver failures").
_SHI_LP_RETRY_SCALE_FLAGS: tuple[int, ...] = (0,)


def _cold_retry(model: Model, ok: tuple[int, ...] = (GRB.OPTIMAL,)) -> list[int]:
    """Re-solve a failed SHI LP ``model`` from scratch under each of :data:`_SHI_LP_RETRY_SCALE_FLAGS`.

    Warm starts and the configured scaling (:data:`relucent.config.advanced.GUROBI_SHI_SCALE_FLAG`) are
    what typically fail on a badly conditioned cell; every answer is certified afterwards, so the
    setting only changes what is proposed. Stops at the first status in ``ok``. Returns every
    status seen (the failed one first); the configured scaling is restored.
    """
    tried = [int(model.status)]
    try:
        for scale_flag in _SHI_LP_RETRY_SCALE_FLAGS:
            model.reset()  # discard the basis
            model.params.ScaleFlag = scale_flag
            model.optimize()
            tried.append(int(model.status))
            if model.status in ok:
                break
    finally:
        model.params.ScaleFlag = cfg.advanced.GUROBI_SHI_SCALE_FLAG
    return tried


def _recover_relaxed_shi_lp(
    model: Model, poly: "Polyhedron", i: int, decide: Callable[[], bool | None] | None = None
) -> bool | None:
    """Re-solve a failed SHI LP for halfspace ``i`` (:func:`_cold_retry`), else decide without it, else raise.

    The LP relaxes one row of a feasible cell and caps its objective, so it is feasible and bounded:
    any non-optimal status is a solver failure. Returns None once a re-solve succeeds (``model``
    then holds the solution). If none does, returns ``decide()``'s verdict (made in exact arithmetic,
    without the LP) when it has one, and otherwise raises.
    """
    tried = _cold_retry(model, (GRB.OPTIMAL, GRB.USER_OBJ_LIMIT))
    if model.status in (GRB.OPTIMAL, GRB.USER_OBJ_LIMIT):
        return None
    verdict = decide() if decide is not None else None
    if verdict is not None:
        return verdict
    from relucent.core.errors import AmbiguousGeometryError

    model.close()
    raise AmbiguousGeometryError(
        f"the SHI LP for halfspace {i} of {poly!r} failed although it relaxes a feasible LP (statuses "
        + f"{tried}: warm, then cold under ScaleFlag {list(_SHI_LP_RETRY_SCALE_FLAGS)}), and exact "
        + "arithmetic could not decide it either: LP solver failure"
    )


def _raise_if_coincident_facet(
    poly: "Polyhedron",
    model: Model,
    z: Any,
    lp_constrs: list[Any],
    j: int,
    i: int,
    partners: list[int],
    h_red: np.ndarray,
    err_red: np.ndarray,
    b_red: np.ndarray,
    push_size: float,
    ineq_rows: np.ndarray,
    work_to_orig: np.ndarray,
    interior_point: Callable[[], np.ndarray],
) -> None:
    """Raise when LP row ``j`` (halfspace ``i``) was judged no facet only because a duplicate hides it.

    ``partners`` are LP rows equal to row ``j`` up to a positive scale within their float64 error
    (:func:`_near_duplicate_rows`). They are relaxed together with row ``j``; if row ``j`` is then
    certified a facet, the cell has a facet made of several rows that are the same hyperplane to
    float64 precision, which the one-sign-flip neighbor structure cannot represent.
    """
    from relucent.core.errors import NonGenericArrangementError

    for jj in partners:
        lp_constrs[jj].RHS = -b_red[jj, 0] + push_size
    try:
        model.optimize()
        verdict = None
        if model.status == GRB.OPTIMAL:
            keep = np.array([r for r in range(h_red.shape[0]) if r not in set(partners)], dtype=np.intp)
            zv = np.asarray(z.X, dtype=np.float64).reshape(-1)
            verdict = _certify_facet(h_red[keep], err_red[keep], int(np.flatnonzero(keep == j)[0]), zv, None, interior_point)
    finally:
        for jj in partners:
            lp_constrs[jj].RHS = -b_red[jj, 0]
    if verdict is not True:
        model.optimize()  # back to row j's own LP for the caller's debug info (collect_info)
    if verdict is True:
        others = [int(work_to_orig[int(ineq_rows[jj])]) for jj in partners]
        raise NonGenericArrangementError(
            f"halfspaces {[i, *others]} of {poly!r} are the same hyperplane to within float64 error, and it is a facet: the "
            + "arrangement is not simple (typically a unit with an exactly zero bias whose only active input "
            + "is the other unit). relucent needs a simple arrangement; the facet cannot be crossed by one "
            + "sign flip."
        )


def _raise_if_coincident_across(poly: "Polyhedron", i: int, zero_units: np.ndarray, eq_orig_idx: list[int]) -> None:
    """Raise when a unit identically zero in this cell coincides with facet ``i`` on its other side.

    Such a unit's preactivation vanishes on the facet and is a nonzero multiple of halfspace ``i``'s
    across it, so the region across is labelled by flipping both signs, not one: the flipped cell
    is empty or has coincident rows. Decided on the float64 rows of the flipped sign sequence
    (restricted to this cell's affine hull): rows within their error of the same hyperplane, in
    either orientation, count as coincident.
    """
    from relucent.core.errors import NonGenericArrangementError
    from relucent.core.poly import Polyhedron

    src = poly.halfspaces_rows_ss
    if poly._net is None or src is None:
        return
    flipped = np.asarray(src, dtype=np.int8).copy().reshape(1, -1)
    flipped[0, i] = -flipped[0, i]
    across = Polyhedron(poly._net, flipped)
    hs, err = across.halfspaces_np, across.halfspaces_err_np
    pos = np.arange(hs.shape[0])
    if eq_orig_idx:
        eq = np.asarray(eq_orig_idx, dtype=np.intp)
        x0, null_basis, ineq_mask = _affine_null_basis(hs, eq, errors=err)
        err = _reduced_rows_error(hs, err, eq, ineq_mask, x0, null_basis)
        ineq = hs[ineq_mask]
        hs = np.hstack((ineq[:, :-1] @ null_basis, ineq[:, :-1] @ x0 + ineq[:, -1:]))
        pos = np.cumsum(ineq_mask) - 1  # original row -> face-coordinate row, for inequality rows
    for jz in zero_units:
        jz = int(jz)
        if jz == i or jz in eq_orig_idx:
            continue
        pair, pair_err = hs[[pos[i], pos[jz]]], err[[pos[i], pos[jz]]]
        if _near_duplicate_rows(pair, pair_err) or _near_duplicate_rows(pair * np.array([[1.0], [-1.0]]), pair_err):
            raise NonGenericArrangementError(
                f"unit {jz} is identically zero in {poly!r} but coincides with its facet {i} on the "
                + "other side (typically a unit with an exactly zero bias whose only active input is unit "
                + f"{i}): the arrangement is not simple, and the region across facet {i} is not one sign flip away."
            )


def certified_bounded(poly: "Polyhedron", env: Env | None = None) -> bool:
    """Whether the nonempty cell ``poly`` is bounded, decided for its exact rows.

    A nonempty cell ``{A x + b <= 0, A_eq x + b_eq = 0}`` is bounded exactly when its recession
    cone ``{d : A d <= 0, A_eq d = 0}`` is ``{0}``. A finite Chebyshev radius does not show this:
    a cell whose recession cone is lower-dimensional, such as a half-infinite prism, has a finite
    inscribed ball. On ReLU networks such cones come from structural coincidences (a deeper
    unit's normal lying exactly in the span of the active normals above it), which float64
    cannot tell from a near-miss.

    The float64 test is Stiemke's lemma: the cone is ``{0}`` iff the rows have full rank and
    ``A^T y + A_eq^T z = 0`` for some ``y > 0``. One LP finds ``y >= 1``; it is accepted when a
    basis of the rows stays nonsingular, and the correction that cancels the exact rows' residual
    on that basis keeps ``y`` positive, under every perturbation within the rows' error
    (:mod:`relucent._internal.rounding`). Anything else is decided by an exact simplex on the
    recession cone (:func:`relucent._internal.exact.exact_recession_cone_is_zero`).

    Raises:
        AmbiguousGeometryError: If the float64 test fails and the exact rows are unavailable or
            the exact simplex does not finish.
    """
    from relucent._internal import exact
    from relucent.core.errors import AmbiguousGeometryError

    hs = np.asarray(poly.halfspaces_np, dtype=np.float64)
    err = np.asarray(poly.halfspaces_err_np, dtype=np.float64)
    zero_idx = np.asarray(poly.zero_indices, dtype=np.intp)
    zero_idx = zero_idx[zero_idx < hs.shape[0]]
    is_eq = np.zeros(hs.shape[0], dtype=bool)
    is_eq[zero_idx] = True
    # A constant row (exactly zero normal) says nothing about directions.
    keep = np.flatnonzero(np.any(hs[:, :-1] != 0.0, axis=1))
    if _bounded_by_stiemke(hs[keep, :-1], err[keep, :-1], is_eq[keep], env=env):
        return True
    rows = poly._exact_rows()
    verdict = exact.exact_recession_cone_is_zero(rows, zero_idx.tolist()) if rows is not None else None
    if verdict is None:
        raise AmbiguousGeometryError(
            f"cannot decide whether {poly!r} is bounded: its recession cone is degenerate to within "
            + "float64 error and its exact rows are unavailable"
        )
    return verdict


def _bounded_by_stiemke(a: np.ndarray, a_err: np.ndarray, is_eq: np.ndarray, *, env: Env | None = None) -> bool:
    """True when ``{d : a_i . d <= 0, a_e . d = 0}`` is certified ``{0}`` for every exact ``a``
    within ``a_err`` of ``a`` (inequality rows ``~is_eq``, equality rows ``is_eq``).

    False means only "not certified": see :func:`certified_bounded`.
    """
    import scipy.linalg

    from relucent._internal import rounding

    m, n = a.shape
    if m < n:
        return False  # rank < n: the cone holds a line
    ineq = ~is_eq
    model = Model("Recession cone", env or get_env())
    try:
        w = model.addMVar(m, lb=np.where(ineq, 1.0, -GRB.INFINITY), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w")
        model.addConstr(a.T @ w == np.zeros(n))
        model.setObjective(ineq.astype(np.float64) @ w, GRB.MINIMIZE)
        model.optimize()
        if model.status != GRB.OPTIMAL:
            return False
        y = np.asarray(w.X, dtype=np.float64).reshape(-1)
    finally:
        model.close()
    if np.any(y[ineq] <= 0.0):
        return False
    # The correction below lives on n rows; pick a well-conditioned set.
    pivots = scipy.linalg.qr(a.T, pivoting=True, mode="economic")[-1]
    basis = np.asarray(pivots, dtype=np.intp)[:n]
    m_b = a[basis]
    # Perturbation of the basis (Frobenius bounds spectral), plus the SVD's own rounding.
    dm = float(np.linalg.norm(a_err[basis])) + 8.0 * n * n * rounding.EPS * float(np.linalg.norm(m_b))
    smin = float(np.linalg.svd(m_b, compute_uv=False)[-1])
    if smin <= 2.0 * dm:
        return False
    # |A*^T y| for the exact rows A*: computed residual, its rounding, and the rows' own error.
    resid = np.abs(a.T @ y) + rounding.gamma(m + 1) * (np.abs(a.T) @ np.abs(y)) + a_err.T @ np.abs(y)
    # Moving the basis multipliers by delta cancels the exact residual (A*_B is nonsingular).
    delta = float(np.linalg.norm(resid)) * (1.0 + 8.0 * rounding.EPS) / (smin - dm) * (1.0 + 8.0 * rounding.EPS)
    return not bool(np.any(y[basis][ineq[basis]] <= delta))


def _certify_not_facet(
    h_red: np.ndarray,
    err_red: np.ndarray,
    j: int,
    s: np.ndarray,
    e: np.ndarray,
    tight: np.ndarray,
) -> bool:
    """True when row ``j`` is certified strictly negative on the whole exact cell, bounded or not.

    ``tight`` names the ``k`` rows tight at the LP point (its optimal basis); ``s`` and ``e`` are
    every row's value and float64 error there. Solving ``M^T y = a_j`` on the tight normals ``M``
    and bounding how far the exact rows move ``y`` (perturbation of ``M`` and ``a_j``, plus the
    solve residual) gives an interval around the exact multipliers ``y*``. When every ``y*`` is
    nonnegative, the exact ``a_j*`` is a nonnegative combination of the exact tight normals, so
    ``row_j*(z') - sum_r y*_r row_r*(z')`` is one constant over all of space and, with each tight
    row nonpositive on the cell, ``row_j*(z') <= row_j*(z) - sum_r y*_r row_r*(z)`` everywhere on
    it. That constant is bounded above from the values at the LP point ``z``.

    A dual residual alone (``a_j - M^T y`` left unabsorbed) proves nothing on an unbounded cell:
    it can tilt row ``j`` positive arbitrarily far away. Dual-degenerate bases (a multiplier
    within its error of zero, e.g. row ``j`` parallel to a tight row) are therefore not
    certified here.
    """
    from relucent._internal import rounding

    a = h_red[:, :-1]
    k = a.shape[1]
    tight = np.asarray(tight, dtype=np.intp)
    if tight.size != k or np.any(tight == j):
        return False
    m = a[tight]
    dm = float(np.linalg.norm(err_red[tight, :-1]))
    smin = float(np.linalg.svd(m, compute_uv=False)[-1])
    if smin <= 2.0 * dm:
        return False
    try:
        y = np.linalg.solve(m.T, a[j])
    except np.linalg.LinAlgError:
        return False
    resid = np.abs(m.T @ y - a[j]) + rounding.gamma(k + 1) * (np.abs(m.T) @ np.abs(y) + np.abs(a[j]))
    da_j = float(np.linalg.norm(err_red[j, :-1]))
    delta = (float(np.linalg.norm(resid)) + da_j + dm * float(np.linalg.norm(y))) / (smin - dm)
    delta *= 1.0 + 8.0 * rounding.EPS
    if np.any(y < delta):
        return False
    slack = e[j] + float(np.sum((y + delta) * (np.abs(s[tight]) + e[tight])))
    return bool(s[j] + slack * (1.0 + 8.0 * rounding.EPS) < 0.0)


def _certify_facet(
    h_red: np.ndarray,
    err_red: np.ndarray,
    j: int,
    z: np.ndarray,
    tight: np.ndarray | None,
    interior_point: Callable[[], np.ndarray],
) -> bool | None:
    """Certify from one LP answer whether row ``j`` of the cell ``h_red <= 0`` is a facet.

    ``z`` is the LP point for "maximise row j with row j relaxed". Returns True when a witness
    exists -- a point where row ``j`` is strictly positive and every other row strictly negative,
    each beyond its float64 error (``err_red``) -- found at ``z`` or on the segment from ``z`` to
    the cell's verified interior point. Returns False when the LP's tight rows (``tight``, from
    its basis) certify row ``j`` strictly negative on the whole cell (:func:`_certify_not_facet`).
    Returns None when neither holds.
    """
    from relucent._internal import rounding

    zv = np.asarray(z, dtype=np.float64).reshape(-1)
    a, b = h_red[:, :-1], h_red[:, -1]
    s = a @ zv + b
    e = rounding.row_errors(err_red, zv)
    others = np.arange(s.shape[0]) != j
    if s[j] > e[j]:
        if np.all(s[others] < -e[others]):
            return True
        zc = interior_point()
        sc = a @ zc + b
        ec = rounding.row_errors(err_red, zc)
        bad = others & (s >= -e)
        # Row values are affine and error bounds convex along the segment, so row r is
        # strictly negative beyond its error for alpha > alpha_lo, and row j strictly
        # positive for alpha < alpha_hi. Take the midpoint (alpha_lo can be exactly 0,
        # e.g. a row that's 0 with zero error at z) and re-check everything there.
        num = s[bad] + e[bad]
        den = num - (sc[bad] + ec[bad])
        alpha_lo = float(np.max(num / den))
        top = s[j] - e[j]
        alpha_hi = min(1.0, top / (top - (sc[j] - ec[j])))
        if not alpha_lo < alpha_hi:
            return None
        za = zv + 0.5 * (alpha_lo + alpha_hi) * (zc - zv)
        side = rounding.classify_rows(h_red, err_red, za, 0.0)
        if side[j] == 1 and np.all(side[others] == -1):
            return True
        return None
    if tight is not None and _certify_not_facet(h_red, err_red, j, s, e, tight):
        return False
    return None


def _read_lp_solution(z: Any, constrs: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """The SHI LP's point ``z.X`` (flat) and constraint basis ``CBasis`` (flat; None if unavailable).

    Read once per row: each attribute read goes through Gurobi's matrix API.
    """
    zv = np.asarray(z.X, dtype=np.float64).reshape(-1)
    try:
        cbasis: np.ndarray | None = np.asarray(constrs.CBasis).reshape(-1)
    except Exception:  # noqa: BLE001 - no basis available
        cbasis = None
    return zv, cbasis


def _certify_facet_from_model(
    status: int,
    zv: np.ndarray,
    cbasis: np.ndarray | None,
    j: int,
    h_red: np.ndarray,
    err_red: np.ndarray,
    interior_point: Callable[[], np.ndarray],
) -> bool | None:
    """Run :func:`_certify_facet` on an SHI LP solution (from :func:`_read_lp_solution`).

    The basis's tight rows feed the non-facet certificate only when the LP is optimal.
    """
    tight = np.flatnonzero(cbasis != 0) if status == GRB.OPTIMAL and cbasis is not None else None
    return _certify_facet(h_red, err_red, j, zv, tight, interior_point)


def compute_properties(poly: "Polyhedron", qhull_mode: str | None = None) -> None:
    """Compute additional geometric properties for low-dimensional polyhedra (vertices, hull, volume).

    Mutates ``poly`` cache fields (``_hs``, ``_vertices``, ``_ch``, ``_volume``,
    ``_attempted_compute_properties``). No-op if already attempted.

    For non-maximal cells (``zero_indices`` non-empty) the halfspace system lives on a
    lower-dimensional affine subspace.  The function projects into that subspace via an
    SVD-derived null basis, runs :class:`~scipy.spatial.HalfspaceIntersection` in the
    reduced coordinates, then maps vertices back to ambient space.  Volume / convex hull
    are computed in the intrinsic (reduced) coordinates so they remain meaningful instead
    of collapsing to zero due to the ambient degeneracy.

    Raises:
        ValueError: If input dimension > 6, interior point is missing, or qhull fails
            (depending on ``qhull_mode``).
    """
    if qhull_mode is None:
        qhull_mode = cfg.QHULL_MODE
    if poly._attempted_compute_properties:
        return
    poly._attempted_compute_properties = True

    assert poly._net is not None
    if poly._net.input_shape[0] > 6:
        raise ValueError("Input shape too large to compute extra properties")

    # Filter degenerate constraints and remap zero_indices accordingly.
    errors = poly.halfspaces_err_np
    halfspaces, old_to_new = _drop_degenerate_halfspaces_tracked(poly.halfspaces_np, errors=errors)
    if halfspaces.shape[0] != errors.shape[0]:
        errors = errors[old_to_new >= 0]
    zero_indices_eff = _remap_zero_indices(poly.zero_indices, old_to_new)

    # For non-maximal cells, project into the affine subspace defined by the equality
    # constraints.  HalfspaceIntersection requires a full-dimensional interior point,
    # which cannot exist when equality hyperplanes reduce the dimension.
    x0: np.ndarray | None = None
    null_basis: np.ndarray | None = None
    projected_halfspaces = halfspaces

    if zero_indices_eff is not None and zero_indices_eff.size > 0:
        x0, null_basis, ineq_mask = _affine_null_basis(halfspaces, zero_indices_eff, errors=errors)

        if null_basis.shape[1] == 0:
            # The equality system pins a unique point; no inequalities can reduce it further.
            poly._vertices = x0.T  # shape (1, ambient_dim)
            poly._volume = 0.0
            return

        # Express inequality halfspaces in reduced coordinates z: x = null_basis @ z + x0.
        inequalities = halfspaces[ineq_mask]
        A_red = inequalities[:, :-1] @ null_basis
        b_red = inequalities[:, :-1] @ x0 + inequalities[:, -1:]
        projected_halfspaces = np.hstack((A_red, b_red))

    # Project interior point to reduced coordinates.
    if poly.interior_point is None:
        raise ValueError("Interior point not found")
    if null_basis is not None and x0 is not None:
        z0, *_ = np.linalg.lstsq(null_basis, poly.interior_point.reshape(-1, 1) - x0, rcond=None)
        projected_interior_point = np.asarray(z0).reshape(-1)
    else:
        projected_interior_point = poly.interior_point

    # Qhull does not support 1-D halfspace intersection; handle analytically.
    reduced_dim = projected_halfspaces.shape[1] - 1
    if reduced_dim == 1:
        a_col = projected_halfspaces[:, 0]
        b_col = projected_halfspaces[:, 1]
        lower, upper = -float("inf"), float("inf")
        for ai, bi in zip(a_col, b_col, strict=True):
            if ai == 0.0:
                # A row parallel to the segment; the segment was certified nonempty.
                if bi > 0.0:
                    raise ValueError("Infeasible 1-D projected halfspace system")
                continue
            cutoff = -bi / ai
            if ai > 0:
                upper = min(upper, cutoff)
            else:
                lower = max(lower, cutoff)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("Projected 1-D intersection is empty or unbounded")
        reduced_verts = np.array([[lower], [upper]], dtype=np.float64)
        if null_basis is not None and x0 is not None:
            poly._vertices = np.unique((null_basis @ reduced_verts.T + x0).T, axis=0)
        else:
            poly._vertices = np.unique(reduced_verts, axis=0)
        poly._volume = float(upper - lower)
        return

    try:
        with warnings.catch_warnings(record=True) as w:
            hs = HalfspaceIntersection(
                projected_halfspaces,
                projected_interior_point,
                qhull_options=None,
            )  # http://www.qhull.org/html/qh-optq.htm
        if w:
            msgs = "; ".join(str(wi.message) for wi in w)
            if qhull_mode == "IGNORE":
                poly.warnings.extend([RuntimeWarning(wi) for wi in w])
            if qhull_mode == "WARN_ALL":
                warnings.warn(f"Halfspace intersection emitted warnings: {msgs}", stacklevel=2)
            elif qhull_mode == "HIGH_PRECISION":
                raise ValueError(f"HalfspaceIntersection emitted warnings in HIGH_PRECISION mode: {msgs}")
            elif qhull_mode == "JITTERED":
                with warnings.catch_warnings(record=True) as w2:
                    new_hs = HalfspaceIntersection(
                        projected_halfspaces,
                        projected_interior_point,
                        # Triangulated output is approximately 1000 times more accurate than joggled input.
                        qhull_options="QJ",
                    )  # http://www.qhull.org/html/qh-optq.htm
                if w2:
                    poly.warnings.append(
                        RuntimeWarning(
                            "Recomputing HalfspaceIntersection with jitter option 'QJ' still had numerical problems"
                        )
                    )
                    poly.warnings.extend([RuntimeWarning(wi) for wi in w2])
                    msgs = "; ".join(str(wi.message) for wi in w)
                else:
                    ## Jittering solved the numerical problems
                    hs = new_hs
    except ValueError:
        raise  # Our HIGH_PRECISION raise - do not retry
    except Exception as e:
        if qhull_mode == "JITTERED":
            try:
                hs = HalfspaceIntersection(
                    projected_halfspaces,
                    projected_interior_point,
                    # Triangulated output is approximately 1000 times more accurate than joggled input.
                    qhull_options="QJ",
                )  # http://www.qhull.org/html/qh-optq.htm
                poly.warnings.append(RuntimeWarning(f"HalfspaceIntersection failed initially, succeeded with QJ retry: {e}"))
            except Exception as e2:
                raise ValueError(f"Error while computing halfspace intersection: {e}") from e2
        else:
            raise ValueError(f"Error while computing halfspace intersection: {e}") from e

    poly._halfspace_intersection = hs
    raw_vertices = hs.intersections  # in reduced coordinates when projected

    # Remap to ambient coordinates for the trust filter and poly._vertices.
    vertices = (null_basis @ raw_vertices.T + x0).T if null_basis is not None and x0 is not None else raw_vertices

    trust_vertices = ~(np.isinf(vertices).any(axis=1) | np.isnan(vertices).any(axis=1))
    # A Qhull vertex is kept unless some row is violated beyond that row's own float64 error
    # there. (Qhull's own error is not bounded here; vertices and volume do not enter topology.)
    trust_vertices_2 = np.array(
        [not np.any(_rounding().classify_rows(halfspaces, errors, v, 0.0) == 1) for v in vertices[trust_vertices]],
        dtype=bool,
    )
    poly._vertices = vertices[trust_vertices][trust_vertices_2]

    # ConvexHull and volume are computed in the intrinsic (reduced) coordinates.
    # Using ambient-space vertices for non-maximal cells would produce a degenerate hull.
    ch_vertices = raw_vertices[trust_vertices][trust_vertices_2] if null_basis is not None else poly._vertices

    if poly.finite and len(ch_vertices) > reduced_dim:
        try:
            poly._convex_hull = ConvexHull(ch_vertices)
            try:
                poly._volume = poly._convex_hull.volume
            except Exception as e:
                raise ValueError(f"Error while computing convex hull volume: {e}") from e
        except Exception as e:
            if qhull_mode == "WARN_ALL":
                warnings.warn(f"Error while computing convex hull: {e}", stacklevel=2)
            elif qhull_mode == "HIGH_PRECISION":
                raise ValueError(f"Error while computing convex hull: {e}") from e
            poly._convex_hull = None
            poly._volume = -1
    else:
        poly._volume = float("inf")
