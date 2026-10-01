"""Exact rational geometry for the two decisions float64 can't make on ReLU rows.

Float64 weights are dyadic rationals, so a cell's composed rows have exact rational
values. ReLU networks produce structural coincidences that float64 can't tell from
near-misses: a deeper unit's normal can lie exactly in the span of the active normals
above it, so hyperplanes end up exactly parallel or exactly concurrent. Two decisions
hit this often. When float64 falls inside its error bound
(:mod:`relucent._internal.rounding`), they're made here with
:class:`fractions.Fraction` arithmetic:

* :func:`exact_point`: where some rows vanish, and which side of every other row that
  point is on (vertex recovery; parallel rows that never meet).
* :func:`exact_facet_by_simplex`: whether a row is a facet (a redundant row parallel
  to a facet has a zero dual multiplier, which float64 can't certify).

Anything else undecidable raises :class:`~relucent.core.errors.AmbiguousGeometryError`.
Exact arithmetic is slow (numbers reach a few hundred bits on deep nets), so it only
runs on ambiguous cases, with each pattern's rows cached.
"""

from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from relucent.core.errors import NonGenericArrangementError

if TYPE_CHECKING:
    from relucent.model.model import ReLUNetwork

__all__ = [
    "exact_facet_by_simplex",
    "exact_point",
    "exact_rows_affordable",
    "exact_rows_for_ss",
]

Row = list[Fraction]

# Keyed by a digest of the network's weights, not ``id(net)``: ids are reused once a network is
# freed, and weights can be reassigned in place, and either would serve another network's rows.
_CACHE: OrderedDict[tuple[bytes, bytes], list[Row]] = OrderedDict()
_CACHE_SIZE = 256

# Rational multiply-adds beyond which rebuilding a pattern's rows exactly is not attempted (each
# costs microseconds, more as numerators grow): a wide network's exact rows would take hours.
MAX_EXACT_OPS = 10_000_000


def _weights_digest(net: ReLUNetwork) -> bytes:
    """Digest of every layer's type, weights and bias, as the exact rows depend on them."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    for name, layer in net.layers.items():
        h.update(f"{name}:{type(layer).__name__};".encode())
        for attr in ("weight", "bias"):
            value = getattr(layer, attr, None)
            if value is not None:
                arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
                h.update(repr(arr.shape).encode())
                h.update(arr.tobytes())
    return h.digest()


def exact_rows_affordable(net: ReLUNetwork) -> bool:
    """Whether :func:`exact_rows_for_ss` stays within :data:`MAX_EXACT_OPS` for ``net``."""
    from relucent.model.model import LinearLayer

    ops = 0
    d: int | None = None
    for layer in net.layers.values():
        if isinstance(layer, LinearLayer):
            n_out, n_in = np.asarray(layer.weight).shape
            if d is None:
                d = int(n_in)  # the identity start makes the first product ~d * n_out
                ops += d * int(n_out)
            else:
                ops += d * int(n_in) * int(n_out)
    return ops <= MAX_EXACT_OPS


def _frac_matrix(a: np.ndarray) -> list[list[Fraction]]:
    return [[Fraction(float(v)) for v in row] for row in np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)]


def _dot(u: list[Fraction], v: list[Fraction]) -> Fraction:
    return sum((ui * vi for ui, vi in zip(u, v, strict=True) if ui and vi), Fraction(0))


def exact_rows_for_ss(net: ReLUNetwork, ss: np.ndarray) -> list[Row]:
    """The exact halfspace rows ``[a | b]`` that ``get_hs`` computes in float64 for ``ss``.

    Mirrors ``relucent.geometry.calculations._get_hs_numpy``: a 0 entry keeps its unit's row
    (with sign +1) but switches the unit off downstream.
    """
    from relucent.model.model import FlattenLayer, LinearLayer, ReLULayer

    row = np.asarray(ss, dtype=np.int8).ravel()
    key = (_weights_digest(net), row.tobytes())
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit

    cur_a: list[list[Fraction]] | None = None  # d x width
    cur_b: list[Fraction] | None = None  # width
    rows: list[Row] = []
    idx = 0
    for layer in net.layers.values():
        if isinstance(layer, LinearLayer):
            w = _frac_matrix(np.asarray(layer.weight))  # out x in
            b = [Fraction(float(v)) for v in np.asarray(layer.bias, dtype=np.float64).ravel()]
            n_in = len(w[0])
            if cur_a is None or cur_b is None:
                cur_a = [[Fraction(int(i == j)) for j in range(n_in)] for i in range(n_in)]
                cur_b = [Fraction(0)] * n_in
            d = len(cur_a)
            cur_a = [
                [sum((cur_a[i][k] * w[j][k] for k in range(n_in) if cur_a[i][k]), Fraction(0)) for j in range(len(w))]
                for i in range(d)
            ]
            cur_b = [sum((cur_b[k] * w[j][k] for k in range(n_in) if cur_b[k]), Fraction(0)) + b[j] for j in range(len(w))]
        elif isinstance(layer, ReLULayer):
            if cur_a is None or cur_b is None:
                raise ValueError("ReLU layer must follow a linear layer")
            width = len(cur_b)
            for j in range(width):
                s = int(row[idx + j])
                sign = 1 if s == 0 else s
                # halfspace = -(sign * preactivation) <= 0
                rows.append([-sign * cur_a[i][j] for i in range(len(cur_a))] + [-sign * cur_b[j]])
            on = [int(row[idx + j]) == 1 for j in range(width)]
            cur_a = [[v if on[j] else Fraction(0) for j, v in enumerate(r)] for r in cur_a]
            cur_b = [v if on[j] else Fraction(0) for j, v in enumerate(cur_b)]
            idx += width
        elif isinstance(layer, FlattenLayer):
            if cur_a is not None:
                raise NotImplementedError("Intermediate flatten layer not supported")
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

    _CACHE[key] = rows
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return rows


def _solve(a: list[list[Fraction]], rhs: list[Fraction]) -> tuple[list[Fraction] | None, int, bool]:
    """Gauss-Jordan on ``a x = rhs``: ``(a solution or None, rank, consistent)``."""
    m = [list(r) + [c] for r, c in zip(a, rhs, strict=True)]
    n_rows, n_cols = len(m), len(a[0])
    piv_cols: list[int] = []
    r = 0
    for c in range(n_cols):
        p = next((i for i in range(r, n_rows) if m[i][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        inv = 1 / m[r][c]
        m[r] = [v * inv for v in m[r]]
        for i in range(n_rows):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [vi - f * vr for vi, vr in zip(m[i], m[r], strict=True)]
        piv_cols.append(c)
        r += 1
        if r == n_rows:
            break
    consistent = all(m[i][n_cols] == 0 for i in range(r, n_rows))
    if not consistent or r < n_cols:
        return None, r, consistent
    x = [Fraction(0)] * n_cols
    for i, c in enumerate(piv_cols):
        x[c] = m[i][n_cols]
    return x, r, consistent


def exact_point(rows: list[Row], eq_indices: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Exact version of ``Polyhedron._halfspace_point_with_error`` on exact rows.

    Returns ``(x, err)`` with ``x`` the float64 rounding of the exact point and ``err`` a bound
    on its distance to it, or ``None`` when the equality rows have no common point or another
    row is violated. Raises :class:`NonGenericArrangementError` when the equality rows meet in
    more than a point, or another row also vanishes there (hyperplanes exactly concurrent).
    """
    eq = [int(i) for i in np.asarray(eq_indices).ravel()]
    a = [rows[i][:-1] for i in eq]
    rhs = [-rows[i][-1] for i in eq]
    x, rank, consistent = _solve(a, rhs)
    if not consistent:
        return None
    if x is None:
        raise NonGenericArrangementError(
            f"equality rows {eq} meet in an affine set of dimension {len(a[0]) - rank}, not a point"
        )
    eq_set = set(eq)
    for j, r in enumerate(rows):
        if j in eq_set:
            continue
        normal = r[:-1]
        if not any(normal):
            continue  # a constant row, decided when the cell was built
        value = sum((ni * xi for ni, xi in zip(normal, x, strict=True)), Fraction(0)) + r[-1]
        if value > 0:
            return None
        if value == 0:
            raise NonGenericArrangementError(f"row {j} vanishes exactly at the point where rows {eq} vanish")
    xf = np.array([float(v) for v in x], dtype=np.float64)
    err = float(sum((abs(Fraction(float(xf[i])) - x[i]) for i in range(len(x))), Fraction(0)))
    return xf, err * (1.0 + 4.0 * float(np.finfo(np.float64).eps)) + float(np.finfo(np.float64).tiny)


def _maximize(
    ineq: list[Row],
    eq: list[Row],
    objective: Row,
    x: list[Fraction],
    *,
    stop_above: Fraction,
    max_iter: int,
) -> str | None:
    """Maximise ``objective`` over ``{ineq rows <= 0, eq rows == 0}`` exactly, from feasible ``x``.

    An active-set primal simplex in rational arithmetic. Holding a set of tight rows, it steps
    along the projection of the objective's normal onto their common null space until another
    row becomes tight; where the normal lies in the span of the held rows it drops one with a
    negative multiplier (smallest index first, as in Bland's rule). Returns ``"above"`` once the
    objective exceeds ``stop_above``, ``"unbounded"`` when a feasible ray from ``x`` increases
    it without bound, and ``"optimal"`` at a point whose multipliers are all nonnegative,
    exactly; None if the equality normals are dependent or ``max_iter`` steps pass. ``x`` must
    satisfy every row exactly.
    """
    c, c0 = objective[:-1], objective[-1]
    eq_normals = [r[:-1] for r in eq]

    def combine(normals: list[list[Fraction]], coef: list[Fraction]) -> list[Fraction]:
        return [sum((w * n[t] for w, n in zip(coef, normals, strict=True)), Fraction(0)) for t in range(len(x))]

    active: list[int] = []
    for _ in range(max_iter):
        if _dot(c, x) + c0 > stop_above:
            return "above"
        normals = eq_normals + [ineq[r][:-1] for r in active]
        lam: list[Fraction] | None = []
        direction = list(c)
        if normals:
            gram = [[_dot(a, b) for b in normals] for a in normals]
            lam, _, _ = _solve(gram, [_dot(a, c) for a in normals])
            if lam is None:
                return None
            direction = [cv - sv for cv, sv in zip(c, combine(normals, lam), strict=True)]
        if any(direction):
            # The objective increases along `direction`, which keeps every held row tight.
            best: tuple[Fraction, int] | None = None
            for r, row in enumerate(ineq):
                if r in active:
                    continue
                rate = _dot(row[:-1], direction)
                if rate > 0:
                    step = -(_dot(row[:-1], x) + row[-1]) / rate
                    if best is None or step < best[0]:
                        best = (step, r)
            if best is None:
                return "unbounded"
            step, r = best
            x = [xv + step * dv for xv, dv in zip(x, direction, strict=True)]
            active.append(r)
            continue
        assert lam is not None
        # The normal is in the span of the held rows (equality multipliers are free).
        negative = [r for r, mu in zip(active, lam[len(eq) :], strict=True) if mu < 0]
        if not negative:
            return "optimal"
        active.remove(min(negative))
    return None


def _feasible_start(ineq: list[Row], eq: list[Row], start: np.ndarray) -> list[Fraction] | None:
    """``start`` moved exactly onto the ``eq`` rows' affine hull, if it then satisfies ``ineq``."""
    x = [Fraction(float(v)) for v in np.asarray(start, dtype=np.float64).ravel()]
    if eq:
        normals = [r[:-1] for r in eq]
        gram = [[_dot(a, b) for b in normals] for a in normals]
        lam, _, _ = _solve(gram, [_dot(r[:-1], x) + r[-1] for r in eq])
        if lam is None:
            return None
        x = [xv - sum((w * n[t] for w, n in zip(lam, normals, strict=True)), Fraction(0)) for t, xv in enumerate(x)]
    if any(_dot(r[:-1], x) + r[-1] > 0 for r in ineq):
        return None
    return x


def exact_facet_by_simplex(
    rows: list[Row],
    i: int,
    eq_indices: list[int],
    start: np.ndarray,
    *,
    max_iter: int | None = None,
) -> bool | None:
    """Exact answer to "is row ``i`` a facet of the cell", by an exact simplex (:func:`_maximize`).

    Row ``i`` is a facet exactly when it is irredundant: some point satisfying every other row
    (and the ``eq_indices`` rows with equality, on a face) has row ``i`` positive. This maximises
    row ``i`` over that polyhedron in rational arithmetic from ``start``, a float64 point inside
    the cell, moved exactly onto the face's affine hull. It needs no simplex basis from the LP
    solver, so it also decides cases whose float64 optimal basis is wrong in exact arithmetic
    (e.g. hyperplanes concurrent to within the LP tolerances).

    Returns None when ``start`` is not exactly feasible, the equality normals are dependent, or
    ``max_iter`` steps pass (default ``50 * (len(rows) + 1)``) without a decision.
    """
    eq_set = {int(e) for e in eq_indices}
    if not any(rows[i][:-1]):
        return None
    ineq = [rows[r] for r in range(len(rows)) if r != i and r not in eq_set and any(rows[r][:-1])]
    eq = [rows[e] for e in sorted(eq_set)]
    x = _feasible_start(ineq, eq, start)
    if x is None:
        return None
    limit = max_iter if max_iter is not None else 50 * (len(rows) + 1)
    result = _maximize(ineq, eq, rows[i], x, stop_above=Fraction(0), max_iter=limit)
    if result is None:
        return None
    return result != "optimal"
