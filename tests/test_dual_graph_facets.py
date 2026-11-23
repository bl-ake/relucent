"""Region adjacency of an explored complex against ground truth computed from the weights, without relucent.

The ambient dual graph must join two activation regions exactly when they share a facet. Two regions whose sign
sequences differ in one unit need not touch: past the first layer, the units after that one change formula across
its hyperplane, so the two regions can lie apart. A dual graph that joined every such pair passed every
self-consistency check (both sides of those checks used the same flip rule), so the ground truth here comes from
the weights alone:

* d = 2, exactly: the plane is cut by every unit's line, layer by layer, in rational arithmetic (float64 weights
  are exact rationals). Every activation region becomes a polygon, and two regions are adjacent when their polygons
  share an edge of positive length. No relucent code, linear programs or tolerances are involved.
* any d, by LP: each region's halfspaces are rebuilt from the weights in NumPy, and two regions are adjacent when a
  point of the hyperplane between them lies strictly inside every other halfspace of both (HiGHS, through SciPy,
  not Gurobi).

The networks are built here from seeded NumPy weights, so both sides read the same numbers, and the seeds are ones
whose networks have regions that differ in one unit without touching.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest
from scipy.optimize import linprog

from relucent import Complex
from relucent.model import LinearLayer, ReLULayer, ReLUNetwork
from relucent.search.exploration import explore_for_topology

Weights = list[tuple[np.ndarray, np.ndarray]]
Point = tuple[Fraction, Fraction]
Signs = tuple[int, ...]


# ── networks ─────────────────────────────────────────────────────────────────────────────
def _weights(widths: list[int], seed: int) -> Weights:
    """PyTorch-style uniform initialization, with a ReLU after every layer (the last one included, so the output
    unit's sign, and with it the decision boundary, is part of each region's sign sequence)."""
    rng = np.random.default_rng(seed)
    out = []
    for fan_in, fan_out in zip(widths[:-1], widths[1:], strict=True):
        bound = 1.0 / np.sqrt(fan_in)
        out.append((rng.uniform(-bound, bound, (fan_out, fan_in)), rng.uniform(-bound, bound, fan_out)))
    return out


def _network(weights: Weights) -> ReLUNetwork:
    layers = []
    for w, b in weights:
        layers += [LinearLayer(weight=w, bias=b.reshape(1, -1)), ReLULayer()]
    return ReLUNetwork(layers=layers, input_shape=(weights[0][0].shape[1],))


def _signs(weights: Weights, x: np.ndarray) -> np.ndarray:
    """Sign of every unit's pre-activation at each row of ``x``, layer by layer."""
    out, h = [], x
    for w, b in weights:
        z = h @ w.T + b
        out.append(np.sign(z))
        h = np.maximum(z, 0.0)
    return np.hstack(out).astype(np.int8)


def _explored(weights: Weights) -> Complex:
    cplx = Complex(_network(weights))
    explore_for_topology(cplx, seed=0)
    return cplx


def _relucent_regions(cplx: Complex) -> dict[Signs, object]:
    return {tuple(int(v) for v in np.asarray(p.ss_np).ravel()): p for p in cplx}


def _relucent_edges(cplx: Complex) -> set[frozenset[Signs]]:
    sig = {p: tuple(int(v) for v in np.asarray(p.ss_np).ravel()) for p in cplx}
    return {frozenset((sig[u], sig[v])) for u, v in cplx.G.edges()}


# ── ground truth at d = 2: the exact arrangement ─────────────────────────────────────────
def _area2(poly: list[Point]) -> Fraction:
    return sum((p[0] * q[1] - q[0] * p[1] for p, q in zip(poly, poly[1:] + poly[:1], strict=True)), Fraction(0))


def _clip(poly: list[Point], a: tuple[Fraction, Fraction], c: Fraction) -> list[Point] | None:
    """The part of a convex polygon where ``a.x + c >= 0``, or None if it has no area."""
    out: list[Point] = []
    for p, q in zip(poly, poly[1:] + poly[:1], strict=True):
        fp, fq = a[0] * p[0] + a[1] * p[1] + c, a[0] * q[0] + a[1] * q[1] + c
        if fp >= 0:
            out.append(p)
        if (fp > 0 > fq) or (fp < 0 < fq):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    dedup = [p for i, p in enumerate(out) if p != out[i - 1]] if out else []
    return dedup if len(dedup) >= 3 and _area2(dedup) != 0 else None


def _exact_regions_2d(weights: Weights, box: int) -> dict[Signs, list[Point]]:
    """Every activation region inside the square ``[-box, box]^2``, as an exact polygon, by sign sequence."""
    q = Fraction
    square = [(q(-box), q(-box)), (q(box), q(-box)), (q(box), q(box)), (q(-box), q(box))]
    # A cell: its polygon, the signs so far, and the previous layer's output on it as A x + c.
    cells = [(square, (), [[q(1), q(0)], [q(0), q(1)]], [q(0), q(0)])]
    for w, b in weights:
        wq = [[q(float(v)) for v in row] for row in w]
        bq = [q(float(v)) for v in b]
        grown = []
        for poly, signs, A, c in cells:
            Z = [[sum((wq[j][k] * A[k][m] for k in range(len(A))), q(0)) for m in range(2)] for j in range(len(wq))]
            z0 = [sum((wq[j][k] * c[k] for k in range(len(c))), q(0)) + bq[j] for j in range(len(wq))]
            parts: list[tuple[list[Point], Signs]] = [(poly, ())]
            for j in range(len(wq)):
                split = []
                for piece, s in parts:
                    for sign in (1, -1):
                        kept = _clip(piece, (sign * Z[j][0], sign * Z[j][1]), sign * z0[j])
                        if kept is not None:
                            split.append((kept, (*s, sign)))
                parts = split
            for piece, s in parts:
                on = [int(v > 0) for v in s]
                grown.append(
                    (
                        piece,
                        signs + s,
                        [[Z[j][m] * on[j] for m in range(2)] for j in range(len(s))],
                        [z0[j] * on[j] for j in range(len(s))],
                    )
                )
        cells = grown
    regions: dict[Signs, list[Point]] = {}
    for poly, signs, _, _ in cells:
        assert signs not in regions, "an activation region is convex, so it is one polygon"
        regions[signs] = poly
    return regions


def _exact_adjacency_2d(regions: dict[Signs, list[Point]]) -> set[frozenset[Signs]]:
    """Pairs of regions whose polygons share an edge of positive length."""
    by_line: dict[tuple[Fraction, Fraction, Fraction], list[tuple[Signs, Fraction, Fraction]]] = defaultdict(list)
    for s, poly in regions.items():
        for p, r in zip(poly, poly[1:] + poly[:1], strict=True):
            a, b = r[1] - p[1], p[0] - r[0]
            if a == 0 and b == 0:
                continue
            c = -(a * p[0] + b * p[1])
            k = a if a != 0 else b
            t = (p[0], r[0]) if b != 0 else (p[1], r[1])  # a coordinate that runs along the line
            by_line[(a / k, b / k, c / k)].append((s, min(t), max(t)))
    adjacent = set()
    for segments in by_line.values():
        for i, (s1, lo1, hi1) in enumerate(segments):
            for s2, lo2, hi2 in segments[i + 1 :]:
                if s1 != s2 and min(hi1, hi2) > max(lo1, lo2):
                    adjacent.add(frozenset((s1, s2)))
    return adjacent


# ── ground truth in any dimension: halfspaces from the weights, facets by HiGHS ─────────
def _region_rows(weights: Weights, signs: Signs) -> np.ndarray:
    """Rows ``[a, c]`` with ``a.x + c <= 0`` inside the region with these signs, rebuilt from the weights."""
    d = weights[0][0].shape[1]
    lin, off = np.eye(d), np.zeros(d)  # the previous layer's output on the region: lin @ x + off
    rows, k = [], 0
    for w, b in weights:
        z_lin, z_off = w @ lin, w @ off + b
        s = np.asarray(signs[k : k + len(b)], dtype=float)
        k += len(b)
        rows.append(np.c_[-s[:, None] * z_lin, -s * z_off])
        lin, off = (s > 0)[:, None] * z_lin, (s > 0) * z_off
    return np.vstack(rows)


def _shared_facet_depth(weights: Weights, s1: Signs, s2: Signs) -> float:
    """How far inside every other halfspace of both regions a point of the hyperplane between them can lie (rows
    normalised, capped at 1): positive when they share a facet, zero or negative when they do not touch."""
    (unit,) = np.flatnonzero(np.asarray(s1) != np.asarray(s2))
    r1, r2 = _region_rows(weights, s1), _region_rows(weights, s2)
    rows = np.vstack([np.delete(r1, unit, 0), np.delete(r2, unit, 0)])
    a, c = rows[:, :-1], rows[:, -1]
    norm = np.linalg.norm(a, axis=1)
    a, c, norm = a[norm > 0], c[norm > 0], norm[norm > 0]
    d = a.shape[1]
    res = linprog(
        np.r_[np.zeros(d), -1.0],
        A_ub=np.c_[a, norm],
        b_ub=-c,
        A_eq=np.r_[r1[unit, :-1], 0.0][None],
        b_eq=[-r1[unit, -1]],
        bounds=[(None, None)] * d + [(None, 1.0)],
        method="highs",
    )
    return float(-res.fun) if res.status == 0 else float("-inf")


def _one_unit_apart(regions) -> list[tuple[Signs, Signs]]:
    keys = set(regions)
    pairs = []
    for s in keys:
        for i, v in enumerate(s):
            t = (*s[:i], -v, *s[i + 1 :])
            if v != 0 and t in keys and s < t:
                pairs.append((s, t))
    return pairs


# ── tests ────────────────────────────────────────────────────────────────────────────────
# Each network has regions that differ in one unit without touching (7, 16 and 14 pairs in the last three); in
# the first, joining them would also merge two region components on one side of the boundary.
CASES_2D = [([2, 4, 4, 1], 6), ([2, 5, 5, 1], 4), ([2, 4, 4, 4, 1], 2)]
CASES_3D = [([3, 5, 5, 1], 2)]
BOX = 2**20  # every vertex of these networks lies far inside it


def test_sign_sequences_follow_the_weights() -> None:
    """The correspondence both ground truths rely on: relucent's sign sequence of a point is the sign of every
    unit's pre-activation, layer by layer, computed from the weights."""
    weights = _weights([3, 5, 5, 1], 0)
    cplx = Complex(_network(weights))
    x = np.random.default_rng(1).normal(size=(500, 3))
    np.testing.assert_array_equal(np.asarray(cplx.point2ss(x), dtype=np.int8), _signs(weights, x))


@pytest.mark.parametrize(("widths", "seed"), CASES_2D)
def test_regions_and_adjacency_match_the_exact_arrangement(widths: list[int], seed: int) -> None:
    weights = _weights(widths, seed)
    cplx = _explored(weights)
    exact = _exact_regions_2d(weights, BOX)
    assert set(_relucent_regions(cplx)) == set(exact)
    truth = _exact_adjacency_2d(exact)
    apart = [p for p in _one_unit_apart(exact) if frozenset(p) not in truth]
    assert apart, "this network has regions that differ in one unit without touching"
    assert _relucent_edges(cplx) == truth


@pytest.mark.parametrize(("widths", "seed"), CASES_3D)
def test_adjacency_matches_facets_found_by_an_independent_lp(widths: list[int], seed: int) -> None:
    weights = _weights(widths, seed)
    cplx = _explored(weights)
    regions = _relucent_regions(cplx)
    sampled = {tuple(int(v) for v in row) for row in _signs(weights, np.random.default_rng(2).normal(scale=3, size=(4000, 3)))}
    assert sampled <= set(regions), "every sampled point lies in a region the search found"
    touching, apart = set(), []
    for pair in _one_unit_apart(regions):
        (touching.add(frozenset(pair)) if _shared_facet_depth(weights, *pair) > 1e-9 else apart.append(pair))
    assert apart, "this network has regions that differ in one unit without touching"
    assert _relucent_edges(cplx) == touching


@pytest.mark.parametrize(("widths", "seed"), CASES_2D + CASES_3D)
def test_regions_on_each_side_separate_as_the_boundary_requires(widths: list[int], seed: int) -> None:
    """The complement of a properly embedded hypersurface with b components has b + 1 components."""
    cplx = _explored(_weights(widths, seed))
    out = cplx.n - 1
    boundary = cplx.boundary_complex(out)
    n_boundary = nx.number_weakly_connected_components(boundary.meta_graph()) if len(boundary) else 0
    same_side = nx.Graph()
    same_side.add_nodes_from(cplx.G.nodes)
    same_side.add_edges_from((u, v) for u, v, shi in cplx.G.edges(data="shi") if shi != out)
    assert nx.number_connected_components(same_side) == 1 + n_boundary
