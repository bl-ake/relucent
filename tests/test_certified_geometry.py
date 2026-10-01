"""Geometric decisions are certified in float64, decided exactly, or raise.

Replaces the threshold tests for ``TOL_SHI_OBJECTIVE`` / ``MIN_SEARCH_INRADIUS``: facets, cell
emptiness and thin cells no longer depend on any tolerance setting. Each decision is made against
the float64 error of the rows involved (``relucent._internal.rounding``); facet and vertex decisions
fall back to exact rational arithmetic when float64 cannot decide (``relucent._internal.exact``),
and anything else undecidable raises ``AmbiguousGeometryError``.
"""

from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

import relucent.config as cfg
from relucent import AmbiguousGeometryError, Polyhedron
from relucent.geometry.calculations import get_shis, solve_radius
from relucent.utils import get_env


@pytest.fixture
def env():
    return get_env()


def _cell(rows: list[list[float]]) -> Polyhedron:
    hs = np.asarray(rows, dtype=np.float64)
    return Polyhedron(None, np.ones((1, hs.shape[0]), dtype=np.int8), halfspaces=hs)


def _rectangle(width: float = 1.0, height: float = 1.0) -> list[list[float]]:
    return [[-1.0, 0.0, 0.0], [1.0, 0.0, -width], [0.0, -1.0, 0.0], [0.0, 1.0, -height]]


@pytest.mark.parametrize("height", [1e-8, 1e-10, 1e-12])
def test_thin_slab_keeps_every_facet(env, height: float) -> None:
    assert set(get_shis(_cell(_rectangle(height=height)), env=env)) == {0, 1, 2, 3}


@pytest.mark.parametrize("height", [1e-3, 1e-9])
def test_facets_do_not_depend_on_push_size(env, height: float) -> None:
    cell = _cell(_rectangle(height=height))
    assert set(get_shis(cell, env=env, push_size=2.0 * height)) == {0, 1, 2, 3}


@pytest.mark.parametrize("delta", [100.0, 1e-4, 1e-12])
def test_looser_parallel_wall_is_not_a_facet(env, delta: float) -> None:
    rows = [*_rectangle(), [0.0, 1.0, -(1.0 + delta)]]
    assert set(get_shis(_cell(rows), env=env)) == {0, 1, 2, 3}


def test_row_touching_one_vertex_is_not_a_facet(env) -> None:
    """``x + y <= 2`` meets the unit square only at (1, 1): its LP optimum is exactly 0."""
    rows = [*_rectangle(), [1.0, 1.0, -2.0]]
    assert set(get_shis(_cell(rows), env=env)) == {0, 1, 2, 3}


@pytest.mark.parametrize("height", [1e-6, 1e-10, 1e-14])
def test_solve_radius_never_returns_an_unverified_center(env, height: float) -> None:
    rows = _rectangle(height=height)
    try:
        center, radius = solve_radius(env, np.asarray(rows))
    except AmbiguousGeometryError:
        return
    assert center is not None and radius is not None and radius > 0
    xq = [Fraction(float(v)) for v in np.asarray(center).ravel()]
    for r in rows:
        value = sum((Fraction(a) * x for a, x in zip(r[:-1], xq, strict=True)), Fraction(0)) + Fraction(r[-1])
        assert value < 0


def test_solve_radius_empty_cell(env) -> None:
    rows = [[-1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 1.0, -1.0]]  # x >= 0 and x <= -1
    assert solve_radius(env, np.asarray(rows)) == (None, None)


def _strip_net(eps: float):
    """[1, 2, 1] net whose middle region is the strip 0 < x < eps."""
    import torch

    from relucent import convert
    from relucent.utils import TorchMLP

    fc1 = torch.nn.Linear(1, 2, dtype=torch.float64)
    fc2 = torch.nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        fc1.weight.fill_(1.0)
        fc1.bias[0] = 0.0
        fc1.bias[1] = -eps
        fc2.weight.fill_(0.01)
        fc2.bias.zero_()
    return convert(TorchMLP(OrderedDict([("fc0", fc1), ("relu0", torch.nn.ReLU()), ("fc1", fc2)]), [1, 2, 1]))


@pytest.mark.parametrize("eps", [1e-7, 1e-9])
def test_bfs_finds_thin_strip(eps: float, monkeypatch: pytest.MonkeyPatch) -> None:
    from relucent import Complex

    # The two ReLU thresholds are parallel by construction; that is the point of the test.
    monkeypatch.setattr(cfg, "CAREFUL_MODE", False)
    monkeypatch.setenv("RELUCENT_CAREFUL_MODE", "0")
    cplx = Complex(_strip_net(eps))
    info = cplx.bfs(start=np.array([-1.0], dtype=np.float64), max_polys=20, nworkers=1, verbose=0, verify=True)
    assert info["Complete"] is True
    assert not info["Bad SHI Computations"]
    assert len(cplx) == 3


def test_exact_rows_follow_the_network_weights() -> None:
    """The exact-row cache must never serve rows of other weights (reused ids, in-place edits)."""
    import gc

    import torch

    from relucent import convert, mlp
    from relucent._internal import exact
    from relucent.model.model import LinearLayer

    ss = np.ones((1, 4), dtype=np.int8)

    def fresh(net) -> list[Any]:
        exact._CACHE.clear()
        return exact.exact_rows_for_ss(net, ss)

    torch.manual_seed(0)
    net = convert(mlp([2, 4, 1]))
    before = exact.exact_rows_for_ss(net, ss)
    for layer in net.layers.values():
        if isinstance(layer, LinearLayer):
            layer.weight = layer.weight * 3.0
    after = exact.exact_rows_for_ss(net, ss)
    assert after != before
    assert after == fresh(net)

    for seed in range(20):  # freed networks' ids get reused
        torch.manual_seed(seed)
        other = convert(mlp([2, 4, 1]))
        got = exact.exact_rows_for_ss(other, ss)
        expected = fresh(other)
        assert got == expected
        exact.exact_rows_for_ss(other, ss)
        del other
        gc.collect()


def test_exact_rows_declined_for_wide_networks() -> None:
    import torch

    from relucent import convert, mlp
    from relucent._internal import exact

    torch.manual_seed(0)
    wide = convert(mlp([100, 400, 400, 1]))
    assert not exact.exact_rows_affordable(wide)
    cell = Polyhedron(wide, np.ones((1, 800), dtype=np.int8))
    assert cell._exact_rows() is None
    assert exact.exact_rows_affordable(convert(mlp([2, 8, 8, 1])))


@pytest.mark.parametrize("seed", [3, 8, 9, 10, 11])
def test_face_contains_its_own_interior_point(seed: int) -> None:
    """A face's zero rows hold to within rounding: no float64 point is exactly on the face.

    For these seeds the face's interior point is on the positive side of the face's plane in
    exact arithmetic (by rounding), which an exact check of every row wrongly rejected.
    """
    rng = np.random.default_rng(seed)
    normal = rng.standard_normal(2)
    offset = float(rng.uniform(-1.0, 1.0))
    # Face {normal . x + offset = 0} of the cell {normal . x + offset <= 0} cut by the box |x_i| <= 1.
    box = [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
    rows = np.array([[normal[0], normal[1], offset], *box])
    face = Polyhedron(None, np.array([[0, 1, 1, 1, 1]], dtype=np.int8), halfspaces=rows, dim=1, _ambient_dim=2)
    point = np.asarray(face.get_interior_point()).reshape(-1)
    assert point in face
    assert (point + 1e-3 * normal) not in face  # clearly off the face, on the outside
    assert np.array([2.0, 0.0]) not in face  # outside the box


@pytest.mark.parametrize("eps", [1e-17, 1e-11])
@pytest.mark.parametrize("bound", [1e3, float("inf")])
def test_unbounded_tilted_row_is_never_called_redundant(env, eps: float, bound: float) -> None:
    """``eps * x + y <= 0.5`` is positive at x ~ 1 / eps inside the unbounded cell: a facet.

    Gurobi stops at x = 0 (the reduced cost eps is below its tolerance), so the non-facet answer
    must not be certified from that LP point: the facet is found, or the decision raises.
    """
    rows = [[0.0, 1.0, 0.0], [0.0, -1.0, -1.0], [-1.0, 0.0, 0.0], [eps, 1.0, -0.5]]
    try:
        shis = get_shis(_cell(rows), env=env, bound=bound)
    except AmbiguousGeometryError:
        return
    assert 3 in shis


def test_robustly_redundant_row_is_certified_in_float64(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-facets with a clear dual certificate are decided in float64, on an unbounded cell."""
    from relucent._internal import exact

    def _no_exact(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("float64 should have decided")

    monkeypatch.setattr(exact, "exact_facet_by_simplex", _no_exact)
    # y <= 0 and x >= 0 cut the cell; the other two rows are redundant with positive multipliers.
    rows = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 1.0, -3.0], [-1.0, 0.5, -1.0]]
    assert set(get_shis(_cell(rows), env=env)) == {0, 1}


@pytest.mark.parametrize("gap", [5e-7, 1e-9])
def test_solve_radius_certifies_cells_empty_within_gurobi_tolerance(env, gap: float) -> None:
    """Gurobi returns a slightly negative radius; its duals prove emptiness, verified here."""
    from relucent._internal import rounding

    slab = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, gap], [0.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    with pytest.raises(AmbiguousGeometryError, match="emptiness could not be verified"):
        solve_radius(env, slab)  # parallel rows: no square Farkas certificate in float64
    triangle = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, gap]])
    noisy = rounding.exact_rows_error(triangle) * 1e3  # rows carrying float64 error
    assert solve_radius(env, triangle, errors=noisy) == (None, None)


def test_careful_mode_flags_duplicate_rows_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from relucent.geometry.calculations import _drop_degenerate_halfspaces_tracked

    monkeypatch.setattr(cfg, "CAREFUL_MODE", True)
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((50, 4))
    _drop_degenerate_halfspaces_tracked(rows)  # distinct rows pass
    from relucent import NonGenericArrangementError

    with pytest.raises(NonGenericArrangementError, match="fails genericity"):
        _drop_degenerate_halfspaces_tracked(np.vstack((rows, 2.0 * rows[7])))
    nearly = rows[7] + np.array([0.0, 0.0, 0.0, 1e-9])
    _drop_degenerate_halfspaces_tracked(np.vstack((rows, nearly)))  # 1e-9 apart is distinct


def test_row_constant_on_a_face_is_decided_exactly(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """A row parallel to the face's equality row is constant on the face: exactly decided, not raised."""
    from relucent._internal import exact

    calls: list[object] = []
    real = exact.exact_facet_by_simplex

    def _counting(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out)
        return out

    monkeypatch.setattr(exact, "exact_facet_by_simplex", _counting)
    rows = np.array(
        [
            [1.0, 2.0, 3.0, 0.5],  # equality row: the face is n . x = -0.5
            [2.0, 4.0, 6.0, 0.0],  # 2 (n . x) = -1 on the face: a negative constant, never a facet
            [1.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, -1.0, 0.0, -1.0],
        ]
    )
    face = Polyhedron(None, np.array([[0, 1, 1, 1, 1, 1]], dtype=np.int8), halfspaces=rows, dim=2, _ambient_dim=3)
    assert set(get_shis(face, env=env)) == {2, 3, 4, 5}
    assert False in calls  # row 1 decided exactly: not a facet


def _coincident_net():
    """2-2-2-1 net where unit 2 = -relu(unit 0) exactly (zero bias, single input).

    Where unit 0 is on (x > 0.5), unit 2's hyperplane is exactly unit 0's; where unit 0 is off,
    unit 2 is identically zero. The arrangement is not simple.
    """
    import torch

    from relucent import convert
    from relucent.utils import TorchMLP

    fc0 = torch.nn.Linear(2, 2, dtype=torch.float64)
    fc1 = torch.nn.Linear(2, 2, dtype=torch.float64)
    fc2 = torch.nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        fc0.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        fc0.bias.copy_(torch.tensor([-0.5, 0.3]))
        fc1.weight.copy_(torch.tensor([[-1.0, 0.0], [0.7, 1.1]]))
        fc1.bias.copy_(torch.tensor([0.0, 0.1]))
        fc2.weight.copy_(torch.tensor([[1.0, 1.0]]))
        fc2.bias.copy_(torch.tensor([-0.4]))
    layers: OrderedDict[str, torch.nn.Module] = OrderedDict(
        [("fc0", fc0), ("relu0", torch.nn.ReLU()), ("fc1", fc1), ("relu1", torch.nn.ReLU()), ("fc2", fc2)]
    )
    return convert(TorchMLP(layers, [2, 2, 2, 1]))


def test_coincident_hyperplanes_are_reported_not_skipped() -> None:
    """From x > 0.5 rows 0 and 2 are the same halfspace (each alone looks redundant): BFS must raise
    NonGenericArrangementError rather than finish without crossing x = 0.5."""
    from relucent import Complex, NonGenericArrangementError

    cplx = Complex(_coincident_net())
    with pytest.raises(NonGenericArrangementError):
        cplx.bfs(start=np.array([[0.9, 0.2]], dtype=np.float64), nworkers=1, verbose=0, verify=False)


def test_unit_zero_here_but_coincident_across_is_reported(env) -> None:
    """For x < 0.5 unit 2 is identically zero (so no BFS can start there); across unit 0's facet it
    is exactly unit 0's hyperplane, so the neighbor is not one sign flip away."""
    from relucent import NonGenericArrangementError

    net = _coincident_net()
    zero_side = Polyhedron(net, np.array([[-1, 1, -1, 1]], dtype=np.int8))
    with pytest.raises(NonGenericArrangementError, match="identically zero"):
        get_shis(zero_side, env=env)


def test_distinct_nearby_rows_are_not_reported(env) -> None:
    """Rows 1e-9 apart are distinct hyperplanes: the tighter one is a facet, no error."""
    rows = [*_rectangle(), [0.0, 1.0, -(1.0 + 1e-9)]]
    assert set(get_shis(_cell(rows), env=env)) == {0, 1, 2, 3}


class _FlakyModel:
    """Stand-in for a Gurobi model whose solves fail (NUMERIC) a given number of times."""

    def __init__(self, failures: int) -> None:
        from types import SimpleNamespace

        self.failures, self.status, self.resets = failures, 12, 0
        self.params = SimpleNamespace(ScaleFlag=None)
        self.seen: list[object] = []

    def reset(self) -> None:
        self.resets += 1

    def optimize(self) -> None:
        self.seen.append(self.params.ScaleFlag)
        self.status = 12 if len(self.seen) <= self.failures else 2


@pytest.mark.parametrize("failures", [0, 1])
def test_lp_cold_retry_uses_automatic_scaling_and_restores_settings(failures: int) -> None:
    from relucent.geometry import calculations as C

    m = _FlakyModel(failures)
    assert C._cold_retry(cast(Any, m)) == [12, 2 if failures == 0 else 12]
    assert m.resets == 1 and m.seen == [-1]  # one cold solve, under automatic scaling
    assert m.params.ScaleFlag == cfg.GUROBI_SHI_SCALE_FLAG


@pytest.mark.parametrize(("d", "n", "scale"), [(2, 7, 1e-4), (3, 6, 1e-4), (2, 7, 1e-8), (2, 7, 1e-12), (3, 6, 1e-8)])
def test_nearly_concurrent_hyperplanes_are_found_or_raise(d: int, n: int, scale: float) -> None:
    """Small biases put every hyperplane through a tiny neighbourhood of the origin. The
    arrangement is still generic, so BFS must find all sum_k C(n, k) regions; only slivers below
    the LP tolerances (biases below ~1e-6) may instead raise, being too thin to decide in float64."""
    import math

    import torch
    import torch.nn as nn

    from relucent import Complex

    for seed in range(3):
        torch.manual_seed(seed)
        lin = nn.Linear(d, n).double()
        with torch.no_grad():
            lin.bias.copy_(torch.randn(n, dtype=torch.float64) * scale)
        cx = Complex(nn.Sequential(lin, nn.ReLU()))
        try:
            cx.bfs(nworkers=1, start=torch.randn(d, dtype=torch.float64))
        except AmbiguousGeometryError:
            if scale >= 1e-4:
                raise
            continue
        assert len(cx) == sum(math.comb(n, k) for k in range(d + 1))


def _exact_rows(rows: list[list[float]]) -> list[list[Fraction]]:
    return [[Fraction(v) for v in r] for r in rows]


@pytest.mark.parametrize(
    ("extra", "facet"),
    [
        ([1.0, 1.0, -2.0], False),  # x + y <= 2 touches the unit square only at its corner (1, 1)
        ([1.0, 1.0, -3.0], False),  # misses the square entirely
        ([1.0, 1.0, -1.5], True),  # cuts the corner off
        ([1.0, 0.0, -1.0], False),  # duplicates the side x <= 1: neither copy alone is irredundant
        ([-1.0, 0.0, -2.0], False),  # parallel to x >= 0, outside it
    ],
)
def test_exact_facet_by_simplex_on_square(extra: list[float], facet: bool) -> None:
    from relucent._internal import exact

    rows = _exact_rows([*_rectangle(), extra])
    start = np.array([0.25, 0.25])
    assert exact.exact_facet_by_simplex(rows, 4, [], start) is facet
    for side in range(4):
        duplicated = extra == [1.0, 0.0, -1.0] and side == 1
        assert exact.exact_facet_by_simplex(rows, side, [], start) is (not duplicated)


def test_exact_facet_by_simplex_unbounded_and_face() -> None:
    from relucent._internal import exact

    wedge = _exact_rows([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, -1.0, -1.0]])  # x, y >= 0, x <= y + 1
    assert exact.exact_facet_by_simplex(wedge, 2, [], np.array([0.5, 0.5])) is True  # recession ray
    # On the face y = 0 (row 1 held at zero) of the wedge, x <= 1 is an endpoint, x >= 0 the other.
    assert exact.exact_facet_by_simplex(wedge, 2, [1], np.array([0.5, 1e-17])) is True
    assert exact.exact_facet_by_simplex(wedge, 0, [1], np.array([0.5, 1e-17])) is True
    # A start outside the cell is refused rather than trusted.
    assert exact.exact_facet_by_simplex(wedge, 2, [], np.array([-1.0, 0.5])) is None


class _RelaxedLPsFail:
    """A real Gurobi model that reports NUMERIC for every solve after the first (the cell's own
    feasibility LP), as when the solver fails on every relaxed SHI LP under every setting."""

    def __init__(self, name: str, env: Any) -> None:
        import gurobipy

        self._model = gurobipy.Model(name, env)
        self._solves = 0

    def __getattr__(self, attr: str) -> object:
        return getattr(self._model, attr)

    def optimize(self) -> None:
        self._model.optimize()
        self._solves += 1

    @property
    def status(self) -> int:
        return 12 if self._solves > 1 else int(self._model.status)


@pytest.mark.parametrize("extra", [[1.0, 1.0, -2.0], [1.0, 1.0, -1.5], [1.0, 0.0, -1.0 - 1e-9]])
def test_failed_lp_raises(env, monkeypatch: pytest.MonkeyPatch, extra) -> None:
    from relucent.geometry import calculations as C

    cell = _cell([*_rectangle(), extra])
    monkeypatch.setattr(C, "Model", _RelaxedLPsFail)
    with pytest.raises(AmbiguousGeometryError, match="LP solver failure"):
        get_shis(cell, env=env)


def test_get_shis_strict_is_deprecated(env) -> None:
    with pytest.warns(FutureWarning, match="strict"):
        assert set(get_shis(_cell(_rectangle()), env=env, strict=True)) == {0, 1, 2, 3}
