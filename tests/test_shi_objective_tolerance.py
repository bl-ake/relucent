"""Adversarial tests for TOL_SHI_OBJECTIVE vs Chebyshev inradius.

These tests pin down where the SHI objective threshold bites:

* **Chebyshev inradius** (``solve_radius``) measures ball size in the cell.
* **SHI LP objective** measures penetration along a face normal after relaxing
  that face by ``push_size``.  It scales with ``push_size``, not inradius.

The thin-slab constructions below have inradius ``≈ height/2`` while face SHI
objectives with default ``push_size=1`` stay ``≈ 1``.  That separates the two
quantities and shows small inradius alone does not force SHI rejection.

The ``push_size`` sweep documents the false-negative regime: when
``push_size`` is tuned to the slab height, horizontal-face objectives fall
near ``push_size`` and can drop below ``TOL_SHI_OBJECTIVE``.

TODO: Test 2D Complexes
"""

from __future__ import annotations

import numpy as np
import pytest
from gurobipy import GRB, Model

import relucent.config as cfg
from relucent import Polyhedron
from relucent.calculations import (
    DegenerateHalfspaceInfeasibility,
    _affine_null_basis,
    _drop_degenerate_halfspaces_tracked,
    _remap_zero_indices,
    get_shis,
    solve_radius,
)
from relucent.search import _enforce_min_search_inradius, search_calculations
from relucent.utils import get_env


def _make_1d_strip_net(eps: float):
    """[1,2,1] MLP with middle cell width eps (Chebyshev inradius eps/2)."""
    from collections import OrderedDict

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
    return convert(
        TorchMLP(
            OrderedDict([("fc0", fc1), ("relu0", torch.nn.ReLU()), ("fc1", fc2)]),
            [1, 2, 1],
        )
    )


def _poly_from_halfspaces(*, halfspaces: np.ndarray, zero_rows: set[int] | None = None) -> Polyhedron:
    halfspaces = np.asarray(halfspaces, dtype=np.float64)
    m = halfspaces.shape[0]
    zero_rows = zero_rows or set()
    ss = np.ones((1, m), dtype=np.int8)
    for i in zero_rows:
        ss[0, int(i)] = 0
    return Polyhedron(None, ss, halfspaces=halfspaces)


def _thin_rectangle_halfspaces(*, width: float = 1.0, height: float) -> np.ndarray:
    """Axis-aligned rectangle [0, width] x [0, height] as a^T x + b <= 0 rows."""
    return np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, -width],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, -height],
        ],
        dtype=np.float64,
    )


def _shi_objective_for_face(
    poly: Polyhedron,
    face_i: int,
    *,
    env,
    push_size: float = 1.0,
) -> float:
    """Run the single-face SHI maximization LP (same logic as ``get_shis``)."""
    hs_np = poly.halfspaces_np
    A_orig = hs_np[:, :-1]
    b_orig = hs_np[:, -1:]
    amb_d = int(A_orig.shape[1])

    try:
        hs_work, old_to_new = _drop_degenerate_halfspaces_tracked(hs_np)
    except DegenerateHalfspaceInfeasibility:
        return float("nan")

    n_work = int(hs_work.shape[0])
    if n_work == 0:
        return float("nan")

    A_work = hs_work[:, :-1]
    b_work = hs_work[:, -1:]
    zero_eff = _remap_zero_indices(poly.zero_indices, old_to_new)

    if zero_eff is not None and zero_eff.size > 0:
        x0, null_basis, ineq_mask = _affine_null_basis(hs_work, zero_eff)
        k = int(null_basis.shape[1])
        if k == 0:
            return float("nan")
        ineq_rows = np.flatnonzero(ineq_mask)
        ineq_half = hs_work[ineq_mask]
        a_red = ineq_half[:, :-1] @ null_basis
        b_red = ineq_half[:, :-1] @ x0 + ineq_half[:, -1:]
        row_lp = {int(r): j for j, r in enumerate(ineq_rows.tolist())}
    else:
        x0 = np.zeros((amb_d, 1), dtype=np.float64)
        null_basis = np.eye(amb_d, dtype=np.float64)
        ineq_rows = np.arange(n_work, dtype=np.intp)
        a_red = A_work
        b_red = b_work
        row_lp = {int(r): r for r in range(n_work)}
        k = amb_d

    wi = int(old_to_new[face_i])
    if wi < 0:
        return float("nan")
    j = row_lp.get(wi)
    if j is None:
        return float("nan")

    model = Model("SHI_single", env)
    z = model.addMVar((k, 1), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")
    constrs = model.addConstr(a_red @ z <= -b_red, name="hyperplanes")
    model.optimize()
    if model.status != GRB.OPTIMAL:
        model.close()
        raise RuntimeError(f"initial SHI interior solve failed: status={model.status}")

    constrs[j].setAttr("RHS", -b_red[j, 0] + push_size)
    ai = A_orig[face_i : face_i + 1, :].T
    c_obj = null_basis.T @ ai
    const_obj = float((ai.T @ x0).item() + b_orig[face_i, 0])
    model.setObjective(c_obj.T @ z + const_obj, GRB.MAXIMIZE)
    model.params.BestObjStop = cfg.GUROBI_SHI_BEST_OBJ_STOP
    model.params.BestBdStop = cfg.GUROBI_SHI_BEST_BD_STOP
    model.optimize()

    if model.status not in (GRB.OPTIMAL, GRB.USER_OBJ_LIMIT):
        model.close()
        raise RuntimeError(f"SHI face {face_i} solve failed: status={model.status}")
    obj = float(model.objVal)
    model.close()
    return obj


@pytest.fixture
def env():
    return get_env()


class TestShiObjectiveVsInradius:
    """Thin real cells: inradius can be far below TOL_SHI_OBJECTIVE while SHIs stay valid."""

    @pytest.mark.parametrize("height", [1e-8, 1e-10, 1e-12])
    def test_thin_slab_inradius_below_tol_still_has_all_faces(self, env, height: float):
        hs = _thin_rectangle_halfspaces(height=height)
        poly = _poly_from_halfspaces(halfspaces=hs)

        _center, inradius = solve_radius(env, hs)
        assert inradius is not None
        assert inradius < cfg.TOL_SHI_OBJECTIVE
        assert np.isclose(inradius, height / 2.0, rtol=0.0, atol=max(1e-14, height * 10))

        shis = get_shis(poly, env=env, push_size=1.0)
        assert set(shis) == {0, 1, 2, 3}

        # Horizontal faces (2, 3): objective ≈ push_size.  Vertical faces: same order.
        for face in (2, 3):
            obj = _shi_objective_for_face(poly, face, env=env, push_size=1.0)
            assert obj > cfg.TOL_SHI_OBJECTIVE
            assert obj > 10 * height

    def test_shi_objective_scales_with_push_size_not_inradius(self, env):
        height = 1e-7
        hs = _thin_rectangle_halfspaces(height=height)
        poly = _poly_from_halfspaces(halfspaces=hs)

        push_small = 3.0 * height
        obj_small = _shi_objective_for_face(poly, 2, env=env, push_size=push_small)
        assert np.isclose(obj_small, push_small, rtol=0.05, atol=1e-12)

        push_large = 1.0
        obj_large = _shi_objective_for_face(poly, 2, env=env, push_size=push_large)
        assert np.isclose(obj_large, push_large, rtol=0.05, atol=1e-9)
        assert obj_large > 1e6 * height


class TestShiObjectiveThresholdBoundary:
    """Where ``TOL_SHI_OBJECTIVE`` accepts or rejects a genuine thin face."""

    def test_threshold_crossover_rejects_when_push_size_below_tol(self, env, monkeypatch):
        height = 1e-9
        push_size = 2.0 * height
        hs = _thin_rectangle_halfspaces(height=height)
        poly = _poly_from_halfspaces(halfspaces=hs)

        obj = _shi_objective_for_face(poly, 2, env=env, push_size=push_size)
        assert np.isclose(obj, push_size, rtol=0.1, atol=1e-12)
        assert obj < cfg.TOL_SHI_OBJECTIVE

        monkeypatch.setattr(cfg, "TOL_SHI_OBJECTIVE", push_size / 10.0)
        shis_low = get_shis(poly, env=env, push_size=push_size)
        assert 2 in shis_low

        monkeypatch.setattr(cfg, "TOL_SHI_OBJECTIVE", push_size * 10.0)
        shis_high = get_shis(poly, env=env, push_size=push_size)
        assert 2 not in shis_high

    def test_tol_shi_objective_not_above_gurobi_stop(self):
        # SHI cert threshold should not exceed the Gurobi early-stop floor.
        assert cfg.TOL_SHI_OBJECTIVE <= cfg.GUROBI_SHI_BEST_OBJ_STOP


class TestShiFalsePositiveRejection:
    """Non-faces whose SHI objective is numerical noise must stay rejected."""

    def test_loose_parallel_constraint_not_shi(self, env):
        # Region is y <= 0; a parallel loose wall y <= 100 is redundant.
        hs = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],  # tight: y <= 0
                [0.0, 1.0, -100.0],  # loose: y <= 100
            ],
            dtype=np.float64,
        )
        poly = _poly_from_halfspaces(halfspaces=hs)
        shis = get_shis(poly, env=env)
        assert 2 in shis
        assert 3 not in shis
        obj_loose = _shi_objective_for_face(poly, 3, env=env, push_size=1.0)
        assert obj_loose <= cfg.TOL_SHI_OBJECTIVE

    def test_near_duplicate_face_objective_below_tol(self, env):
        # Two almost-coincident top walls; only the tighter one is a face.
        # delta must be large enough to survive float64 and CAREFUL_MODE duplicate detection.
        delta = 1e-4
        hs = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, -1.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, -1.0],  # y <= 1
                [0.0, 1.0, -(1.0 + delta)],  # y <= 1 + delta (looser)
            ],
            dtype=np.float64,
        )
        poly = _poly_from_halfspaces(halfspaces=hs)
        shis = get_shis(poly, env=env)
        assert 3 in shis
        assert 4 not in shis
        obj_dup = _shi_objective_for_face(poly, 4, env=env, push_size=1.0)
        assert obj_dup <= cfg.TOL_SHI_OBJECTIVE


class TestChebyshevVsShiIndependence:
    """solve_radius can fail or return ~0 while get_shis still works on the same geometry."""

    def test_solve_radius_raises_while_get_shis_succeeds(self, env):
        # Extreme thinness: Chebyshev LP returns y≈0 (raises), but faces are real.
        height = 1e-14
        hs = _thin_rectangle_halfspaces(height=height)
        poly = _poly_from_halfspaces(halfspaces=hs)
        with pytest.raises(ValueError, match="Inradius"):
            solve_radius(env, hs)
        assert set(get_shis(poly, env=env)) == {0, 1, 2, 3}

    def test_zero_inradius_segment_still_has_two_shis(self, env):
        # Degenerate 1D cell embedded as a segment with zero-width limit is different;
        # use a genuinely thin but positive-width segment.
        height = 1e-11
        hs = _thin_rectangle_halfspaces(height=height)
        poly = _poly_from_halfspaces(halfspaces=hs)
        _center, inradius = solve_radius(env, hs)
        assert inradius is not None and inradius < 1e-10
        assert len(get_shis(poly, env=env)) == 4


class TestMinSearchInradiusGuard:
    """Search-time guard when Chebyshev inradius falls below ``MIN_SEARCH_INRADIUS``."""

    def test_enforce_raises_below_half_shi_tolerance(self, env):
        height = 1e-10
        hs = _thin_rectangle_halfspaces(height=height)
        poly = _poly_from_halfspaces(halfspaces=hs)
        _center, inradius = solve_radius(env, hs)
        assert inradius is not None
        assert inradius < cfg.MIN_SEARCH_INRADIUS
        poly._finite = True
        poly._finite_computed = True
        poly._center = _center
        poly._inradius = inradius
        with pytest.raises(ValueError, match="MIN_SEARCH_INRADIUS"):
            _enforce_min_search_inradius(poly, env=env)

    def test_enforce_passes_at_tolerance_scale(self, env):
        height = 1e-7
        hs = _thin_rectangle_halfspaces(height=height)
        poly = _poly_from_halfspaces(halfspaces=hs)
        _center, inradius = solve_radius(env, hs)
        assert inradius is not None
        assert inradius >= cfg.MIN_SEARCH_INRADIUS
        poly._finite = True
        poly._finite_computed = True
        poly._center = _center
        poly._inradius = inradius
        _enforce_min_search_inradius(poly, env=env)

    def test_enforce_skips_zero_cells(self, env):
        hs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        poly = _poly_from_halfspaces(halfspaces=hs, zero_rows={0, 1})
        poly._finite = True
        poly._finite_computed = True
        poly._inradius = 0.0
        _enforce_min_search_inradius(poly, env=env)

    def test_1d_relu_strip_triggers_search_worker_guard(self, env):
        """Two parallel ReLU thresholds spaced by eps give a strip with inradius eps/2."""
        from relucent import Complex
        from relucent.worker_context import worker_context_scope

        eps = 1e-9
        assert eps / 2 < cfg.MIN_SEARCH_INRADIUS

        net = _make_1d_strip_net(eps)
        with worker_context_scope(net) as ctx:
            ctx.env = env
            thin = Complex(net).add_point(np.array([eps / 2], dtype=np.float64))
            assert thin.inradius is not None
            assert thin.inradius < cfg.MIN_SEARCH_INRADIUS

            result = search_calculations((thin.ss_np, 0, 0))
            assert not isinstance(result[0], Polyhedron)
            assert "MIN_SEARCH_INRADIUS" in str(result[0])


class TestStripNetworkBfs:
    """End-to-end BFS on the 1D strip network (stall + perturbation regressions)."""

    @pytest.fixture(autouse=True)
    def _disable_careful_mode(self, monkeypatch):
        # Parallel ReLU thresholds produce nearly duplicate halfspace rows at fat
        # start regions; the strip geometry is intentional for inradius testing.
        monkeypatch.setattr(cfg, "CAREFUL_MODE", False)
        # Spawned workers (macOS default) re-import config and read RELUCENT_* env
        # vars; CI sets RELUCENT_CAREFUL_MODE=1, so patch the env as well as cfg.
        monkeypatch.setenv("RELUCENT_CAREFUL_MODE", "0")

    def test_bfs_rejects_thin_strip_neighbor_and_completes(self):
        from collections import deque

        from relucent import Complex
        from relucent.search import searcher
        from relucent.utils import BlockingQueue

        eps = 1e-9
        net = _make_1d_strip_net(eps)
        cplx = Complex(net)
        # At eps=1e-9 only the middle strip has SHIs; fat regions at x=-1 have none.
        # Pre-queue a thin-cell task so the pool worker hits _enforce_min_search_inradius.
        thin = cplx.add_point(np.array([eps / 2], dtype=np.float64))
        queue = BlockingQueue(
            queue_class=deque,
            pop=deque.popleft,
            push=deque.append,
        )
        queue.push((thin.ss_np.copy(), 0, 1, cplx.ssm[thin.ss_np]))
        info = searcher(
            cplx,
            start=thin,
            queue=queue,
            max_polys=20,
            nworkers=1,
            verbose=0,
            verify=False,
        )

        assert info["Complete"] is True
        assert len(info["Bad SHI Computations"]) >= 1
        assert any("MIN_SEARCH_INRADIUS" in str(entry[-1]) for entry in info["Bad SHI Computations"])

    def test_bfs_completes_after_strip_widening_perturbation(self):
        from relucent import Complex

        eps = 1e-7
        assert eps / 2 >= cfg.MIN_SEARCH_INRADIUS

        net = _make_1d_strip_net(eps)
        cplx = Complex(net)
        info = cplx.bfs(start=np.array([-1.0], dtype=np.float64), max_polys=20, nworkers=1, verbose=0, verify=False)

        assert info["Complete"] is True
        assert not any("MIN_SEARCH_INRADIUS" in str(entry[-1]) for entry in info["Bad SHI Computations"])
        assert len(cplx) >= 2


def test_shi_objective_diagnostic_table(env):
    """Print per-face objectives for a thin slab (visible with ``pytest -s``)."""
    height = 1e-8
    hs = _thin_rectangle_halfspaces(height=height)
    poly = _poly_from_halfspaces(halfspaces=hs)
    _center, inradius = solve_radius(env, hs)

    rows: list[str] = []
    for face in range(hs.shape[0]):
        obj = _shi_objective_for_face(poly, face, env=env, push_size=1.0)
        accepted = obj > cfg.TOL_SHI_OBJECTIVE
        rows.append(f"face={face} obj={obj:.6e} accepted={accepted}")

    print(f"inradius={inradius:.6e} TOL_SHI_OBJECTIVE={cfg.TOL_SHI_OBJECTIVE:.6e}")
    print("\n".join(rows))
    assert all(_shi_objective_for_face(poly, f, env=env) > cfg.TOL_SHI_OBJECTIVE for f in range(4))
