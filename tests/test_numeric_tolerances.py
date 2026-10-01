"""Tests for relucent.config.numeric_tolerances."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

import relucent.config as cfg
from relucent import convert, mlp
from relucent.config import update_settings
from relucent.config.numeric_tolerances import apply_tolerances, compute_tolerances


def test_ordering_invariants() -> None:
    tol = compute_tolerances()
    for name, value in tol.items():
        assert value == value
        if name == "GUROBI_SHI_BEST_BD_STOP":
            assert value < 0
        else:
            assert value > 0
    assert tol["TOL_INTERIOR_VERIFY"] >= tol["TOL_HALFSPACE_CONTAINMENT"]
    assert tol["TOL_SHI_OBJECTIVE"] <= tol["GUROBI_SHI_BEST_OBJ_STOP"]
    assert tol["GUROBI_SHI_BEST_BD_STOP"] <= -tol["GUROBI_SHI_BEST_OBJ_STOP"]
    assert tol["MIN_SEARCH_INRADIUS"] == tol["TOL_SHI_OBJECTIVE"] / 2.0


def test_monotonicity_with_ambient_dim() -> None:
    tol_low = compute_tolerances(ambient_dim=2)
    tol_high = compute_tolerances(ambient_dim=500)
    assert tol_high["TOL_HALFSPACE_CONTAINMENT"] >= tol_low["TOL_HALFSPACE_CONTAINMENT"]


def test_containment_probe_accepts_feasible_point() -> None:
    tol = compute_tolerances(max_coord=1.0, ambient_dim=2)["TOL_HALFSPACE_CONTAINMENT"]
    a = np.array([1.0, 0.0], dtype=np.float64)
    x = np.array([0.0, 0.0], dtype=np.float64)
    b = -tol / 4.0
    assert float(a @ x + b) <= tol


def test_apply_tolerances_updates_config() -> None:
    snapshot = {name: getattr(cfg, name) for name in compute_tolerances()}
    try:
        expected = compute_tolerances(max_coord=1.0, ambient_dim=2)["TOL_HALFSPACE_CONTAINMENT"]
        apply_tolerances(max_coord=1.0, ambient_dim=2, respect_env=False)
        assert expected == cfg.TOL_HALFSPACE_CONTAINMENT
    finally:
        update_settings(**snapshot)


def test_apply_tolerances_with_net() -> None:
    net = convert(mlp([2, 4, 1]))
    snapshot = {name: getattr(cfg, name) for name in compute_tolerances()}
    try:
        expected = compute_tolerances(net=net)["TOL_HALFSPACE_CONTAINMENT"]
        apply_tolerances(net=net, respect_env=False)
        assert expected == cfg.TOL_HALFSPACE_CONTAINMENT
    finally:
        update_settings(**snapshot)


def test_apply_respects_per_key_env_override() -> None:
    snapshot = {name: getattr(cfg, name) for name in compute_tolerances()}
    try:
        os.environ["RELUCENT_TOL_HALFSPACE_CONTAINMENT"] = "0.42"
        update_settings(TOL_HALFSPACE_CONTAINMENT=0.42)
        apply_tolerances(respect_env=True)
        assert cfg.TOL_HALFSPACE_CONTAINMENT == 0.42
        assert compute_tolerances()["TOL_DEAD_RELU"] == cfg.TOL_DEAD_RELU
    finally:
        os.environ.pop("RELUCENT_TOL_HALFSPACE_CONTAINMENT", None)
        update_settings(**snapshot)


def test_import_bootstrap_runs_by_default() -> None:
    code = (
        "import relucent; "
        "import relucent.config as cfg; "
        "from relucent.config.numeric_tolerances import compute_tolerances; "
        "tol = compute_tolerances(); "
        "assert cfg.TOL_HALFSPACE_CONTAINMENT == tol['TOL_HALFSPACE_CONTAINMENT']; "
        "assert cfg.MIN_SEARCH_INRADIUS == tol['MIN_SEARCH_INRADIUS']"
    )
    env = os.environ.copy()
    env.pop("RELUCENT_SKIP_NUMERIC_BOOTSTRAP", None)
    env.pop("RELUCENT_TOL_HALFSPACE_CONTAINMENT", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_complex_auto_tolerances_false_skips_network_tune(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def _record(*, net=None, **_kwargs: object) -> None:
        calls.append(net)

    monkeypatch.setattr("relucent.config.numeric_tolerances.apply_tolerances", _record)
    from relucent.core.complex import Complex

    Complex(convert(mlp([2, 3, 1])), auto_tolerances=False)
    assert calls == []
    Complex(convert(mlp([2, 3, 1])), auto_tolerances=True)
    assert len(calls) == 1


def test_boundary_mip_bound_margin_default() -> None:
    assert cfg.BOUNDARY_MIP_BOUND_MARGIN == 5.0


def test_shi_proof_tolerance_covers_lp_feasibility_on_small_weight_networks() -> None:
    """SHI proof points come from an LP, so slack can hit gurobi's feasibility tol."""
    from relucent.model.model import LinearLayer

    net = convert(mlp([3, 8, 8, 1]))
    for layer in net.layers.values():
        if isinstance(layer, LinearLayer):
            layer.weight = layer.weight * 1e-3
            layer.bias = layer.bias * 1e-3
    tol = compute_tolerances(net=net)
    gurobi_feasibility_tol = 1e-6
    assert tol["TOL_SHI_HYPERPLANE"] >= (3 + 1) * gurobi_feasibility_tol


def test_shi_objective_stays_below_push_size_for_large_input_box() -> None:
    """The SHI objective is at most push_size (1), so an acceptance threshold near it rejects every facet."""
    small = compute_tolerances(max_coord=1.0)
    large = compute_tolerances(max_coord=1e15)
    assert small["TOL_SHI_OBJECTIVE"] < 1e-6
    assert large["TOL_SHI_OBJECTIVE"] <= 1e-6
    assert large["MIN_SEARCH_INRADIUS"] <= 5e-7


def test_bfs_explores_deep_large_weight_network() -> None:
    """Uncapped, this net's TOL_SHI_OBJECTIVE is 15 and BFS stopped at its start region."""
    import torch

    from relucent.core.complex import Complex
    from relucent.model.model import LinearLayer

    torch.manual_seed(0)
    net = convert(mlp([2, 6, 6, 6, 6, 6, 1]))
    for layer in net.layers.values():
        if isinstance(layer, LinearLayer):
            layer.weight = layer.weight * 35.0
            layer.bias = layer.bias * 35.0
    saved = {name: getattr(cfg, name) for name in [*compute_tolerances(), "CAREFUL_MODE"]}
    try:
        # CAREFUL_MODE's checks are not what this test is about; keep it to the search itself.
        update_settings(CAREFUL_MODE=False)
        cplx = Complex(net, auto_tolerances=True)
        assert cfg.TOL_SHI_OBJECTIVE <= 1e-6
        cplx.bfs(start=torch.zeros(2, dtype=torch.float64) + 0.123, verbose=False)
    finally:
        update_settings(**saved)
    ss = np.asarray(cplx.point2ss(np.random.default_rng(0).standard_normal((512, 2))))
    assert len(cplx) > 1
    assert all(s in cplx for s in ss)


def test_tolerances_do_not_scale_with_the_network() -> None:
    """No setting follows a network's weights or depth: that one-number-per-network scaling was the defect."""
    from relucent.model.model import LinearLayer

    small = convert(mlp([2, 8, 8, 1]))
    big = convert(mlp([2, 8, 8, 1]))
    for layer in big.layers.values():
        if isinstance(layer, LinearLayer):
            layer.weight = layer.weight * 1e3
            layer.bias = layer.bias * 1e3
    a, b = compute_tolerances(net=small), compute_tolerances(net=big)
    for name in a:
        if name != "BOUNDARY_MIP_EPS":
            assert a[name] == b[name], name
    assert compute_tolerances(max_coord=1.0) == compute_tolerances(max_coord=1e15)


def test_genericity_tells_vertices_apart_by_their_own_error() -> None:
    """Two distinct segment endpoints 1e-9 apart are distinct; the old 2e-6 grid merged them."""
    from relucent.core.poly import Polyhedron
    from relucent.verify.certify import verify_arrangement_genericity

    # Two collinear unit-length segments on the x-axis meeting near x = 0: [-1, -1e-9] and [0, 1].
    left = Polyhedron(
        None,
        np.array([[0, 1, 1]], dtype=np.int8),
        halfspaces=np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, -1.0], [1.0, 0.0, 1e-9]]),
        dim=1,
        _ambient_dim=2,
    )
    right = Polyhedron(
        None,
        np.array([[0, 1, 1, 1]], dtype=np.int8),
        halfspaces=np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        dim=1,
        _ambient_dim=2,
    )
    verify_arrangement_genericity([left, right])
