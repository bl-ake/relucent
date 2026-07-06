"""Tests for relucent.numeric_tolerances."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

import relucent.config as cfg
from relucent import convert, mlp
from relucent.config import update_settings
from relucent.numeric_tolerances import apply_tolerances, compute_tolerances


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


def test_monotonicity_with_max_coord() -> None:
    tol_small = compute_tolerances(max_coord=1e6)
    tol_large = compute_tolerances(max_coord=1e7)
    assert tol_large["TOL_HALFSPACE_CONTAINMENT"] >= tol_small["TOL_HALFSPACE_CONTAINMENT"]
    assert tol_large["TOL_VERIFY_AB_ATOL"] >= tol_small["TOL_VERIFY_AB_ATOL"]


def test_monotonicity_with_ambient_dim() -> None:
    tol_low = compute_tolerances(ambient_dim=2)
    tol_high = compute_tolerances(ambient_dim=500)
    assert tol_high["TOL_HALFSPACE_CONTAINMENT"] >= tol_low["TOL_HALFSPACE_CONTAINMENT"]


def test_network_tolerances_use_tighter_coord_scale() -> None:
    net = convert(mlp([2, 8, 1]))
    global_tol = compute_tolerances()
    net_tol = compute_tolerances(net=net)
    assert net_tol["TOL_HALFSPACE_CONTAINMENT"] < global_tol["TOL_HALFSPACE_CONTAINMENT"]


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
        "from relucent.numeric_tolerances import compute_tolerances; "
        "assert cfg.TOL_HALFSPACE_CONTAINMENT == compute_tolerances()['TOL_HALFSPACE_CONTAINMENT']"
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

    monkeypatch.setattr("relucent.numeric_tolerances.apply_tolerances", _record)
    from relucent.complex import Complex

    Complex(convert(mlp([2, 3, 1])), auto_tolerances=False)
    assert calls == []
    Complex(convert(mlp([2, 3, 1])), auto_tolerances=True)
    assert len(calls) == 1
