"""Tests for relucent.config and relucent.config.advanced."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

import relucent.config as cfg
from relucent import convert, torch_mlp
from relucent.config import update_settings


def _settings() -> dict[str, object]:
    public = {name: getattr(cfg, name) for name in cfg.__all__ if name not in ("advanced", "update_settings")}
    return public | {name: getattr(cfg.advanced, name) for name in cfg.advanced.__all__}


def _run(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    full_env = {k: v for k, v in os.environ.items() if not k.startswith("RELUCENT_")} | env
    return subprocess.run([sys.executable, "-c", code], env=full_env, check=False, capture_output=True, text=True)


def test_import_and_complex_leave_settings_alone() -> None:
    """Neither ``import relucent`` nor ``Complex(net)`` rewrites settings a user has made."""
    code = (
        "import relucent, relucent.config as cfg; "
        "assert cfg.POLYHEDRON_BOUND_MARGIN == 5.0, cfg.POLYHEDRON_BOUND_MARGIN; "
        "cfg.update_settings(MAX_RADIUS=7.0, POLYHEDRON_BOUND_MARGIN=0.5); "
        "relucent.Complex(relucent.torch_mlp([2, 4, 1])); "
        "assert (cfg.MAX_RADIUS, cfg.POLYHEDRON_BOUND_MARGIN) == (7.0, 0.5)"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr


def test_complex_does_not_change_any_setting() -> None:
    before = _settings()
    from relucent.core.complex import Complex

    Complex(convert(torch_mlp([2, 3, 1])))
    assert _settings() == before


def test_update_settings_routes_advanced_settings() -> None:
    saved = cfg.advanced.ASTAR_BIAS_WEIGHT
    try:
        update_settings(ASTAR_BIAS_WEIGHT=0.25)
        assert cfg.advanced.ASTAR_BIAS_WEIGHT == 0.25
        assert "ASTAR_BIAS_WEIGHT" not in vars(cfg)
    finally:
        update_settings(ASTAR_BIAS_WEIGHT=saved)


def test_update_settings_rejects_unknown_and_invalid() -> None:
    with pytest.raises(TypeError, match="Unknown config keys"):
        update_settings(NOT_A_SETTING=1)
    with pytest.raises(ValueError, match="QHULL_MODE"):
        update_settings(QHULL_MODE="SOMETIMES")
    assert cfg.QHULL_MODE == "IGNORE"


def test_env_vars_use_the_relucent_prefix() -> None:
    code = "from relucent.config import advanced as a; print(a.ASTAR_BIAS_WEIGHT, a.GUROBI_SHI_SCALE_FLAG)"
    proc = _run(code, RELUCENT_ASTAR_BIAS_WEIGHT="0.5", RELUCENT_GUROBI_SHI_SCALE_FLAG="1")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["0.5", "1"]
    proc = _run("import relucent.config", RELUCENT_QHULL_MODE="SOMETIMES")
    assert proc.returncode != 0 and "QHULL_MODE" in proc.stderr


def test_polyhedron_bound_margin_default() -> None:
    assert cfg.POLYHEDRON_BOUND_MARGIN == 5.0


def test_bfs_explores_deep_large_weight_network() -> None:
    """BFS must get past its start region on a deep net with large weights."""
    import torch

    from relucent.core.complex import Complex
    from relucent.model.model import LinearLayer

    torch.manual_seed(0)
    net = convert(torch_mlp([2, 6, 6, 6, 6, 6, 1]))
    for layer in net.layers.values():
        if isinstance(layer, LinearLayer):
            layer.weight = layer.weight * 35.0
            layer.bias = layer.bias * 35.0
    saved = cfg.CAREFUL_MODE
    try:
        # CAREFUL_MODE's checks are not what this test is about; keep it to the search itself.
        update_settings(CAREFUL_MODE=False)
        cplx = Complex(net)
        cplx.bfs(start=torch.zeros(2, dtype=torch.float64) + 0.123, verbose=False)
    finally:
        update_settings(CAREFUL_MODE=saved)
    ss = np.asarray(cplx.point2ss(np.random.default_rng(0).standard_normal((512, 2))))
    assert len(cplx) > 1
    assert all(s in cplx for s in ss)


def test_genericity_tells_vertices_apart_by_their_own_error() -> None:
    """Two distinct segment endpoints 1e-9 apart are distinct; the old 2e-6 grid merged them."""
    from relucent.core.poly import Polyhedron
    from relucent.verify.certify import verify_arrangement_genericity

    # Two collinear unit-length segments on the x-axis meeting near x = 0: [-1, -1e-9] and [0, 1].
    left = Polyhedron(
        None,
        np.array([[0, 1, 1]], dtype=np.int8),
        halfspaces=np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, -1.0], [1.0, 0.0, 1e-9]]),
        ambient_dim=2,
    )
    right = Polyhedron(
        None,
        np.array([[0, 1, 1, 1]], dtype=np.int8),
        halfspaces=np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        ambient_dim=2,
    )
    verify_arrangement_genericity([left, right])
