"""Tests for relucent.config and relucent.config.advanced."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

import relucent.config as cfg
from relucent import convert, torch_mlp
from relucent._internal.network_scale import boundary_mip_eps
from relucent.config import update_settings
from relucent.model.model import LinearLayer


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
        "assert cfg.BOUNDARY_MIP_EPS is None, cfg.BOUNDARY_MIP_EPS; "
        "cfg.update_settings(MAX_RADIUS=7.0, BOUNDARY_MIP_EPS=0.5); "
        "relucent.Complex(relucent.torch_mlp([2, 4, 1])); "
        "assert (cfg.MAX_RADIUS, cfg.BOUNDARY_MIP_EPS) == (7.0, 0.5)"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr


def test_complex_does_not_change_any_setting() -> None:
    before = _settings()
    from relucent.core.complex import Complex

    Complex(convert(torch_mlp([2, 3, 1])))
    assert _settings() == before


def test_boundary_mip_eps_floor_and_scaling() -> None:
    small = convert(torch_mlp([2, 4, 1]))
    assert boundary_mip_eps(small) == 2e-4
    big = convert(torch_mlp([2, 8, 8, 1]))
    for layer in big.layers.values():
        if isinstance(layer, LinearLayer):
            layer.weight = layer.weight * 1e3
            layer.bias = layer.bias * 1e3
    assert boundary_mip_eps(big) > 2e-4


def test_update_settings_routes_advanced_settings() -> None:
    saved = cfg.advanced.BOUNDARY_MIP_EXCLUSION_WORKERS
    try:
        update_settings(BOUNDARY_MIP_EXCLUSION_WORKERS=3)
        assert cfg.advanced.BOUNDARY_MIP_EXCLUSION_WORKERS == 3
        assert "BOUNDARY_MIP_EXCLUSION_WORKERS" not in vars(cfg)
    finally:
        update_settings(BOUNDARY_MIP_EXCLUSION_WORKERS=saved)


def test_update_settings_rejects_unknown_and_invalid() -> None:
    with pytest.raises(TypeError, match="Unknown config keys"):
        update_settings(NOT_A_SETTING=1)
    with pytest.raises(ValueError, match="QHULL_MODE"):
        update_settings(QHULL_MODE="SOMETIMES")
    with pytest.raises(ValueError, match="BOUNDARY_MIP_CUT_ORDER"):
        update_settings(BOUNDARY_MIP_CUT_ORDER="sideways")
    assert cfg.QHULL_MODE == "IGNORE"


def test_env_vars_use_the_relucent_prefix() -> None:
    code = "from relucent.config import advanced as a; print(a.BOUNDARY_MIP_CUT_ORDER, a.BOUNDARY_MIP_BULK_NOGOOD_EMIT)"
    proc = _run(code, RELUCENT_BOUNDARY_MIP_CUT_ORDER="random", RELUCENT_BOUNDARY_MIP_BULK_NOGOOD_EMIT="off")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["random", "off"]
    proc = _run("import relucent.config", RELUCENT_QHULL_MODE="SOMETIMES")
    assert proc.returncode != 0 and "QHULL_MODE" in proc.stderr


def test_boundary_mip_bound_margin_default() -> None:
    assert cfg.BOUNDARY_MIP_BOUND_MARGIN == 5.0


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
