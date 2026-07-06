"""Regression tests for combinatorially feasible but geometrically spurious boundary cells."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from relucent import Complex, set_seeds
from relucent.poly import Polyhedron

os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")

_PHANTOM_SS = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 0], dtype=np.int8).reshape(1, -1)
_BOUNDARY_SHI = 8


def _synthetic_progress_ckpt0_model():
    try:
        from analysis.core.experiment_store import ExperimentArray
        from analysis.paths import ANALYSIS_DIR, BETTI_TRAINING_EXP
    except ImportError as exc:
        pytest.skip(f"analysis package not available: {exc}")
    if not BETTI_TRAINING_EXP.is_dir():
        pytest.skip(f"experiment directory missing: {BETTI_TRAINING_EXP}")
    exp = ExperimentArray.load(str(BETTI_TRAINING_EXP), ncached=1)[0]
    return exp.get_model(
        device="cpu",
        root=str(ANALYSIS_DIR),
        fallback_exp_dir=str(BETTI_TRAINING_EXP),
    )


def _populate_input_complex(cplx: Complex) -> None:
    torch.manual_seed(0)
    cplx.bfs(verbose=False, geometry_properties=["finite"], start=torch.randn(1, 2))
    cplx.get_dual_graph(verbose=False)


def test_both_ambient_cofaces_feasible_rejects_phantom_ss() -> None:
    from relucent.boundary_search import _both_ambient_cofaces_feasible

    model = _synthetic_progress_ckpt0_model()
    poly = Polyhedron(model, _PHANTOM_SS)
    assert poly.feasible
    assert not _both_ambient_cofaces_feasible(poly, _BOUNDARY_SHI)


def test_discover_boundary_complex_matches_reference_on_ckpt0_phantom() -> None:
    set_seeds(0)
    model = _synthetic_progress_ckpt0_model()
    cplx = Complex(model)
    _populate_input_complex(cplx)
    ref = cplx.get_boundary_complex(_BOUNDARY_SHI, verbose=False)
    new = Complex(model).discover_boundary_complex(_BOUNDARY_SHI, verbose=False, nworkers=1)
    assert {p.tag for p in ref} == {p.tag for p in new}
    assert len(new) == len(ref) == 12
    assert _PHANTOM_SS.tobytes() not in {p.tag for p in new}
