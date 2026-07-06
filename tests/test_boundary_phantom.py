"""Regression tests for combinatorially feasible but geometrically spurious boundary cells."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from relucent import Complex, mlp, set_seeds
from relucent.poly import Polyhedron
from relucent.utils import TorchMLP

os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")

_BUNDLED_CKPT = Path(__file__).parent / "data" / "synthetic_progress_ckpt0.pt"
_PHANTOM_SS = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 0], dtype=np.int8).reshape(1, -1)
_BOUNDARY_SHI = 8


def _synthetic_progress_ckpt0_model() -> TorchMLP:
    if not _BUNDLED_CKPT.is_file():
        pytest.skip(f"bundled checkpoint missing: {_BUNDLED_CKPT}")
    model = mlp(widths=[2, 8, 1], add_last_relu=True)
    assert isinstance(model, TorchMLP)
    state_dict = torch.load(_BUNDLED_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


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
