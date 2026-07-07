"""Regression: network-scaled SHI bound detects output-neuron facets."""

from __future__ import annotations

import os

import numpy as np

from relucent import Complex, mlp

os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")


def test_default_polyhedron_bound_used_by_lazy_shis() -> None:
    model = mlp(widths=[2, 4, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False)
    top = next(p for p in cplx if p.dim == cplx.dim)
    assert top.bound is not None
    assert len(top.shis) > 0
