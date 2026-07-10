"""Regression: network-scaled SHI bound detects output-neuron facets."""

from __future__ import annotations

import os

import numpy as np

from relucent import Complex, Polyhedron, mlp, set_seeds
from relucent._network_scale import default_polyhedron_bound
from relucent.calculations import get_shis

os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")


def test_default_polyhedron_bound_used_by_lazy_shis() -> None:
    model = mlp(widths=[2, 4, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False)
    top = next(p for p in cplx if p.dim == cplx.dim)
    assert top.bound is not None
    assert len(top.shis) > 0


def test_get_shis_escalate_bound_false_uses_single_box(seeded: int) -> None:
    """``escalate_bound=False`` keeps SHI LPs at the requested box radius."""
    set_seeds(seeded)
    from relucent import convert

    model = mlp(widths=[2, 4, 1], add_last_relu=False)
    relu_net = convert(model)
    bound = default_polyhedron_bound(relu_net)
    ss = np.array([[1, -1, -1, 1]], dtype=np.int8)
    poly = Polyhedron(relu_net, ss, bound=bound)
    poly.get_geometry(("finite",), env=None)
    shis_no_esc = get_shis(poly, bound=bound, escalate_bound=False)
    shis_esc = get_shis(poly, bound=bound, escalate_bound=True)
    assert len(shis_no_esc) > 0
    assert len(shis_esc) > 0


def test_get_shis_escalates_bound_for_unbounded_arrangement_cell(seeded: int) -> None:
    """Unbounded hyperplane cells must not fail SHI LPs at the network-scaled box."""
    set_seeds(seeded)
    model = mlp(widths=[2, 4, 1], add_last_relu=False)
    cplx = Complex(model)
    net = cplx._net
    bound = default_polyhedron_bound(net)
    # All-sign cell: unbounded in generic 2D arrangement with 4 hyperplanes.
    ss = np.array([[1, -1, -1, 1]], dtype=np.int8)
    poly = Polyhedron(net, ss, bound=bound)
    poly.get_geometry(("finite",), env=None)
    assert poly.finite is False
    shis = get_shis(poly, bound=bound)
    assert len(shis) > 0
