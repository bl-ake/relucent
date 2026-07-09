"""Tests for contracted-SHI symmetrization on 1-cells."""

from __future__ import annotations

import numpy as np

from relucent import Complex, mlp
from relucent.meta_graph import _symmetrize_top_cell_flip_shis
from relucent.poly import Polyhedron
from relucent.utils import flip_ss_at_shi


def test_symmetrize_top_cell_flip_shis_drops_one_sided_listing() -> None:
    net = mlp(widths=[2, 4, 1], add_last_relu=True)
    cplx = Complex(net)
    ss_a = np.array([[1, -1, 1, -1, 1, -1, 1, -1, 1]], dtype=np.int8)
    ss_b = flip_ss_at_shi(ss_a, 0)
    halfspaces = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    a = Polyhedron(
        net,
        ss_a,
        halfspaces=halfspaces,
        shis=[0, 1],
        codim=3,
        dim=1,
        _ambient_dim=2,
    )
    b = Polyhedron(
        net,
        ss_b,
        halfspaces=halfspaces,
        shis=[1],
        codim=3,
        dim=1,
        _ambient_dim=2,
    )
    cplx.add_polyhedron(a, check_exists=False)
    cplx.add_polyhedron(b, check_exists=False)

    changed = _symmetrize_top_cell_flip_shis(cplx)
    assert changed >= 1
    assert 0 not in a.shis
    assert 0 not in b.shis
    for poly in (a, b):
        for shi in poly.shis:
            neighbor = cplx[flip_ss_at_shi(poly.ss_np, int(shi))]
            assert int(shi) in neighbor.shis
