"""Tests for contracted-SHI compute and verify on 1-cells."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from relucent import Complex, mlp
from relucent.meta_graph import (
    CubicalConsistencyError,
    set_contracted_shis,
    set_shis_from_dual_graph,
    verify_contracted_shis,
    verify_shi_flip_neighbors,
    verify_shis_from_dual_graph,
    verify_top_cell_flip_shi_symmetry,
)
from relucent.poly import Polyhedron
from relucent.utils import encode_ss, flip_ss_at_shi


def _asymmetric_one_cell_fixture() -> tuple[Complex, Polyhedron, Polyhedron]:
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
    return cplx, a, b


def test_verify_top_cell_flip_shi_symmetry_raises_on_one_sided_listing() -> None:
    cplx, _a, _b = _asymmetric_one_cell_fixture()
    with pytest.raises(CubicalConsistencyError, match="Asymmetric SHI"):
        verify_top_cell_flip_shi_symmetry(cplx)


def test_set_contracted_shis_then_verify_succeeds_on_flip_pair() -> None:
    cplx, a, b = _asymmetric_one_cell_fixture()
    set_contracted_shis(cplx)
    verify_contracted_shis(cplx)
    for poly in (a, b):
        for shi in poly.shis:
            neighbor = cplx[flip_ss_at_shi(poly.ss_np, int(shi))]
            assert int(shi) in neighbor.shis


def test_verify_shi_flip_neighbors_raises_on_mismatch() -> None:
    ss = np.array([[1, -1, 1]], dtype=np.int8)
    neighbor_tags = {encode_ss(flip_ss_at_shi(ss, 0))}
    with pytest.raises(CubicalConsistencyError, match="flip-neighbor mismatch"):
        verify_shi_flip_neighbors(ss, [2], neighbor_tags=neighbor_tags)


def test_verify_shis_from_dual_graph_matches_edge_labels() -> None:
    net = mlp(widths=[2, 4, 1], add_last_relu=True)
    cplx = Complex(net)
    ss_a = np.array([[1, -1, 1, -1, 1, -1, 1, -1, 1]], dtype=np.int8)
    ss_b = flip_ss_at_shi(ss_a, 0)
    a = Polyhedron(net, ss_a, shis=[0], codim=2, dim=1, _ambient_dim=2)
    b = Polyhedron(net, ss_b, shis=[0], codim=2, dim=1, _ambient_dim=2)
    cplx.add_polyhedron(a, check_exists=False)
    cplx.add_polyhedron(b, check_exists=False)
    graph = nx.Graph()
    graph.add_node(a)
    graph.add_node(b)
    graph.add_edge(a, b, shi=0)
    set_shis_from_dual_graph(graph)
    verify_shis_from_dual_graph(graph)
