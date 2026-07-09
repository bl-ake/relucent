"""Strict invariants for 1-cells in meta-graphs."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from relucent import Complex, Polyhedron, mlp, set_seeds
from relucent import meta_graph as mg
from relucent.exploration import explore_for_topology
from relucent.meta_graph import CubicalConsistencyError


def _segment_meta(
    *,
    shis: list[int],
    finite: bool,
    zero_face_nodes: list[object],
) -> nx.MultiDiGraph:
    """Minimal meta-graph with one 1-cell and optional 0-face endpoints."""
    meta: nx.MultiDiGraph = nx.MultiDiGraph()
    one = b"seg"
    meta.add_node(one, dim=1, shis=list(shis), finite=finite, ss=np.array([[1, 0, 0]], dtype=np.int8))
    for i, z in enumerate(zero_face_nodes):
        meta.add_node(z, dim=0, shis=[], finite=True, ss=np.array([[0, 0, 0]], dtype=np.int8))
        meta.add_edge(one, z, shi=i)
    return meta


def test_verify_meta_graph_one_cells_accepts_bounded_segment() -> None:
    meta = _segment_meta(shis=[0, 2], finite=True, zero_face_nodes=[b"z0", b"z1"])
    mg.verify_meta_graph_one_cells(meta)


def test_verify_meta_graph_one_cells_accepts_unbounded_ray() -> None:
    meta = _segment_meta(shis=[1], finite=False, zero_face_nodes=[b"z0"])
    mg.verify_meta_graph_one_cells(meta)


def test_verify_meta_graph_one_cells_rejects_too_many_shis() -> None:
    meta = _segment_meta(shis=[0, 1, 2], finite=False, zero_face_nodes=[b"z0"])
    with pytest.raises(CubicalConsistencyError, match="at most 2"):
        mg.verify_meta_graph_one_cells(meta)


def test_verify_meta_graph_one_cells_rejects_bounded_with_one_shi() -> None:
    meta = _segment_meta(shis=[0], finite=True, zero_face_nodes=[b"z0", b"z1"])
    with pytest.raises(CubicalConsistencyError, match="Bounded 1-cell"):
        mg.verify_meta_graph_one_cells(meta)


def test_verify_meta_graph_one_cells_rejects_unbounded_with_two_endpoints() -> None:
    meta = _segment_meta(shis=[0, 1], finite=False, zero_face_nodes=[b"z0", b"z1"])
    with pytest.raises(CubicalConsistencyError, match="marked unbounded"):
        mg.verify_meta_graph_one_cells(meta)


def test_classify_one_cells_finite_one_zero_face_is_unbounded() -> None:
    """A single combinatorial 0-face classifies the 1-cell as an unbounded ray."""
    halfspaces = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    ss_z0 = np.array([[1, 0, 1, 0]], dtype=np.int8)
    ss_ray = np.array([[1, 0, 0, 0]], dtype=np.int8)
    p0 = Polyhedron(None, ss_z0, halfspaces=halfspaces, finite=True)
    ray = Polyhedron(None, ss_ray, halfspaces=halfspaces, shis=[1], finite=None)
    ray._finite_computed = False

    by_dim = {0: [p0], 1: [ray]}
    edges_by_dim = {1: ([(ray.tag, p0.tag, 2)], [])}

    n = mg.classify_one_cells_finite_from_face_edges(by_dim, edges_by_dim)[0]
    assert n == 1
    assert ray._finite is False
    assert len(ray._shis or []) < 2


@pytest.mark.filterwarnings("ignore:Working with k<d polyhedron\\.:UserWarning")
def test_explored_complex_meta_graph_one_cells(seeded: int) -> None:
    """Explored complexes satisfy 1-cell SHI / boundedness invariants."""
    set_seeds(seeded)
    net = mlp(widths=[4, 5, 5, 1], add_last_relu=True)
    cplx = Complex(net)
    start = torch.randn(4, dtype=torch.float64)
    explore_for_topology(cplx, start.numpy(), max_polys=5000)
    meta = cplx.get_meta_graph(verify=True, verbose=False)
    mg.verify_meta_graph_one_cells(meta)
