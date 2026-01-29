"""Tests for relucent.complex (Complex, BFS/DFS, dual graph, pathfinding)."""

import warnings

import networkx as nx
import numpy as np
import pytest
import torch

from relucent import Complex, get_mlp_model


# ----- Original tests (reorganized and renamed) -----


def test_bfs_dfs_dual_graph_isomorphic(seeded):
    """BFS/DFS equivalence, conversion to dual graph."""
    model = get_mlp_model(widths=[4, 8], add_last_relu=True)
    cplx1 = Complex(model)
    start1 = torch.rand((1, 4), device=model.device, dtype=model.dtype)
    cplx1.bfs(start=start1)
    G1 = cplx1.get_dual_graph()

    cplx2 = Complex(model)
    start2 = torch.rand((1, 4), device=model.device, dtype=model.dtype)
    cplx2.dfs(start=start2)
    G2 = cplx2.get_dual_graph()

    p1 = cplx1.point2ss(start1)
    assert p1 in cplx2
    assert nx.is_isomorphic(G1, G2)


def test_recover_from_dual_graph(seeded):
    """Recovery of full complex from dual graph."""
    model = get_mlp_model(widths=[5, 9], add_last_relu=True)
    cplx1 = Complex(model)
    start1 = torch.rand((1, 5), device=model.device, dtype=model.dtype)
    cplx1.bfs(start=start1)
    assert len(cplx1) == 382

    G1 = cplx1.get_dual_graph(relabel=True)
    cplx2 = Complex(model)
    cplx2.recover_from_dual_graph(G1, initial_ss=cplx1.point2ss(start1), source=0)
    G2 = cplx2.get_dual_graph(relabel=True)
    assert nx.is_isomorphic(G1, G2, edge_match=lambda u, v: u["shi"] == v["shi"])


def test_bfs_polyhedron_affine_and_membership(seeded):
    """BFS with larger network, point2poly, affine map, max_polys."""
    model = get_mlp_model(widths=[16, 64, 64, 64, 10])
    cplx = Complex(model)
    start = torch.rand(16, device=model.device, dtype=model.dtype)
    p = cplx.point2poly(start)
    assert len(p.halfspaces) == cplx.n
    assert p.ss_np.size == cplx.n
    assert torch.allclose(start @ p.W + p.b, model(start))

    cplx.bfs(max_polys=100, start=start)
    assert p in cplx
    assert len(cplx) == 100
    assert len(set(cplx.index2poly)) == len(cplx)


def test_dfs_max_depth_and_shis(seeded):
    """DFS with max_depth, nworkers=1, get_volumes=False"""
    model = get_mlp_model(widths=[6, 8, 10])
    cplx = Complex(model)
    result = cplx.dfs(max_depth=2, nworkers=1, get_volumes=False)
    assert result["Search Depth"] == 2
    assert [poly.shis is not None for poly in cplx]


def test_hamming_astar_path(seeded):
    """Pathfinding between two polyhedra via Hamming A*."""
    model = get_mlp_model(widths=[16, 32, 32, 1])
    cplx = Complex(model)
    start = torch.rand(16, device=model.device, dtype=model.dtype)
    end = torch.rand(16, device=model.device, dtype=model.dtype)
    path = cplx.hamming_astar(start=start, end=end)
    assert start in path[0]
    assert end in path[-1]
    for p1, p2 in zip(path[:-1], path[1:]):
        assert (p1.ss_np != p2.ss_np).sum().item() == 1


def test_plot_and_dual_graph_smoke(seeded):
    """Test starter code from the readme."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*[Ii]nterior point.*out of bounds.*",
            category=UserWarning,
        )
        model = get_mlp_model(widths=[2, 10, 5, 1])
        cplx = Complex(model)
        cplx.bfs()
        fig = cplx.plot(bound=10000)
        assert fig is not None
        _ = sum(len(p.shis) for p in cplx) / len(cplx)
        x = np.random.random(model.input_shape).astype(np.float32)
        p = cplx.point2poly(x)
        _ = p.halfspaces[p.shis, :]
        G = cplx.get_dual_graph()
        assert G.number_of_nodes() == len(cplx)


# ----- Additional Complex API tests -----


class TestComplexCreationAndIndexing:
    def test_len_iter(self, small_mlp):
        cplx = Complex(small_mlp)
        start = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        cplx.bfs(start=start, max_polys=20)
        assert len(cplx) == 20
        polys = list(cplx)
        assert len(polys) == 20

    def test_dim_n(self, small_mlp):
        cplx = Complex(small_mlp)
        assert cplx.dim == 4
        assert cplx.n == 8

    def test_getitem_by_ss(self, small_mlp):
        cplx = Complex(small_mlp)
        x = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        p = cplx.add_point(x)
        q = cplx[p.ss_np]
        assert q is p

    def test_getitem_by_polyhedron(self, small_mlp):
        cplx = Complex(small_mlp)
        x = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        p = cplx.add_point(x)
        q = cplx[p]
        assert q is p

    def test_contains(self, small_mlp):
        cplx = Complex(small_mlp)
        x = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        p = cplx.add_point(x)
        assert p.ss_np in cplx
        assert p in cplx

    def test_getitem_keyerror(self, small_mlp):
        cplx = Complex(small_mlp)
        x = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        p = cplx.add_point(x)
        bad_ss = p.ss_np.copy()
        bad_ss[0, 0] = -bad_ss[0, 0]  # flip one sign; neighbor not in complex yet
        with pytest.raises(KeyError):
            _ = cplx[bad_ss]


class TestComplexPointAndSS:
    def test_point2ss_tensor(self, small_mlp):
        cplx = Complex(small_mlp)
        x = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        ss = cplx.point2ss(x)
        assert isinstance(ss, torch.Tensor)
        assert ss.shape[1] == cplx.n

    def test_point2ss_ndarray(self, small_mlp):
        cplx = Complex(small_mlp)
        x = np.random.randn(1, 4).astype(np.float32)
        ss = cplx.point2ss(x)
        assert isinstance(ss, np.ndarray)
        assert ss.shape[1] == cplx.n

    def test_point2poly_check_exists(self, small_mlp):
        cplx = Complex(small_mlp)
        x = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        cplx.add_point(x)
        p = cplx.point2poly(x, check_exists=True)
        assert p in cplx

    def test_add_point_add_ss(self, small_mlp):
        cplx = Complex(small_mlp)
        x = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        ss = cplx.point2ss(x)
        p1 = cplx.add_point(x)
        p2 = cplx.add_ss(ss)
        assert p1 is p2


class TestComplexDualGraph:
    def test_dual_graph_basic(self, small_mlp):
        cplx = Complex(small_mlp)
        start = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        cplx.bfs(start=start, max_polys=30)
        G = cplx.get_dual_graph()
        assert G.number_of_nodes() == len(cplx)
        for u, v, d in G.edges(data=True):
            assert "shi" in d

    def test_dual_graph_relabel(self, small_mlp):
        cplx = Complex(small_mlp)
        start = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        cplx.bfs(start=start, max_polys=15)
        G = cplx.get_dual_graph(relabel=True)
        assert set(G.nodes()) == set(range(len(cplx)))


class TestComplexGetPolyAttrs:
    def test_get_poly_attrs(self, small_mlp):
        cplx = Complex(small_mlp)
        start = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        cplx.bfs(start=start, max_polys=10)
        attrs = cplx.get_poly_attrs(["finite", "Wl2"])
        assert "finite" in attrs
        assert "Wl2" in attrs
        assert len(attrs["finite"]) == len(attrs["Wl2"]) == len(cplx)


class TestComplexAdjacent:
    def test_adjacent_polyhedra(self, small_mlp):
        cplx = Complex(small_mlp)
        start = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        cplx.bfs(start=start, max_polys=25)
        p = next(iter(cplx))
        neighbors = cplx.adjacent_polyhedra(p)
        assert isinstance(neighbors, set)
        for q in neighbors:
            assert (p.ss_np != q.ss_np).sum() >= 1


class TestComplexMisc:
    def test_random_walk_smoke(self, small_mlp):
        """Smoke test for random_walk search."""
        cplx = Complex(small_mlp)
        start = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        result = cplx.random_walk(start=start, max_polys=15, nworkers=1, get_volumes=False)
        assert "Search Depth" in result
        assert len(cplx) <= 15

    def test_clean_data(self, small_mlp):
        """clean_data clears cached data on polyhedra."""
        cplx = Complex(small_mlp)
        start = torch.rand((1, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        cplx.bfs(start=start, max_polys=10)
        p = next(iter(cplx))
        _ = p.halfspaces
        _ = p.W
        cplx.clean_data()
        assert p._halfspaces is None
        assert p._W is None

    def test_ss_iterator(self, small_mlp):
        """ss_iterator yields sign sequences per ReLU layer."""
        cplx = Complex(small_mlp)
        x = torch.rand((2, 4), device=small_mlp.device, dtype=small_mlp.dtype)
        layers = list(cplx.ss_iterator(x))
        assert len(layers) == len(cplx.ss_layers)
        for ss in layers:
            assert ss.shape[0] == 2
            assert ss.shape[1] == 8
