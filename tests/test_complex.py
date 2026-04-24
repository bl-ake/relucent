"""Tests for relucent.complex (Complex, BFS/DFS, dual graph, pathfinding)."""

import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import cast

import networkx as nx
import numpy as np
import pytest
import torch
import torch.nn as nn

from relucent import Complex, Polyhedron, mlp, set_seeds
from relucent.calculations import adjacent_polyhedra
from relucent.model import Layer, LinearLayer, ReLULayer, ReLUNetwork


def _rand_batch(dim: int, batch: int = 1) -> torch.Tensor:
    return torch.rand((batch, dim), dtype=torch.float64)


def test_bfs_dfs_dual_graph_isomorphic(seed: int):
    """BFS/DFS equivalence, conversion to dual graph."""
    set_seeds(seed)
    model = mlp(widths=[4, 8], add_last_relu=True)
    cplx1 = Complex(model)
    start1 = _rand_batch(4)
    cplx1.bfs(start=start1)
    G1 = cplx1.get_dual_graph()

    cplx2 = Complex(model)
    start2 = _rand_batch(4)
    cplx2.dfs(start=start2)
    G2 = cplx2.get_dual_graph()

    p1 = cplx1.point2ss(start1)
    assert p1 in cplx2
    assert nx.is_isomorphic(G1, G2)


def test_recover_from_dual_graph(seed: int):
    """Recovery of full complex from dual graph."""
    set_seeds(seed)
    model = mlp(widths=[5, 9], add_last_relu=True)
    cplx1 = Complex(model)
    start1 = _rand_batch(5)
    cplx1.bfs(start=start1)
    assert len(cplx1) > 0

    G1 = cplx1.get_dual_graph(relabel=True)
    cplx2 = Complex(model)
    cplx2.recover_from_dual_graph(G1, initial_ss=cplx1.point2ss(start1), source=0)
    G2 = cplx2.get_dual_graph(relabel=True, auto_add=True)
    assert all(p.feasible for p in cplx2)
    assert nx.is_isomorphic(G1, G2, edge_match=lambda u, v: u["shi"] == v["shi"])


def test_bfs_polyhedron_affine_and_membership(seed: int):
    """BFS with larger network, point2poly, affine map, max_polys."""
    set_seeds(seed)
    model = mlp(widths=[16, 64, 64, 64, 10])
    cplx = Complex(model)
    start = torch.rand(16, dtype=torch.float64)
    p = cplx.point2poly(start)
    assert len(p.halfspaces) == cplx.n
    assert p.ss_np.size == cplx.n
    assert isinstance(p.W, torch.Tensor)
    assert isinstance(p.b, torch.Tensor)
    assert torch.allclose(start @ p.W + p.b, model(start))

    cplx.bfs(max_polys=100, start=start)
    assert p in cplx
    assert len(cplx) == 100
    assert len(set(cplx.index2poly)) == len(cplx)


def test_dfs_max_depth_and_shis(seed: int):
    """DFS with max_depth and nworkers=1."""
    set_seeds(seed)
    model = mlp(widths=[6, 8, 10])
    cplx = Complex(model)
    result = cplx.dfs(max_depth=2, nworkers=1)
    assert result["Search Depth"] == 2
    assert all(poly.shis is not None for poly in cplx)


def test_hamming_astar_path(seed: int):
    """Pathfinding between two polyhedra via Hamming A*."""
    set_seeds(seed)
    model = mlp(widths=[16, 32, 32, 1])
    cplx = Complex(model)
    start = torch.rand(16, dtype=torch.float64)
    end = torch.rand(16, dtype=torch.float64)
    result = cplx.hamming_astar(start=start, end=end)
    assert result["succeeded"], result
    assert result["path"] is not None, result
    path = result["path"]
    assert start in path[0]
    assert end in path[-1]
    for p1, p2 in zip(path[:-1], path[1:], strict=True):
        assert (p1.ss_np != p2.ss_np).sum().item() == 1


def test_plot_and_dual_graph_smoke(seed: int):
    """Test starter code from the readme."""
    set_seeds(seed)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*[Ii]nterior point.*out of bounds.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*divide by zero encountered in divide.*",
            category=RuntimeWarning,
        )
        network = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )
        cplx = Complex(network)
        cplx.bfs()
        fig = cplx.plot(plot_mode="cells", bound=10000)
        assert fig is not None
        _ = sum(len(p.shis) for p in cplx) / len(cplx)
        x = np.random.random(cplx._net.input_shape).astype(np.float32)
        p = cplx.point2poly(x)
        _ = p.halfspaces[p.shis, :]
        _ = p.W
        _ = p.b
        _ = p.finite
        _ = p.center
        _ = p.inradius
        _ = p.interior_point
        _ = p.interior_point_norm
        _ = sum(len(p.shis) for p in cplx) / len(cplx)
        G = cplx.get_dual_graph()
        assert G.number_of_nodes() == len(cplx)


class TestComplexCreationAndIndexing:
    def test_len_iter(self, small_mlp):
        cplx = Complex(small_mlp)
        start = _rand_batch(4)
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
        x = _rand_batch(4)
        p = cplx.add_point(x)
        q = cplx[p.ss_np]
        assert q is p

    def test_getitem_by_polyhedron(self, small_mlp):
        cplx = Complex(small_mlp)
        x = _rand_batch(4)
        p = cplx.add_point(x)
        q = cplx[p]
        assert q is p

    def test_contains(self, small_mlp):
        cplx = Complex(small_mlp)
        x = _rand_batch(4)
        p = cplx.add_point(x)
        assert p.ss_np in cplx
        assert p in cplx

    def test_getitem_keyerror(self, small_mlp):
        cplx = Complex(small_mlp)
        x = _rand_batch(4)
        p = cplx.add_point(x)
        bad_ss = p.ss_np.copy()
        bad_ss[0, 0] = -bad_ss[0, 0]  # flip one sign; neighbor not in complex yet
        with pytest.raises(KeyError):
            _ = cplx[bad_ss]


class TestComplexPointAndSS:
    def test_point2ss_tensor(self, small_mlp):
        cplx = Complex(small_mlp)
        x = _rand_batch(4)
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
        x = _rand_batch(4)
        cplx.add_point(x)
        p = cplx.point2poly(x, check_exists=True)
        assert p in cplx

    def test_add_point_add_ss(self, small_mlp):
        cplx = Complex(small_mlp)
        x = _rand_batch(4)
        ss = cplx.point2ss(x)
        p1 = cplx.add_point(x)
        p2 = cplx.add_ss(ss)
        assert p1 is p2


class TestComplexDualGraph:
    def test_dual_graph_basic(self, small_mlp):
        cplx = Complex(small_mlp)
        start = _rand_batch(4)
        cplx.bfs(start=start, max_polys=30)
        with pytest.warns(UserWarning, match=r"Dual graph is incomplete\. .* boundary cells were not added"):
            G = cplx.get_dual_graph()
        assert G.number_of_nodes() == len(cplx)
        for _, _, d in G.edges(data=True):
            assert "shi" in d

    def test_dual_graph_relabel(self, small_mlp):
        cplx = Complex(small_mlp)
        start = _rand_batch(4)
        cplx.bfs(start=start, max_polys=15)
        with pytest.warns(UserWarning, match=r"Dual graph is incomplete\. .* boundary cells were not added"):
            G = cplx.get_dual_graph(relabel=True)
        assert set(G.nodes()) == set(range(len(cplx)))


class TestComplexGetPolyAttrs:
    def test_get_poly_attrs(self, small_mlp):
        cplx = Complex(small_mlp)
        start = _rand_batch(4)
        cplx.bfs(start=start, max_polys=10)
        attrs = cplx.get_poly_attrs(["finite", "Wl2"])
        assert "finite" in attrs
        assert "Wl2" in attrs
        assert len(attrs["finite"]) == len(attrs["Wl2"]) == len(cplx)


class TestComplexAdjacent:
    def test_adjacent_polyhedra(self, small_mlp):
        cplx = Complex(small_mlp)
        start = _rand_batch(4)
        cplx.bfs(start=start, max_polys=25)
        p = next(iter(cplx))
        neighbors = adjacent_polyhedra(p, cplx.ss2poly)
        assert isinstance(neighbors, set)
        for q in neighbors:
            assert (p.ss_np != q.ss_np).sum() >= 1


class TestComplexAutoConversion:
    def test_sequential_is_accepted(self, seeded):
        """Complex accepts a plain nn.Sequential and auto-converts it."""
        assert seeded is not None
        sequential = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        cplx = Complex(sequential)
        assert cplx.net is sequential
        assert isinstance(cplx._net, ReLUNetwork)

    def test_module_dict_is_accepted(self, seeded):
        """Complex accepts an OrderedDict of layers and auto-converts it."""
        assert seeded is not None
        layers: OrderedDict[str, nn.Module] = OrderedDict(
            [
                ("fc0", nn.Linear(4, 8)),
                ("relu0", nn.ReLU()),
                ("fc1", nn.Linear(8, 2)),
            ]
        )
        cplx = Complex(nn.Sequential(layers))
        assert isinstance(cplx._net, ReLUNetwork)

    def test_auto_converted_dim_and_n(self, seeded):
        """Auto-converted sequential has correct input dim and neuron count."""
        assert seeded is not None
        sequential = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        cplx = Complex(sequential)
        assert cplx.dim == 4
        assert cplx.n == 8

    def test_auto_converted_matches_explicit_nn(self, seeded):
        """Auto-converting a Sequential gives the same complex as the explicit NN."""
        assert seeded is not None
        net = mlp(widths=[4, 8, 2])
        # Build an equivalent torch Sequential from canonical weights.
        fc0 = nn.Linear(4, 8)
        fc0.weight.data.copy_(torch.as_tensor(net.layers["fc0"].weight, dtype=torch.float64))
        fc0.bias.data.copy_(torch.as_tensor(net.layers["fc0"].bias.reshape(-1), dtype=torch.float64))
        fc1 = nn.Linear(8, 2)
        fc1.weight.data.copy_(torch.as_tensor(net.layers["fc1"].weight, dtype=torch.float64))
        fc1.bias.data.copy_(torch.as_tensor(net.layers["fc1"].bias.reshape(-1), dtype=torch.float64))
        sequential = nn.Sequential(fc0, nn.ReLU(), fc1).to(dtype=torch.float64)
        cplx_nn = Complex(net)
        cplx_seq = Complex(sequential)
        assert cplx_nn.dim == cplx_seq.dim
        assert cplx_nn.n == cplx_seq.n
        # Both should compute the same polyhedron for the same input point
        x = _rand_batch(4)
        ss_nn = cplx_nn.point2ss(x)
        ss_seq = cplx_seq.point2ss(x)
        assert torch.equal(torch.as_tensor(ss_nn), torch.as_tensor(ss_seq))

    def test_nn_module_passed_directly_unchanged(self, seeded):
        """An NN instance is not re-wrapped; cplx.net is the same object."""
        assert seeded is not None
        net = mlp(widths=[4, 8, 2])
        cplx = Complex(net)
        assert cplx.net is net


def _exercise_complex_for_model(model: nn.Module | ReLUNetwork) -> None:
    """Run a minimal end-to-end Complex usage flow for a model."""
    cplx = Complex(model)
    x = torch.randn((1, cplx.dim), dtype=torch.float64)
    y = cplx._net(x)
    ss = cplx.point2ss(x)
    p = cplx.add_point(x)
    q = cplx.point2poly(x, check_exists=True)
    assert y.shape[0] == 1
    assert ss.shape[1] == cplx.n
    assert p is q
    assert p in cplx


def _build_canonical_nn() -> ReLUNetwork:
    return mlp(widths=[4, 8, 2], add_last_relu=False)


def _build_plain_sequential() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def _build_ordered_sequential() -> nn.Module:
    return nn.Sequential(
        OrderedDict(
            [
                ("fc0", nn.Linear(4, 8)),
                ("relu0", nn.ReLU()),
                ("fc1", nn.Linear(8, 2)),
            ]
        )
    )


def _build_module_list() -> nn.Module:
    return nn.ModuleList(
        [
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        ]
    )


def _build_deepcopy_sequential() -> nn.Module:
    return deepcopy(_build_plain_sequential())


def _build_state_dict_clone() -> nn.Module:
    source = _build_plain_sequential()
    cloned = _build_plain_sequential()
    cloned.load_state_dict(source.state_dict())
    return cloned


class TestComplexPytorchCompatibility:
    @pytest.mark.parametrize(
        ("builder_name", "builder"),
        [
            ("canonical_nn", _build_canonical_nn),
            ("plain_sequential", _build_plain_sequential),
            ("ordered_sequential", _build_ordered_sequential),
            ("module_list", _build_module_list),
            ("deepcopy_sequential", _build_deepcopy_sequential),
            ("state_dict_clone", _build_state_dict_clone),
        ],
    )
    def test_complex_handles_common_pytorch_model_creation_paths(self, seeded, builder_name, builder):
        """Complex can be created and used across common PyTorch model construction paths."""
        assert seeded is not None
        model = builder()
        try:
            _exercise_complex_for_model(model)
        except Exception as exc:  # pragma: no cover - failure path aid
            pytest.fail(f"Complex failed for model builder '{builder_name}': {exc!r}")


class TestComplexParallelAdd:
    def test_parallel_add_returns_polyhedra(self, small_mlp):
        """parallel_add adds polyhedra to the complex and returns them in input order."""
        cplx = Complex(small_mlp)
        points = [np.random.randn(1, 4) for _ in range(4)]
        results = cplx.parallel_add(points, nworkers=2)
        assert len(results) == 4
        # All successful results are Polyhedron instances
        assert all(p is None or isinstance(p, Polyhedron) for p in results)
        # At least some should have been added
        assert len(cplx) > 0

    def test_parallel_add_order_matches_input(self, small_mlp):
        """parallel_add preserves input point order in its return list."""
        cplx1 = Complex(small_mlp)
        cplx2 = Complex(small_mlp)
        x0 = np.random.randn(1, 4)
        x1 = np.random.randn(1, 4)
        polys_parallel = cplx1.parallel_add([x0, x1], nworkers=1)
        p0 = cplx2.add_point(x0)
        p1 = cplx2.add_point(x1)
        assert polys_parallel[0] is not None and p0.tag == polys_parallel[0].tag
        assert polys_parallel[1] is not None and p1.tag == polys_parallel[1].tag


class TestComplexMisc:
    def test_random_walk_smoke(self, small_mlp):
        """Smoke test for random_walk search."""
        cplx = Complex(small_mlp)
        start = _rand_batch(4)
        result = cplx.random_walk(start=start, max_polys=15, nworkers=1)
        assert "Search Depth" in result
        assert len(cplx) <= 15

    def test_clean_data(self, small_mlp):
        """clean_data clears cached data on polyhedra."""
        cplx = Complex(small_mlp)
        start = _rand_batch(4)
        cplx.bfs(start=start, max_polys=10)
        p = next(iter(cplx))
        _ = p.halfspaces
        _ = p.W
        cplx.clean_data()
        assert p._halfspaces is None
        assert p._w is None

    def test_ss_iterator(self, small_mlp):
        """ss_iterator yields sign sequences per ReLU layer."""
        cplx = Complex(small_mlp)
        x = _rand_batch(4, batch=2)
        layers = list(cplx.ss_iterator(x))
        assert len(layers) == len(cplx.ss_layers)
        for ss in layers:
            assert ss.shape[0] == 2
            assert ss.shape[1] == 8


class TestComplexNumpyCanonicalNetworks:
    def test_build_complex_from_numpy_defined_network(self, seeded):
        assert seeded is not None
        layers = OrderedDict(
            [
                ("fc0", LinearLayer(weight=np.random.randn(6, 4), bias=np.random.randn(1, 6))),
                ("relu0", ReLULayer()),
                ("fc1", LinearLayer(weight=np.random.randn(3, 6), bias=np.random.randn(1, 3))),
                ("relu1", ReLULayer()),
            ]
        )
        net = ReLUNetwork(layers=cast(dict[str, Layer], layers), input_shape=(4,))
        cplx = Complex(net)
        p = cplx.add_point(np.random.randn(1, 4))
        assert cplx.dim == 4
        assert cplx.n == 9
        assert p in cplx

    def test_numpy_and_torch_inputs_match_sign_sequence(self, seeded):
        assert seeded is not None
        layers = OrderedDict(
            [
                ("fc0", LinearLayer(weight=np.random.randn(5, 4), bias=np.random.randn(1, 5))),
                ("relu0", ReLULayer()),
                ("fc1", LinearLayer(weight=np.random.randn(2, 5), bias=np.random.randn(1, 2))),
                ("relu1", ReLULayer()),
            ]
        )
        cplx = Complex(ReLUNetwork(layers=cast(dict[str, Layer], layers), input_shape=(4,)))
        x_np = np.random.randn(2, 4)
        x_torch = torch.as_tensor(x_np, dtype=torch.float64)
        ss_np = cplx.point2ss(x_np)
        ss_torch = cplx.point2ss(x_torch)
        ss_torch_np = ss_torch.detach().cpu().numpy() if isinstance(ss_torch, torch.Tensor) else np.asarray(ss_torch)
        assert np.array_equal(ss_np, ss_torch_np)

    def test_build_complex_from_affine_tuple_sequence(self, seeded):
        assert seeded is not None
        w0 = np.random.randn(6, 4)
        b0 = np.random.randn(6)
        w1 = np.random.randn(3, 6)
        b1 = np.random.randn(3)
        cplx = Complex([(w0, b0), (w1, b1)])
        p = cplx.add_point(np.random.randn(1, 4))
        assert cplx.dim == 4
        assert cplx.n == 6
        assert p in cplx
