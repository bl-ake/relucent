"""Tests for relucent.poly (Polyhedron, solve_radius)."""

import pickle

import numpy as np
import pytest
import torch

from relucent import Complex, Polyhedron, mlp
from tests.helpers import ss_to_numpy


def _set_linear_params(net, layer_name: str, weight: torch.Tensor, bias: torch.Tensor) -> None:
    layer = net.layers[layer_name]
    if isinstance(layer.weight, torch.Tensor):
        layer.weight.data.copy_(weight.to(net.device, net.dtype))
        layer.bias.data.copy_(bias.to(net.device, net.dtype).reshape(-1))
    else:
        layer.weight = weight.to(net.device, net.dtype).cpu().numpy()
        layer.bias = bias.to(net.device, net.dtype).cpu().numpy().reshape(1, -1)


class TestPolyhedronBasics:
    """Creation, affine map, tag, equality, hashing."""

    def test_create_from_ss(self, seeded):
        assert seeded is not None
        net = mlp(widths=[3, 6, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 3), device=net.device, dtype=net.dtype)
        ss = cplx.point2ss(x)
        p = Polyhedron(net, ss)
        assert p._net is not None
        assert np.array_equal(ss_to_numpy(p.ss), ss_to_numpy(ss))

    def test_affine_map_matches_forward(self, seeded):
        assert seeded is not None
        net = mlp(widths=[4, 8, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 4), device=net.device, dtype=net.dtype)
        ss = cplx.point2ss(x)
        p = Polyhedron(net, ss)
        assert isinstance(p.W, torch.Tensor)
        assert isinstance(p.b, torch.Tensor)
        y_affine = x @ p.W + p.b
        y_net = net(x)
        assert torch.allclose(y_affine, y_net, atol=1e-5)

    def test_tag_stable(self, seeded):
        assert seeded is not None
        net = mlp(widths=[2, 4, 1], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        ss = cplx.point2ss(x)
        p = Polyhedron(net, ss)
        t = p.tag
        assert isinstance(t, bytes)
        assert p.tag == t

    def test_eq_hash(self, seeded):
        assert seeded is not None
        net = mlp(widths=[2, 4, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        p1 = cplx.add_point(x)
        p2 = Polyhedron(net, p1.ss_np)
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_neq(self, seeded):
        assert seeded is not None
        net = mlp(widths=[2, 4, 2], add_last_relu=True)
        cplx = Complex(net)
        x1 = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        x2 = x1 + 0.1
        p1 = cplx.add_point(x1)
        p2 = cplx.add_point(x2)
        if p1 != p2:
            assert hash(p1) != hash(p2)

    def test_eq_other_type_raises(self, seeded):
        assert seeded is not None
        net = mlp(widths=[2, 4, 1], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        p = cplx.add_point(x)
        with pytest.raises(ValueError, match="Cannot compare Polyhedron"):
            _ = p == 1

    def test_volume(self):
        W = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]])
        b = torch.tensor([1, 1, 1, 1])
        net = mlp(widths=[2, 4], add_last_relu=True)
        _set_linear_params(net, "fc0", W, b)
        cplx = Complex(net)
        p = cplx.add_point(torch.zeros((1, 2), device=net.device, dtype=net.dtype))
        assert p.volume is not None
        assert p.volume == 4


class TestPolyhedronContainment:
    def test_interior_point_in_polyhedron(self, seeded):
        assert seeded is not None
        net = mlp(widths=[3, 6, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 3), device=net.device, dtype=net.dtype)
        p = cplx.add_point(x)
        pt = p.interior_point
        assert pt is not None
        assert np.asarray(pt).reshape(1, -1) in p

    def test_point_containment_tensor(self, seeded):
        assert seeded is not None
        net = mlp(widths=[2, 4, 1], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        p = cplx.add_point(x)
        assert x in p


@pytest.mark.filterwarnings("ignore:Working with k<d polyhedron\\.:UserWarning")
class TestPolyhedronBoundedVertices:
    def test_bounded_vertices_supports_codim1_polyhedron(self, seeded):
        assert seeded is not None
        net = mlp(widths=[2, 1], add_last_relu=True)
        _set_linear_params(
            net,
            "fc0",
            torch.tensor([[1.0, 0.0]], device=net.device, dtype=net.dtype),
            torch.tensor([0.0], device=net.device, dtype=net.dtype),
        )
        net.save_numpy_weights()
        p = Polyhedron(net, np.array([[0]], dtype=np.int8))

        verts = p.get_bounded_vertices(bound=1.0)
        assert verts is not None
        assert verts.shape[1] == 2
        assert np.allclose(verts[:, 0], 0.0, atol=1e-6)
        assert np.isclose(np.max(verts[:, 1]), 1.0, atol=1e-4)
        assert np.isclose(np.min(verts[:, 1]), -1.0, atol=1e-4)

    def test_bounded_vertices_supports_point_polyhedron(self, seeded):
        assert seeded is not None
        net = mlp(widths=[2, 2], add_last_relu=True)
        _set_linear_params(
            net,
            "fc0",
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=net.device, dtype=net.dtype),
            torch.tensor([0.0, 0.0], device=net.device, dtype=net.dtype),
        )
        net.save_numpy_weights()
        p = Polyhedron(net, np.array([[0, 0]], dtype=np.int8))

        verts = p.get_bounded_vertices(bound=1.0)
        assert verts is not None
        assert verts.shape == (1, 2)
        assert np.allclose(verts[0], np.array([0.0, 0.0]), atol=1e-6)


class TestPolyhedronOps:
    def test_nflips(self, seeded):
        assert seeded is not None
        net = mlp(widths=[2, 4, 2], add_last_relu=True)
        cplx = Complex(net)
        x1 = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        x2 = x1 + 0.2
        p1 = cplx.add_point(x1)
        p2 = cplx.add_point(x2)
        n = p1.nflips(p2)
        assert isinstance(n, (int, np.integer))
        assert n >= 0


class TestPolyhedronCleanData:
    def test_clean_data_clears_caches(self, seeded):
        assert seeded is not None
        net = mlp(widths=[2, 4, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        p = cplx.add_point(x)
        _ = p.halfspaces
        _ = p.W
        _ = p.b
        p.clean_data()
        assert p._halfspaces is None
        assert p._w is None
        assert p._b is None


class TestPolyhedronPickle:
    """Pickle roundtrip (from original test_save_load)."""

    def test_pickle_roundtrip(self, seeded):
        assert seeded is not None
        net = mlp(widths=[3, 6, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 3), device=net.device, dtype=net.dtype)
        ss = cplx.point2ss(x)
        p = Polyhedron(net, ss)
        assert isinstance(p.W, torch.Tensor)
        assert isinstance(p.b, torch.Tensor)
        y1 = x @ p.W + p.b
        assert torch.allclose(y1, net(x))

        blob = pickle.dumps(p)
        p2 = pickle.loads(blob)

        assert p2._net is None
        assert isinstance(p2.ss, (np.ndarray, torch.Tensor))
        assert np.array_equal(ss_to_numpy(p2.ss), ss_to_numpy(p.ss))
        assert p2.tag == p.tag

        p2._net = cplx._net
        W2 = torch.as_tensor(p2.W, device=net.device, dtype=net.dtype)
        b2 = torch.as_tensor(p2.b, device=net.device, dtype=net.dtype)
        y2 = x @ W2 + b2
        assert torch.allclose(y2, net(x))
        assert p2.halfspaces.shape == p.halfspaces.shape
