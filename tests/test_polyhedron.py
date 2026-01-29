"""Tests for relucent.poly (Polyhedron, solve_radius)."""

import pickle

import numpy as np
import pytest
import torch

from relucent import Complex, Polyhedron, get_mlp_model

from tests.helpers import ss_to_numpy


class TestPolyhedronBasics:
    """Creation, affine map, tag, equality, hashing."""

    def test_create_from_ss(self, seeded):
        net = get_mlp_model(widths=[3, 6, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 3), device=net.device, dtype=net.dtype)
        ss = cplx.point2ss(x)
        p = Polyhedron(net, ss)
        assert p.net is net
        assert np.array_equal(ss_to_numpy(p.ss), ss_to_numpy(ss))

    def test_affine_map_matches_forward(self, seeded):
        net = get_mlp_model(widths=[4, 8, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 4), device=net.device, dtype=net.dtype)
        ss = cplx.point2ss(x)
        p = Polyhedron(net, ss)
        y_affine = x @ p.W + p.b
        y_net = net(x)
        assert torch.allclose(y_affine, y_net, atol=1e-5)

    def test_tag_stable(self, seeded):
        net = get_mlp_model(widths=[2, 4, 1], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        ss = cplx.point2ss(x)
        p = Polyhedron(net, ss)
        t = p.tag
        assert isinstance(t, bytes)
        assert p.tag == t

    def test_eq_hash(self, seeded):
        net = get_mlp_model(widths=[2, 4, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        p1 = cplx.add_point(x)
        p2 = Polyhedron(net, p1.ss_np)
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_neq(self, seeded):
        net = get_mlp_model(widths=[2, 4, 2], add_last_relu=True)
        cplx = Complex(net)
        x1 = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        x2 = x1 + 0.1
        p1 = cplx.add_point(x1)
        p2 = cplx.add_point(x2)
        if p1 != p2:
            assert hash(p1) != hash(p2)

    def test_eq_other_type_raises(self, seeded):
        net = get_mlp_model(widths=[2, 4, 1], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        p = cplx.add_point(x)
        with pytest.raises(ValueError, match="Cannot compare Polyhedron"):
            _ = p == 1


class TestPolyhedronContainment:
    def test_interior_point_in_polyhedron(self, seeded):
        net = get_mlp_model(widths=[3, 6, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 3), device=net.device, dtype=net.dtype)
        p = cplx.add_point(x)
        pt = p.interior_point
        assert pt is not None
        assert np.asarray(pt).reshape(1, -1) in p

    def test_point_containment_tensor(self, seeded):
        net = get_mlp_model(widths=[2, 4, 1], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        p = cplx.add_point(x)
        assert x in p


class TestPolyhedronOps:
    def test_nflips(self, seeded):
        net = get_mlp_model(widths=[2, 4, 2], add_last_relu=True)
        cplx = Complex(net)
        x1 = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        x2 = x1 + 0.2
        p1 = cplx.add_point(x1)
        p2 = cplx.add_point(x2)
        n = p1.nflips(p2)
        assert isinstance(n, (int, np.integer))
        assert n >= 0

    def test_mul_raises_non_polyhedron(self, seeded):
        net = get_mlp_model(widths=[2, 4, 1], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        p = cplx.add_point(x)
        with pytest.raises(ValueError, match="Cannot multiply Polyhedron"):
            _ = p * 1


class TestPolyhedronCleanData:
    def test_clean_data_clears_caches(self, seeded):
        net = get_mlp_model(widths=[2, 4, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        p = cplx.add_point(x)
        _ = p.halfspaces
        _ = p.W
        _ = p.b
        p.clean_data()
        assert p._halfspaces is None
        assert p._W is None
        assert p._b is None
        assert p._center is None


class TestPolyhedronPickle:
    """Pickle roundtrip (from original test_save_load)."""

    def test_pickle_roundtrip(self, seeded):
        net = get_mlp_model(widths=[3, 6, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 3), device=net.device, dtype=net.dtype)
        ss = cplx.point2ss(x)
        p = Polyhedron(net, ss)
        y1 = x @ p.W + p.b
        assert torch.allclose(y1, net(x))

        blob = pickle.dumps(p)
        p2 = pickle.loads(blob)

        assert p2.net is None
        assert isinstance(p2.ss, (np.ndarray, torch.Tensor))
        assert np.array_equal(ss_to_numpy(p2.ss), ss_to_numpy(p.ss))
        assert p2.tag == p.tag

        p2.net = net
        W2 = torch.as_tensor(p2.W, device=net.device, dtype=net.dtype)
        b2 = torch.as_tensor(p2.b, device=net.device, dtype=net.dtype)
        y2 = x @ W2 + b2
        assert torch.allclose(y2, net(x))
        assert p2.halfspaces.shape == p.halfspaces.shape
