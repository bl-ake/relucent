"""Tests for relucent.core.poly (Polyhedron, solve_radius)."""

import pickle
from collections.abc import Iterable

import numpy as np
import pytest
import torch
import torch.nn as nn

from relucent import Complex, Polyhedron
from relucent import torch_mlp as _mlp
from relucent._internal.cache import UNSET
from relucent.model.builders import TorchMLP
from tests.helpers import ss_to_numpy


def torch_mlp(widths: Iterable[int], add_last_relu: bool = False) -> TorchMLP:
    result = _mlp(widths, add_last_relu=add_last_relu)
    assert isinstance(result, TorchMLP)
    return result


def _set_linear_params(net: TorchMLP, layer_name: str, weight: torch.Tensor, bias: torch.Tensor) -> None:
    layer = net.layers[layer_name]
    assert isinstance(layer, nn.Linear)
    layer.weight.data.copy_(weight.to(net.device, net.dtype))
    assert layer.bias is not None
    layer.bias.data.copy_(bias.to(net.device, net.dtype).reshape(-1))


class TestPolyhedronBasics:
    """Creation, affine map, tag, equality, hashing."""

    def test_create_from_ss(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[3, 6, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 3), device=net.device, dtype=net.dtype)
        ss = cplx.point2ss(x)
        p = Polyhedron(net, ss)
        assert p._net is not None
        assert np.array_equal(ss_to_numpy(p.ss), ss_to_numpy(ss).ravel())

    def test_affine_map_matches_forward(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[4, 8, 2], add_last_relu=True)
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
        net = torch_mlp(widths=[2, 4, 1], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        ss = cplx.point2ss(x)
        p = Polyhedron(net, ss)
        t = p.tag
        assert isinstance(t, bytes)
        assert p.tag == t

    def test_eq_hash(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[2, 4, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        p1 = cplx.add_point(x)
        p2 = Polyhedron(net, p1.ss_np)
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_neq(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[2, 4, 2], add_last_relu=True)
        cplx = Complex(net)
        x1 = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        x2 = x1 + 0.1
        p1 = cplx.add_point(x1)
        p2 = cplx.add_point(x2)
        if p1 != p2:
            assert hash(p1) != hash(p2)

    def test_eq_other_type_raises(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[2, 4, 1], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        p = cplx.add_point(x)
        # Comparing with an unsupported type returns NotImplemented (Python falls back to identity check)
        assert (p == 1) is False

    def test_volume(self):
        W = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]])
        b = torch.tensor([1, 1, 1, 1])
        net = torch_mlp(widths=[2, 4], add_last_relu=True)
        _set_linear_params(net, "fc0", W, b)
        cplx = Complex(net)
        p = cplx.add_point(torch.zeros((1, 2), device=net.device, dtype=net.dtype))
        assert p.volume is not None
        assert p.volume == 4


class TestPolyhedronContainment:
    def test_interior_point_in_polyhedron(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[3, 6, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 3), device=net.device, dtype=net.dtype)
        p = cplx.add_point(x)
        pt = p.interior_point
        assert pt is not None
        assert np.asarray(pt).reshape(1, -1) in p

    def test_point_containment_tensor(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[2, 4, 1], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        p = cplx.add_point(x)
        assert x in p


@pytest.mark.filterwarnings("ignore:Working with k<d polyhedron\\.:UserWarning")
class TestPolyhedronBoundedVertices:
    def test_bounded_vertices_supports_codim1_polyhedron(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[2, 1], add_last_relu=True)
        _set_linear_params(
            net,
            "fc0",
            torch.tensor([[1.0, 0.0]], device=net.device, dtype=net.dtype),
            torch.tensor([0.0], device=net.device, dtype=net.dtype),
        )
        p = Polyhedron(net, np.array([[0]], dtype=np.int8))

        verts = p.bounded_vertices(bound=1.0)
        assert verts is not None
        assert verts.shape[1] == 2
        assert np.allclose(verts[:, 0], 0.0, atol=1e-6)
        assert np.isclose(np.max(verts[:, 1]), 1.0, atol=1e-4)
        assert np.isclose(np.min(verts[:, 1]), -1.0, atol=1e-4)

    def test_bounded_vertices_supports_point_polyhedron(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[2, 2], add_last_relu=True)
        _set_linear_params(
            net,
            "fc0",
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=net.device, dtype=net.dtype),
            torch.tensor([0.0, 0.0], device=net.device, dtype=net.dtype),
        )
        p = Polyhedron(net, np.array([[0, 0]], dtype=np.int8))

        verts = p.bounded_vertices(bound=1.0)
        assert verts is not None
        assert verts.shape == (1, 2)
        assert np.allclose(verts[0], np.array([0.0, 0.0]), atol=1e-6)


class TestPolyhedronOps:
    def test_nflips(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[2, 4, 2], add_last_relu=True)
        cplx = Complex(net)
        x1 = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        x2 = x1 + 0.2
        p1 = cplx.add_point(x1)
        p2 = cplx.add_point(x2)
        n = p1.nflips(p2)
        assert isinstance(n, (int, np.integer))
        assert n >= 0


class TestPolyhedronRetainGeometryCaches:
    def test_retain_geometry_caches_clears_heavy_caches(self, seeded):
        from relucent.search.engine import retain_geometry_caches

        assert seeded is not None
        net = torch_mlp(widths=[2, 4, 2], add_last_relu=True)
        cplx = Complex(net)
        x = torch.rand((1, 2), device=net.device, dtype=net.dtype)
        p = cplx.add_point(x)
        _ = p.halfspaces
        _ = p.W
        _ = p.b
        retain_geometry_caches(p, ())
        assert p._halfspaces is None
        assert p._w is None
        assert p._b is None


class TestPolyhedronPickle:
    """Pickle roundtrip (from original test_save_load)."""

    def test_pickle_roundtrip(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[3, 6, 2], add_last_relu=True)
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

    def test_pickle_roundtrip_preserves_dimension_caches(self, seeded):
        assert seeded is not None
        net = torch_mlp(widths=[2, 4, 1])
        cplx = Complex(net)
        p = cplx.add_point(torch.randn(1, 2, device=net.device, dtype=net.dtype))
        _ = p.halfspaces
        p._ambient_dim = p.ambient_dim
        _ = p.codim, p.dim

        blob = pickle.dumps(p)
        p2 = pickle.loads(blob)

        assert p2._ambient_dim == p._ambient_dim
        assert p2.codim == p.codim
        assert p2.dim == p.dim
        assert "codim" in p2.__dict__
        assert "dim" in p2.__dict__


def _data_cell(rows: list[list[float]]) -> Polyhedron:
    hs = np.asarray(rows, dtype=np.float64)
    return Polyhedron(None, np.ones(hs.shape[0], dtype=np.int8), halfspaces=hs)


_UNIT_SQUARE = [[-1.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, 0.0], [0.0, 1.0, -1.0]]


class TestPolyhedronCaching:
    """Each computed property is cached once; "not computed" and a computed ``None`` stay distinct."""

    def test_empty_cell_properties_are_none(self):
        cell = _data_cell([[1.0, 0.0, 0.0], [-1.0, 0.0, 1.0]])  # x <= 0 and x >= 1
        assert cell.feasible is False
        assert cell.finite is None
        assert cell.center is None and cell.inradius is None
        assert cell.interior_point is None and cell.interior_point_norm is None
        assert cell.vertices is None and cell.convex_hull is None and cell.volume is None

    def test_bounded_and_unbounded_cells(self):
        square = _data_cell(_UNIT_SQUARE)
        assert square.finite is True
        assert square.volume == pytest.approx(1.0)
        assert square.convex_hull is not None and square.halfspace_intersection is not None
        assert square.vertices is not None and len(square.vertices) == 4
        quadrant = _data_cell([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        assert quadrant.finite is False
        assert quadrant.volume == float("inf") and quadrant.convex_hull is None

    def test_qhull_runs_once(self, monkeypatch: pytest.MonkeyPatch):
        from relucent.geometry.calculations import compute_properties

        calls: list[int] = []

        def counting(*args, **kwargs):
            calls.append(1)
            return compute_properties(*args, **kwargs)

        monkeypatch.setattr("relucent.core.poly.compute_properties", counting)
        square = _data_cell(_UNIT_SQUARE)
        _ = square.vertices, square.volume, square.convex_hull, square.halfspace_intersection
        assert len(calls) == 1

    def test_compute_geometric_properties_rejects_unknown_names(self):
        with pytest.raises(ValueError, match="Unknown geometry properties"):
            _data_cell(_UNIT_SQUARE).compute_geometric_properties(["volume", "hs"])

    def test_compute_geometric_properties_fills_every_name(self):
        square = Polyhedron(None, np.ones(4, dtype=np.int8), halfspaces=np.asarray(_UNIT_SQUARE), W=np.eye(2), b=np.zeros(2))
        square.compute_geometric_properties(Polyhedron.GEOMETRY_PROPERTIES)
        assert square._finite is True and square._chebyshev is not UNSET
        assert square._interior_point is not UNSET and square._qhull is not UNSET

    def test_sign_sequence_is_read_only(self):
        cell = _data_cell(_UNIT_SQUARE)
        with pytest.raises(AttributeError):
            setattr(cell, "ss", np.zeros(4, dtype=np.int8))  # noqa: B010 - the assignment is the test

    def test_hash_is_not_pickled(self):
        """hash(bytes) differs between processes, so a pickled hash breaks sets after loading."""
        cell = _data_cell(_UNIT_SQUARE)
        assert hash(cell) == hash(cell.tag)
        state = cell.__getstate__()
        assert "_hash" not in state and "_tag" not in state

    def test_pickle_keeps_computed_geometry(self):
        square = _data_cell(_UNIT_SQUARE)
        _ = square.finite, square.center, square.interior_point, square.volume
        copy = pickle.loads(pickle.dumps(square))
        assert copy._finite is True and copy._chebyshev is not UNSET and copy._qhull is not UNSET
        assert copy.volume == pytest.approx(1.0)

    def test_legacy_pickled_state_is_upgraded(self):
        """State as relucent 0.9 pickled it: separate "computed" flags and a cached hash."""
        center = np.array([[0.5], [0.5]])
        legacy = {
            "_finite_computed": True,
            "_finite": True,
            "_center": center,
            "_inradius": 0.5,
            "_chebyshev_done": True,
            "_interior_point": None,
            "_attempted_compute_properties": True,
            "_volume": 1.0,
            "_hash": 12345,
            "_tag": b"stale",
            "_Wl2": 3.0,
            "_halfspaces_np": np.asarray(_UNIT_SQUARE),
            "_rows_data": True,
        }
        cell = Polyhedron(None, np.ones(4, dtype=np.int8))
        cell.__setstate__(legacy)
        assert cell._finite is True
        assert cell._chebyshev is not UNSET and cell._chebyshev[1] == 0.5
        assert cell._interior_point is UNSET and cell._qhull is UNSET
        assert "_hash" not in cell.__dict__ and hash(cell) == hash(cell.tag)
        assert cell.volume == pytest.approx(1.0)

    def test_legacy_uncomputed_finite_becomes_unset(self):
        cell = Polyhedron(None, np.ones(4, dtype=np.int8))
        cell.__setstate__(
            {"_finite_computed": False, "_finite": None, "_halfspaces_np": np.asarray(_UNIT_SQUARE), "_rows_data": True}
        )
        assert cell._finite is UNSET
        assert cell.finite is True
