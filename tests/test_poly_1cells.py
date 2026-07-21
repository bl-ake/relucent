"""Regression tests for 1-cells (line segments) in polyhedra.

These tests focus on the "manual halfspaces" path (constructing Polyhedron with
explicit halfspaces + a sign sequence) and verify that key Polyhedron behaviors
remain correct for 1-dimensional cells both in 1D ambient space and when
embedded as lower-dimensional cells in higher ambient dimensions.
"""

import numpy as np
import pytest

from relucent import Polyhedron


def _poly_from_halfspaces(*, halfspaces: np.ndarray, zero_rows: set[int] | None = None) -> Polyhedron:
    """Create a Polyhedron with a consistent sign-sequence for zero/equality rows."""
    halfspaces = np.asarray(halfspaces, dtype=np.float64)
    if halfspaces.ndim != 2 or halfspaces.shape[1] < 2:
        raise ValueError("halfspaces must have shape (m, d+1)")
    m = halfspaces.shape[0]
    zero_rows = zero_rows or set()
    ss = np.ones((1, m), dtype=np.int8)
    for i in zero_rows:
        ss[0, int(i)] = 0
    return Polyhedron(None, ss, halfspaces=halfspaces)


def _sorted_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expected 2D array")
    # lexsort wants last key first
    keys = [x[:, j] for j in range(x.shape[1] - 1, -1, -1)]
    return x[np.lexsort(keys)]


class TestOneCellsInAmbient1D:
    def test_segment_vertices_center_inradius_contains(self):
        # Segment [0, 1] in R^1:
        # -x <= 0   (x >= 0)
        #  x - 1 <= 0 (x <= 1)
        halfspaces = np.array(
            [
                [-1.0, 0.0],
                [1.0, -1.0],
            ]
        )
        p = _poly_from_halfspaces(halfspaces=halfspaces)

        assert p.ambient_dim == 1
        assert p.codim == 0
        assert p.dim == 1

        # Chebyshev center/inradius in 1D should be midpoint and half-length.
        assert p.finite is True
        assert p.center is not None
        assert np.allclose(p.center.reshape(-1), np.array([0.5]), atol=1e-7)
        assert p.inradius is not None
        assert np.isclose(p.inradius, 0.5, atol=1e-7)

        verts = p.get_bounded_vertices(bound=2.0)
        assert verts is not None
        verts = _sorted_rows(np.unique(verts, axis=0))
        assert verts.shape == (2, 1)
        assert np.allclose(verts[:, 0], np.array([0.0, 1.0]), atol=1e-7)

        assert np.array([0.5]) in p
        assert np.array([0.0]) in p
        assert np.array([1.0]) in p
        assert np.array([-0.1]) not in p
        assert np.array([1.1]) not in p

        # SHIs should be both inequalities (order not guaranteed).
        assert set(p.shis) == {0, 1}
        faces = p.faces
        assert len(faces) == 2
        for f in faces:
            assert f.ambient_dim == 1
            assert f.dim == 0
            fv = f.get_bounded_vertices(bound=2.0)
            assert fv is not None
            assert fv.shape == (1, 1)
        endpoints_list: list[np.ndarray] = []
        for f in faces:
            fv = f.get_bounded_vertices(bound=2.0)
            if fv is not None:
                endpoints_list.append(fv)
        endpoints = _sorted_rows(np.vstack(endpoints_list))
        assert np.allclose(endpoints[:, 0], np.array([0.0, 1.0]), atol=1e-7)


@pytest.mark.filterwarnings("ignore:Working with k<d polyhedron\\.:UserWarning")
class TestOneCellsEmbeddedInHigherDimensions:
    def test_segment_in_R2(self):
        # Segment from (0,0) to (1,0) embedded in R^2.
        # Inequalities:
        #   -x <= 0  (x >= 0)
        #    x - 1 <= 0 (x <= 1)
        # Equality (sign=0 row):
        #    y <= 0 treated as y == 0 by zero_indices in solve_radius / get_bounded_vertices.
        halfspaces = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        p = _poly_from_halfspaces(halfspaces=halfspaces, zero_rows={2})

        assert p.ambient_dim == 2
        assert p.codim == 1
        assert p.dim == 1

        assert p.finite is True
        assert p.center is not None
        assert np.allclose(p.center.reshape(-1), np.array([0.5, 0.0]), atol=1e-7)
        assert p.inradius is not None
        assert np.isclose(p.inradius, 0.5, atol=1e-7)

        verts = p.get_bounded_vertices(bound=2.0)
        assert verts is not None
        verts = _sorted_rows(np.unique(verts, axis=0))
        assert verts.shape == (2, 2)
        assert np.allclose(verts, np.array([[0.0, 0.0], [1.0, 0.0]]), atol=1e-7)

        assert np.array([0.25, 0.0]) in p
        assert np.array([0.25, 1e-2]) not in p

        assert set(p.shis) == {0, 1}
        faces = p.faces
        assert len(faces) == 2
        endpoints = []
        for f in faces:
            assert f.ambient_dim == 2
            assert f.dim == 0
            fv = f.get_bounded_vertices(bound=2.0)
            assert fv is not None
            assert fv.shape == (1, 2)
            endpoints.append(fv[0])
        endpoints = _sorted_rows(np.asarray(endpoints))
        assert np.allclose(endpoints, np.array([[0.0, 0.0], [1.0, 0.0]]), atol=1e-7)

    def test_bounded_clip_remaps_zero_indices_past_degenerate_row(self):
        """Degenerate row before an equality must not shift zero_indices onto a box face.

        ``get_bounded_halfspaces`` drops near-zero normals then calls ``solve_radius``
        with equality indices. If those indices are not remapped, they can point at a
        newly stacked bounding-box row (e.g. ``x = ±bound``), falsely marking a cell
        that clearly intersects the plot box as empty. This showed up when plotting
        flat decision-boundary cells that had a dead ReLU row before the active SHI.
        """
        halfspaces = np.array(
            [
                [0.0, 0.0, 0.0],  # degenerate; dropped before feasibility
                [-1.0, 0.0, 0.0],  # x >= 0
                [1.0, 0.0, -1.0],  # x <= 1
                [0.0, 1.0, 0.0],  # y == 0 (equality)
            ]
        )
        p = _poly_from_halfspaces(halfspaces=halfspaces, zero_rows={3})
        assert list(p.zero_indices) == [3]

        bounded = p.get_bounded_halfspaces(bound=2.0)
        assert bounded.shape[0] >= 4  # inequalities + box; degenerate removed
        verts = p.get_bounded_vertices(bound=2.0)
        assert verts is not None
        verts = _sorted_rows(np.unique(verts, axis=0))
        assert verts.shape == (2, 2)
        assert np.allclose(verts, np.array([[0.0, 0.0], [1.0, 0.0]]), atol=1e-7)

    def test_segment_in_R3(self):
        # Segment from (0,0,0) to (1,0,0) embedded in R^3 with two equality rows.
        halfspaces = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],  # x >= 0
                [1.0, 0.0, 0.0, -1.0],  # x <= 1
                [0.0, 1.0, 0.0, 0.0],  # y == 0 (via zero index)
                [0.0, 0.0, 1.0, 0.0],  # z == 0 (via zero index)
            ]
        )
        p = _poly_from_halfspaces(halfspaces=halfspaces, zero_rows={2, 3})

        assert p.ambient_dim == 3
        assert p.codim == 2
        assert p.dim == 1

        assert p.finite is True
        assert p.center is not None
        assert np.allclose(p.center.reshape(-1), np.array([0.5, 0.0, 0.0]), atol=1e-7)
        assert p.inradius is not None
        assert np.isclose(p.inradius, 0.5, atol=1e-7)

        verts = p.get_bounded_vertices(bound=2.0)
        assert verts is not None
        verts = _sorted_rows(np.unique(verts, axis=0))
        assert verts.shape == (2, 3)
        assert np.allclose(verts, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), atol=1e-7)

        assert set(p.shis) == {0, 1}


def test_finite_true_without_chebyshev_cache_interior_and_center():
    """``finite=True`` from construction (e.g. boundary propagation) must not require ``_center``.

    ``vertices`` / Qhull paths require a network; this checks the LP interior path used by them.
    """
    halfspaces = np.array(
        [
            [-1.0, 0.0],
            [1.0, -1.0],
        ],
        dtype=np.float64,
    )
    ss = np.ones((1, 2), dtype=np.int8)
    p = Polyhedron(None, ss, halfspaces=halfspaces, finite=True)
    assert p._center is None
    ip = np.asarray(p.interior_point).reshape(-1)
    assert np.allclose(ip, np.array([0.5]), atol=1e-6)
    assert np.allclose(p.get_interior_point().reshape(-1), np.array([0.5]), atol=1e-6)
    assert p.center is not None
    assert p.inradius is not None
    assert np.isclose(float(p.inradius), 0.5, atol=1e-6)
