"""Unit tests for relucent.geometry.calculations helpers."""

import numpy as np
import pytest

from relucent._internal.gurobi import get_env
from relucent.core.complex import Complex
from relucent.geometry.calculations import (
    _drop_degenerate_halfspaces_tracked,
    _remap_zero_indices,
    solve_radius,
)
from relucent.model.builders import torch_mlp


def test_drop_degenerate_halfspaces_tracked_filters_and_maps():
    halfspaces = np.array(
        [
            [1.0, 0.0, -1.0],  # keep
            [0.0, 0.0, -0.5],  # drop: degenerate, always true
            [0.0, 1.0, -2.0],  # keep
            [0.0, 0.0, 0.0],  # drop: degenerate, neutral
        ]
    )

    filtered, old_to_new = _drop_degenerate_halfspaces_tracked(halfspaces)

    np.testing.assert_allclose(filtered, halfspaces[[0, 2]])
    np.testing.assert_array_equal(old_to_new, np.array([0, -1, 1, -1], dtype=np.intp))


def test_drop_degenerate_halfspaces_tracked_raises_on_infeasible_degenerate():
    halfspaces = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.0, 0.0, 0.1],  # 0*x + 0*y + 0.1 <= 0 is infeasible
        ]
    )

    with pytest.raises(ValueError, match="Degenerate halfspace\\(s\\) imply infeasibility"):
        _drop_degenerate_halfspaces_tracked(halfspaces)


def test_remap_zero_indices_drops_removed_rows():
    old_to_new = np.array([0, -1, 1, -1], dtype=np.intp)

    remapped = _remap_zero_indices(np.array([0, 1, 2, 3], dtype=np.intp), old_to_new)
    assert remapped is not None
    np.testing.assert_array_equal(remapped, np.array([0, 1], dtype=np.intp))

    assert _remap_zero_indices(np.array([1, 3], dtype=np.intp), old_to_new) is None
    assert _remap_zero_indices(None, old_to_new) is None
    assert _remap_zero_indices(np.array([], dtype=np.intp), old_to_new) is None


def test_solve_radius_raises_on_nonfinite_halfspaces():
    """NaN/Inf in halfspaces must fail fast with a clear error (not passed to Gurobi)."""
    env = get_env()
    hs_nan = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [float("nan"), float("nan"), float("nan")],
        ]
    )
    with pytest.raises(ValueError, match="Halfspaces contain NaN or Inf coefficients"):
        solve_radius(env, hs_nan)

    hs_inf = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [float("inf"), 0.0, -1.0],
        ]
    )
    with pytest.raises(ValueError, match="Halfspaces contain NaN or Inf coefficients"):
        solve_radius(env, hs_inf)


def test_solve_radius_no_inequalities_after_degenerate_drop():
    """All halfspaces degenerate and redundant → full space; must not call Gurobi with 0-row mats."""
    env = get_env()
    hs = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    center, r = solve_radius(env, hs, max_radius=50.0)
    assert center is not None
    np.testing.assert_allclose(center.ravel(), 0.0)
    assert r == 50.0


@pytest.mark.filterwarnings("ignore:Working with k<d polyhedron\\.:UserWarning")
def test_solve_radius_dependent_equalities_raise():
    """Dependent equality rows (x == 0 twice) are not a transversal intersection: raise, don't guess."""
    from relucent import AmbiguousGeometryError

    env = get_env()
    hs = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    with pytest.raises(AmbiguousGeometryError, match="dependent"):
        solve_radius(env, hs, max_radius=10.0, zero_indices=np.array([0, 1], dtype=np.intp))


@pytest.mark.filterwarnings("ignore:Working with k<d polyhedron\\.:UserWarning")
def test_solve_radius_equalities_only_unique_point():
    """Independent equalities pin a unique point → relative inradius 0."""
    env = get_env()
    hs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    center, r = solve_radius(env, hs, max_radius=100.0, zero_indices=np.array([0, 1], dtype=np.intp))
    assert center is not None
    np.testing.assert_allclose(center.ravel(), [0.0, 0.0], atol=1e-8)
    assert r == 0.0


def test_retain_geometry_caches_retains_requested_heavy_caches(seeded):
    """Requested geometry properties are kept; unrequested heavy caches are dropped."""
    from relucent.search.engine import retain_geometry_caches

    assert seeded is not None
    net = torch_mlp(widths=[2, 4, 1])
    cplx = Complex(net)
    p = cplx.add_point(np.zeros((1, 2)))
    p.get_geometry(["halfspaces", "W", "b", "interior_point"])

    retain_geometry_caches(p, ["halfspaces", "W", "b"])
    assert p._halfspaces is not None
    assert p._w is not None
    assert p._b is not None

    retain_geometry_caches(p, ["interior_point"])
    assert p._halfspaces is None
    assert p._w is None
    assert p._b is None
    assert p._interior_point is not None


def test_default_search_is_topology_only(seeded):
    """Default search skips optional geometry caches."""
    assert seeded is not None
    net = torch_mlp(widths=[2, 4, 1])
    cplx = Complex(net)
    cplx.bfs(max_polys=3, nworkers=1, verbose=0)
    for poly in cplx:
        assert poly._halfspaces is None
        assert poly._w is None
        assert poly._b is None
        assert poly.finite is not None


def test_search_all_geometry_properties_retains_caches(seeded):
    """Passing ALL_GEOMETRY_PROPERTIES computes and retains optional geometry caches during search."""
    from relucent.search import ALL_GEOMETRY_PROPERTIES

    assert seeded is not None
    net = torch_mlp(widths=[2, 4, 1])
    cplx = Complex(net)
    cplx.bfs(max_polys=3, nworkers=1, verbose=0, geometry_properties=ALL_GEOMETRY_PROPERTIES)
    for poly in cplx:
        assert poly._w is not None
        assert poly._b is not None
        assert poly._halfspaces is not None or poly._halfspaces_np is not None
