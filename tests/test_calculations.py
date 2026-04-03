"""Unit tests for relucent.calculations helpers."""

import numpy as np
import pytest

from relucent.calculations import (
    _drop_degenerate_halfspaces_tracked,
    _halfspaces_feasible,
    _remap_zero_indices,
    solve_radius,
)
from relucent.utils import get_env


def test_drop_degenerate_halfspaces_tracked_filters_and_maps():
    halfspaces = np.array(
        [
            [1.0, 0.0, -1.0],  # keep
            [0.0, 0.0, -0.5],  # drop: degenerate, always true
            [0.0, 1.0, -2.0],  # keep
            [0.0, 0.0, 0.0],  # drop: degenerate, neutral
        ]
    )

    filtered, old_to_new = _drop_degenerate_halfspaces_tracked(
        halfspaces,
        tol_normal=1e-12,
        tol_bias=1e-12,
    )

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
        _drop_degenerate_halfspaces_tracked(
            halfspaces,
            tol_normal=1e-12,
            tol_bias=1e-12,
        )


def test_remap_zero_indices_drops_removed_rows():
    old_to_new = np.array([0, -1, 1, -1], dtype=np.intp)

    remapped = _remap_zero_indices(np.array([0, 1, 2, 3], dtype=np.intp), old_to_new)
    assert remapped is not None
    np.testing.assert_array_equal(remapped, np.array([0, 1], dtype=np.intp))

    assert _remap_zero_indices(np.array([1, 3], dtype=np.intp), old_to_new) is None
    assert _remap_zero_indices(None, old_to_new) is None
    assert _remap_zero_indices(np.array([], dtype=np.intp), old_to_new) is None


def test_halfspaces_feasible_true_and_false():
    env = get_env()

    feasible = np.array(
        [
            [1.0, -1.0],  # x <= 1
            [-1.0, 0.0],  # x >= 0
        ]
    )
    infeasible = np.array(
        [
            [1.0, -1.0],  # x <= 1
            [-1.0, 2.0],  # x >= 2
        ]
    )

    assert _halfspaces_feasible(env, feasible, zero_indices=None) is True
    assert _halfspaces_feasible(env, infeasible, zero_indices=None) is False


def test_halfspaces_feasible_with_zero_indices_equalities():
    env = get_env()

    # Row 0 is treated as equality x == 1; row 1 enforces x <= 2.
    feasible = np.array(
        [
            [1.0, -1.0],
            [1.0, -2.0],
        ]
    )
    # Row 0 is equality x == 2; row 1 enforces x <= 1 (infeasible).
    infeasible = np.array(
        [
            [1.0, -2.0],
            [1.0, -1.0],
        ]
    )

    assert _halfspaces_feasible(env, feasible, zero_indices=np.array([0], dtype=np.intp)) is True
    assert _halfspaces_feasible(env, infeasible, zero_indices=np.array([0], dtype=np.intp)) is False


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
def test_solve_radius_equalities_only_after_split():
    """Inequalities empty but equalities remain — avoid 0-row norm_vector * y in Gurobi."""
    env = get_env()
    # x == 0, y free: two equalities in R^2, no strict inequalities.
    hs = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    center, r = solve_radius(env, hs, max_radius=10.0, zero_indices=np.array([0, 1], dtype=np.intp))
    assert center is not None
    np.testing.assert_allclose(center.ravel(), [0.0, 0.0], atol=1e-8)
    # Redundant equalities x=0: rank 1 → affine line; relative inradius hits max_radius cap.
    assert r == 10.0


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
