"""Tests for complex invariant verification."""

from __future__ import annotations

import numpy as np
import pytest

from relucent import Complex, mlp, set_seeds
from relucent.verify import ComplexNotCompleteError

os_environ = __import__("os").environ
os_environ.setdefault("DISABLE_RESEARCH_WARNING", "1")


def test_bfs_sets_verified_on_small_network() -> None:
    model = mlp(widths=[2, 4, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False)
    assert cplx.complete is True
    assert cplx.verified is True


def test_max_polys_disables_verify() -> None:
    model = mlp(widths=[2, 8, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, max_polys=3)
    assert cplx.complete is False
    assert cplx.verified is False


def test_incomplete_search_raises_with_verify() -> None:
    model = mlp(widths=[2, 8, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, max_polys=3)
    assert cplx.complete is False
    assert cplx.verified is False
    cplx2 = Complex(model)
    cplx2.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, max_polys=5000)
    assert cplx2.complete is True
    assert cplx2.verified is True


def test_empty_boundary_complex_verifies() -> None:
    """Output ReLU constant on all regions → empty boundary, not a verify crash."""
    set_seeds(3)
    model = mlp(widths=[2, 4, 1], add_last_relu=True, init="uniform")
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False)
    assert len(cplx.get_boundary_edges(cplx.n - 1)) == 0
    boundary = cplx.get_boundary_complex(cplx.n - 1)
    assert len(boundary) == 0
    assert boundary.complete is True
    assert boundary.verified is True
    assert boundary.get_dual_graph(verbose=False, verify=False).number_of_nodes() == 0


def test_assert_topology_ready_blocks_unverified() -> None:
    model = mlp(widths=[2, 4, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, max_polys=3)
    assert cplx.complete is False
    with pytest.raises(ComplexNotCompleteError):
        cplx.get_boundary_complex(cplx.n - 1)


def test_finalize_sync_corrects_asymmetric_shi_cache() -> None:
    """Top-cell ``_shis`` are re-derived from the dual graph, repairing asymmetric LP cache."""
    from relucent.utils import flip_ss_at_shi
    from relucent.verify import ShiFlipInvariantError, verify_shi_flip_symmetry

    model = mlp(widths=[2, 4, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, verify=False)
    top = next(p for p in cplx if p.dim == cplx.dim)
    if not top.shis:
        pytest.skip("no shis on top cell")
    shi = int(top.shis[0])
    neighbor = cplx[flip_ss_at_shi(np.asarray(top.ss_np, dtype=np.int8), shi)]
    neighbor._shis = [s for s in neighbor.shis if int(s) != shi]
    with pytest.raises(ShiFlipInvariantError):
        verify_shi_flip_symmetry(cplx)
    cplx.get_dual_graph(verbose=False, require_complete=False, verify=False, cubical=False)
    verify_shi_flip_symmetry(cplx)
