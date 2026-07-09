"""Tests for complex invariant verification."""

from __future__ import annotations

import numpy as np
import pytest

from relucent import Complex, mlp, set_seeds
from relucent.complex import IncompleteDualGraphError
from relucent.exploration import finalize_ambient_search
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


def test_assert_topology_ready_blocks_add_point_only() -> None:
    """Point sampling without BFS does not satisfy get_boundary_complex prerequisites."""
    import torch
    import torch.nn as nn

    fc = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    model = nn.Sequential(fc, nn.ReLU())
    cplx = Complex(model)
    for x in np.array([[-0.1, 0.0], [0.1, 0.0], [-0.1, 1.0], [0.1, -1.0]]):
        cplx.add_point(x.reshape(1, -1), check_exists=True)
    with pytest.raises(ComplexNotCompleteError, match="explore_for_topology|BFS"):
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


def test_finalize_ambient_search_reuses_dual_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    model = mlp(widths=[2, 4, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, verify=False)

    orig_get_dual_graph = cplx.get_dual_graph
    calls = {"count": 0}

    def _counting_get_dual_graph(*args, **kwargs):
        calls["count"] += 1
        return orig_get_dual_graph(*args, **kwargs)

    monkeypatch.setattr(cplx, "get_dual_graph", _counting_get_dual_graph)

    finalize_ambient_search(cplx, complete=True, verify=True)

    assert calls["count"] == 1
    assert cplx.verified is True


def test_complete_verify_fails_closed_on_shi_recompute_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import relucent.calculations as calc
    from relucent.verify import verify_lp_flip_neighbors_in_complex

    model = mlp(widths=[2, 4, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, verify=False)
    cplx.set_exploration_state(complete=True, verified=False)
    orig_get_shis = calc.get_shis

    def _boom(poly, *args, **kwargs):
        if getattr(poly, "_shis_strict", False):
            return orig_get_shis(poly, *args, **kwargs)
        raise ValueError("synthetic SHI failure")

    monkeypatch.setattr(calc, "get_shis", _boom)

    with pytest.raises(IncompleteDualGraphError, match="failed to recompute SHIs"):
        verify_lp_flip_neighbors_in_complex(cplx, nworkers=1)


def test_get_boundary_complex_reuses_strict_cached_shis_from_verified_bfs(monkeypatch: pytest.MonkeyPatch) -> None:
    import relucent.calculations as calc

    model = mlp(widths=[2, 4, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, verify=True)

    def _boom(*args, **kwargs):
        raise AssertionError("strict SHIs should be reused instead of recomputed")

    monkeypatch.setattr(calc, "get_shis", _boom)

    boundary = cplx.get_boundary_complex(cplx.n - 1)
    assert boundary.verified is True


def test_get_boundary_complex_reuses_strict_shis_after_dual_graph_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    import relucent.calculations as calc

    model = mlp(widths=[2, 4, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, verify=True)
    graph = cplx.get_dual_graph(relabel=True, verbose=False)
    initial_ss = cplx.index2poly[0].ss_np

    reloaded = Complex(model)
    reloaded.recover_from_dual_graph(graph, initial_ss=initial_ss, source=0)
    assert reloaded.verified is True
    top = next(p for p in reloaded if p.dim == reloaded.dim)
    assert top._shis_strict is True

    reloaded.get_poly_attrs(["finite"])
    top = next(p for p in reloaded if p.dim == reloaded.dim)
    assert top._shis_strict is True

    def _boom(*args, **kwargs):
        raise AssertionError("strict SHIs should be reused instead of recomputed")

    monkeypatch.setattr(calc, "get_shis", _boom)

    boundary = reloaded.get_boundary_complex(reloaded.n - 1)
    assert boundary.verified is True


def test_lp_verify_reuses_strict_cached_shis_from_verified_bfs(monkeypatch: pytest.MonkeyPatch) -> None:
    import relucent.calculations as calc
    from relucent.verify import verify_lp_flip_neighbors_in_complex

    model = mlp(widths=[2, 4, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, verify=True)
    cplx.set_exploration_state(complete=True, verified=False)

    def _boom(*args, **kwargs):
        raise AssertionError("strict SHIs should be reused instead of recomputed")

    monkeypatch.setattr(calc, "get_shis", _boom)

    verify_lp_flip_neighbors_in_complex(cplx, nworkers=1)


def test_lp_verify_serial_and_parallel_agree_on_complete_complex() -> None:
    from relucent.verify import verify_lp_flip_neighbors_in_complex

    model = mlp(widths=[2, 4, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, verify=False)
    cplx.set_exploration_state(complete=True, verified=False)

    verify_lp_flip_neighbors_in_complex(cplx, nworkers=1)
    verify_lp_flip_neighbors_in_complex(cplx, nworkers=2)


def test_lp_verify_serial_and_parallel_agree_on_incomplete_complex() -> None:
    from relucent.verify import verify_lp_flip_neighbors_in_complex

    model = mlp(widths=[2, 8, 1], add_last_relu=True)
    cplx = Complex(model)
    cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, max_polys=3, verify=False)
    cplx.set_exploration_state(complete=True, verified=False)

    with pytest.raises(IncompleteDualGraphError) as serial_err:
        verify_lp_flip_neighbors_in_complex(cplx, nworkers=1)
    with pytest.raises(IncompleteDualGraphError) as parallel_err:
        verify_lp_flip_neighbors_in_complex(cplx, nworkers=2)

    assert "Dual graph is incomplete relative to LP facets" in str(serial_err.value)
    assert str(serial_err.value) == str(parallel_err.value)
