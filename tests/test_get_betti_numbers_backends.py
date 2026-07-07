"""``get_betti_numbers`` must agree when GF(2) rank uses the C vs Python backends.

The C path is selected inside :func:`~relucent.topology.gf2_rank_boundary` when
``topology._c_backend`` is true (see :data:`~relucent.topology.C_BACKEND_AVAILABLE`).
These tests force each backend and compare full Betti dictionaries on small complexes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import networkx as nx
import numpy as np
import pytest
import torch
import torch.nn as nn

import relucent.topology as topology
from relucent import Complex, set_seeds
from relucent.exploration import explore_for_topology
from relucent.topology import C_BACKEND_AVAILABLE, ConnectedComponentsMismatch, get_betti_numbers


def _make_meta(dim_edges: list[tuple[int, int, int]]) -> nx.MultiDiGraph[Any]:
    """Build a synthetic meta-graph from (source_dim, source_id, target_id) triples.

    Nodes are (dim, id) pairs with a ``dim`` attribute; edges go from k-cells to
    (k-1)-cells to represent face incidences, matching the convention expected by
    :func:`get_betti_numbers`.
    """
    g: nx.MultiDiGraph[Any] = nx.MultiDiGraph()
    for src_dim, src_id, tgt_id in dim_edges:
        src = (src_dim, src_id)
        tgt = (src_dim - 1, tgt_id)
        if src not in g:
            g.add_node(src, dim=src_dim)
        if tgt not in g:
            g.add_node(tgt, dim=src_dim - 1)
        g.add_edge(src, tgt, shi=0)
    return g


def _isolated_meta(dim: int, n: int) -> nx.MultiDiGraph[Any]:
    """Return a meta-graph with ``n`` isolated nodes of ``dim`` (no edges, no lower cells)."""
    g: nx.MultiDiGraph[Any] = nx.MultiDiGraph()
    for i in range(n):
        g.add_node((dim, i), dim=dim)
    return g


def _make_unbounded_two_component_meta() -> nx.MultiDiGraph[Any]:
    """kmin=1 meta-graph with two disconnected groups, all cells unbounded (for truncation)."""
    meta: nx.MultiDiGraph[Any] = nx.MultiDiGraph()
    for dim, idx in [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)]:
        meta.add_node(
            (dim, idx),
            dim=dim,
            finite=False,
            ss=np.array([[1]], dtype=np.int8),
        )
    meta.add_edge((2, 0), (1, 0), shi=0)
    meta.add_edge((2, 0), (1, 1), shi=1)
    meta.add_edge((2, 1), (1, 2), shi=0)
    meta.add_edge((2, 1), (1, 3), shi=1)
    return meta


def _set_gf2_backend(monkeypatch: pytest.MonkeyPatch, *, use_c: bool) -> None:
    if use_c and not C_BACKEND_AVAILABLE:
        pytest.skip("C GF(2) backend not available (gcc compile/load failed)")
    monkeypatch.setattr(topology, "_c_backend", use_c)


def _betti_for_backend(
    monkeypatch: pytest.MonkeyPatch,
    cplx: Complex,
    *,
    use_c: bool,
    **kwargs: Any,
) -> dict[int, int]:
    _set_gf2_backend(monkeypatch, use_c=use_c)
    return cplx.get_betti_numbers(**kwargs)


def _add_points(cplx: Complex, pts: np.ndarray) -> None:
    for x in np.asarray(pts, dtype=np.float64):
        ss = cplx.point2ss(x.reshape(1, -1))
        if (np.asarray(ss) == 0).any():
            continue
        cplx.add_point(x.reshape(1, -1), check_exists=True)


def _diamond_boundary_model(radius: float = 1.0) -> nn.Sequential:
    fc0 = nn.Linear(2, 6, bias=False, dtype=torch.float64)
    base = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    fc0.weight.data[:] = base + 1e-3 * torch.randn_like(base)
    fc1 = nn.Linear(6, 2, bias=False, dtype=torch.float64)
    fc1.weight.data[:] = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    fc2 = nn.Linear(2, 1, bias=True, dtype=torch.float64)
    fc2.weight.data[:] = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    fc2.bias.data[:] = torch.tensor([-float(radius)], dtype=torch.float64)
    return nn.Sequential(fc0, nn.ReLU(), fc1, fc2, nn.ReLU())


def _populate_diamond_boundary(seed: int) -> Complex:
    rng = np.random.default_rng(seed)
    cplx = Complex(_diamond_boundary_model())
    thetas = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    _add_points(cplx, np.vstack([0.9 * dirs, 1.1 * dirs, rng.standard_normal((80, 2))]))
    return cplx.get_boundary_complex(cplx.n - 1)


def _populate_line_boundary(seed: int) -> Complex:
    rng = np.random.default_rng(seed)
    fc = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    cplx = Complex(nn.Sequential(fc, nn.ReLU()))
    xs = np.linspace(-2.0, 2.0, 21)
    ys = np.linspace(-2.0, 2.0, 21)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    eps = 1e-2
    left = grid.copy()
    left[:, 0] = -eps
    right = grid.copy()
    right[:, 0] = eps
    _add_points(cplx, np.vstack([left, right, rng.standard_normal((80, 2))]))
    return cplx.get_boundary_complex(cplx.n - 1)


def _populate_small_1d_complex(seed: int) -> Complex:
    set_seeds(seed)
    model = nn.Sequential(nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 1), nn.ReLU())
    cplx = Complex(model)
    explore_for_topology(cplx, np.array([0.0]))
    return cplx


@pytest.mark.python_gf2
def test_get_betti_numbers_python_backend_smoke(seeded: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pure-Python GF(2) rank path runs end-to-end (no C required)."""
    db = _populate_diamond_boundary(seeded)
    betti = _betti_for_backend(monkeypatch, db, use_c=False)
    assert isinstance(betti, dict)
    assert int(betti.get(1, 0)) >= 1


@pytest.mark.requires_c_gf2
def test_c_gf2_backend_available() -> None:
    """CI on Linux must compile and load ``_gf2_rank.c`` (see workflow job ``gf2-backend``)."""
    assert C_BACKEND_AVAILABLE, "C GF(2) backend failed to compile or load; get_betti_numbers would use slow Python rank only"


@pytest.mark.requires_c_gf2
@pytest.mark.parametrize(
    "build_cplx,kwargs",
    [
        (_populate_diamond_boundary, {}),
        (_populate_diamond_boundary, {"compactify": True, "reduced": True}),
        (_populate_diamond_boundary, {"verify_chain_complex": True}),
        (_populate_line_boundary, {}),
        (_populate_line_boundary, {"compactify": True}),
        (_populate_line_boundary, {"respect_finite": True}),
        (_populate_small_1d_complex, {}),
        (_populate_small_1d_complex, {"compactify": True}),
    ],
)
def test_get_betti_numbers_c_matches_python(
    seeded: int,
    monkeypatch: pytest.MonkeyPatch,
    build_cplx: Callable[[int], Complex],
    kwargs: dict[str, Any],
) -> None:
    """C and Python ``gf2_rank_boundary`` backends yield identical Betti numbers."""
    cplx = build_cplx(seeded)
    assert len(cplx) > 0
    betti_c = _betti_for_backend(monkeypatch, cplx, use_c=True, **kwargs)
    betti_py = _betti_for_backend(monkeypatch, cplx, use_c=False, **kwargs)
    assert betti_c == betti_py, f"C {betti_c} != Python {betti_py} (kwargs={kwargs})"


def test_complex_get_betti_numbers_delegates_to_topology(seeded: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Public :meth:`~relucent.complex.Complex.get_betti_numbers`` matches topology module."""
    db = _populate_diamond_boundary(seeded)
    _set_gf2_backend(monkeypatch, use_c=C_BACKEND_AVAILABLE)
    meta = db.get_meta_graph(verbose=False)
    Complex.truncate_meta_graph(meta)
    via_topology = get_betti_numbers(meta)
    via_complex = db.get_betti_numbers(compactify=False)
    assert via_topology == via_complex


@pytest.mark.requires_c_gf2
@pytest.mark.parametrize(
    "build_cplx,kwargs",
    [
        (_populate_diamond_boundary, {}),
        (_populate_diamond_boundary, {"compactify": True, "reduced": True}),
        (_populate_diamond_boundary, {"verify_chain_complex": True}),
        (_populate_line_boundary, {}),
        (_populate_small_1d_complex, {}),
    ],
)
def test_get_betti_numbers_parallel_matches_sequential(
    seeded: int,
    monkeypatch: pytest.MonkeyPatch,
    build_cplx: Callable[[int], Complex],
    kwargs: dict[str, Any],
) -> None:
    """Parallel ranking (nworkers>1) yields the same Betti numbers as sequential (nworkers=1)."""
    _set_gf2_backend(monkeypatch, use_c=True)
    cplx = build_cplx(seeded)
    assert len(cplx) > 0
    betti_seq = cplx.get_betti_numbers(nworkers=1, **kwargs)
    betti_par = cplx.get_betti_numbers(nworkers=4, **kwargs)
    assert betti_seq == betti_par, f"sequential {betti_seq} != parallel {betti_par} (kwargs={kwargs})"


# ---------------------------------------------------------------------------
# Tests for complexes with kmin > 0 (no 0-cells in the chain)
# ---------------------------------------------------------------------------


def test_get_betti_numbers_kmin1_two_isolated_1cells() -> None:
    """Two isolated 1-cells (no 0-cells, no 2-cells) → β₁ = 2.

    When kmin = 1 there is no ∂₁ (C₀ = 0), so the "bottom" of the chain is
    dimension 1.  β_{kmin} = n_{kmin} − rank(∂_{kmin+1}) = 2 − 0 = 2.
    The result must NOT be an empty dict.
    """
    meta = _isolated_meta(dim=1, n=2)
    betti = get_betti_numbers(meta)
    assert betti.get(1) == 2, f"expected {{1: 2}}, got {betti}"
    assert 0 not in betti, f"key 0 should be absent when kmin=1, got {betti}"


def test_get_betti_numbers_kmin1_two_components_via_2cells() -> None:
    """kmin = 1 with two groups of 1-cells connected internally by 2-cells.

    Group P: 1-cells (1,0) and (1,1) both face 2-cell (2,0).
    Group Q: 1-cells (1,2) and (1,3) both face 2-cell (2,1).
    The two groups are disconnected → β₁ = 2 (two connected components).

    Chain: C₂ →^{∂₂} C₁ →^{∂₁=0} 0
    n₁=4, n₂=2, rank(∂₂)=2 (each 2-cell contributes one independent boundary).
    β₁ = 4 − 0 − 2 = 2.
    """
    meta: nx.MultiDiGraph[Any] = nx.MultiDiGraph()
    for dim, idx in [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)]:
        meta.add_node((dim, idx), dim=dim)
    # 2-cell (2,0) is bounded by 1-cells (1,0) and (1,1)
    meta.add_edge((2, 0), (1, 0), shi=0)
    meta.add_edge((2, 0), (1, 1), shi=1)
    # 2-cell (2,1) is bounded by 1-cells (1,2) and (1,3)
    meta.add_edge((2, 1), (1, 2), shi=0)
    meta.add_edge((2, 1), (1, 3), shi=1)

    betti = get_betti_numbers(meta)
    assert betti.get(1) == 2, f"expected β₁=2 (two disconnected groups of 1-cells), got {betti}"
    assert 0 not in betti, f"key 0 should be absent when kmin=1, got {betti}"


def test_get_betti_numbers_kmin1_single_component() -> None:
    """kmin = 1, four 1-cells connected into one component via three 2-cells → β₁ = 1."""
    # Chain: C₂ →^{∂₂} C₁;  n₁=4, n₂=3
    # 2-cell (2,0): faces (1,0),(1,1);  (2,1): faces (1,1),(1,2);  (2,2): faces (1,2),(1,3)
    # rank(∂₂) = 3  → β₁ = 4 − 3 = 1
    meta: nx.MultiDiGraph[Any] = nx.MultiDiGraph()
    for dim, idx in [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2)]:
        meta.add_node((dim, idx), dim=dim)
    meta.add_edge((2, 0), (1, 0), shi=0)
    meta.add_edge((2, 0), (1, 1), shi=1)
    meta.add_edge((2, 1), (1, 1), shi=0)
    meta.add_edge((2, 1), (1, 2), shi=1)
    meta.add_edge((2, 2), (1, 2), shi=0)
    meta.add_edge((2, 2), (1, 3), shi=1)

    betti = get_betti_numbers(meta)
    assert betti.get(1) == 1, f"expected β₁=1 (one component), got {betti}"
    assert 0 not in betti, f"key 0 should be absent when kmin=1, got {betti}"


def test_get_betti_numbers_kmin0_unaffected() -> None:
    """When kmin = 0 (0-cells present), behaviour is identical to before the fix.

    Two isolated 0-cells → β₀ = 2 (standard connected-components formula).
    """
    meta = _isolated_meta(dim=0, n=2)
    betti = get_betti_numbers(meta)
    assert betti.get(0) == 2, f"expected {{0: 2}}, got {betti}"


def test_beta0_truncated_two_components() -> None:
    """Truncated unbounded complex: β₀ equals graph component count (not rank formula)."""
    meta = _make_unbounded_two_component_meta()
    Complex.truncate_meta_graph(meta)
    betti = get_betti_numbers(meta)
    assert betti.get(0) == 2, f"expected β₀=2 after truncation, got {betti}"


def test_verify_connected_components_raises() -> None:
    """verify_connected_components surfaces rank-formula vs graph mismatch."""
    meta = _make_unbounded_two_component_meta()
    Complex.truncate_meta_graph(meta)
    with pytest.raises(ConnectedComponentsMismatch) as exc_info:
        get_betti_numbers(meta, verify_connected_components=True)
    err = exc_info.value
    assert err.rank_beta0 == 0
    assert err.graph_beta0 == 2


def test_verify_connected_components_passes_for_proper_complex() -> None:
    """Proper CW complex: rank formula and graph connectivity agree for β₀."""
    meta = _isolated_meta(dim=0, n=2)
    betti = get_betti_numbers(meta, verify_connected_components=True)
    assert betti.get(0) == 2, f"expected {{0: 2}}, got {betti}"
