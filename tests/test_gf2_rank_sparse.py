"""Sparse GF(2) rank matches packed elimination on random and boundary-like matrices."""

from __future__ import annotations

import numpy as np
import pytest

from relucent.topology import (
    _boundary_row_sets,
    _packed_boundary_matrix,
    _row_sets_to_packed,
    _transpose_packed,
    gf2_rank_boundary,
    gf2_rank_packed,
    gf2_rank_sparse_rowsets,
)


def _dense_rank_gf2(a: np.ndarray) -> int:
    m = (np.asarray(a) != 0).astype(np.uint8).copy()
    nrows, ncols = m.shape
    rank = 0
    for col in range(ncols):
        if rank >= nrows:
            break
        pivot = None
        for r in range(rank, nrows):
            if m[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != rank:
            m[[rank, pivot]] = m[[pivot, rank]]
        for r in range(nrows):
            if r != rank and m[r, col]:
                m[r] ^= m[rank]
        rank += 1
    return rank


def _random_sparse_rowsets(
    rng: np.random.Generator,
    nrows: int,
    ncols: int,
    *,
    nnz_per_row: int,
) -> list[set[int]]:
    row_sets: list[set[int]] = []
    for _ in range(nrows):
        cols = rng.choice(ncols, size=min(nnz_per_row, ncols), replace=False)
        row_sets.append(set(int(c) for c in cols))
    return row_sets


@pytest.mark.parametrize("nrows,ncols,nnz", [(40, 50, 3), (200, 180, 4), (500, 600, 2)])
def test_sparse_rank_matches_packed_random(nrows: int, ncols: int, nnz: int) -> None:
    rng = np.random.default_rng((nrows << 12) + ncols)
    row_sets = _random_sparse_rowsets(rng, nrows, ncols, nnz_per_row=nnz)
    packed = _row_sets_to_packed(row_sets, ncols)
    r_sparse = gf2_rank_sparse_rowsets([s.copy() for s in row_sets], ncols)
    r_packed = gf2_rank_packed(packed.copy(), ncols)
    dense = np.zeros((nrows, ncols), dtype=np.uint8)
    for i, cols in enumerate(row_sets):
        for j in cols:
            dense[i, j] = 1
    assert r_sparse == r_packed == _dense_rank_gf2(dense)


def test_sparse_rank_matches_packed_on_meta_like_grid() -> None:
    """Boundary-shaped matrix: few ones per column (face incidences)."""
    import networkx as nx

    meta = nx.MultiDiGraph()
    n_2 = 80
    n_1 = 120
    n_0 = 40
    nodes_2 = [f"2_{i}" for i in range(n_2)]
    nodes_1 = [f"1_{i}" for i in range(n_1)]
    nodes_0 = [f"0_{i}" for i in range(n_0)]
    for n, d in [*((x, 2) for x in nodes_2), *((x, 1) for x in nodes_1), *((x, 0) for x in nodes_0)]:
        meta.add_node(n, dim=d)
    rng = np.random.default_rng(0)
    for u in nodes_2:
        for v in rng.choice(nodes_1, size=3, replace=False):
            meta.add_edge(u, v)
    for u in nodes_1:
        for v in rng.choice(nodes_0, size=2, replace=False):
            meta.add_edge(u, v)

    nodes_by_dim: dict[int, list[object]] = {
        2: list(nodes_2),
        1: list(nodes_1),
        0: list(nodes_0),
    }
    for k in (1, 2):
        row_sets, ncols = _boundary_row_sets(meta, nodes_by_dim, k=k)
        packed, nc = _packed_boundary_matrix(meta, nodes_by_dim, k=k)
        assert nc == ncols
        assert gf2_rank_sparse_rowsets([s.copy() for s in row_sets], ncols) == gf2_rank_packed(packed.copy(), ncols)


@pytest.mark.requires_c_gf2
def test_c_backend_matches_dense_reference_random() -> None:
    """C backend returns same rank as brute-force GF(2) dense elimination."""
    from relucent._gf2 import gf2_rank_packed_c

    rng = np.random.default_rng(999)
    # Tiny matrices so _dense_rank_gf2 (pure Python) is fast.
    configs = [
        (15, 15, 3, 200),
        (20, 30, 4, 100),
        (30, 20, 4, 100),
        (25, 25, 8, 100),  # denser
        (25, 25, 2, 100),  # very sparse
    ]
    for nrows, ncols, nnz_per_row, trials in configs:
        for _ in range(trials):
            row_sets = _random_sparse_rowsets(rng, nrows, ncols, nnz_per_row=nnz_per_row)
            packed = _row_sets_to_packed(row_sets, ncols)
            dense = np.zeros((nrows, ncols), dtype=np.uint8)
            for i, cols in enumerate(row_sets):
                for j in cols:
                    dense[i, j] = 1
            expected = _dense_rank_gf2(dense)
            rc = gf2_rank_packed_c(packed.copy(), ncols)
            assert rc == expected, f"Mismatch ({nrows}×{ncols} nnz/row={nnz_per_row}): dense={expected} c={rc}"


@pytest.mark.requires_c_gf2
def test_c_backend_flip_parity_edge_case() -> None:
    """C Phase 1 must handle the flip-parity case: a row with bit_c=0 can still
    be a valid pivot when an existing pivot has bit_c set (any_pv_bit_c != 0)."""
    from relucent._gf2 import gf2_rank_packed_c

    # Build a 3×3 GF(2) matrix (packed into 1 uint64 per row):
    #   row 0: cols {0, 1}  → binary 011
    #   row 1: cols {0}     → binary 001  (bit_c1 = 0, but valid pivot for col 1
    #                                      after XOR with row 0)
    #   row 2: cols {2}     → binary 100
    # rank should be 3.
    ncols = 3
    p = np.array(
        [
            [np.uint64(0b011)],
            [np.uint64(0b001)],
            [np.uint64(0b100)],
        ],
        dtype=np.uint64,
    )
    expected = _dense_rank_gf2(np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.uint8))
    assert expected == 3
    rc = gf2_rank_packed_c(p.copy(), ncols)
    assert rc == 3, f"Expected rank 3, got {rc}"

    # Also: a 2×2 case where the only pivot for col 1 has bit_c1=0 originally.
    #   row 0: cols {0, 1}  → 011
    #   row 1: cols {0}     → 001
    # rank = 2
    p2 = np.array([[np.uint64(0b11)], [np.uint64(0b01)]], dtype=np.uint64)
    assert gf2_rank_packed_c(p2.copy(), 2) == 2


def test_transpose_rank_matches_direct() -> None:
    nrows, ncols = 500, 8000
    rng = np.random.default_rng(1)
    row_sets = _random_sparse_rowsets(rng, nrows, ncols, nnz_per_row=4)
    packed = _row_sets_to_packed(row_sets, ncols)
    assert gf2_rank_boundary(packed.copy(), ncols) == gf2_rank_packed(packed.copy(), ncols)
    transposed, ncols_t = _transpose_packed(packed, ncols)
    assert gf2_rank_packed(transposed.copy(), ncols_t) == gf2_rank_packed(packed.copy(), ncols)
