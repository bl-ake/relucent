"""GF(2) packed / sparse multiply vs dense reference."""

from __future__ import annotations

import numpy as np
import pytest

from relucent.topology.betti import (
    _mask_trailing_bits_in_last_word,
    _packed_to_dense_mod2,
    _packed_to_sparse_rowlists,
    gf2_matmul_packed_stacked_rows,
    gf2_matmul_sparse_rowlists,
)


def _dense_bool_to_packed(a01: np.ndarray) -> tuple[np.ndarray, int]:
    """``a01`` is uint8/bool 0/1 matrix; return packed uint64 rows and column count."""
    a = (np.asarray(a01) != 0).astype(np.uint8)
    m, n = int(a.shape[0]), int(a.shape[1])
    nwords = (n + 63) // 64
    out = np.zeros((m, nwords), dtype=np.uint64)
    for j in range(n):
        w = j >> 6
        sh = j & 63
        out[:, w] |= a[:, j].astype(np.uint64) << np.uint64(sh)
    return out, n


@pytest.mark.parametrize("m,n,p", [(3, 5, 4), (12, 40, 33), (1, 65, 64), (8, 127, 1)])
def test_gf2_matmul_packed_stacked_rows_matches_dense(m: int, n: int, p: int) -> None:
    rng = np.random.default_rng((m << 16) + (n << 8) + p)
    a = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
    b = rng.integers(0, 2, size=(n, p), dtype=np.uint8)
    dense_c = (a.astype(np.int64) @ b.astype(np.int64)) % 2

    p_a, nc_a = _dense_bool_to_packed(a)
    p_b, nc_b = _dense_bool_to_packed(b)
    p_c = gf2_matmul_packed_stacked_rows(p_a, nc_a, p_b, nc_b)
    _mask_trailing_bits_in_last_word(p_c, p)
    unpacked = _packed_to_dense_mod2(p_c, p)
    np.testing.assert_array_equal(unpacked, dense_c.astype(np.uint8))


def test_gf2_matmul_sparse_rowlists_matches_dense() -> None:
    rng = np.random.default_rng(7)
    m, n, p = 20, 30, 25
    a = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
    b = rng.integers(0, 2, size=(n, p), dtype=np.uint8)
    dense_c = (a.astype(np.int64) @ b.astype(np.int64)) % 2

    left_rows = [list(np.flatnonzero(a[i])) for i in range(m)]
    right_rows = [list(np.flatnonzero(b[t])) for t in range(n)]
    packed = gf2_matmul_sparse_rowlists(left_rows, right_rows, p)
    _mask_trailing_bits_in_last_word(packed, p)
    np.testing.assert_array_equal(_packed_to_dense_mod2(packed, p), dense_c.astype(np.uint8))


def test_gf2_matmul_large_sparse_chain_square_is_fast() -> None:
    """∂ with ~2 ones/column: sparse multiply must finish quickly on large dims."""
    rng = np.random.default_rng(11)
    n0, n1, n2 = 8_000, 12_000, 8_000
    # Build sparse ∂1 (n0 × n1) and ∂2 (n1 × n2) with two 1s per column.
    left_rows: list[list[int]] = [[] for _ in range(n0)]
    for j in range(n1):
        i0 = int(rng.integers(0, n0))
        i1 = int(rng.integers(0, n0))
        left_rows[i0].append(j)
        if i1 != i0:
            left_rows[i1].append(j)

    right_rows: list[list[int]] = [[] for _ in range(n1)]
    for j in range(n2):
        t0 = int(rng.integers(0, n1))
        t1 = int(rng.integers(0, n1))
        right_rows[t0].append(j)
        if t1 != t0:
            right_rows[t1].append(j)

    packed = gf2_matmul_sparse_rowlists(left_rows, right_rows, n2)
    assert packed.shape == (n0, (n2 + 63) // 64)


def test_packed_to_sparse_rowlists_roundtrip() -> None:
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, size=(9, 70), dtype=np.uint8)
    packed, ncols = _dense_bool_to_packed(a)
    rows = _packed_to_sparse_rowlists(packed, ncols)
    for i in range(9):
        assert sorted(rows[i]) == list(np.flatnonzero(a[i]))


def test_gf2_matmul_wrong_right_height_raises() -> None:
    left = np.zeros((2, 1), dtype=np.uint64)
    right = np.zeros((3, 1), dtype=np.uint64)
    with pytest.raises(ValueError, match="ncols_left"):
        gf2_matmul_packed_stacked_rows(left, 2, right, 5)
