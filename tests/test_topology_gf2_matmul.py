"""GF(2) packed multiply (stacked-rows) vs dense reference."""

from __future__ import annotations

import numpy as np
import pytest

from relucent.topology.betti import _mask_trailing_bits_in_last_word, _packed_to_dense_mod2, gf2_matmul_packed_stacked_rows


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


def test_gf2_matmul_wrong_right_height_raises() -> None:
    left = np.zeros((2, 1), dtype=np.uint64)
    right = np.zeros((3, 1), dtype=np.uint64)
    with pytest.raises(ValueError, match="ncols_left"):
        gf2_matmul_packed_stacked_rows(left, 2, right, 5)
