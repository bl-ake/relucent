"""Topology helpers: Betti numbers over GF(2) for ReLU cell complexes.

This module implements the "direct boundary matrix" approach described in the
user-provided pseudocode:

- build a chosen subcomplex and close under faces
- construct boundary operators ∂_k over GF(2)
- compute Betti numbers from boundary ranks

Notes:
- Coefficients are in GF(2), so orientations are irrelevant.
- Cells are represented by :class:`relucent.poly.Polyhedron` objects; a codimension-1
  facet of a cell is obtained by setting one nonzero sign entry to 0.
- For ``compactify=False``, :func:`get_betti_numbers` uses a truncated meta-graph so
  non-compact complexes are modeled combinatorially without separate ``infinity`` modes.
- Pass ``verify_chain_complex=True`` to require ``∂²=0`` on the assembled boundary maps;
  the check uses a packed GF(2) multiply (stacked rows), not dense integer matmuls.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from relucent.complex import Complex

# Try to load the C backend once at import time.
try:
    from relucent._gf2 import (
        available as _c_available,
    )
    from relucent._gf2 import (
        gf2_rank_boundary_c as _gf2_rank_boundary_c,
    )
    # from relucent._gf2 import (
    #     gf2_transpose_packed_c as _gf2_transpose_packed_c,
    # )

    _C_BACKEND: bool = _c_available()
except Exception:
    _C_BACKEND = False

__all__ = [
    "ChainComplexInconsistent",
    "C_BACKEND_AVAILABLE",
    "get_betti_numbers",
    "gf2_matmul_packed_stacked_rows",
    "gf2_rank_boundary",
    "gf2_rank_packed",
    "gf2_rank_sparse_rowsets",
]

# Warn when a single ∂_k rank may take a long time (nearly square, large, pure-Python only).
_SLOW_RANK_MIN_DIM = 50_000

# Public flag: True when _gf2_rank.c compiled and loaded successfully.
C_BACKEND_AVAILABLE: bool = _C_BACKEND


def _verbose_line(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"relucent.topology: {msg}", file=sys.stderr, flush=True)


class ChainComplexInconsistent(RuntimeError):
    """Meta-graph boundary maps do not compose to zero (∂²≠0 over GF(2)).

    In that situation the rank formula ``β_k = n_k - rank(∂_k) - rank(∂_{k+1})`` is not
    guaranteed to count homology and can even be negative. See ``violations`` for where
    ``∂_k ∘ ∂_{k+1}`` is nonzero mod 2.
    """

    def __init__(self, violations: list[dict[str, Any]], message: str | None = None) -> None:
        self.violations = violations
        if message is None:
            parts = [f"k={v['k']} (∂_{v['k']}∘∂_{v['k'] + 1}): nnz={v['nnz']} shape={v['shape']}" for v in violations]
            message = "Meta-graph is not a cellular chain complex over GF(2): " + "; ".join(parts)
        super().__init__(message)


def _packed_boundary_matrix(
    meta: Any,
    nodes_by_dim: dict[int, list[object]],
    *,
    k: int,
    compactify: bool,
) -> tuple[np.ndarray, int]:
    """Bit-packed ∂_k: C_k → C_{k-1} (rows index (k−1)-cells, columns index k-cells)."""
    rows = nodes_by_dim.get(k - 1, [])
    cols = nodes_by_dim.get(k, [])
    if not rows or not cols:
        return np.zeros((0, 0), dtype=np.uint64), 0

    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: j for j, c in enumerate(cols)}

    nrows = len(rows)
    ncols = len(cols)
    nwords = (ncols + 63) // 64
    packed = np.zeros((nrows, nwords), dtype=np.uint64)

    inc_count: dict[object, int] | None = None
    if compactify:
        inc_count = {r: 0 for r in rows}
        for u, v, _ in meta.edges(data=True):
            if u in col_index and v in row_index:
                inc_count[v] += 1

    for u, v, _ in meta.edges(data=True):
        if u not in col_index or v not in row_index:
            continue
        if inc_count is not None and inc_count[v] < 2:
            continue
        j = int(col_index[u])
        i = int(row_index[v])
        w = j >> 6
        bit = np.uint64(1) << (j & 63)
        packed[i, w] ^= bit

    return packed, ncols


def _boundary_row_sets(
    meta: Any,
    nodes_by_dim: dict[int, list[object]],
    *,
    k: int,
    compactify: bool,
) -> tuple[list[set[int]], int]:
    """Sparse ∂_k as a list of row sets (row index → column indices with a 1)."""
    rows = nodes_by_dim.get(k - 1, [])
    cols = nodes_by_dim.get(k, [])
    if not rows or not cols:
        return [], 0

    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: j for j, c in enumerate(cols)}

    nrows = len(rows)
    ncols = len(cols)
    row_sets: list[set[int]] = [set() for _ in range(nrows)]

    inc_count: dict[object, int] | None = None
    if compactify:
        inc_count = {r: 0 for r in rows}
        for u, v, _ in meta.edges(data=True):
            if u in col_index and v in row_index:
                inc_count[v] += 1

    for u, v, _ in meta.edges(data=True):
        if u not in col_index or v not in row_index:
            continue
        if inc_count is not None and inc_count[v] < 2:
            continue
        j = int(col_index[u])
        i = int(row_index[v])
        row_sets[i].add(j)

    return row_sets, ncols


def _swap_rows_in_col_index(
    col_to_rows: dict[int, set[int]],
    row_sets: list[set[int]],
    i: int,
    j: int,
) -> None:
    row_sets[i], row_sets[j] = row_sets[j], row_sets[i]
    for c in row_sets[i] | row_sets[j]:
        rows = col_to_rows.setdefault(c, set())
        rows.discard(i)
        rows.discard(j)
        if c in row_sets[i]:
            rows.add(i)
        if c in row_sets[j]:
            rows.add(j)
        if not rows:
            col_to_rows.pop(c, None)


def gf2_rank_sparse_rowsets(
    row_sets: list[set[int]],
    ncols: int,
    *,
    progress: bool = False,
    progress_desc: str | None = None,
) -> int:
    """Gaussian elimination rank over GF(2) on sparse row sets.

    Each row is a set of column indices where the matrix entry is 1.  An inverted
    index (column → rows) avoids scanning all rows on every elimination step, which
    matters when boundaries have hundreds of thousands of cells but only a few
    incidences per column.
    """
    nrows = len(row_sets)
    if nrows == 0 or ncols == 0:
        return 0

    col_to_rows: dict[int, set[int]] = {}
    for r, cols in enumerate(row_sets):
        for c in cols:
            col_to_rows.setdefault(c, set()).add(r)

    rank = 0
    col_iter: Iterable[int] = range(ncols)
    if progress:
        col_iter = tqdm(
            col_iter,
            desc=progress_desc or "GF(2) rank",
            leave=False,
            total=ncols,
        )

    for col in col_iter:
        if rank >= nrows:
            break
        candidates = [r for r in col_to_rows.get(col, ()) if r >= rank]
        if not candidates:
            continue
        pivot = min(candidates)
        if pivot != rank:
            _swap_rows_in_col_index(col_to_rows, row_sets, rank, pivot)

        pivot_row = row_sets[rank]
        for r in list(col_to_rows.get(col, ())):
            if r == rank or r < rank:
                continue
            if col not in row_sets[r]:
                continue
            old = row_sets[r]
            new = old ^ pivot_row
            if new is old:
                continue
            row_sets[r] = new
            for c in old.symmetric_difference(new):
                rows = col_to_rows.setdefault(c, set())
                if c in new:
                    rows.add(r)
                else:
                    rows.discard(r)
                    if not rows:
                        col_to_rows.pop(c, None)
        rank += 1

    return rank


def _row_sets_to_packed(row_sets: list[set[int]], ncols: int) -> np.ndarray:
    nrows = len(row_sets)
    nwords = (ncols + 63) // 64
    packed = np.zeros((nrows, nwords), dtype=np.uint64)
    for i, cols in enumerate(row_sets):
        for j in cols:
            w = j >> 6
            packed[i, w] ^= np.uint64(1) << (j & 63)
    return packed


def _transpose_packed(packed: np.ndarray, ncols: int) -> tuple[np.ndarray, int]:
    """Return ``A^T`` for a bit-packed GF(2) matrix ``A`` with ``ncols`` logical columns."""
    nrows_in = int(packed.shape[0])
    nwords_in = int(packed.shape[1])
    nrows_out = int(ncols)
    ncols_out = nrows_in
    nwords_out = (ncols_out + 63) // 64
    out = np.zeros((nrows_out, nwords_out), dtype=np.uint64)
    for r in range(nrows_in):
        for w in range(nwords_in):
            word = int(packed[r, w])
            if word == 0:
                continue
            base = w << 6
            while word:
                b = (word & -word).bit_length() - 1
                word &= word - 1
                c = base + b
                if c >= ncols:
                    continue
                tw = r >> 6
                tb = r & 63
                out[c, tw] ^= np.uint64(1) << tb
    return out, ncols_out


def _packed_to_dense_mod2(packed: np.ndarray, ncols: int) -> np.ndarray:
    if packed.size == 0 or ncols == 0:
        return np.zeros((int(packed.shape[0]), ncols), dtype=np.uint8)
    out = np.zeros((packed.shape[0], ncols), dtype=np.uint8)
    for j in range(ncols):
        w = j >> 6
        sh = j & 63
        out[:, j] = ((packed[:, w] >> sh) & np.uint64(1)).astype(np.uint8)
    return out


def _mask_trailing_bits_in_last_word(packed: np.ndarray, ncols: int) -> None:
    """In-place: clear garbage bits strictly beyond column ``ncols-1`` in the last u64 word."""
    if packed.size == 0 or ncols == 0:
        return
    tail = int(ncols) & 63
    if tail:
        wmask = (np.uint64(1) << np.uint64(tail)) - np.uint64(1)
        packed[:, -1] &= wmask


def gf2_matmul_packed_stacked_rows(
    left: np.ndarray,
    ncols_left: int,
    right: np.ndarray,
    ncols_right: int,
) -> np.ndarray:
    """Matrix product ``left @ right`` over GF(2) using row-packed ``uint64`` blocks.

    Both operands use the same layout as :func:`_packed_boundary_matrix`: each row is a
    bit vector of length ``ncols_*`` stored in ``ceil(ncols_*/64)`` little-endian words
    (column ``j`` lives in bit ``j & 63`` of word ``j >> 6``).

    This implements the **stacked-rows** / ``A``-as-selector view: row ``i`` of the
    product is the XOR (sum in GF(2)) of those rows ``t`` of ``right`` for which
    ``left[i, t] == 1``.  No dense ``int64`` matmul and no full expansion of either
    factor, so ∂² checks scale with column weight of ``left`` rather than forcing a
    dense ``O(m * n * p)`` integer multiply.

    Args:
        left: Shape ``(m, nwords_L)`` with ``nwords_L = ceil(ncols_left / 64)``.
        right: Shape ``(n_mid, nwords_R)`` with ``n_mid == ncols_left`` (rows of
            ``right`` index the same ``t`` as columns of ``left``) and
            ``nwords_R = ceil(ncols_right / 64)``.

    Returns:
        Packed product of shape ``(m, nwords_R)``, ``uint64``.  Bits beyond
        ``ncols_right`` in the last word may be nonzero; callers that need a strict
        width should use :func:`_mask_trailing_bits_in_last_word`.
    """
    if left.size == 0 or right.size == 0 or ncols_left == 0 or ncols_right == 0:
        return np.zeros((int(left.shape[0]), (int(ncols_right) + 63) // 64), dtype=np.uint64)

    m = int(left.shape[0])
    n_mid = int(ncols_left)
    if int(right.shape[0]) != n_mid:
        raise ValueError(f"right must have ncols_left={n_mid} rows, got {right.shape[0]}")

    nwords_r = (int(ncols_right) + 63) // 64
    out = np.zeros((m, nwords_r), dtype=np.uint64)

    # Column t of `left` selects row t of `right` into every output row where the bit is 1.
    for t in range(n_mid):
        w = t >> 6
        sh = t & 63
        col_bits = (left[:, w] >> np.uint64(sh)) & np.uint64(1)
        rows = np.flatnonzero(col_bits)
        if rows.size == 0:
            continue
        out[rows, :] ^= right[t, :]

    return out


def _chain_square_violations(
    *,
    packed_by_k: dict[int, np.ndarray],
    ncols_by_k: dict[int, int],
    kmin: int,
    kmax: int,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Return nonempty list if any ∂_k ∘ ∂_{k+1} is nonzero over GF(2) for kmin < k < kmax."""
    violations: list[dict[str, Any]] = []
    for k in range(max(1, kmin + 1), kmax + 1):
        p_lo = packed_by_k.get(k)
        p_hi = packed_by_k.get(k + 1)
        if p_lo is None or p_hi is None:
            continue
        n_mid = int(ncols_by_k.get(k, 0))
        n_hi = int(ncols_by_k.get(k + 1, 0))
        if n_mid == 0 or n_hi == 0:
            continue
        nrows_lo = int(p_lo.shape[0])
        nrows_hi = int(p_hi.shape[0])
        _verbose_line(
            verbose,
            f"chain_square: checking ∂_{k}∘∂_{k + 1} (packed multiply), " + f"shapes ({nrows_lo},{n_mid})@({nrows_hi},{n_hi})",
        )
        comp_packed = gf2_matmul_packed_stacked_rows(p_lo, n_mid, p_hi, n_hi)
        _mask_trailing_bits_in_last_word(comp_packed, n_hi)
        nonzero = bool(comp_packed.any())
        _verbose_line(
            verbose,
            f"chain_square: ∂_{k}∘∂_{k + 1} composition is {'nonzero' if nonzero else 'zero'}",
        )
        if nonzero:
            dense = _packed_to_dense_mod2(comp_packed, n_hi)
            nnz = int(np.count_nonzero(dense))
            violations.append(
                {
                    "k": k,
                    "nnz": nnz,
                    "shape": [int(comp_packed.shape[0]), int(n_hi)],
                }
            )
    return violations


def get_betti_numbers(
    cplx: Complex,
    *,
    reduced: bool = False,
    compactify: bool = False,
    respect_finite: bool = False,
    verify_chain_complex: bool = False,
    verbose: bool = False,
) -> dict[int, int]:
    """Single Betti-number entrypoint.

    Args:
        reduced: If True, return reduced homology (β̃₀ = β₀ - 1 for nonempty complexes).
        compactify: If True, use the Borel–Moore-style boundary operator on the usual
            meta-graph (faces shared by at least two top cells only). If False, use the
            meta-graph from :meth:`~relucent.complex.Complex.get_meta_graph` with
            ``truncate=True`` and the full cellular boundary (every meta edge counts),
            so non-compact spaces are modeled by an explicit far-truncation without
            ad-hoc ``infinity`` bookkeeping.
        respect_finite: If True, restrict each chain group to cells with ``finite is True``
            before constructing boundary maps.
        verify_chain_complex: If True, require ``∂_k ∘ ∂_{k+1} = 0`` (mod 2) for every ``k``
            where both maps exist; otherwise raise :class:`ChainComplexInconsistent`. This uses
            :func:`gf2_matmul_packed_stacked_rows` (GF(2) multiply via XOR of packed rows),
            which avoids dense ``int64`` matmuls. When False, the legacy rank formula is used
            even if ∂²≠0 (then some ``β_k`` may be meaningless or negative).
        verbose: If True, print short progress lines to stderr (meta-graph, boundaries, chain
            checks, result). Also passed through to :meth:`~relucent.complex.Complex.get_meta_graph`.

    Note:
        **BFS-complete does not imply ∂²=0.** Breadth-first search can enumerate every
        full-dimensional activation region in a connected component, but
        :meth:`~relucent.complex.Complex.get_meta_graph` only adds a codimension-1 face
        edge when that face appears as a polyhedron in the contracted chain or in the same
        :class:`~relucent.complex.Complex`. Missing face nodes or incidences yield boundary
        matrices that are not a chain complex, so homology formulas do not apply until the
        face data is completed (see the ``Note`` on ``get_meta_graph``).
    """
    if len(cplx) == 0:
        return {}

    _verbose_line(
        verbose,
        f"get_betti_numbers: len(cplx)={len(cplx)} compactify={compactify} reduced={reduced} "
        + f"respect_finite={respect_finite} verify_chain_complex={verify_chain_complex}",
    )

    # ``truncate=True`` adds synthetic ``finite=True`` 0-cell copies of unbounded cells.
    # When ``respect_finite=True`` those orphan 0-cells (their infinite parent is
    # filtered out) would spuriously inflate β_0.  Use the plain meta-graph instead:
    # ``respect_finite`` is already a direct finiteness filter so no truncation is needed.
    truncate = not compactify and not respect_finite
    _verbose_line(verbose, "get_betti_numbers: building meta-graph …")
    meta = cplx.get_meta_graph(enrich=True, verbose=verbose, truncate=truncate)
    _verbose_line(
        verbose,
        f"get_betti_numbers: meta-graph |V|={meta.number_of_nodes()} |E|={meta.number_of_edges()} " + f"truncate={truncate}",
    )

    # Collect nodes by dimension (optionally filtered by finiteness).
    nodes_by_dim: dict[int, list[object]] = {}
    for n, attrs in meta.nodes(data=True):
        k = int(attrs.get("dim", -1))
        if k < 0:
            continue
        if respect_finite and attrs.get("finite", None) is not True:
            continue
        nodes_by_dim.setdefault(k, []).append(n)

    if not nodes_by_dim:
        return {}

    kmin = min(nodes_by_dim.keys())
    kmax = max(nodes_by_dim.keys())
    counts = ", ".join(f"{k}d:{len(nodes_by_dim[k])}" for k in sorted(nodes_by_dim))
    _verbose_line(verbose, f"get_betti_numbers: cells by dim kmin={kmin} kmax={kmax} ({counts})")

    boundary_rank: dict[int, int] = {k: 0 for k in range(kmin, kmax + 2)}
    packed_by_k: dict[int, np.ndarray] = {}
    ncols_by_k: dict[int, int] = {}

    k_values = list(range(max(1, kmin), kmax + 1))
    k_iter: Iterable[int] = k_values
    if verbose:
        k_iter = tqdm(k_values, desc="Betti: boundary maps", unit="∂", leave=False)

    for k in k_iter:
        packed, ncols = _packed_boundary_matrix(meta, nodes_by_dim, k=k, compactify=compactify)
        ncols_by_k[k] = ncols
        if ncols == 0:
            boundary_rank[k] = 0
            _verbose_line(verbose, f"get_betti_numbers: ∂_{k} skipped (no columns)")
            continue
        nrows = int(packed.shape[0])
        if verify_chain_complex:
            packed_by_k[k] = packed.copy()
        if verbose and not _C_BACKEND and nrows >= _SLOW_RANK_MIN_DIM and ncols >= _SLOW_RANK_MIN_DIM:
            ratio = ncols / max(nrows, 1)
            if 0.5 <= ratio <= 2.0:
                _verbose_line(
                    verbose,
                    f"get_betti_numbers: ∂_{k} is large and nearly square ({nrows}×{ncols}); "
                    + "C backend unavailable—pure-Python GF(2) rank may take hours.",
                )
        boundary_rank[k] = int(
            gf2_rank_boundary(
                packed,
                ncols,
                progress=verbose,
                progress_desc=f"GF(2) rank ∂_{k}",
            )
        )
        _verbose_line(
            verbose,
            f"get_betti_numbers: ∂_{k} shape ({nrows},{ncols}) rank={boundary_rank[k]}",
        )

    if verify_chain_complex:
        _verbose_line(verbose, "get_betti_numbers: verifying ∂²=0 (chain_square) …")
        viol = _chain_square_violations(
            packed_by_k=packed_by_k,
            ncols_by_k=ncols_by_k,
            kmin=kmin,
            kmax=kmax,
            verbose=verbose,
        )
        _verbose_line(verbose, "get_betti_numbers: chain_square checks finished")
        if viol:
            raise ChainComplexInconsistent(viol)

    beta: dict[int, int] = {}
    for k in range(kmin, kmax + 1):
        n_k = len(nodes_by_dim.get(k, []))
        r_dk = boundary_rank.get(k, 0) if k >= 1 else 0
        r_dk1 = boundary_rank.get(k + 1, 0) if k < kmax else 0
        beta[k] = int(n_k - r_dk - r_dk1)

    if reduced and int(beta.get(0, 0)) > 0:
        # Reduced homology: β̃0 = β0 - 1 (when β0 is represented explicitly).
        b0 = int(beta[0]) - 1
        if b0 == 0:
            beta.pop(0, None)
        else:
            beta[0] = b0

    trimmed = {k: v for k, v in beta.items() if v != 0}
    _verbose_line(verbose, f"get_betti_numbers: done (nonzero Betti entries: {trimmed})")

    # Trim zeros for cleanliness.
    return trimmed


def gf2_rank_packed(
    packed: np.ndarray,
    ncols: int,
    *,
    progress: bool = False,
    progress_desc: str | None = None,
) -> int:
    """Gaussian elimination rank over GF(2) on row-major bit-packed rows (uint64 words)."""
    if packed.size == 0 or ncols == 0:
        return 0
    nrows = int(packed.shape[0])
    rank = 0
    col_iter: Iterable[int] = range(ncols)
    if progress:
        col_iter = tqdm(
            col_iter,
            desc=progress_desc or "GF(2) rank",
            leave=False,
            total=ncols,
        )
    for col in col_iter:
        if rank >= nrows:
            break
        word = col >> 6
        sh = col & 63
        bitm = np.uint64(1) << sh
        colbits = packed[rank:, word] & bitm
        pivot_offs = np.flatnonzero(colbits)
        if pivot_offs.size == 0:
            continue
        pivot = rank + int(pivot_offs[0])
        if pivot != rank:
            packed[[rank, pivot], :] = packed[[pivot, rank], :]
        mask = (packed[:, word] & bitm) != 0
        mask[rank] = False
        inds = np.flatnonzero(mask)
        if inds.size > 0:
            packed[inds, :] ^= packed[rank, :]
        rank += 1
    return rank


def gf2_rank_boundary(
    packed: np.ndarray,
    ncols: int,
    *,
    progress: bool = False,
    progress_desc: str | None = None,
) -> int:
    """Rank of a boundary matrix.

    Uses the C backend (``_gf2_rank.c``) when available—typically 100–500×
    faster than the pure-Python path.  The C backend automatically transposes
    tall matrices to reduce column count, which speeds up ∂₁ and ∂₃.
    Falls back gracefully to pure Python if the C library could not be
    compiled or loaded.
    """
    if _C_BACKEND:
        return _gf2_rank_boundary_c(packed, ncols, progress=progress, progress_desc=progress_desc)
    # Pure-Python fallback: transpose when it reduces column count.
    nrows = int(packed.shape[0])
    if nrows > ncols and ncols > 0:
        transposed, ncols_t = _transpose_packed(packed, ncols)
        desc = f"{progress_desc} (A^T)" if progress_desc else "GF(2) rank (A^T)"
        return gf2_rank_packed(transposed, ncols_t, progress=progress, progress_desc=desc)
    return gf2_rank_packed(packed, ncols, progress=progress, progress_desc=progress_desc)


#
# NOTE: This module intentionally contains only the minimal set of helpers needed
# for the current Betti-number computation paths.
