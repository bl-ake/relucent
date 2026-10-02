"""Betti numbers over GF(2) for ReLU cell complexes.

Builds a subcomplex, closes it under faces, forms the boundary operators ∂_k over
GF(2), and reads Betti numbers off their ranks.

* GF(2) coefficients, so orientations don't matter.
* A codimension-1 facet of a cell comes from setting one nonzero sign entry to 0.
* :func:`get_betti_numbers` uses every codimension-one incidence in ``meta``.
  Truncation and Borel–Moore boundaries live on :class:`~relucent.core.complex.Complex`.
* ``verify_chain_complex=True`` checks ``∂²=0`` with sparse GF(2) products (nonzero
  pattern only, no dense matmuls).
"""

from __future__ import annotations

import heapq
from typing import Any, Literal, overload

import numpy as np

from relucent._internal.logging import logger, progress, show_progress, with_verbosity

# Try to load the C backend once at import time.
_c_backend = False
_gf2_rank_boundary_c = None

try:
    from relucent.topology._gf2 import (
        available as _c_available,
    )
    from relucent.topology._gf2 import (
        gf2_rank_boundary_c as _gf2_rank_boundary_c_impl,
    )
    # from relucent.topology._gf2 import (
    #     gf2_transpose_packed_c as _gf2_transpose_packed_c,
    # )

    _gf2_rank_boundary_c = _gf2_rank_boundary_c_impl
    _c_backend = _c_available()
except Exception:
    pass

__all__ = [
    "ChainComplexInconsistent",
    "ConnectedComponentsMismatch",
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
C_BACKEND_AVAILABLE: bool = _c_backend


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


class ConnectedComponentsMismatch(RuntimeError):
    """Rank-formula β₀ disagrees with the graph component count.

    Indicates the truncated meta-graph 1-skeleton is not a closed CW complex
    (e.g. missing materialized 0-faces after truncation).
    """

    def __init__(self, rank_beta0: int, graph_beta0: int) -> None:
        self.rank_beta0 = rank_beta0
        self.graph_beta0 = graph_beta0
        super().__init__(
            f"β₀ from rank formula ({rank_beta0}) != graph components ({graph_beta0}). "
            + "Truncation closure may be incomplete; use verify_connected_components=True to surface this."
        )


def _count_weakly_connected_components(meta: Any) -> int:
    """Count path-connected components of the meta-graph (edges treated as undirected)."""
    neighbors: dict[object, list[object]] = {n: [] for n in meta.nodes()}
    for u, v in meta.edges():
        neighbors[u].append(v)
        neighbors[v].append(u)
    visited: set[object] = set()
    count = 0
    for start in meta.nodes():
        if start in visited:
            continue
        count += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(neighbors[node])
    return count


@overload
def _packed_boundary_matrix(
    meta: Any,
    nodes_by_dim: dict[int, list[object]],
    *,
    k: int,
    require_shared_faces: bool = False,
    also_sparse: Literal[False] = False,
) -> tuple[np.ndarray, int]: ...


@overload
def _packed_boundary_matrix(
    meta: Any,
    nodes_by_dim: dict[int, list[object]],
    *,
    k: int,
    require_shared_faces: bool = False,
    also_sparse: Literal[True],
) -> tuple[np.ndarray, int, list[list[int]]]: ...


def _packed_boundary_matrix(
    meta: Any,
    nodes_by_dim: dict[int, list[object]],
    *,
    k: int,
    require_shared_faces: bool = False,
    also_sparse: bool = False,
) -> tuple[np.ndarray, int] | tuple[np.ndarray, int, list[list[int]]]:
    """Bit-packed ∂_k: C_k → C_{k-1} (rows index (k−1)-cells, columns index k-cells).

    When ``also_sparse=True``, also return per-row column-index lists with the same
    GF(2) XOR fill as the packed matrix (for sparse ∂² checks).
    """
    rows = nodes_by_dim.get(k - 1, [])
    cols = nodes_by_dim.get(k, [])
    if not rows or not cols:
        empty = np.zeros((0, 0), dtype=np.uint64)
        return (empty, 0, []) if also_sparse else (empty, 0)

    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: j for j, c in enumerate(cols)}

    nrows = len(rows)
    ncols = len(cols)
    nwords = (ncols + 63) // 64
    packed = np.zeros((nrows, nwords), dtype=np.uint64)
    row_sets: list[set[int]] | None = [set() for _ in range(nrows)] if also_sparse else None

    inc_count: dict[object, int] | None = None
    if require_shared_faces:
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
        if row_sets is not None:
            if j in row_sets[i]:
                row_sets[i].discard(j)
            else:
                row_sets[i].add(j)

    if also_sparse:
        assert row_sets is not None
        return packed, ncols, [list(s) for s in row_sets]
    return packed, ncols


def _boundary_row_sets(
    meta: Any,
    nodes_by_dim: dict[int, list[object]],
    *,
    k: int,
    require_shared_faces: bool = False,
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
    if require_shared_faces:
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


def _sparse_boundary_maps(
    meta: Any,
    nodes_by_dim: dict[int, list[object]],
    *,
    require_shared_faces: bool = False,
) -> dict[int, tuple[list[set[int]], int]]:
    """Every ∂_k as ``(row sets, ncols)`` from one pass over ``meta``'s edges.

    Matches :func:`_packed_boundary_matrix` entry for entry: rows index (k−1)-cells, columns
    k-cells, and repeated incidences cancel mod 2. With ``require_shared_faces`` a (k−1)-cell
    keeps its incidences only when at least two edges come into it from k-cells.
    """
    where: dict[object, tuple[int, int]] = {}
    for k, cells in nodes_by_dim.items():
        for i, n in enumerate(cells):
            where[n] = (k, i)
    maps: dict[int, tuple[list[set[int]], int]] = {
        k: ([set() for _ in nodes_by_dim.get(k - 1, ())], len(nodes_by_dim[k])) for k in nodes_by_dim if k - 1 in nodes_by_dim
    }
    incidences: list[tuple[int, int, int]] = []
    coface_count: dict[tuple[int, int], int] = {}
    for u, v in meta.edges():
        hi, lo = where.get(u), where.get(v)
        if hi is None or lo is None or lo[0] != hi[0] - 1:
            continue
        incidences.append((hi[0], lo[1], hi[1]))
        if require_shared_faces:
            coface_count[lo] = coface_count.get(lo, 0) + 1
    for k, i, j in incidences:
        if require_shared_faces and coface_count[(k - 1, i)] < 2:
            continue
        row = maps[k][0][i]
        if j in row:
            row.discard(j)
        else:
            row.add(j)
    return maps


def _gf2_product_nnz(left_rows: list[list[int]], right_rows: list[list[int]]) -> int:
    """Number of nonzeros in the GF(2) product of two matrices given as row index lists."""
    nnz = 0
    for ts in left_rows:
        ones: set[int] = set()
        for t in ts:
            ones.symmetric_difference_update(right_rows[t])
        nnz += len(ones)
    return nnz


def _packed_to_sparse_rowlists(packed: np.ndarray, ncols: int) -> list[list[int]]:
    """Extract per-row column indices of 1-bits from a row-packed ``uint64`` matrix."""
    m = int(packed.shape[0])
    nwords = int(packed.shape[1])
    out: list[list[int]] = [[] for _ in range(m)]
    ncols_i = int(ncols)
    for i in range(m):
        cols_i = out[i]
        row = packed[i]
        for w in range(nwords):
            word = int(row[w])
            if word == 0:
                continue
            base = w << 6
            while word:
                bit = (word & -word).bit_length() - 1
                t = base + bit
                if t < ncols_i:
                    cols_i.append(t)
                word &= word - 1
    return out


def gf2_matmul_sparse_rowlists(
    left_rows: list[list[int]],
    right_rows: list[list[int]],
    ncols_right: int,
) -> np.ndarray:
    """GF(2) product of sparse matrices given as per-row column-index lists.

    Cost is proportional to the number of intermediate nonzeros
    ``∑_i ∑_{t ∈ left[i]} |right[t]|``, not to the dense dimensions.
    """
    m = len(left_rows)
    n_mid = len(right_rows)
    nwords_r = (int(ncols_right) + 63) // 64
    out = np.zeros((m, nwords_r), dtype=np.uint64)
    if m == 0 or n_mid == 0 or ncols_right == 0:
        return out

    for i, ts in enumerate(left_rows):
        if not ts:
            continue
        ones: set[int] = set()
        for t in ts:
            if t < 0 or t >= n_mid:
                raise ValueError(f"left column index {t} out of range for right with {n_mid} rows")
            for j in right_rows[t]:
                if j in ones:
                    ones.discard(j)
                else:
                    ones.add(j)
        if not ones:
            continue
        out_row = out[i]
        for j in ones:
            out_row[j >> 6] ^= np.uint64(1) << np.uint64(j & 63)
    return out


# Sparse elimination stores each nonzero twice (row and column sets) at roughly this many
# bytes apiece; once fill-in makes that dearer than one bit per entry of the remaining matrix,
# the remainder is ranked densely, provided its packed form stays under the byte cap.
_SPARSE_BYTES_PER_NONZERO = 128
_DENSE_REMAINDER_MAX_BYTES = 1 << 30


@with_verbosity
def gf2_rank_sparse_rowsets(
    row_sets: list[set[int]],
    ncols: int,
    *,
    verbose: int | None = None,
    progress_desc: str | None = None,
) -> int:
    """Gaussian elimination rank over GF(2) on sparse row sets.

    Each row is a set of column indices where the matrix entry is 1. Pivots are chosen to
    keep fill-in low (Markowitz-style): always a column of least remaining degree, on its row
    of least degree. A degree-one column, such as a free face in a cell complex, costs no fill,
    so the boundary maps of manifold-like complexes reduce with almost none, in time and memory
    proportional to their nonzeros rather than to rows × columns. If fill-in nonetheless makes
    the remaining matrix dense enough that bits are cheaper than sets, that remainder (the
    Schur complement, whose rank adds to the pivots so far) is ranked bit-packed instead.

    ``row_sets`` is consumed.
    """
    del verbose  # applied by @with_verbosity
    nrows = len(row_sets)
    if nrows == 0 or ncols == 0:
        return 0
    rows = row_sets
    cols: list[set[int]] = [set() for _ in range(ncols)]
    for r, cs in enumerate(rows):
        for c in cs:
            cols[c].add(r)
    heap = [(len(s), c) for c, s in enumerate(cols) if s]
    heapq.heapify(heap)
    nnz = sum(len(s) for s in cols)
    next_density_check = 2 * nnz

    pbar = progress(desc=progress_desc or "GF(2) rank", total=len(heap), leave=False)
    rank = 0
    try:
        while heap:
            d, c = heapq.heappop(heap)
            col = cols[c]
            if len(col) != d or d == 0:
                continue  # stale heap entry
            r = min(col, key=lambda x: len(rows[x]))
            # Clear row r from every other column by adding column c to it.
            for c2 in list(rows[r]):
                if c2 == c:
                    continue
                target = cols[c2]
                for x in col:
                    if x in target:
                        target.discard(x)
                        rows[x].discard(c2)
                        nnz -= 1
                    else:
                        target.add(x)
                        rows[x].add(c2)
                        nnz += 1
                heapq.heappush(heap, (len(target), c2))
            for x in col:
                rows[x].discard(c)
            nnz -= len(col)
            cols[c] = set()
            rows[r] = set()
            rank += 1
            pbar.update(1)
            if nnz > next_density_check:
                next_density_check = 2 * nnz
                live_rows = [i for i, s in enumerate(rows) if s]
                live_cols = [j for j, s in enumerate(cols) if s]
                dense_bytes = len(live_rows) * ((len(live_cols) + 63) // 64) * 8
                if dense_bytes <= min(nnz * _SPARSE_BYTES_PER_NONZERO, _DENSE_REMAINDER_MAX_BYTES):
                    del cols, heap
                    renumber = {j: k for k, j in enumerate(live_cols)}
                    packed = _row_sets_to_packed([{renumber[j] for j in rows[i]} for i in live_rows], len(live_cols))
                    return rank + int(gf2_rank_boundary(packed, len(live_cols)))
    finally:
        pbar.close()
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

    Multiplication extracts the sparse nonzero patterns of both factors and composes
    them with :func:`gf2_matmul_sparse_rowlists`, so cost tracks intermediate nonzeros
    rather than a dense ``O(m · n · p)`` pass or an ``O(m · n)`` packed-column scan.

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

    n_mid = int(ncols_left)
    if int(right.shape[0]) != n_mid:
        raise ValueError(f"right must have ncols_left={n_mid} rows, got {right.shape[0]}")

    left_rows = _packed_to_sparse_rowlists(left, n_mid)
    right_rows = _packed_to_sparse_rowlists(right, int(ncols_right))
    return gf2_matmul_sparse_rowlists(left_rows, right_rows, int(ncols_right))


def _chain_square_violations(
    *,
    sparse_by_k: dict[int, list[list[int]]],
    ncols_by_k: dict[int, int],
    kmin: int,
    kmax: int,
) -> list[dict[str, Any]]:
    """Return nonempty list if any ∂_k ∘ ∂_{k+1} is nonzero over GF(2) for kmin < k < kmax."""
    violations: list[dict[str, Any]] = []
    for k in range(max(1, kmin + 1), kmax + 1):
        left_rows = sparse_by_k.get(k)
        right_rows = sparse_by_k.get(k + 1)
        if left_rows is None or right_rows is None:
            continue
        n_mid = int(ncols_by_k.get(k, 0))
        n_hi = int(ncols_by_k.get(k + 1, 0))
        if n_mid == 0 or n_hi == 0:
            continue
        nrows_lo = len(left_rows)
        nrows_hi = len(right_rows)
        logger.debug(
            f"chain_square: checking ∂_{k}∘∂_{k + 1} (sparse multiply), shapes ({nrows_lo},{n_mid})@({nrows_hi},{n_hi})"
        )
        nnz = _gf2_product_nnz(left_rows, right_rows)
        logger.debug(f"chain_square: ∂_{k}∘∂_{k + 1} composition is {'nonzero' if nnz else 'zero'}")
        if nnz:
            violations.append({"k": k, "nnz": nnz, "shape": [nrows_lo, n_hi]})
    return violations


@with_verbosity
def get_betti_numbers(
    meta: Any,
    *,
    require_shared_faces: bool = False,
    reduced: bool = False,
    verify_chain_complex: bool = False,
    verify_connected_components: bool = True,  ## TODO: How slow is this?
    verbose: int | None = None,
    nworkers: int | None = None,
    method: Literal["sparse", "dense"] = "sparse",
) -> dict[int, int]:
    """Compute Betti numbers from face incidences in ``meta``.

    Args:
        meta: Face poset as a NetworkX ``MultiDiGraph`` from
            :meth:`~relucent.core.complex.Complex.get_meta_graph` (optionally truncated or
            restricted to finite cells first).
        require_shared_faces: If True, only incidences where a codimension-one face has at
            least two cofaces (Borel–Moore-style). Default False counts every meta edge.
            Set by :meth:`~relucent.core.complex.Complex.get_betti_numbers_from_meta` when
            ``compactify=True``.
        reduced: If True, return reduced homology (β̃₀ = β₀ - 1 for nonempty complexes).
        verify_chain_complex: If True, require ``∂_k ∘ ∂_{k+1} = 0`` (mod 2) for every ``k``
            where both maps exist; otherwise raise :class:`ChainComplexInconsistent`.
            Uses sparse GF(2) matrix multiplication over the nonzero incidence pattern.
        verify_connected_components: If True, require rank-formula β₀ to agree with the
            number of path-connected components when ``kmin == 0``; otherwise raise
            :class:`ConnectedComponentsMismatch`.
        verbose: Output level: ``0`` quiet, ``1`` progress bars, ``2`` per-map detail.
            ``None`` uses :data:`relucent.config.VERBOSE`.
        nworkers: ``method="dense"`` only. Threads for ranking boundary maps concurrently.
            ``None`` (default) uses one per non-trivial map if the C backend is available,
            else runs sequentially. ``0`` or ``1`` is always sequential; ``N > 1`` uses up
            to N. (ctypes releases the GIL, so the threads really run in parallel.)
        method: ``"sparse"`` (default) ranks each map by sparse elimination with low-fill
            pivots (:func:`gf2_rank_sparse_rowsets`); cost follows the number of incidences.
            ``"dense"`` ranks bit-packed matrices (:func:`gf2_rank_boundary`), needing
            ``rows × columns / 8`` bytes per map (59 GB for a 688k × 688k ∂₂); kept for
            cross-checking.

    Note:
        Truncation and finite-cell restriction are done on
        :class:`~relucent.core.complex.Complex` before calling this.

        With no 0-cells (``kmin > 0``), the lowest Betti number is keyed by ``kmin``, not
        ``0``. E.g. a boundary complex of only 1- and 2-cells returns ``{1: n}``, where
        ``n`` is the number of connected components of the 1-skeleton.

        With ``kmin == 0``, β₀ comes from the cellular rank formula on the (possibly
        truncated and closed) meta-graph. ``verify_connected_components=True`` checks it
        against the path-component count.
    """
    del verbose  # applied by @with_verbosity
    if meta.number_of_nodes() == 0:
        return {}

    logger.debug(
        f"get_betti_numbers: |V|={meta.number_of_nodes()} |E|={meta.number_of_edges()} "
        + f"require_shared_faces={require_shared_faces} reduced={reduced} "
        + f"verify_chain_complex={verify_chain_complex} "
        + f"verify_connected_components={verify_connected_components}",
    )

    nodes_by_dim: dict[int, list[object]] = {}
    for n, attrs in meta.nodes(data=True):
        k = int(attrs.get("dim", -1))
        if k < 0:
            continue
        nodes_by_dim.setdefault(k, []).append(n)

    if not nodes_by_dim:
        return {}

    kmin = min(nodes_by_dim.keys())
    kmax = max(nodes_by_dim.keys())
    counts = ", ".join(f"{k}d:{len(nodes_by_dim[k])}" for k in sorted(nodes_by_dim))
    logger.debug(f"get_betti_numbers: cells by dim kmin={kmin} kmax={kmax} ({counts})")

    boundary_rank: dict[int, int] = {k: 0 for k in range(kmin, kmax + 2)}
    sparse_by_k: dict[int, list[list[int]]] = {}
    ncols_by_k: dict[int, int] = {}

    k_values = list(range(max(1, kmin), kmax + 1))

    if method == "sparse":
        maps = _sparse_boundary_maps(meta, nodes_by_dim, require_shared_faces=require_shared_faces)
        for k in progress(k_values, desc="Betti: boundary ranks", unit="∂", leave=False):
            row_sets, ncols = maps.pop(k, ([], 0))
            ncols_by_k[k] = ncols
            if verify_chain_complex and ncols:
                sparse_by_k[k] = [list(s) for s in row_sets]  # ranking consumes the sets
            boundary_rank[k] = gf2_rank_sparse_rowsets(row_sets, ncols, progress_desc=f"GF(2) rank ∂_{k}")
            logger.debug(f"get_betti_numbers: ∂_{k} shape ({len(row_sets)},{ncols}) rank={boundary_rank[k]}")
            del row_sets
        return _finish_betti_numbers(
            meta,
            nodes_by_dim,
            boundary_rank,
            sparse_by_k=sparse_by_k,
            ncols_by_k=ncols_by_k,
            kmin=kmin,
            kmax=kmax,
            reduced=reduced,
            verify_chain_complex=verify_chain_complex,
            verify_connected_components=verify_connected_components,
        )

    # Phase A: build all boundary matrices (fast, sequential). When verifying, also keep
    # sparse XOR row lists for the ∂² checks (rank may mutate the packed arrays).
    # Phase A: build all boundary matrices (sequential – fast).
    # When verifying, also keep sparse XOR row lists for ∂² checks (no
    # packed copies; rank may mutate the packed arrays in place).
    # -------------------------------------------------------------------
    matrices: dict[int, tuple[np.ndarray, int]] = {}
    for k in k_values:
        if verify_chain_complex:
            packed, ncols, sparse_rows = _packed_boundary_matrix(
                meta,
                nodes_by_dim,
                k=k,
                require_shared_faces=require_shared_faces,
                also_sparse=True,
            )
        else:
            packed, ncols = _packed_boundary_matrix(meta, nodes_by_dim, k=k, require_shared_faces=require_shared_faces)
            sparse_rows = None
        ncols_by_k[k] = ncols
        if ncols == 0:
            boundary_rank[k] = 0
            logger.debug(f"get_betti_numbers: ∂_{k} skipped (no columns)")
        else:
            if sparse_rows is not None:
                sparse_by_k[k] = sparse_rows
            matrices[k] = (packed, ncols)

    # Phase B: rank each non-trivial boundary map. With the C backend and several maps,
    # use threads (ctypes releases the GIL, so they overlap with each other and with
    # OpenMP inside the rank call).
    # Phase B: rank each non-trivial boundary map.
    # When the C backend is available and there are multiple maps, run them
    # in parallel threads (ctypes releases the GIL, so threads truly
    # overlap with each other and with OpenMP inside the C rank call).
    # -------------------------------------------------------------------
    non_trivial_ks = [k for k in k_values if k in matrices]

    if nworkers is None:
        _nw = len(non_trivial_ks) if _c_backend else 1
    elif nworkers <= 0:
        _nw = 1
    else:
        _nw = nworkers

    _parallel = _nw > 1 and len(non_trivial_ks) > 1

    def _rank_one(k: int) -> tuple[int, int]:
        packed, ncols = matrices[k]
        nrows = int(packed.shape[0])
        if not _c_backend and nrows >= _SLOW_RANK_MIN_DIM and ncols >= _SLOW_RANK_MIN_DIM:
            ratio = ncols / max(nrows, 1)
            if 0.5 <= ratio <= 2.0:
                logger.info(
                    f"get_betti_numbers: ∂_{k} is large and nearly square ({nrows}×{ncols}); "
                    + "C backend unavailable—pure-Python GF(2) rank may take hours.",
                )
        rank = int(
            gf2_rank_boundary(
                packed,
                ncols,
                # Individual per-map progress bars look garbled when multiple
                # threads write to the terminal simultaneously; suppress them
                # in parallel mode and show a single outer completion bar instead.
                verbose=0 if _parallel else None,
                progress_desc=f"GF(2) rank ∂_{k}",
            )
        )
        logger.debug(f"get_betti_numbers: ∂_{k} shape ({nrows},{ncols}) rank={rank}")
        return k, rank

    if _parallel:
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import as_completed as _as_completed

        pbar = progress(total=len(non_trivial_ks), desc="Betti: boundary ranks", unit="∂", leave=False)
        try:
            with ThreadPoolExecutor(max_workers=min(_nw, len(non_trivial_ks))) as executor:
                futures = {executor.submit(_rank_one, k): k for k in non_trivial_ks}
                for fut in _as_completed(futures):
                    k_done, r = fut.result()
                    boundary_rank[k_done] = r
                    pbar.update(1)
        finally:
            pbar.close()
    else:
        for k in progress(non_trivial_ks, desc="Betti: boundary ranks", unit="∂", leave=False):
            _, boundary_rank[k] = _rank_one(k)

    return _finish_betti_numbers(
        meta,
        nodes_by_dim,
        boundary_rank,
        sparse_by_k=sparse_by_k,
        ncols_by_k=ncols_by_k,
        kmin=kmin,
        kmax=kmax,
        reduced=reduced,
        verify_chain_complex=verify_chain_complex,
        verify_connected_components=verify_connected_components,
    )


def _finish_betti_numbers(
    meta: Any,
    nodes_by_dim: dict[int, list[object]],
    boundary_rank: dict[int, int],
    *,
    sparse_by_k: dict[int, list[list[int]]],
    ncols_by_k: dict[int, int],
    kmin: int,
    kmax: int,
    reduced: bool,
    verify_chain_complex: bool,
    verify_connected_components: bool,
) -> dict[int, int]:
    """Check ∂²=0 if asked, then Betti numbers from the boundary ranks (zeros trimmed)."""
    if verify_chain_complex:
        logger.debug("get_betti_numbers: verifying ∂²=0 (chain_square) …")
        viol = _chain_square_violations(
            sparse_by_k=sparse_by_k,
            ncols_by_k=ncols_by_k,
            kmin=kmin,
            kmax=kmax,
        )
        logger.debug("get_betti_numbers: chain_square checks finished")
        if viol:
            raise ChainComplexInconsistent(viol)

    beta: dict[int, int] = {}
    for k in range(kmin, kmax + 1):
        n_k = len(nodes_by_dim.get(k, []))
        r_dk = boundary_rank.get(k, 0) if k > kmin else 0
        r_dk1 = boundary_rank.get(k + 1, 0) if k < kmax else 0
        beta[k] = int(n_k - r_dk - r_dk1)

    if kmin == 0 and verify_connected_components:
        n_cc = _count_weakly_connected_components(meta)
        rank_beta0 = beta.get(0, 0)
        if rank_beta0 != n_cc:
            raise ConnectedComponentsMismatch(rank_beta0, n_cc)

    if reduced and int(beta.get(0, 0)) > 0:
        # Reduced homology: β̃0 = β0 - 1 (when β0 is represented explicitly).
        b0 = int(beta[0]) - 1
        if b0 == 0:
            beta.pop(0, None)
        else:
            beta[0] = b0

    trimmed = {k: v for k, v in beta.items() if v != 0}
    logger.debug(f"get_betti_numbers: done (nonzero Betti entries: {trimmed})")

    # Trim zeros for cleanliness.
    return trimmed


@with_verbosity
def gf2_rank_packed(
    packed: np.ndarray,
    ncols: int,
    *,
    verbose: int | None = None,
    progress_desc: str | None = None,
) -> int:
    """Gaussian elimination rank over GF(2) on row-major bit-packed rows (uint64 words)."""
    del verbose  # applied by @with_verbosity
    if packed.size == 0 or ncols == 0:
        return 0
    nrows = int(packed.shape[0])
    rank = 0
    for col in progress(range(ncols), desc=progress_desc or "GF(2) rank", leave=False, total=ncols):
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


@with_verbosity
def gf2_rank_boundary(
    packed: np.ndarray,
    ncols: int,
    *,
    verbose: int | None = None,
    progress_desc: str | None = None,
) -> int:
    """Rank of a boundary matrix.

    Uses the C backend (``_gf2_rank.c``) when available—typically 100–500×
    faster than the pure-Python path.  The C backend automatically transposes
    tall matrices to reduce column count, which speeds up ∂₁ and ∂₃.
    Falls back gracefully to pure Python if the C library could not be
    compiled or loaded.
    """
    del verbose  # applied by @with_verbosity
    if _c_backend and _gf2_rank_boundary_c is not None:
        return _gf2_rank_boundary_c(packed, ncols, show_bar=show_progress(), progress_desc=progress_desc)
    # Pure-Python fallback: transpose when it reduces column count.
    nrows = int(packed.shape[0])
    if nrows > ncols and ncols > 0:
        transposed, ncols_t = _transpose_packed(packed, ncols)
        desc = f"{progress_desc} (A^T)" if progress_desc else "GF(2) rank (A^T)"
        return gf2_rank_packed(transposed, ncols_t, progress_desc=desc)
    return gf2_rank_packed(packed, ncols, progress_desc=progress_desc)


#
# NOTE: This module intentionally contains only the minimal set of helpers needed
# for the current Betti-number computation paths.
