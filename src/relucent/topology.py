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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from relucent.complex import Complex

__all__ = [
    "get_betti_numbers",
    "gf2_rank_packed",
]


def get_betti_numbers(
    cplx: Complex,
    *,
    reduced: bool = False,
    compactify: bool = False,
    respect_finite: bool = False,
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
    """
    if len(cplx) == 0:
        return {}

    meta = cplx.get_meta_graph(enrich=True, verbose=False, truncate=not compactify)

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

    boundary_rank: dict[int, int] = {k: 0 for k in range(kmin, kmax + 2)}

    for k in range(max(1, kmin), kmax + 1):
        rows = nodes_by_dim.get(k - 1, [])
        cols = nodes_by_dim.get(k, [])
        if not rows or not cols:
            boundary_rank[k] = 0
            continue

        row_index = {r: i for i, r in enumerate(rows)}
        col_index = {c: j for j, c in enumerate(cols)}

        nrows = len(rows)
        ncols = len(cols)
        nwords = (ncols + 63) // 64
        packed = np.zeros((nrows, nwords), dtype=np.uint64)

        # Borel–Moore (``compactify``): only (k−1)-faces shared by ≥2 k-cells. Ordinary homology
        # on a far-truncation (``not compactify``): ``meta`` already has ``truncate=True``; count
        # every face incidence (full cellular ∂; multi-edges still XOR over GF(2)).
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

        boundary_rank[k] = int(gf2_rank_packed(packed, ncols))

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

    # Trim zeros for cleanliness.
    return {k: v for k, v in beta.items() if v != 0}


def gf2_rank_packed(packed: np.ndarray, ncols: int) -> int:
    """Gaussian elimination rank over GF(2) on row-major bit-packed rows (uint64 words)."""
    if packed.size == 0 or ncols == 0:
        return 0
    nrows = int(packed.shape[0])
    rank = 0
    for col in range(ncols):
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


#
# NOTE: This module intentionally contains only the minimal set of helpers needed
# for the current Betti-number computation paths.
