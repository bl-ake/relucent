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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
    infinity: Literal["link", "one_point"] = "link",
) -> dict[int, int]:
    """Single Betti-number entrypoint.

    Args:
        reduced: If True, return reduced homology (β̃₀ = β₀ - 1 for nonempty complexes).
        compactify: If True, use the Borel–Moore-style contracted-chain boundary operator
            (relucent's contracted representation).
        respect_finite: If True, restrict each chain group to cells with ``finite is True``
            before constructing boundary maps.
        infinity: Only used when ``compactify=False``. How to model the boundary at infinity
            for non-compact 1D complexes when computing ordinary homology:
            - "link" (default): add distinct formal endpoints for each missing end.
            - "one_point": glue all missing ends in a dual-graph component to a single
              formal point at infinity (one-point compactification).
    """
    if len(cplx) == 0:
        return {}

    meta = cplx.get_meta_graph(enrich=True, verbose=False)

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

        # For 1D non-compact complexes, the meta-graph can include unbounded 1-cells with
        # only one (or zero) incident 0-face. Ordinary homology of the stabilized
        # far-truncation is captured by adding "boundary at infinity" 0-cells.
        synthetic_edges: list[tuple[object, object]] = []
        if not compactify and k == 1:
            # Edge case: the contracted chain/meta-graph can contain explicit 0-cells with
            # ``finite is False`` (a "vertex at infinity") that are shared by multiple 1-cells.
            #
            # For ordinary homology with ``infinity="link"``, we do NOT want such vertices to
            # glue different missing ends together. Instead, treat incidences to finite=False
            # 0-cells as "missing endpoints" and attach distinct synthetic endpoints.
            #
            # For ``infinity="one_point"``, gluing is intended, so we keep them.
            if infinity == "link":
                # Treat explicit finite=False 0-cells as "at infinity": they should not
                # serve as shared vertices for ordinary homology under link semantics.
                #
                # Remove them from the 0-chain group and let the synthetic endpoint
                # logic below add distinct ends per missing incidence.
                rows = [
                    r
                    for r in rows
                    if (not isinstance(r, (bytes, bytearray)))
                    or (meta.nodes.get(bytes(r), {}) or {}).get("finite", None) != False  # noqa: E712
                ]
                # Keep counts consistent with the modified chain group used for ∂₁.
                nodes_by_dim[0] = list(rows)

            end_count: dict[object, int] = {c: 0 for c in cols}
            for u, v, _data in meta.edges(data=True):
                if u in end_count and v in rows:
                    end_count[u] += 1

            # Use dual-graph components (on 1-cells) to optionally glue ends.
            comp_of: dict[object, int] = {}
            if infinity == "one_point":
                import networkx as nx

                G = cplx.get_dual_graph(verbose=False)
                for ci, comp in enumerate(nx.connected_components(G)):
                    for node in comp:
                        comp_of[getattr(node, "tag", node)] = ci

            next_idx = 0
            for u, cnt in end_count.items():
                if isinstance(u, (bytes, bytearray)) and meta.nodes[bytes(u)].get("finite", None) is True:
                    continue
                if cnt == 2:
                    continue
                if infinity == "one_point":
                    # Attach missing ends to a single infinity vertex per component.
                    comp_id = comp_of.get(u, 0)
                    inf = ("__inf0__", int(comp_id))
                    if inf not in rows:
                        rows.append(inf)
                    # If cnt==1 add one incidence; if cnt==0 add two incidences (same vertex twice cancels in GF(2)),
                    # so in the cnt==0 case we add a second distinct incidence key to preserve parity.
                    synthetic_edges.append((u, inf))
                    if cnt == 0:
                        inf2 = ("__inf0__", int(comp_id), "b")
                        if inf2 not in rows:
                            rows.append(inf2)
                        synthetic_edges.append((u, inf2))
                    continue

                # infinity == "link": add distinct endpoints per missing end.
                if cnt == 1:
                    s = ("__end__", u, next_idx)
                    next_idx += 1
                    rows.append(s)
                    synthetic_edges.append((u, s))
                elif cnt == 0:
                    s1 = ("__end__", u, next_idx)
                    next_idx += 1
                    s2 = ("__end__", u, next_idx)
                    next_idx += 1
                    rows.append(s1)
                    rows.append(s2)
                    synthetic_edges.append((u, s1))
                    synthetic_edges.append((u, s2))
                else:
                    raise RuntimeError(f"1-cell has >2 incident 0-faces in meta-graph: {cnt}")

            # ``rows`` is the actual (k-1)-chain basis used to build ∂₁, including any
            # synthetic boundary-at-infinity vertices we appended above. Ensure the
            # β₀ formula uses the same basis size.
            nodes_by_dim[0] = list(rows)

        row_index = {r: i for i, r in enumerate(rows)}
        col_index = {c: j for j, c in enumerate(cols)}

        nrows = len(rows)
        ncols = len(cols)
        nwords = (ncols + 63) // 64
        packed = np.zeros((nrows, nwords), dtype=np.uint64)

        if compactify:
            # Count how many k-cells are incident to each (k-1)-cell.
            inc_count: dict[object, int] = {r: 0 for r in rows}
            for u, v, _data in meta.edges(data=True):
                if u not in col_index or v not in row_index:
                    continue
                inc_count[v] += 1

            # Only include incidences for (k-1)-faces that are shared by >=2 k-cells.
            for u, v, _data in meta.edges(data=True):
                if u not in col_index or v not in row_index:
                    continue
                if inc_count.get(v, 0) < 2:
                    continue
                j = int(col_index[u])
                i = int(row_index[v])
                w = j >> 6
                bit = np.uint64(1) << (j & 63)
                packed[i, w] ^= bit
        else:
            # Each meta edge is a face relation: dim(u)=k, dim(v)=k-1.
            for u, v, _data in meta.edges(data=True):
                if u not in col_index or v not in row_index:
                    continue
                j = int(col_index[u])
                i = int(row_index[v])
                w = j >> 6
                bit = np.uint64(1) << (j & 63)
                packed[i, w] ^= bit  # Multi-edges XOR naturally over GF(2)
            for u, v in synthetic_edges:
                if u not in col_index or v not in row_index:
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
