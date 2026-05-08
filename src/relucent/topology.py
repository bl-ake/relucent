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

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from relucent.complex import Complex
    from relucent.poly import Polyhedron

HomologyMode = Literal[
    "standard",
    "borel_moore",
    # Back-compat aliases used in older code.
    "traditional",
    "contracted_standard",
    "contracted_borel_moore",
    "traditional_truncated",
]

__all__ = ["betti_numbers", "betti_numbers_complex", "gf2_rank", "gf2_rank_packed"]


def _auto_truncation_bound_from_interior_points(cplx: "Complex") -> float:
    """Choose a truncation radius from max |interior_point|_inf."""
    import relucent.config as cfg

    mx = 0.0
    for p in cplx:
        ip = getattr(p, "interior_point", None)
        if ip is None:
            continue
        v = float(np.max(np.abs(np.asarray(ip).reshape(-1))))
        if v > mx:
            mx = v
    if mx <= 0.0:
        mx = float(cfg.DEFAULT_PLOT_BOUND)
    return float(cfg.PLOT_MARGIN_FACTOR) * mx


def _betti_numbers_truncated_1d(cplx: "Complex", *, bound: float | None = None) -> dict[int, int]:
    """Traditional (truncated) Betti numbers for 1D complexes via endpoint graph."""
    b = float(bound) if bound is not None else float(_auto_truncation_bound_from_interior_points(cplx))

    # Endpoint coordinates come from Qhull/Gurobi and can differ slightly across
    # adjacent cells that share a vertex. Use a moderately loose snapping scale
    # so shared endpoints merge reliably.
    def _snap(x: np.ndarray, tol: float = 1e-4) -> tuple[int, ...]:
        return tuple(np.round(np.asarray(x, dtype=float) / tol).astype(int).tolist())

    def _endpoints(poly: "Polyhedron") -> list[tuple[int, ...]]:
        verts_arr = poly.get_bounded_vertices(b)
        if verts_arr is None or len(verts_arr) == 0:
            return []
        verts_arr = np.asarray(verts_arr, dtype=float)
        uniq = np.unique(np.round(verts_arr, decimals=6), axis=0)
        if uniq.shape[0] == 1:
            return [_snap(uniq[0])]
        best = (0, 1)
        best_d = -1.0
        for i in range(uniq.shape[0]):
            for j in range(i + 1, uniq.shape[0]):
                d = float(np.sum((uniq[i] - uniq[j]) ** 2))
                if d > best_d:
                    best_d = d
                    best = (i, j)
        return [_snap(uniq[best[0]]), _snap(uniq[best[1]])]

    one_cells = [p for p in cplx if int(p.dim) == 1]
    edges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    verts: dict[tuple[int, ...], int] = {}

    for poly in one_cells:
        eps = _endpoints(poly)
        if len(eps) < 2:
            continue
        u, v = eps[0], eps[1]
        verts.setdefault(u, len(verts))
        verts.setdefault(v, len(verts))
        edges.append((u, v))

    n0 = len(verts)
    n1 = len(edges)
    if n0 == 0:
        return {}

    # Build ∂1 as an (n0 x n1) incidence matrix over GF(2).
    nwords = (n1 + 63) // 64
    packed = np.zeros((n0, nwords), dtype=np.uint64)
    for j, (u, v) in enumerate(edges):
        w = j >> 6
        bit = np.uint64(1) << (j & 63)
        packed[verts[u], w] ^= bit
        packed[verts[v], w] ^= bit

    r1 = int(gf2_rank_packed(packed, n1))
    beta0 = int(n0 - r1)
    beta1 = int(n1 - r1)
    out: dict[int, int] = {}
    if beta0 != 0:
        out[0] = beta0
    if beta1 != 0:
        out[1] = beta1
    return out


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


def gf2_rank(matrix: np.ndarray) -> int:
    """Matrix rank over GF(2) via Gaussian elimination with bit-packed rows."""
    if matrix.size == 0:
        return 0
    nrows, ncols = matrix.shape
    nwords = (int(ncols) + 63) // 64
    packed = np.zeros((nrows, nwords), dtype=np.uint64)
    m = np.asarray(matrix)
    for w in range(nwords):
        c0 = w * 64
        c1 = min(c0 + 64, int(ncols))
        acc = np.zeros(nrows, dtype=np.uint64)
        for j, bc in enumerate(range(c0, c1)):
            acc |= (m[:, bc].astype(np.uint8, copy=False) & 1).astype(np.uint64) << j
        packed[:, w] = acc
    return int(gf2_rank_packed(packed, int(ncols)))


@dataclass(frozen=True, slots=True)
class _InfinityCell:
    """A formal 0-cell used for one-point compactification (per component)."""

    id: int

    @property
    def dim(self) -> int:  # noqa: D401 - property is a simple constant
        return 0

    @property
    def tag(self) -> bytes:
        # Must be unique per instance so ∂_1 can reference distinct rows.
        return b"__RELUCENT_INFINITY__" + self.id.to_bytes(8, "little", signed=False)


def _iter_codim1_faces(cell: Polyhedron) -> Iterator[Polyhedron]:
    """Enumerate codimension-1 faces (facets) of a polyhedron.

    Use the polyhedron's supporting halfspace indices (SHIs) so we only generate
    true facets. This can rely on Gurobi in some configurations.
    """
    # 0-cells have no codimension-1 faces.
    if int(cell.dim) <= 0:
        return
    ss = np.asarray(cell.ss_np)
    if ss.ndim != 2 or ss.shape[0] != 1:
        raise ValueError(f"Expected sign sequence shape (1, m), got {ss.shape}.")
    for shi in cell.shis:
        face_ss = ss.copy()
        face_ss[0, int(shi)] = 0
        yield cell.__class__(cell._net, face_ss, bound=cell.bound)


def _iter_codim1_faces_by_ss(cell: Polyhedron) -> Iterator[Polyhedron]:
    """Enumerate codimension-1 faces by zeroing any nonzero SS entry.

    For low-dimensional cells (especially 1-cells), ``cell.shis`` can be
    over/under-inclusive because SHI detection is performed in the ambient space
    rather than the cell's affine hull. This SS-based rule is slower in high
    dimension but reliable for 1D incidence.
    """
    ss = np.asarray(cell.ss_np)
    if ss.ndim != 2 or ss.shape[0] != 1:
        raise ValueError(f"Expected sign sequence shape (1, m), got {ss.shape}.")
    for shi in np.flatnonzero(ss[0] != 0):
        face_ss = ss.copy()
        face_ss[0, int(shi)] = 0
        face = cell.__class__(cell._net, face_ss, bound=cell.bound)
        if face.is_face_of(cell):
            yield face


def _facet_tags_for_1cell(sigma: "Polyhedron", cells: dict[bytes, "Polyhedron"]) -> set[bytes]:
    """Return tags of 0-faces that are *actual* facets of a 1-cell in the subcomplex.

    SS-zeroing can propose faces that are not true facets of ``sigma``. We filter
    using the sign-sequence face relation (cheap; no geometry).
    """
    tags: set[bytes] = set()
    for tau in _iter_codim1_faces_by_ss(sigma):
        t = tau.tag
        if t not in cells:
            continue
        # Reuse pooled object and verify face relation to avoid spurious endpoints.
        f = cells[t]
        if f.is_face_of(sigma):
            tags.add(t)
    return tags


def _build_subcomplex(
    all_cells: Iterable[Polyhedron] | Complex,
    selector: Callable[[Polyhedron], bool],
) -> dict[bytes, Polyhedron]:
    """Select cells, then close downward under faces.

    If ``all_cells`` is a :class:`relucent.complex.Complex`, this function uses
    :meth:`relucent.complex.Complex.get_chain_complex` to enumerate cells across
    dimensions (and to reuse existing ``Polyhedron`` objects when possible).
    """
    pool_by_tag: dict[bytes, Polyhedron] | None = None
    max_dim: int | None = None
    all_iter: Iterable[Polyhedron] = cast(Iterable["Polyhedron"], all_cells)
    if hasattr(all_cells, "get_chain_complex"):
        cplx = cast("Complex", all_cells)
        # Avoid calling contraction-chain logic for 1D complexes: contracting a
        # 1D boundary complex relies on a dual-graph construction that is tuned
        # for top-dimensional (ambient) cells. For Betti computations on 1D
        # boundary complexes, we only need the provided 1-cells plus their faces.
        try:
            max_dim_here = max(int(p.dim) for p in cplx)
        except ValueError:
            max_dim_here = 0
        if max_dim_here == 1:
            # For 1D complexes, we *must* allow creating fresh 0-faces on demand
            # (they are typically not stored in the complex). So we do not use a
            # pooled chain-complex mapping here.
            pool_by_tag = None
            all_iter = cast(Iterable["Polyhedron"], cplx)
        else:
            # Prefer reusing Polyhedron objects already stored in the chain complex
            # instead of generating fresh face objects repeatedly.
            chain = cplx.get_chain_complex(verbose=False)
            pool_by_tag = {cell.tag: cell for cx in chain for cell in cx}  # type: ignore[attr-defined]
        # Important: build the subcomplex as the downward closure of the
        # *top-dimensional* selected cells. Lower-dimensional cells that appear in
        # the contraction chain but are not faces of any selected top cell should
        # not be included, or β0 can be inflated by isolated "artifact" vertices.
        if pool_by_tag is not None:
            max_dim = max(int(c.dim) for c in pool_by_tag.values()) if pool_by_tag else 0
            all_iter = (c for c in pool_by_tag.values() if int(c.dim) == max_dim)

    chosen: dict[bytes, Polyhedron] = {}
    stack: list[Polyhedron] = []
    for c in all_iter:
        if selector(c):
            t = c.tag
            if t not in chosen:
                chosen[t] = c
                stack.append(c)

    while stack:
        cell = stack.pop()
        if pool_by_tag is not None:
            # When we have a pool from get_chain_complex(), avoid calling `face.tag`
            # on freshly constructed faces (which can trigger geometry/SHI work via
            # cached polyhedron state). Instead, compute the face tag directly from
            # the sign sequence and reuse the pooled object if present.
            # For 1-cells, use SS-based faces; for higher-dimensional cells use SHIs.
            face_iter: Iterable[Polyhedron]
            if int(cell.dim) == 1:
                face_iter = _iter_codim1_faces_by_ss(cell)
            else:
                face_iter = _iter_codim1_faces(cell)

            for face in face_iter:
                t = face.tag
                if t in chosen:
                    continue
                f = pool_by_tag.get(t)
                if f is None:
                    continue
                chosen[t] = f
                stack.append(f)
        else:
            face_iter2: Iterable[Polyhedron]
            if int(cell.dim) == 1:
                face_iter2 = _iter_codim1_faces_by_ss(cell)
            else:
                face_iter2 = _iter_codim1_faces(cell)
            for face in face_iter2:
                t = face.tag
                if t in chosen:
                    continue
                chosen[t] = face
                stack.append(face)
    return chosen


def _count_facets_in_subcomplex(sigma: Polyhedron, cells: dict[bytes, Polyhedron]) -> int:
    if int(sigma.dim) == 1:
        return len(_facet_tags_for_1cell(sigma, cells))
    return sum(1 for tau in _iter_codim1_faces(sigma) if tau.tag in cells)


def _infinity_attachment_for_compactify(
    all_cells: object,
    *,
    cells: dict[bytes, Polyhedron],
    max_dim: int,
) -> tuple[list[_InfinityCell], dict[bytes, _InfinityCell | None]]:
    """Formal 0-cells at infinity so unbounded 1-cells get two GF(2) boundary endpoints.

    In the assembled sign-sequence cell complex, unbounded 1-cells can have
    too few codimension-1 faces present *inside the subcomplex*:

    - 1 visible facet: one endpoint is missing (typical "ray" case)
    - 0 visible facets: both endpoints are missing (typical "affine line" case)

    Attaching a formal infinity 0-cell per *dual-graph component* provides a
    consistent GF(2) incidence map for Betti-number computation, without
    incorrectly gluing ends of disjoint components (which would happen with a
    single global infinity 0-cell).

    Returns:
        (infinity_0_cells, map top-dimensional tag -> infinity cell or None).
    """
    if max_dim < 1:
        return [], {}

    top_in_sub = [c for c in cells.values() if int(c.dim) == max_dim]
    if not top_in_sub:
        return [], {}

    tag_to_inf: dict[bytes, _InfinityCell | None] = {c.tag: None for c in top_in_sub}

    # Any top-dimensional 1-cell with <= 1 visible endpoint facets indicates that
    # we will need an infinity vertex to make ∂_1 well-defined.
    dangling: list[Polyhedron] = [
        c for c in top_in_sub if _count_facets_in_subcomplex(c, cells) <= 1
    ]
    if not dangling:
        return [], tag_to_inf

    if not hasattr(all_cells, "get_dual_graph"):
        inf0 = _InfinityCell(id=0)
        for c in dangling:
            tag_to_inf[c.tag] = inf0
        return [inf0], tag_to_inf

    cplx = cast("Complex", all_cells)
    G = cplx.get_dual_graph(verbose=False)
    top_tags = frozenset(c.tag for c in top_in_sub)
    nodes = [p for p in G.nodes() if p.tag in top_tags]
    if not nodes:
        inf0 = _InfinityCell(id=0)
        for c in dangling:
            tag_to_inf[c.tag] = inf0
        return [inf0], tag_to_inf

    Gsub = G.subgraph(nodes)
    infs: list[_InfinityCell] = []
    next_id = 0
    for comp in nx.connected_components(Gsub):
        comp_dangling = [
            sigma
            for sigma in comp
            if _count_facets_in_subcomplex(cast("Polyhedron", sigma), cells) <= 1
        ]
        if not comp_dangling:
            continue
        inf_c = _InfinityCell(id=next_id)
        next_id += 1
        infs.append(inf_c)
        for sigma in comp_dangling:
            tag_to_inf[cast("Polyhedron", sigma).tag] = inf_c
    return infs, tag_to_inf


def betti_numbers(
    all_cells: Iterable[Polyhedron] | Complex,
    selector: Callable[[Polyhedron], bool] | None = None,
    *,
    reduced: bool = False,
    compactify: bool = False,
) -> dict[int, int]:
    """Compute Betti numbers over GF(2) for a selected subcomplex.

    Args:
        all_cells: Iterable of available cells (polyhedra). These may include only
            top-dimensional cells; this function will close under faces as needed.
        selector: Predicate deciding which cells to keep *before* downward closure.
            If None, keep all provided cells.
        reduced: If True, return reduced homology Betti numbers (β̃_0 = β_0 - 1 for
            nonempty complexes).
        compactify: If True, add formal 0-cells at infinity and (in dimension 1) attach
            them to 1-cells that otherwise have only one 0-face in the subcomplex.
            When ``all_cells`` is a :class:`~relucent.complex.Complex`, one infinity
            0-cell is introduced per dual-graph component that needs it, so disjoint
            boundary pieces are not glued at infinity.

    Returns:
        Mapping k -> β_k.
    """
    if selector is None:

        def _all(_: Polyhedron) -> bool:
            return True

        selector = _all

    cells = _build_subcomplex(all_cells, selector)
    if not cells:
        return {}

    # Group by dimension.
    cells_by_dim: dict[int, list[Polyhedron | _InfinityCell]] = {}
    max_dim = 0
    for c in cells.values():
        k = int(c.dim)
        cells_by_dim.setdefault(k, []).append(c)
        max_dim = max(max_dim, k)

    infs: list[_InfinityCell] = []
    tag_to_infinity: dict[bytes, _InfinityCell | None] = {}
    if compactify:
        infs, tag_to_infinity = _infinity_attachment_for_compactify(
            all_cells, cells=cells, max_dim=max_dim
        )
        cells_by_dim.setdefault(0, []).extend(infs)

    # Boundary ranks over GF(2) using bit-packed columns.
    boundary_rank: dict[int, int] = {}
    boundary_rank[0] = 0

    for k in range(1, max_dim + 1):
        rows = cells_by_dim.get(k - 1, [])
        cols = cells_by_dim.get(k, [])
        if not rows or not cols:
            boundary_rank[k] = 0
            continue

        # Index rows by tag (Polyhedron) or infinity tag.
        row_index: dict[bytes, int] = {}
        for i, r in enumerate(rows):
            t = r.tag if isinstance(r, _InfinityCell) else cast("Polyhedron", r).tag
            row_index[t] = int(i)

        nrows = len(rows)
        ncols = len(cols)
        nwords = (ncols + 63) // 64
        packed = np.zeros((nrows, nwords), dtype=np.uint64)

        for j, sigma in enumerate(cols):
            w = j >> 6
            bit = np.uint64(1) << (j & 63)
            if isinstance(sigma, _InfinityCell):
                continue

            # Facets = codim-1 faces that are present in K.
            # For 1-cells, use SS-based face enumeration and de-duplicate by tag.
            facet_tags: set[bytes] = set()
            sigma_poly = cast("Polyhedron", sigma)
            if k == 1:
                facet_tags = _facet_tags_for_1cell(sigma_poly, cells)
            else:
                for tau in _iter_codim1_faces(sigma_poly):
                    if tau.tag in cells:
                        facet_tags.add(tau.tag)

            # Compactification: unbounded 1-cells get a second endpoint on the formal
            # infinity 0-cell for their dual component.
            if compactify and k == 1 and len(facet_tags) == 1:
                inf_att = tag_to_infinity.get(sigma_poly.tag)
                if inf_att is not None:
                    ii_inf = row_index.get(inf_att.tag)
                    if ii_inf is not None:
                        packed[ii_inf, w] ^= bit

            for t in facet_tags:
                ii = row_index.get(t)
                if ii is not None:
                    packed[ii, w] ^= bit

        boundary_rank[k] = int(gf2_rank_packed(packed, ncols))

    # Betti numbers: β_k = n_k - rank(∂_k) - rank(∂_{k+1})
    beta: dict[int, int] = {}
    for k in range(0, max_dim + 1):
        n_k = len(cells_by_dim.get(k, []))
        r_dk = boundary_rank.get(k, 0) if k >= 1 else 0
        r_dk1 = boundary_rank.get(k + 1, 0) if k < max_dim else 0
        beta[k] = int(n_k - r_dk - r_dk1)

    if reduced and sum(len(v) for v in cells_by_dim.values()) > 0:
        beta[0] = int(beta.get(0, 0) - 1)

    # Trim trailing zeros for cleanliness.
    return {k: v for k, v in beta.items() if v != 0}


def betti_numbers_complex(
    cplx: Complex,
    *,
    selector: Callable[[Polyhedron], bool] | None = None,
    homology: HomologyMode = "standard",
) -> dict[int, int]:
    """Compatibility wrapper for :class:`relucent.complex.Complex`."""
    mode = cast(str, homology)
    if mode in {"standard", "contracted_standard"}:
        return betti_numbers(cplx, selector, reduced=False, compactify=False)
    if mode in {"borel_moore", "contracted_borel_moore"}:
        # Borel--Moore ≅ reduced homology of one-point compactification (for nice spaces).
        return betti_numbers(cplx, selector, reduced=True, compactify=True)
    if mode == "traditional_truncated":
        # Geometric truncation + endpoint simplicial complex (1D).
        # This uses an automatic radius chosen from interior points.
        try:
            max_dim = max(int(p.dim) for p in cplx)
        except ValueError:
            return {}
        if max_dim == 1:
            return _betti_numbers_truncated_1d(cplx, bound=None)
        # Fallback: old behavior for higher-dimensional truncated complexes (not implemented yet).
        return betti_numbers(cplx, selector, reduced=False, compactify=True)
    if mode == "traditional":
        # Legacy behavior: one-point compactification-style fixups for unbounded directions.
        return betti_numbers(cplx, selector, reduced=False, compactify=True)
    raise ValueError(f"Unknown homology mode: {homology!r}")
