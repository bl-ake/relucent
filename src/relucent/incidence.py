"""Cubical incidence engine: combinatorial dual-graph and meta-graph primitives.

Every cell is identified by its sign sequence (``ss``); a codimension-1 face is
obtained by zeroing one nonzero entry (a supporting-hyperplane index, or SHI).
This module is the single source of truth for that combinatorics. Consumers
take *derived views*:

- **Dual-graph edges** (:func:`dual_edges_top_dim`): group top cells by shared
  face tag (1-cells) or flip neighbor (dim >= 2).
- **Meta-graph face edges** (:func:`collect_meta_face_edges`): zero each
  ``ss_i != 0`` and keep the edge iff the resulting face tag is a known cell.
- **Node metadata** (:func:`meta_node_attrs`): flip-neighbor SHIs via
  :func:`cubical_cell_shis`, never the LP-derived ``poly._shis`` cache.

Two SHI semantics matter and must not be confused:

- **Cubical flip SHIs** (:func:`cubical_cell_shis`): authoritative for dual-graph
  adjacency, meta-graph node metadata, and contracted slices.
- **LP facet SHIs** (``get_shis(..., strict=True)`` in :mod:`relucent.calculations`):
  geometric facets on *ambient* top cells; can be a strict subset of the cubical
  set, so they must never drive meta-graph face-edge or node-metadata assembly.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

import relucent.config as cfg
from relucent._logging import logger
from relucent.errors import CubicalConsistencyError, DualGraphAsymmetricEdgeError, ShiFlipInvariantError
from relucent.poly import Polyhedron
from relucent.utils import encode_ss, flip_ss_at_shi, get_mp_context

if TYPE_CHECKING:
    from relucent.complex import Complex

__all__ = [
    "META_FACE_PARALLEL_MIN_CELLS",
    "build_dual_graph",
    "certify_dual_graph",
    "classify_finite_ascending",
    "classify_finite_combinatorial",
    "classify_finite_lp_fallback",
    "propagate_infeasible_exclusion",
    "classify_lazy_face_polys",
    "classify_one_cells_finite_from_face_edges",
    "collect_meta_face_edges",
    "cubical_cell_shis",
    "dual_edges_top_dim",
    "dual_graph_edge_top_dim",
    "face_tag",
    "flip_tag",
    "geometric_infeasible_one_cells",
    "meta_node_attrs",
    "parallel_collect_meta_face_edges",
    "set_contracted_shis",
    "ss_nonzero_indices",
    "sync_shis_from_dual_graph",
    "verify_contracted_shis",
    "verify_dual_graph_cubical",
    "verify_flip_shi_symmetry",
    "verify_shi_flip_neighbors",
    "verify_shis_from_dual_graph",
]

META_FACE_PARALLEL_MIN_CELLS = 64


# ---------------------------------------------------------------------------
# Combinatorial primitives
# ---------------------------------------------------------------------------


def ss_nonzero_indices(ss: np.ndarray) -> tuple[int, ...]:
    """All sign-sequence indices with ``ss_i != 0``.

    Used when assembling meta-graph face **edges**: try zeroing each nonzero
    entry. Also drives dual-graph flip-neighbor adjacency for top-dimensional
    cells.

    Not used when building the contraction chain
    (:meth:`~relucent.complex.Complex._codim_one_face_kwargs` seeds candidates
    from this, then :func:`set_contracted_shis` finalizes). Propagated
    ``poly._shis`` can be a strict subset after coface intersection; using it
    for edge discovery omits valid faces and breaks ``∂² = 0``.
    """
    row = np.asarray(ss, dtype=np.int8).ravel()
    return tuple(int(i) for i in np.flatnonzero(row != 0))


def face_tag(ss: np.ndarray, shi: int) -> bytes:
    """Tag of the codimension-one face obtained by zeroing ``shi`` in ``ss``."""
    ss_arr = np.asarray(ss, dtype=np.int8)
    row = ss_arr.reshape(-1).copy()
    row[int(shi)] = 0
    return encode_ss(row.reshape(ss_arr.shape))


def flip_tag(ss: np.ndarray, shi: int) -> bytes:
    """Tag of the same-dimension neighbor across hyperplane ``shi``."""
    row = np.asarray(ss, dtype=np.int8).ravel()
    return encode_ss(flip_ss_at_shi(row, int(shi)).reshape(np.asarray(ss).shape))


def dual_graph_edge_top_dim(*, cell_top_dim: int, ambient_dim: int) -> int:
    """``top_dim`` argument for :func:`dual_edges_top_dim` / :func:`verify_dual_graph_cubical`.

    1-cells always use flip-neighbor adjacency so ``_shis`` match flip-SHI verification.
    """
    _ = ambient_dim
    return cell_top_dim if cell_top_dim >= 2 else 2


def cubical_cell_shis(ss: np.ndarray, *, neighbor_tags: set[bytes]) -> list[int]:
    """Flip-neighbor SHIs: nonzero sign-sequence crossings with a same-dimension neighbor."""
    row = np.asarray(ss, dtype=np.int8).ravel()
    kept: list[int] = []
    for shi in ss_nonzero_indices(np.asarray(ss)):
        shi_i = int(shi)
        if shi_i >= row.shape[0] or row[shi_i] == 0:
            continue
        if encode_ss(flip_ss_at_shi(row, shi_i)) in neighbor_tags:
            kept.append(shi_i)
    return sorted(kept)


# ---------------------------------------------------------------------------
# Dual graph: build + repair + certify
# ---------------------------------------------------------------------------


def dual_edges_top_dim(
    cells: Iterable[Polyhedron],
    neighbor_tags: set[bytes],
    *,
    top_dim: int | None = None,
    prefer_cell_shis: bool = False,
) -> list[tuple[Polyhedron, Polyhedron, int]]:
    """Cubical dual edges for top-dimensional cells.

    For ``top_dim == 1``, pairs cells sharing a combinatorial 0-face tag.  For
    ``top_dim >= 2``, pairs cells that are flip neighbors across an existing
    ``ss_i != 0`` crossing (combinatorial cubical adjacency).
    """
    cell_list = list(cells)
    if not cell_list:
        return []
    if top_dim is None:
        top_dim = int(cell_list[0].dim)
    if int(top_dim) == 1:
        return _dual_edges_one_dim(cell_list)
    return _dual_edges_flip_neighbors(cell_list, neighbor_tags, prefer_cell_shis=prefer_cell_shis)


def _dual_edges_flip_neighbors(
    cell_list: list[Polyhedron],
    neighbor_tags: set[bytes],
    *,
    prefer_cell_shis: bool = False,
) -> list[tuple[Polyhedron, Polyhedron, int]]:
    tag_to_poly = {p.tag: p for p in cell_list}
    edges: list[tuple[Polyhedron, Polyhedron, int]] = []
    seen: set[tuple[bytes, bytes]] = set()
    for u in cell_list:
        ss = np.asarray(u.ss_np)
        candidates = list(u._shis) if prefer_cell_shis and u._shis is not None else list(ss_nonzero_indices(ss))
        for shi in candidates:
            shi_i = int(shi)
            if shi_i >= ss.shape[-1] or int(ss.ravel()[shi_i]) == 0:
                raise ValueError(f"SHI {shi_i} is out of bounds for sign sequence {ss}.")
            if int(u.dim) == 1 and u.halfspaces is not None and not u.is_shi_face_feasible(shi_i):
                continue
            vt = flip_tag(ss, shi_i)
            if vt not in neighbor_tags:
                continue
            v = tag_to_poly.get(vt)
            if v is None or u.tag == v.tag:
                continue
            if int(v.dim) == 1 and v.halfspaces is not None and not v.is_shi_face_feasible(shi_i):
                continue
            lo, hi = (u.tag, v.tag) if u.tag < v.tag else (v.tag, u.tag)
            if (lo, hi) in seen:
                continue
            seen.add((lo, hi))
            edges.append((u, v, shi_i))
    return edges


def _dual_edges_one_dim(cell_list: list[Polyhedron]) -> list[tuple[Polyhedron, Polyhedron, int]]:
    face_to_witness: dict[bytes, dict[bytes, int]] = defaultdict(dict)
    for poly in cell_list:
        ss = np.asarray(poly.ss_np)
        for shi in ss_nonzero_indices(ss):
            shi_i = int(shi)
            ft = face_tag(ss, shi_i)
            face_to_witness[ft][poly.tag] = shi_i

    tag_to_poly = {p.tag: p for p in cell_list}
    edges: list[tuple[Polyhedron, Polyhedron, int]] = []
    seen: set[tuple[bytes, bytes]] = set()
    for _ft, witnesses in face_to_witness.items():
        if len(witnesses) > 2:
            if cfg.CAREFUL_MODE:
                logger.warning(
                    "Skipping dual edge for 0-face with %d incident 1-cells (expected <= 2).",
                    len(witnesses),
                )
            continue
        if len(witnesses) != 2:
            continue
        (tag_u, shi_u), (tag_v, _) = list(witnesses.items())
        u, v = tag_to_poly[tag_u], tag_to_poly[tag_v]
        lo, hi = (u.tag, v.tag) if u.tag < v.tag else (v.tag, u.tag)
        if (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        edges.append((u, v, shi_u))
    return edges


def build_dual_graph(
    top_cells: list[Polyhedron],
    *,
    top_dim: int,
    ambient_dim: int,
    repair: bool = True,
) -> nx.Graph[Polyhedron]:
    """Build the combinatorial dual graph on ``top_cells``.

    Nodes are the given polyhedra; edges connect cubical flip neighbors (or
    0-face-sharing neighbors when ``top_dim == 1``). When ``repair`` is True
    (the default, and the only supported repair in this package), each node's
    ``poly._shis`` is overwritten from the assembled edges via
    :func:`sync_shis_from_dual_graph` -- this is the one place relucent
    corrects an asymmetric or stale LP-derived SHI cache.
    """
    graph: nx.Graph[Polyhedron] = nx.Graph()
    graph.add_nodes_from(top_cells)
    if not top_cells:
        return graph
    neighbor_tags = {p.tag for p in top_cells}
    edge_top_dim = dual_graph_edge_top_dim(cell_top_dim=top_dim, ambient_dim=ambient_dim)
    prefer_cell_shis = top_dim < ambient_dim
    edges = dual_edges_top_dim(top_cells, neighbor_tags, top_dim=edge_top_dim, prefer_cell_shis=prefer_cell_shis)
    for u, v, shi in edges:
        graph.add_edge(u, v, shi=shi)
    if repair:
        sync_shis_from_dual_graph(graph)
    return graph


def sync_shis_from_dual_graph(graph: nx.Graph[Any]) -> None:
    """Overwrite each node's ``poly._shis`` from the dual-graph edge ``shi`` attributes.

    This is relucent's one sanctioned repair: it replaces a possibly
    asymmetric or stale LP-derived SHI cache with the symmetric, combinatorially
    consistent set implied by the assembled dual-graph edges.
    """
    shis_per_node: dict[Any, list[int]] = {n: [] for n in graph}
    for u, v, data in graph.edges(data=True):
        shi = data.get("shi")
        if shi is None:
            continue
        shis_per_node[u].append(int(shi))
        shis_per_node[v].append(int(shi))
    for node, shis in shis_per_node.items():
        poly = node if isinstance(node, Polyhedron) else None
        if poly is None and isinstance(graph.nodes[node], dict):
            poly = graph.nodes[node].get("poly")
        if isinstance(poly, Polyhedron):
            assigned = sorted(set(shis))
            keep_strict = (
                bool(getattr(poly, "_shis_strict", False))
                and poly._shis is not None
                and set(int(s) for s in poly._shis) == set(assigned)
            )
            poly._shis = assigned
            poly._shis_strict = keep_strict


def verify_shis_from_dual_graph(graph: nx.Graph[Any]) -> None:
    """Require each polyhedron's ``_shis`` to match incident dual-graph edge labels."""
    shis_per_node: dict[Any, list[int]] = {n: [] for n in graph}
    for u, v, data in graph.edges(data=True):
        shi = data.get("shi")
        if shi is None:
            raise CubicalConsistencyError("Dual-graph edge is missing 'shi' attribute.")
        shis_per_node[u].append(int(shi))
        shis_per_node[v].append(int(shi))
    for node, expected in shis_per_node.items():
        poly = node if isinstance(node, Polyhedron) else None
        if poly is None and isinstance(graph.nodes[node], dict):
            poly = graph.nodes[node].get("poly")
        if not isinstance(poly, Polyhedron):
            continue
        actual = sorted(int(s) for s in (poly._shis or []))
        if actual != sorted(set(expected)):
            raise CubicalConsistencyError(
                f"Dual-graph SHI sync mismatch on {poly!r}: cached {actual!r} != edges {sorted(set(expected))!r}."
            )


def verify_dual_graph_cubical(cells: Iterable[Polyhedron], graph: nx.Graph[Any], *, top_dim: int) -> None:
    """Each dual edge's ``shi`` must zero to a shared face tag on both endpoints."""
    if top_dim <= 0:
        return
    endtags: dict[bytes, set[bytes]] = {}
    if top_dim == 1:
        for p in cells:
            if int(p.dim) != 1:
                continue
            endtags[p.tag] = {face_tag(np.asarray(p.ss_np), int(shi)) for shi in ss_nonzero_indices(p.ss_np)}
        for u, v in graph.edges():
            inter = endtags.get(u.tag, set()) & endtags.get(v.tag, set())
            if len(inter) != 1:
                raise CubicalConsistencyError(
                    "Dual-graph edge "
                    + f"({u.tag!r}, {v.tag!r}) is inconsistent with 0-face tags: "
                    + f"|intersection|={len(inter)}."
                )
        return

    for u, v, shi in graph.edges(data="shi"):
        if shi is None:
            raise CubicalConsistencyError("Dual-graph edge is missing 'shi' attribute.")
        ss_u = np.asarray(u.ss_np)
        ft_u = face_tag(ss_u, int(shi))
        ss_v = np.asarray(v.ss_np)
        ft_v = face_tag(ss_v, int(shi))
        if ft_u != ft_v:
            raise CubicalConsistencyError(
                "Dual-graph edge " + f"shi={int(shi)} does not induce a common face tag on ({u.tag!r}, {v.tag!r})."
            )


def verify_flip_shi_symmetry(cplx: Complex) -> None:
    """Every top-cell SHI must flip to a same-dimension neighbor that lists the SHI too.

    Applies to any complex whose cells share a single dimension slice: full
    ambient complexes as well as contracted chain-complex slices.
    """
    if len(cplx) == 0:
        return
    top_dim = max(int(p.dim) for p in cplx)
    tag2poly = {p.tag: p for p in cplx}
    for poly in cplx:
        if int(poly.dim) != top_dim:
            continue
        ss = np.asarray(poly.ss_np, dtype=np.int8).ravel()
        for shi in poly._shis or []:
            shi_i = int(shi)
            if shi_i >= ss.shape[0] or int(ss[shi_i]) == 0:
                raise ShiFlipInvariantError(f"SHI {shi_i} on {poly!r} has ss[{shi_i}]=0.")
            neighbor = tag2poly.get(encode_ss(flip_ss_at_shi(ss, shi_i)))
            if neighbor is None:
                raise ShiFlipInvariantError(f"SHI {shi_i} on {poly!r} has no flip neighbor in the complex.")
            if int(neighbor.dim) != top_dim:
                raise ShiFlipInvariantError(
                    f"SHI {shi_i} on {poly!r} flips to wrong dimension {neighbor.dim} (expected {top_dim})."
                )
            if shi_i not in (neighbor._shis or []):
                raise ShiFlipInvariantError(
                    f"Asymmetric SHI {shi_i}: listed on {poly!r} but not on flip neighbor {neighbor!r}."
                )


def certify_dual_graph(graph: nx.Graph[Polyhedron], cplx: Complex, *, top_dim: int | None = None) -> None:
    """The one full dual-graph check: bidirectional SHI support + cubical face-tag consistency + sync."""
    if graph.number_of_edges() == 0:
        return
    if top_dim is None:
        top_dim = max(int(p.dim) for p in cplx)
    cubical_top_dim = dual_graph_edge_top_dim(cell_top_dim=top_dim, ambient_dim=int(cplx.dim))
    top_cells = [p for p in cplx if int(p.dim) == top_dim]
    for u, v, data in graph.edges(data=True):
        shi = data.get("shi")
        if shi is None:
            raise DualGraphAsymmetricEdgeError("Dual-graph edge is missing 'shi' attribute.")
        shi_i = int(shi)
        if shi_i not in u.shis:
            raise DualGraphAsymmetricEdgeError(f"Dual edge shi={shi_i} on ({u!r}, {v!r}) is not in u.shis.")
        if shi_i not in v.shis:
            raise DualGraphAsymmetricEdgeError(f"Dual edge shi={shi_i} on ({u!r}, {v!r}) is not in v.shis.")
    verify_dual_graph_cubical(top_cells, graph, top_dim=cubical_top_dim)
    verify_shis_from_dual_graph(graph)


def verify_shi_flip_neighbors(ss: np.ndarray, shis: Iterable[int], *, neighbor_tags: set[bytes]) -> None:
    """Raise if ``shis`` is not the cubical flip-neighbor list for ``ss`` in the slice."""
    expected = cubical_cell_shis(ss, neighbor_tags=neighbor_tags)
    actual = sorted(int(s) for s in shis)
    if actual != expected:
        raise CubicalConsistencyError(f"SHI flip-neighbor mismatch: expected {expected!r}, got {actual!r}.")


# ---------------------------------------------------------------------------
# Contract path: authoritative SHIs on a contracted chain-complex slice
# ---------------------------------------------------------------------------


def _contracted_shis_for_poly(poly: Polyhedron, *, neighbor_tags: set[bytes]) -> list[int]:
    assigned = cubical_cell_shis(poly.ss_np, neighbor_tags=neighbor_tags)
    if int(poly.dim) == 1 and poly.halfspaces is not None:
        assigned = [s for s in assigned if poly.is_shi_face_feasible(int(s))]
    return assigned


def set_contracted_shis(cplx: Complex) -> int:
    """Set authoritative ``_shis`` on a contracted slice after :meth:`~relucent.complex.Complex.contract`.

    Used only on the **contract path** (``contract``, ``get_boundary_complex``).
    Assigns :func:`cubical_cell_shis` once the full dimension slice is known.
    Call :func:`verify_contracted_shis` to assert flip-neighbor and symmetry invariants.

    Returns the number of cells whose ``_shis`` list was changed.
    """
    if len(cplx) == 0:
        return 0
    neighbor_tags = {p.tag for p in cplx}
    n_changed = 0
    for poly in cplx:
        assigned = _contracted_shis_for_poly(poly, neighbor_tags=neighbor_tags)
        if assigned != poly._shis:
            poly._shis = assigned
            poly._shis_strict = False
            n_changed += 1
    return n_changed


def verify_contracted_shis(cplx: Complex) -> None:
    """Check contracted-slice ``_shis`` match cubical flip neighbors and mutual listing."""
    if len(cplx) == 0:
        return
    neighbor_tags = {p.tag for p in cplx}
    for poly in cplx:
        expected = _contracted_shis_for_poly(poly, neighbor_tags=neighbor_tags)
        actual = sorted(int(s) for s in (poly._shis or []))
        if actual != expected:
            raise CubicalConsistencyError(f"Contracted SHI mismatch on {poly!r}: expected {expected!r}, got {actual!r}.")
    if any(int(p.dim) == 1 for p in cplx):
        verify_flip_shi_symmetry(cplx)


# ---------------------------------------------------------------------------
# Meta-graph face-edge collection and node metadata
# ---------------------------------------------------------------------------


def collect_meta_face_edges(
    cells: list[tuple[bytes, np.ndarray, tuple[int, ...]]],
    valid_face_tags: set[bytes],
) -> tuple[list[tuple[bytes, bytes, int]], list[bytes]]:
    """Return face edges (src, dst, shi) for one chunk of k-cells.

    The ``shis`` tuple on each cell should come from :func:`ss_nonzero_indices`,
    not from propagated ``poly._shis``.
    """
    edges: list[tuple[bytes, bytes, int]] = []
    extra_tags: list[bytes] = []
    for src_tag, ss, shis in cells:
        ss_arr = np.asarray(ss)
        for shi in shis:
            shi_i = int(shi)
            ft = face_tag(ss_arr, shi_i)
            if ft not in valid_face_tags:
                continue
            edges.append((src_tag, ft, shi_i))
            extra_tags.append(ft)
    return edges, extra_tags


def parallel_collect_meta_face_edges(
    cells: list[tuple[bytes, np.ndarray, tuple[int, ...]]],
    valid_face_tags: set[bytes],
    *,
    nworkers: int,
) -> tuple[list[tuple[bytes, bytes, int]], list[bytes]]:
    n = len(cells)
    chunk_size = max(n // (nworkers * 4), 1)
    chunks = [cells[i : i + chunk_size] for i in range(0, n, chunk_size)]
    edges: list[tuple[bytes, bytes, int]] = []
    extra_tags: list[bytes] = []
    with get_mp_context().Pool(nworkers) as pool:
        for chunk_edges, chunk_extras in pool.starmap(
            collect_meta_face_edges,
            [(chunk, valid_face_tags) for chunk in chunks],
        ):
            edges.extend(chunk_edges)
            extra_tags.extend(chunk_extras)
    return edges, extra_tags


def meta_node_attrs(poly: Polyhedron, *, neighbor_tags: set[bytes]) -> dict[str, Any]:
    """Meta-graph node attributes. ``shis`` comes from :func:`cubical_cell_shis`.

    Node ``shis`` use cubical flip-neighbor SHIs, never the propagated LP-derived
    ``poly._shis`` cache -- LP facets can be a strict subset that would silently
    under-report meta-graph adjacency. For contracted 1-cells, crossings where the
    corresponding 0-face endpoint is geometrically infeasible are filtered out, matching
    the behavior of :func:`_contracted_shis_for_poly`.
    """
    if poly.dim == 0:
        finite: bool | None = True
    elif poly._finite_computed:
        finite = poly._finite
    else:
        finite = poly.finite
    ss_arr = np.asarray(poly.ss_np)
    crossings = list(ss_nonzero_indices(ss_arr))
    node_shis = cubical_cell_shis(ss_arr, neighbor_tags=neighbor_tags)
    # 1-cells: filter out crossings whose 0-face endpoint is geometrically infeasible.
    if int(poly.dim) == 1 and poly.halfspaces is not None:
        node_shis = [s for s in node_shis if poly.is_shi_face_feasible(int(s))]
    return {
        "poly": poly,
        "dim": int(poly.dim),
        "ss": ss_arr,
        "finite": finite,
        "crossings": crossings,
        "shis": node_shis,
    }


# ---------------------------------------------------------------------------
# Boundedness classification (combinatorial, no LP)
# ---------------------------------------------------------------------------


def geometric_infeasible_one_cells(
    by_dim: Mapping[int, Iterable[Polyhedron]],
    edges_by_dim: dict[int, tuple[list[tuple[bytes, bytes, int]], list[bytes]]],
) -> set[bytes]:
    """Return 1-cell tags that are geometrically empty (no Chebyshev center).

    Only 1-cells with no combinatorial 0-faces need a geometric check; segments
    and rays are decided from face incidence alone.
    """
    if 1 not in by_dim:
        return set()

    zero_face_tags = {p.tag for p in by_dim.get(0, ())}
    zero_faces_by_coface: dict[bytes, set[bytes]] = defaultdict(set)
    if 1 in edges_by_dim:
        for coface_tag, face_tag_, _ in edges_by_dim[1][0]:
            if face_tag_ in zero_face_tags:
                zero_faces_by_coface[coface_tag].add(face_tag_)

    infeasible: set[bytes] = set()
    for p in by_dim[1]:
        if len(zero_faces_by_coface.get(p.tag, ())) > 0:
            continue
        # Use compute-only Chebyshev; ``poly.finite`` would cache ``_finite`` and
        # skip combinatorial classification in :func:`classify_one_cells_finite_from_face_edges`.
        try:
            _center, inradius = p.get_center_inradius()
        except ValueError as exc:
            # Borderline phantom 1-cells can trip the Chebyshev LP with a tiny
            # negative radius (e.g. "Inradius -8e-07"). Treat those exactly as
            # geometrically infeasible so meta-graph construction can continue.
            if str(exc).startswith("Inradius "):
                infeasible.add(p.tag)
                continue
            raise
        if _center is None and inradius is None:
            infeasible.add(p.tag)
    return infeasible


def classify_one_cells_finite_from_face_edges(
    by_dim: Mapping[int, Iterable[Polyhedron]],
    edges_by_dim: dict[int, tuple[list[tuple[bytes, bytes, int]], list[bytes]]],
    *,
    geometric_infeasible: set[bytes] | None = None,
) -> tuple[int, set[bytes]]:
    """Classify ``_finite`` on 1-cells from codimension-one 0-face incidence.

    Uses face edges in ``edges_by_dim[1]`` (sign-sequence zeros), not ``_shis``.
    1-cells with no combinatorial 0-faces are checked geometrically: empty
    phantoms get ``_finite is None`` so truncation does not duplicate them.

    Returns ``(n_classified, infeasible_tags)``.
    """
    if 1 not in by_dim:
        return 0, set()

    zero_face_tags: set[bytes] = {p.tag for p in by_dim.get(0, ())}
    zero_faces_by_coface: dict[bytes, set[bytes]] = defaultdict(set)
    if 1 in edges_by_dim:
        for coface_tag, face_tag_, _ in edges_by_dim[1][0]:
            if face_tag_ in zero_face_tags:
                zero_faces_by_coface[coface_tag].add(face_tag_)

    infeasible = geometric_infeasible
    if infeasible is None:
        infeasible = geometric_infeasible_one_cells(by_dim, edges_by_dim)
    n_classified = 0
    for p in by_dim[1]:
        if p._finite_computed:
            continue
        if p.tag in infeasible:
            p._finite = None
            p._finite_computed = True
            n_classified += 1
            continue
        n_zero = len(zero_faces_by_coface.get(p.tag, ()))
        p._finite = n_zero >= 2
        p._finite_computed = True
        n_classified += 1
    return n_classified, infeasible


def classify_finite_ascending(
    by_dim: Mapping[int, Iterable[Polyhedron]],
    lookup: dict[bytes, Polyhedron],
    edges_by_dim: dict[int, tuple[list[tuple[bytes, bytes, int]], list[bytes]]],
) -> int:
    """Classify ``_finite`` for contracted cells by an ascending sweep.

    Starting from 1-dim cells (whose ``_finite`` must already be set via
    :func:`classify_one_cells_finite_from_face_edges`), proceeds dimension by
    dimension from k = 2 upward:

    * k-dim cell is **unbounded** if ANY `(k−1)`-dim face is unbounded.
    * k-dim cell is **bounded** if ALL `(k−1)`-dim faces are bounded (infeasible
      faces are ignored; a cell whose faces are exclusively infeasible is itself
      marked infeasible).

    Because every (k-1)-dim cell is already classified before the k-dim pass
    (induction from the 1-dim base case), this single ascending sweep fully
    classifies all contracted cells without LP.

    Returns the total number of cells newly classified.
    """
    total = 0
    for k in sorted(by_dim.keys()):
        if k <= 1:
            continue  # 1-dim already handled; 0-dim always bounded
        if k not in edges_by_dim:
            continue

        coface_faces: dict[bytes, set[bytes]] = defaultdict(set)
        for coface_tag, face_tag_, _ in edges_by_dim[k][0]:
            coface_faces[coface_tag].add(face_tag_)

        for coface_tag, face_tags in coface_faces.items():
            coface = lookup.get(coface_tag)
            if coface is None or coface._finite_computed:
                continue

            n_bounded = 0
            n_unbounded = 0
            n_infeasible = 0
            unknown = False
            for ft in face_tags:
                face = lookup.get(ft)
                if face is None or not face._finite_computed:
                    unknown = True
                    continue
                if face._finite is False:
                    n_unbounded += 1
                elif face._finite is None:
                    n_infeasible += 1
                else:
                    n_bounded += 1

            if n_unbounded > 0:
                coface._finite = False
                coface._finite_computed = True
                total += 1
            elif n_bounded > 0 and not unknown:
                coface._finite = True
                coface._finite_computed = True
                total += 1
            elif n_infeasible > 0 and n_bounded == 0 and n_unbounded == 0 and not unknown:
                coface._finite = None
                coface._finite_computed = True
                total += 1

    return total


def classify_finite_combinatorial(
    by_dim: Mapping[int, Iterable[Polyhedron]],
    lookup: dict[bytes, Polyhedron],
    edges_by_dim: dict[int, tuple[list[tuple[bytes, bytes, int]], list[bytes]]],
) -> int:
    """Run :func:`classify_finite_ascending` to a fixed point.

    A single ascending pass can leave cells pending when some ``(k-1)``-faces
    were classified in the same pass (ordering dependency).  Repeat until no
    progress.
    """
    total = 0
    while True:
        n = classify_finite_ascending(by_dim, lookup, edges_by_dim)
        if n == 0:
            break
        total += n
    return total


def classify_finite_lp_fallback(polys: Iterable[Polyhedron]) -> int:
    """Classify any remaining cells via Chebyshev LP (``poly.finite``)."""
    n = 0
    for poly in polys:
        if poly._finite_computed:
            continue
        _ = poly.finite
        n += 1
    return n


def format_pending_finite_polys(polys: Iterable[Polyhedron], *, limit: int = 10) -> str:
    """Short diagnostic string for cells still missing ``_finite``."""
    pending = [p for p in polys if not p._finite_computed]
    if not pending:
        return ""
    samples = ", ".join(f"({p.tag.hex()!r}, dim={int(p.dim)})" for p in pending[:limit])
    suffix = f" ... +{len(pending) - limit} more" if len(pending) > limit else ""
    return f" Pending: {samples}{suffix}."


def propagate_infeasible_exclusion(
    infeasible_tags: set[bytes],
    edges_by_dim: dict[int, tuple[list[tuple[bytes, bytes, int]], list[bytes]]],
) -> set[bytes]:
    """Expand infeasible tags upward through face incidences.

      Phantom ``(k-1)``-cells with ``_finite is None`` must not appear in the
      meta-graph chain complex.  Any ``k``-cell that lists such a face as a
    boundary component is also excluded; otherwise dropping only the face edge
      leaves cofaces with open boundaries and breaks ``∂² = 0``.
    """
    excluded = set(infeasible_tags)
    changed = True
    while changed:
        changed = False
        for edges, _ in edges_by_dim.values():
            for coface_tag, face_tag_, _ in edges:
                if face_tag_ in excluded and coface_tag not in excluded:
                    excluded.add(coface_tag)
                    changed = True
    return excluded


def classify_lazy_face_polys(
    face_tags: Iterable[bytes],
    lookup: dict[bytes, Polyhedron],
    edges_by_dim: dict[int, tuple[list[tuple[bytes, bytes, int]], list[bytes]]],
) -> None:
    """Classify boundedness on lazily discovered face polys before meta nodes are added."""
    pending: dict[int, list[Polyhedron]] = defaultdict(list)
    for tag in face_tags:
        poly = lookup.get(tag)
        if poly is None or poly._finite_computed:
            continue
        if int(poly.dim) == 0:
            poly._finite = True
            poly._finite_computed = True
            continue
        pending[int(poly.dim)].append(poly)
    if not pending:
        return
    if 1 in pending:
        classify_one_cells_finite_from_face_edges(pending, edges_by_dim)
    classify_finite_combinatorial(pending, lookup, edges_by_dim)
    classify_finite_lp_fallback(p for polys in pending.values() for p in polys)
