"""Meta-graph construction, truncation, and face-incidence helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

import relucent.config as cfg
from relucent._logging import logger
from relucent.poly import Polyhedron
from relucent.utils import encode_ss, flip_ss_at_shi, get_mp_context

if TYPE_CHECKING:
    from relucent.complex import Complex

# ``shi`` edge attribute for truncation incidences (not a network SHI).
TRUNCATION_META_SHI: int = -1

INFINITY_POINT_META_NODE: tuple[str] = ("infty",)
INFINITY_POINT_META_SHI: int = -2


class CubicalConsistencyError(ValueError):
    """Dual-graph or incidence data violates the cubical face-star convention."""


class CubicalAmbiguityError(CubicalConsistencyError):
    """More than two top-dimensional cells share one codimension-one face tag."""


class NonGenericArrangementError(ValueError):
    """Geometric genericity / transversality is violated (degenerate endpoints or junctions)."""


__all__ = [
    "INFINITY_POINT_META_NODE",
    "INFINITY_POINT_META_SHI",
    "TRUNCATION_META_SHI",
    "CubicalAmbiguityError",
    "CubicalConsistencyError",
    "NonGenericArrangementError",
    "classify_finite",
    "classify_finite_ascending",
    "assign_contracted_shis",
    "collect_meta_face_edges",
    "cubical_cell_shis",
    "dual_edges_top_dim",
    "face_tag",
    "filter_shi_candidates",
    "finite_cells_subgraph",
    "complete_truncated_one_cell_boundaries",
    "flip_tag",
    "META_FACE_PARALLEL_MIN_CELLS",
    "meta_node_attrs",
    "one_point_compactify_meta_graph",
    "parallel_collect_meta_face_edges",
    "propagate_finite_from_coface_edges",
    "ss_nonzero_indices",
    "sync_shis_from_dual_graph",
    "truncate_meta_graph",
    "verify_arrangement_genericity",
    "verify_dual_graph_cubical",
    "verify_meta_graph_incidence",
]

META_FACE_PARALLEL_MIN_CELLS = 64


# ---------------------------------------------------------------------------
# Cubical incidence engine — one SS+lookup source, derived views
# ---------------------------------------------------------------------------
#
# Primitives (:func:`ss_nonzero_indices`, :func:`face_tag`, :func:`flip_tag`) are purely
# combinatorial.  Consumers take *derived views*:
#
# - **Dual graph** (:func:`dual_edges_top_dim`): group top cells by shared face tag.
# - **Meta-graph face edges** (:func:`collect_meta_face_edges`):
#   zero each ``ss_i != 0``; keep edge iff face tag exists in lookup.
# - **Node metadata** (:func:`meta_node_attrs`): derived flip-neighbor SHIs.
#
# **Contract path** (``contract`` / ``get_boundary_complex`` only):
# With :data:`~relucent.config.CUBICAL_DUAL_GRAPH`, :func:`assign_contracted_shis`
# assigns :func:`cubical_cell_shis`.  Legacy mode seeds coface-intersection SHIs
# in :meth:`~relucent.complex.Complex._codim_one_face_kwargs`, then flip-filters.
# Do not reuse that propagation in :meth:`~relucent.complex.Complex.get_meta_graph`.
#
# **Meta-graph path** (stateless):
#
# 1. **Face edges** — :func:`ss_nonzero_indices` + lookup (complete for ``∂² = 0``).
# 2. **Node ``shis``** — :func:`cubical_cell_shis` (flip neighbors in the dim slice).
# 3. **Boundedness** — :func:`classify_one_cells_finite_from_face_edges` and
#    :func:`classify_finite_ascending` from face edges only.
#
# See ``tests/test_meta_graph_shi_regression.py`` and ``negative-betti-meta-graph-bug.md``.


def ss_nonzero_indices(ss: np.ndarray) -> tuple[int, ...]:
    """All sign-sequence indices with ``ss_i != 0``.

    Used when assembling meta-graph face **edges** (role 2): try zeroing each
    nonzero entry. Also drives dual-graph flip-neighbor adjacency for
    top-dimensional cells.

    Not used when building the contraction chain
    (:meth:`Complex._codim_one_face_kwargs`). Propagated ``_shis`` can be a
    strict subset after coface intersection; using only ``_shis`` for edge
    discovery omits valid faces and breaks ``∂² = 0``.
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


def dual_edges_top_dim(
    cells: Iterable[Polyhedron],
    neighbor_tags: set[bytes],
    *,
    top_dim: int | None = None,
) -> tuple[list[tuple[Polyhedron, Polyhedron, int]], list[tuple[Polyhedron, int]]]:
    """Cubical dual edges for top-dimensional cells.

    For ``top_dim == 1``, pairs cells sharing a combinatorial 0-face tag.  For
    ``top_dim >= 2``, pairs cells that are flip neighbors across an existing
    ``ss_i != 0`` crossing (combinatorial cubical adjacency).
    """
    cell_list = list(cells)
    if not cell_list:
        return [], []
    if top_dim is None:
        top_dim = int(cell_list[0].dim)
    if int(top_dim) == 1:
        return _dual_edges_one_dim(cell_list), []
    return _dual_edges_flip_neighbors(cell_list, neighbor_tags), []


def _dual_edges_flip_neighbors(
    cell_list: list[Polyhedron],
    neighbor_tags: set[bytes],
) -> list[tuple[Polyhedron, Polyhedron, int]]:
    tag_to_poly = {p.tag: p for p in cell_list}
    edges: list[tuple[Polyhedron, Polyhedron, int]] = []
    seen: set[tuple[bytes, bytes]] = set()
    for u in cell_list:
        ss = np.asarray(u.ss_np)
        for shi in ss_nonzero_indices(ss):
            shi_i = int(shi)
            vt = flip_tag(ss, shi_i)
            if vt not in neighbor_tags:
                continue
            v = tag_to_poly.get(vt)
            if v is None or u.tag == v.tag:
                continue
            lo, hi = (u.tag, v.tag) if u.tag < v.tag else (v.tag, u.tag)
            if (lo, hi) in seen:
                continue
            seen.add((lo, hi))
            edges.append((u, v, shi_i))
    return edges


def _dual_edges_one_dim(
    cell_list: list[Polyhedron],
) -> list[tuple[Polyhedron, Polyhedron, int]]:
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


def sync_shis_from_dual_graph(graph: nx.Graph[Any]) -> None:
    """Populate ``poly._shis`` on nodes from dual-graph edge ``shi`` attributes."""
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


def verify_dual_graph_cubical(
    cells: Iterable[Polyhedron],
    graph: nx.Graph[Any],
    *,
    top_dim: int,
) -> None:
    """Layer-1 check: each dual edge's ``shi`` zeroes to a shared face tag."""
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


def _quantize_point_for_genericity(pt: np.ndarray) -> tuple[int, ...]:
    vec = np.asarray(pt, dtype=np.float64).ravel()
    scale = max(1.0, float(np.max(np.abs(vec))))
    step = float(cfg.TOL_INTERIOR_VERIFY) * scale
    return tuple(int(round(float(x) / step)) for x in vec)


def _one_cell_endpoint_map(poly: Polyhedron) -> dict[bytes, tuple[int, np.ndarray]]:
    """Map combinatorial 0-face tags to ``(witness shi, geometric point)`` for a 1-cell."""
    if int(poly.dim) != 1:
        return {}
    out: dict[bytes, tuple[int, np.ndarray]] = {}
    ss = np.asarray(poly.ss_np)
    hs = np.asarray(poly.halfspaces_np)
    for shi in ss_nonzero_indices(ss):
        shi_i = int(shi)
        active = np.array(list(poly.zero_indices) + [shi_i], dtype=np.intp)
        pt = poly._halfspace_point(hs, active)
        if pt is None:
            continue
        out[face_tag(ss, shi_i)] = (shi_i, np.asarray(pt, dtype=np.float64).reshape(-1))
    return out


## TODO: Move to topology.py
def verify_arrangement_genericity(polys: Iterable[Polyhedron]) -> None:
    """Layer-2 geometric check for 1-dimensional arrangements (transversality)."""
    cells = [p for p in polys if int(p.dim) == 1]
    if not cells:
        return

    endpoint_maps = {p.tag: _one_cell_endpoint_map(p) for p in cells}

    for poly in cells:
        ep_map = endpoint_maps[poly.tag]
        if not ep_map:
            continue
        geom_buckets: dict[tuple[int, ...], list[bytes]] = defaultdict(list)
        for tag, (_shi, pt) in ep_map.items():
            geom_buckets[_quantize_point_for_genericity(pt)].append(tag)
        if len(geom_buckets) < len(ep_map):
            raise NonGenericArrangementError(
                "1-cell "
                + f"{poly!r} has {len(ep_map)} combinatorial endpoint(s) but only "
                + f"{len(geom_buckets)} distinct geometric location(s); hyperplanes likely "
                + "concur at a vertex. Try a later training epoch or a non-degenerate "
                + "initialization."
            )

    for i, left in enumerate(cells):
        left_map = endpoint_maps[left.tag]
        left_geom = {_quantize_point_for_genericity(pt) for _shi, pt in left_map.values()}
        left_tags = set(left_map)
        for right in cells[i + 1 :]:
            right_map = endpoint_maps[right.tag]
            right_geom = {_quantize_point_for_genericity(pt) for _shi, pt in right_map.values()}
            right_tags = set(right_map)
            shared_geom = left_geom & right_geom
            shared_tags = left_tags & right_tags
            if not shared_geom:
                continue
            if len(shared_tags) == 0:
                raise NonGenericArrangementError(
                    "1-cells "
                    + f"{left!r} and {right!r} share a geometric endpoint but no combinatorial "
                    + "0-face tag (non-transversal junction). Try a later training epoch or "
                    + "more exploration."
                )
            if len(shared_tags) > 1:
                raise NonGenericArrangementError(
                    "1-cells "
                    + f"{left!r} and {right!r} share {len(shared_tags)} combinatorial 0-face "
                    + "tags; adjacency is ambiguous."
                )


def cubical_cell_shis(
    ss: np.ndarray,
    *,
    neighbor_tags: set[bytes],
) -> list[int]:
    """Flip-neighbor SHIs: ``ss_nonzero_indices`` with same-dimension slice filter."""
    return filter_shi_candidates(ss, ss_nonzero_indices(ss), neighbor_tags=neighbor_tags)


def meta_node_attrs(
    poly: Polyhedron,
    *,
    neighbor_tags: set[bytes],
) -> dict[str, Any]:
    if poly.dim == 0:
        finite: bool | None = True
    elif poly._finite_computed:
        finite = poly._finite
    else:
        finite = poly.finite
    ss_arr = np.asarray(poly.ss_np)
    crossings = list(ss_nonzero_indices(ss_arr))
    flip_shis = cubical_cell_shis(ss_arr, neighbor_tags=neighbor_tags)
    if int(poly.dim) == 1 and poly.halfspaces is not None:
        flip_shis = [s for s in flip_shis if poly.is_shi_face_feasible(int(s))]
    return {
        "poly": poly,
        "dim": int(poly.dim),
        "ss": ss_arr,
        "finite": finite,
        "crossings": crossings,
        "shis": flip_shis,
    }


def collect_meta_face_edges(
    cells: list[tuple[bytes, np.ndarray, tuple[int, ...]]],
    valid_face_tags: set[bytes],
) -> tuple[list[tuple[bytes, bytes, int]], list[bytes]]:
    """Return face edges (src, dst, shi) for one chunk of k-cells (role 2).

    The ``shis`` tuple on each cell should come from
    :func:`ss_nonzero_indices`, not from propagated ``poly._shis``.
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
        for coface_tag, face_tag, _ in edges_by_dim[1][0]:
            if face_tag in zero_face_tags:
                zero_faces_by_coface[coface_tag].add(face_tag)

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
        if n_zero >= 2:
            p._finite = True
        else:
            p._finite = False
        p._finite_computed = True
        n_classified += 1
    return n_classified, infeasible


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
        for coface_tag, face_tag, _ in edges_by_dim[1][0]:
            if face_tag in zero_face_tags:
                zero_faces_by_coface[coface_tag].add(face_tag)

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

        # Build coface_tag → set of face_tags at this level.
        coface_faces: dict[bytes, set[bytes]] = defaultdict(set)
        for coface_tag, face_tag, _ in edges_by_dim[k][0]:
            coface_faces[coface_tag].add(face_tag)

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


def propagate_finite_from_coface_edges(
    lookup: dict[bytes, Polyhedron],
    edges: Iterable[tuple[bytes, bytes, int]],
) -> None:
    """Mark faces as bounded when any codimension-1 coface is already known bounded."""
    for u, v, _ in edges:
        face = lookup.get(v)
        if face is None or face._finite_computed:
            continue
        coface = lookup.get(u)
        if coface is not None and (coface.dim == 0 or (coface._finite_computed and coface._finite is True)):
            face._finite = True
            face._finite_computed = True


def classify_finite(poly: Polyhedron, env: Any) -> None:
    """Resolve ``poly.finite`` via Chebyshev, without caching center/inradius.

    No-ops when boundedness is already known (``_finite_computed`` is set or
    ``dim == 0``). 0-cells are handled by :meth:`~relucent.poly.Polyhedron._apply_zero_cell_finite_hint`
    at construction, but this is a safe fallback.
    """
    if poly._finite_computed or poly.dim == 0:
        return
    center, inradius = poly.get_center_inradius(env=env)
    if center is not None:
        poly._finite = True
    elif inradius is None:
        poly._finite = None
    elif inradius == float("inf"):
        poly._finite = False
    else:
        raise ValueError(f"Unexpected Chebyshev result (center={center!r}, inradius={inradius!r})")
    poly._finite_computed = True


def filter_shi_candidates(
    ss: np.ndarray,
    candidates: Iterable[int],
    *,
    neighbor_tags: set[bytes],
) -> list[int]:
    """Keep SHIs whose sign-flip neighbor exists among ``neighbor_tags``.

    A candidate ``shi`` is retained only when flipping ``ss[shi]`` yields the tag
    of another cell at the same dimension in the contracted complex.
    """
    row = np.asarray(ss, dtype=np.int8).ravel()
    kept: list[int] = []
    for shi in candidates:
        shi_i = int(shi)
        if shi_i >= row.shape[0] or row[shi_i] == 0:
            continue
        if encode_ss(flip_ss_at_shi(row, shi_i)) in neighbor_tags:
            kept.append(shi_i)
    return sorted(kept)


def _symmetrize_top_cell_flip_shis(cplx: Complex) -> int:
    """Drop SHIs that are not listed on both endpoints of a same-dimension flip edge."""
    if len(cplx) == 0:
        return 0
    top_dim = max(int(p.dim) for p in cplx)
    tag2poly = {p.tag: p for p in cplx}
    n_changed = 0
    for poly in cplx:
        if int(poly.dim) != top_dim or not poly._shis:
            continue
        ss = np.asarray(poly.ss_np, dtype=np.int8).ravel()
        kept: list[int] = []
        for shi in poly._shis:
            shi_i = int(shi)
            if shi_i >= ss.shape[0] or int(ss[shi_i]) == 0:
                continue
            neighbor = tag2poly.get(encode_ss(flip_ss_at_shi(ss, shi_i)))
            if neighbor is None or shi_i not in (neighbor._shis or []):
                continue
            kept.append(shi_i)
        assigned = sorted(set(kept))
        if assigned != list(poly._shis):
            poly._shis = assigned
            poly._shis_strict = False
            n_changed += 1
    return n_changed


def assign_contracted_shis(cplx: Complex) -> int:
    """Finalize ``_shis`` on a contracted slice after :meth:`~relucent.complex.Complex.contract`.

    Used only on the **contract path** (``contract``, ``get_boundary_complex``).
    When :data:`~relucent.config.CUBICAL_DUAL_GRAPH` is True, assigns
    :func:`cubical_cell_shis` (SS crossings filtered to same-dimension flip
    neighbors).  Otherwise filters coface-intersected candidates from
    :meth:`~relucent.complex.Complex._codim_one_face_kwargs`.  Meta-graph node
    labels use :func:`cubical_cell_shis` as well — do not call this from
    :meth:`~relucent.complex.Complex.get_meta_graph`.

    Returns the number of cells whose ``_shis`` list was changed.
    """
    if len(cplx) == 0:
        return 0
    neighbor_tags = {p.tag for p in cplx}
    n_changed = 0
    for poly in cplx:
        if cfg.CUBICAL_DUAL_GRAPH:
            assigned = cubical_cell_shis(poly.ss_np, neighbor_tags=neighbor_tags)
            if int(poly.dim) == 1 and poly.halfspaces is not None:
                assigned = [s for s in assigned if poly.is_shi_face_feasible(int(s))]
        elif poly._shis is None:
            continue
        else:
            assigned = filter_shi_candidates(poly.ss_np, poly._shis, neighbor_tags=neighbor_tags)
        if assigned != poly._shis:
            poly._shis = assigned
            poly._shis_strict = False
            n_changed += 1
    if any(int(p.dim) == 1 for p in cplx):
        # ``is_shi_face_feasible`` is evaluated per 1-cell with coface halfspaces;
        # adjacent cells can disagree, so enforce mutual flip-SHI listing.
        n_changed += _symmetrize_top_cell_flip_shis(cplx)
    return n_changed


def verify_meta_graph_incidence(
    meta: nx.MultiDiGraph[Any],
    by_dim: Mapping[int, Iterable[Polyhedron]],
    lookup: dict[bytes, Polyhedron],
) -> None:
    """Check assembled meta-graph matches the stateless incidence engine.

    - Face edges equal ``ss_nonzero_indices`` zeroings kept by lookup.
    - Node ``shis`` equal :func:`cubical_cell_shis` on each dimension slice.
    - ``finite`` on chain cells matches combinatorial classification from face edges.
    """
    valid_face_tags = set(lookup.keys())
    for k, c_k in by_dim.items():
        if int(k) <= 0:
            continue
        expected_edges: set[tuple[bytes, bytes, int]] = set()
        meta_nodes = set(meta.nodes)
        for poly in c_k:
            if poly.tag not in meta_nodes:
                continue
            ss_arr = np.asarray(poly.ss_np)
            for shi in ss_nonzero_indices(ss_arr):
                shi_i = int(shi)
                ft = face_tag(ss_arr, shi_i)
                if ft in valid_face_tags and ft in meta_nodes:
                    expected_edges.add((poly.tag, ft, shi_i))
        actual_edges: set[tuple[bytes, bytes, int]] = set()
        for u, v, data in meta.edges(data=True):
            if int(meta.nodes[u].get("dim", -1)) != int(k):
                continue
            actual_edges.add((u, v, int(data["shi"])))
        if expected_edges != actual_edges:
            missing = len(expected_edges - actual_edges)
            extra = len(actual_edges - expected_edges)
            raise AssertionError(
                f"get_meta_graph verify: dim-{int(k)} face edges mismatch " + f"(missing={missing}, extra={extra})"
            )

    for c_k in by_dim.values():
        neighbor_tags = {p.tag for p in c_k}
        for poly in c_k:
            if poly.tag not in meta.nodes:
                continue
            expected_shis = cubical_cell_shis(poly.ss_np, neighbor_tags=neighbor_tags)
            if int(poly.dim) == 1 and poly.halfspaces is not None:
                expected_shis = [s for s in expected_shis if poly.is_shi_face_feasible(int(s))]
            actual_shis = sorted(int(s) for s in meta.nodes[poly.tag].get("shis", []))
            if actual_shis != expected_shis:
                raise AssertionError(f"get_meta_graph verify: node shis mismatch dim-{int(poly.dim)} tag={poly.tag!r}")

    for c_k in by_dim.values():
        for poly in c_k:
            if poly.tag not in meta.nodes or not poly._finite_computed:
                continue
            node_finite = meta.nodes[poly.tag].get("finite")
            if node_finite != poly._finite:
                raise AssertionError(
                    f"get_meta_graph verify: finite mismatch tag={poly.tag!r} "
                    + f"expected={poly._finite!r} got={node_finite!r}"
                )


def finite_cells_subgraph(meta: nx.MultiDiGraph[Any]) -> nx.MultiDiGraph[Any]:
    """Return the subcomplex induced by nodes with ``finite is True``."""
    finite = [n for n, a in meta.nodes(data=True) if a.get("finite", None) is True]
    return meta.subgraph(finite).copy()


def truncate_meta_graph(meta: nx.MultiDiGraph[Any]) -> None:
    """Augment ``meta`` in place with combinatorial truncation at infinity.

    Every node's ``ss`` gains a trailing ``1`` (strictly inside the truncation halfspace).
    The induced subgraph on nodes with ``finite is False`` (unbounded cells) is duplicated:
    each copy has trailing ``0`` on ``ss``, dimension decremented by one, and node keys
    ``("trunc", tag)``. Face edges among duplicates mirror the induced subgraph; each
    original unbounded node ``n`` gains an edge ``n → ("trunc", n)`` with ``shi`` equal to
    :data:`TRUNCATION_META_SHI`. Duplicates are not created for 0-cells.
    """
    if meta.number_of_nodes() == 0:
        return

    unbounded = {n for n, a in meta.nodes(data=True) if a.get("finite", None) is False}
    ub_faces = meta.subgraph(unbounded).copy()

    def _ss_with_extra_bit(ss: np.ndarray, bit: int) -> np.ndarray:
        a = np.asarray(ss)
        dt = np.int8 if np.issubdtype(a.dtype, np.integer) else a.dtype
        return np.hstack([a, np.full((a.shape[0], 1), bit, dtype=dt)])

    for attrs in meta.nodes.values():
        if (ss0 := attrs.get("ss")) is not None:
            attrs["ss"] = _ss_with_extra_bit(np.asarray(ss0), 1)

    dup_keys: set[Any] = set()
    for orig in unbounded:
        oa = meta.nodes[orig]
        k = int(oa.get("dim", -1))
        ss_in = oa.get("ss")
        if k <= 0 or ss_in is None:
            continue
        dup = ("trunc", orig)
        ss_on_cut = np.asarray(ss_in).copy()
        ss_on_cut[..., -1] = 0
        dup_keys.add(dup)
        meta.add_node(
            dup,
            poly=oa.get("poly"),
            dim=k - 1,
            ss=ss_on_cut,
            finite=True,
            shis=list(oa.get("shis", [])),
            truncation_duplicate=True,
        )
        meta.add_edge(orig, dup, shi=TRUNCATION_META_SHI)

    for u, v, ed in ub_faces.edges(data=True):
        tu, tv = ("trunc", u), ("trunc", v)
        if tu in dup_keys and tv in dup_keys:
            meta.add_edge(tu, tv, **dict(ed))

    complete_truncated_one_cell_boundaries(meta)


def complete_truncated_one_cell_boundaries(meta: nx.MultiDiGraph[Any]) -> tuple[int, int]:
    """Materialize missing 0-faces so every 1-cell has two 0-cell endpoints in ``meta``.

    After :func:`truncate_meta_graph`, unbounded 1-cells often have only the truncation
    duplicate as a boundary vertex.  Combinatorially close each open end by linking to an
    existing 0-face from sign-sequence zeroing when present, otherwise a per-cell endpoint.
    """
    nodes_added = 0
    edges_added = 0

    def _zero_neighbors(one_cell: object) -> set[object]:
        return {v for _u, v, _ in meta.out_edges(one_cell, data=True) if int(meta.nodes[v].get("dim", -1)) == 0}

    def _trunc_partner(one_cell: object) -> object | None:
        key = ("trunc", one_cell)
        return key if key in meta else None

    for one_cell, attrs in list(meta.nodes(data=True)):
        if int(attrs.get("dim", -1)) != 1:
            continue
        ss_in = attrs.get("ss")
        if ss_in is None:
            continue
        ss_arr = np.asarray(ss_in, dtype=np.int8)
        zero_faces = _zero_neighbors(one_cell)

        # Link to interior 0-cells already present in the meta-graph.
        for shi in ss_nonzero_indices(ss_arr):
            if len(zero_faces) >= 2:
                break
            ft = face_tag(ss_arr, int(shi))
            if ft not in meta or int(meta.nodes[ft].get("dim", -1)) != 0:
                continue
            if ft in zero_faces:
                continue
            meta.add_edge(one_cell, ft, shi=int(shi))
            zero_faces.add(ft)
            edges_added += 1

        # Match a face zeroing to the truncation duplicate on the cut (tuple node key).
        trunc = _trunc_partner(one_cell)
        if trunc is not None and trunc not in zero_faces:
            for shi in ss_nonzero_indices(ss_arr):
                row = ss_arr.reshape(-1).copy()
                row[int(shi)] = 0
                trunc_ss = np.asarray(meta.nodes[trunc].get("ss"), dtype=np.int8)
                if trunc_ss.shape == row.reshape(ss_arr.shape).shape and np.array_equal(trunc_ss, row.reshape(ss_arr.shape)):
                    meta.add_edge(one_cell, trunc, shi=int(shi))
                    zero_faces.add(trunc)
                    edges_added += 1
                    break

        # One materialized endpoint per still-open end (unique per 1-cell).
        while len(zero_faces) < 2:
            mat_key: tuple[str, object, int] = ("mat", one_cell, len(zero_faces))
            shi = ss_nonzero_indices(ss_arr)[0] if ss_nonzero_indices(ss_arr) else 0
            row = ss_arr.reshape(-1).copy()
            row[int(shi)] = 0
            if mat_key not in meta:
                meta.add_node(
                    mat_key,
                    poly=None,
                    dim=0,
                    ss=row.reshape(ss_arr.shape),
                    finite=True,
                    shis=[],
                    materialized_face=True,
                )
                nodes_added += 1
            if mat_key not in zero_faces:
                meta.add_edge(one_cell, mat_key, shi=int(shi))
                zero_faces.add(mat_key)
                edges_added += 1
            else:
                break

    return nodes_added, edges_added


def one_point_compactify_meta_graph(meta: nx.MultiDiGraph[Any]) -> bool:
    """Augment ``meta`` in place with a single point-at-infinity 0-cell.

    Mirrors ``canonicalpoly2.0/polyhedra/topology.get_coboundary_matrices``: each
    1-cell whose boundary consists of a single 0-cell (an unbounded end) gains a
    second incidence to one new 0-cell representing infinity.  Returns whether the
    infinity node was added.
    """
    if meta.number_of_nodes() == 0:
        return False

    zero_cells = {n for n, a in meta.nodes(data=True) if int(a.get("dim", -1)) == 0}
    one_cells = [n for n, a in meta.nodes(data=True) if int(a.get("dim", -1)) == 1]
    if not one_cells or not zero_cells:
        return False

    needing_infinity: list[Any] = []
    for u in one_cells:
        n_zero = sum(1 for _u, v, _ in meta.out_edges(u, data=True) if v in zero_cells)
        if n_zero == 1:
            needing_infinity.append(u)
    if not needing_infinity:
        return False

    if INFINITY_POINT_META_NODE not in meta:
        meta.add_node(
            INFINITY_POINT_META_NODE,
            dim=0,
            finite=True,
            infinity_point=True,
            ss=None,
            shis=[],
        )
    for u in needing_infinity:
        meta.add_edge(u, INFINITY_POINT_META_NODE, shi=INFINITY_POINT_META_SHI)
    return True
