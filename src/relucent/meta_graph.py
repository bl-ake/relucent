"""Meta-graph construction, truncation, and face-incidence helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

import relucent.config as cfg
from relucent._logging import logger
from relucent.poly import Polyhedron
from relucent.utils import encode_ss, flip_ss_at_shi, get_env, get_mp_context, get_thread_env

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
    "collect_meta_face_edges",
    "compute_contracted_shis_top_down",
    "contracted_shis_from_coface_intersection",
    "dual_edges_top_dim",
    "face_tag",
    "filter_complex_shis_by_flip_neighbor",
    "filter_shi_candidates",
    "finite_cells_subgraph",
    "flip_tag",
    "known_bounded",
    "META_FACE_PARALLEL_MIN_CELLS",
    "meta_face_edges",
    "meta_node_attrs",
    "meta_node_shis_for_meta_node",
    "one_point_compactify_meta_graph",
    "parallel_collect_meta_face_edges",
    "precompute_finite",
    "propagate_finite_from_coface_edges",
    "propagate_unbounded_from_face_edges",
    "ss_nonzero_indices",
    "sync_shis_from_dual_graph",
    "truncate_meta_graph",
    "verify_arrangement_genericity",
    "verify_dual_graph_cubical",
]

META_FACE_PARALLEL_MIN_CELLS = 64


def known_bounded(poly: Polyhedron) -> bool:
    """True when boundedness is already known (no Chebyshev LP needed)."""
    if poly.dim == 0:
        return True
    return bool(poly._finite_computed and poly._finite is True)


# ---------------------------------------------------------------------------
# Cubical incidence engine — one SS+lookup source, derived views
# ---------------------------------------------------------------------------
#
# Primitives (:func:`ss_nonzero_indices`, :func:`face_tag`, :func:`flip_tag`) are purely
# combinatorial.  Consumers take *derived views*:
#
# - **Dual graph** (:func:`dual_edges_top_dim`): group top cells by shared face tag.
# - **Meta-graph face edges** (:func:`meta_face_edges` / :func:`collect_meta_face_edges`):
#   zero each ``ss_i != 0``; keep edge iff face tag exists in lookup.
# - **Contraction metadata** (:func:`contracted_shis_from_coface_intersection`):
#   coface SHI intersection minus crossing, flip-neighbor filtered (conservative).
# - **Node metadata** (:func:`meta_node_shis_for_meta_node`): propagated ``_shis``.
#
# The historical "three roles" tension is expressed as filter stacks on the same
# cubical primitives, not unrelated SHI lists.
#
# 1. **Contraction face kwargs** (:meth:`Complex._codim_one_face_kwargs`)
#    - Rule: coface SHI intersection minus crossing, then flip-neighbor filter.
#    - Conservative: expanding beyond coface intersection creates phantom cells.
#
# 2. **Meta-graph face edges** (:meth:`Complex.get_meta_graph`)
#    - Rule: :func:`ss_nonzero_indices` + lookup existence (complete for ``∂² = 0``).
#
# 3. **Meta-graph node metadata** (:func:`meta_node_shis_for_meta_node`)
#    - Propagated ``_shis`` for boundedness heuristics; strict subset of crossings.
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


def meta_face_edges(
    cells: list[tuple[bytes, np.ndarray, tuple[int, ...]]],
    valid_face_tags: set[bytes],
) -> tuple[list[tuple[bytes, bytes, int]], list[bytes]]:
    """Assemble meta-graph face edges from SS crossings (delegates to collector)."""
    return collect_meta_face_edges(cells, valid_face_tags)


def contracted_shis_from_coface_intersection(
    ss: np.ndarray,
    coface_shis_sets: Iterable[set[int]],
    crossing_shi: int,
    *,
    neighbor_tags: set[bytes],
) -> list[int]:
    """Conservative contracted-cell SHIs: ∩(SHI(coface) \\ {crossing}), flip-filtered."""
    sets = [set(s) - {int(crossing_shi)} for s in coface_shis_sets]
    if not sets:
        return []
    inferred = set.intersection(*sets) if len(sets) > 1 else sets[0].copy()
    return filter_shi_candidates(ss, inferred, neighbor_tags=neighbor_tags)


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
            poly._shis = sorted(set(shis))


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


def meta_node_shis_for_meta_node(poly: Polyhedron) -> list[int]:
    """SHI list stored on meta-graph **nodes** (role 3: propagated metadata).

    Uses ``poly._shis`` when set (from search or :meth:`Complex.contract`).
    Falls back to :func:`ss_nonzero_indices` only when ``_shis`` is
    missing.  This is **not** the rule for meta-graph face **edges**; see the
    module comment above role 2.
    """
    shis = poly._shis
    if shis is not None:
        return sorted(int(s) for s in shis)
    return list(ss_nonzero_indices(np.asarray(poly.ss_np)))


def meta_node_attrs(poly: Polyhedron) -> dict[str, Any]:
    if poly.dim == 0:
        finite: bool | None = True
    elif poly._finite_computed:
        finite = poly._finite
    else:
        finite = poly.finite
    return {
        "poly": poly,
        "dim": int(poly.dim),
        "ss": np.asarray(poly.ss_np),
        "finite": finite,
        "shis": meta_node_shis_for_meta_node(poly),
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


def classify_finite_ascending(
    by_dim: dict[int, Complex],
    lookup: dict[bytes, Polyhedron],
    edges_by_dim: dict[int, tuple[list[tuple[bytes, bytes, int]], list[bytes]]],
) -> int:
    """Classify ``_finite`` for contracted cells by an ascending sweep.

    Starting from 1-dim cells (whose ``_finite`` must already be set via the
    SHI-count heuristic), proceeds dimension by dimension from k = 2 upward:

    * k-dim cell is **unbounded** if ANY (k-1)-dim face is unbounded.
    * k-dim cell is **bounded** if ALL (k-1)-dim faces are bounded.

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

            all_bounded = True
            any_unbounded = False
            for ft in face_tags:
                face = lookup.get(ft)
                if face is not None and face._finite_computed:
                    if face._finite is False:
                        any_unbounded = True
                        break
                    if face._finite is not True:
                        all_bounded = False
                else:
                    all_bounded = False  # face unknown → can't conclude bounded

            if any_unbounded:
                coface._finite = False
                coface._finite_computed = True
                total += 1
            elif all_bounded:
                coface._finite = True
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
        if coface is not None and known_bounded(coface):
            face._finite = True
            face._finite_computed = True


def propagate_unbounded_from_face_edges(
    lookup: dict[bytes, Polyhedron],
    edges: Iterable[tuple[bytes, bytes, int]],
) -> None:
    """Mark cofaces as unbounded when any codimension-1 face is already known unbounded."""
    for u, v, _ in edges:
        coface = lookup.get(u)
        if coface is None or coface._finite_computed:
            continue
        face = lookup.get(v)
        if face is not None and face._finite_computed and face._finite is False:
            coface._finite = False
            coface._finite_computed = True


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


def precompute_finite(polys: list[Polyhedron], nworkers: int) -> None:
    """Classify ``poly.finite`` for a list of polyhedra.

    Skips polys with ``_finite_computed`` already set (including ``finite=True`` hints
    from coface propagation or construction). Gurobi releases the GIL during LP solves,
    so a :class:`~concurrent.futures.ThreadPoolExecutor` gives genuine parallelism
    without pickling overhead; each thread uses its own
    :func:`~relucent.utils.get_thread_env`.
    """
    pending = [p for p in polys if not p._finite_computed]
    if not pending:
        return

    if nworkers <= 1:
        env = get_env()
        for poly in pending:
            classify_finite(poly, env)
        return

    def _worker(poly: Polyhedron) -> None:
        classify_finite(poly, get_thread_env())

    with ThreadPoolExecutor(max_workers=nworkers) as executor:
        list(executor.map(_worker, pending))


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


def filter_complex_shis_by_flip_neighbor(cplx: Complex) -> int:
    """Post-process contracted-slice ``_shis`` after :meth:`Complex.contract` (role 1).

    Drops SHIs on each cell with no same-dimension flip neighbour, then applies
    :meth:`~relucent.poly.Polyhedron.is_shi_face_feasible` to remove any SHI whose
    induced face would be geometrically infeasible.  Called at the end of
    :meth:`Complex.contract` and :meth:`Complex.get_boundary_complex`.

    Do not replace with full SS flip-neighbour membership; see the module comment
    above role 1.

    Returns the number of cells whose ``_shis`` list was changed.
    """
    if len(cplx) == 0:
        return 0
    neighbor_tags = {p.tag for p in cplx}
    n_changed = 0
    for poly in cplx:
        if poly._shis is None:
            continue
        filtered = filter_shi_candidates(poly.ss_np, poly._shis, neighbor_tags=neighbor_tags)
        filtered = [s for s in filtered if poly.is_shi_face_feasible(s)]
        if filtered != list(poly._shis):
            poly._shis = filtered
            n_changed += 1
    return n_changed


def compute_contracted_shis_top_down(by_dim: dict[int, Complex]) -> None:
    """Fill missing ``_shis`` on chain cells (role 3).

    The chain complex from :meth:`Complex.contract` already sets ``_shis`` via
    coface intersection and flip-neighbor filtering (role 1).  This pass is a
    safety net for contracted cells that still lack ``_shis``.  It does **not**
    drive meta-graph face-edge discovery; that uses
    :func:`ss_nonzero_indices` (role 2).  Boundedness is classified
    separately via the 1-cell SHI-count rule and :func:`classify_finite_ascending`.

    For each face still missing ``_shis``:

        SHI(face) = filter_flip( ∩{ SHI(coface) \\ {crossing_shi} : coface ⊃ face } )

    In :data:`~relucent.config.CAREFUL_MODE`, an :exc:`AssertionError` is raised
    if any k-dim cell (k > 1) ends up with fewer than k SHIs after propagation,
    which indicates false negatives in the original maximal-cell SHI computation.

    Args:
        by_dim: Mapping from dimension to Complex (chain complex).  Top-dim cells
            must have ``_shis`` already set by the searcher.
    """
    for k in sorted(by_dim.keys(), reverse=True):
        if k <= 0:
            continue
        if k - 1 not in by_dim:
            continue

        # For each (k-1)-dim face: accumulate the SHI intersection over all cofaces.
        face_shis: dict[bytes, set[int]] = {}

        for coface in by_dim[k]:
            if coface._shis is None:
                logger.warning(
                    "get_meta_graph: dim-%d cell is missing _shis (tag=%r); "
                    + "its contracted faces will have incomplete SHI information",
                    k,
                    coface.tag[:8],
                )
                continue

            ss_coface = np.asarray(coface.ss_np, dtype=np.int8).ravel()

            for shi in coface._shis:
                ss_face = ss_coface.copy()
                ss_face[shi] = 0
                ft = face_tag(coface.ss_np, int(shi))

                contrib = set(coface._shis) - {shi}
                if ft not in face_shis:
                    face_shis[ft] = contrib.copy()
                else:
                    face_shis[ft] &= contrib

        # Apply the computed SHIs to the (k-1)-dim cells.
        face_lookup = {p.tag: p for p in by_dim[k - 1]}
        neighbor_tags = set(face_lookup.keys())
        n_set = 0
        for face_tag_key, shis in face_shis.items():
            face = face_lookup.get(face_tag_key)
            if face is None:
                continue
            if face._shis is None:
                face._shis = filter_shi_candidates(face.ss_np, shis, neighbor_tags=neighbor_tags)
                n_set += 1

        if n_set:
            logger.info(
                "get_meta_graph: top-down pass set SHIs for %d dim-%d cells (no LP)",
                n_set,
                k - 1,
            )

        if cfg.CAREFUL_MODE:
            for face in by_dim[k - 1]:
                if face._shis is None:
                    continue
                n_shis = len(face._shis)
                # A k-dim cell needs ≥ k SHIs to be geometrically consistent.
                # 1-dim cells can legitimately have 0 or 1 SHIs (full lines or
                # rays), so the assertion only applies to k > 1.
                if (k - 1) > 1 and n_shis < (k - 1):
                    raise AssertionError(
                        f"get_meta_graph: dim-{k - 1} cell has only {n_shis} SHIs "
                        + f"(expected ≥ {k - 1}). This indicates false negatives in the "
                        + f"maximal-cell SHI computation. Tag: {face.tag!r}"
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
