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

__all__ = [
    "INFINITY_POINT_META_NODE",
    "INFINITY_POINT_META_SHI",
    "TRUNCATION_META_SHI",
    "classify_finite",
    "classify_finite_ascending",
    "collect_meta_face_edges",
    "compute_contracted_shis_top_down",
    "filter_complex_shis_by_flip_neighbor",
    "filter_shi_candidates",
    "finite_cells_subgraph",
    "known_bounded",
    "META_FACE_PARALLEL_MIN_CELLS",
    "meta_node_attrs",
    "meta_node_shis_for_meta_node",
    "one_point_compactify_meta_graph",
    "parallel_collect_meta_face_edges",
    "precompute_finite",
    "propagate_finite_from_coface_edges",
    "propagate_unbounded_from_face_edges",
    "remove_phantom_shis",
    "ss_face_crossing_indices",
    "truncate_meta_graph",
]

META_FACE_PARALLEL_MIN_CELLS = 64


def known_bounded(poly: Polyhedron) -> bool:
    """True when boundedness is already known (no Chebyshev LP needed)."""
    if poly.dim == 0:
        return True
    return bool(poly._finite_computed and poly._finite is True)


# ---------------------------------------------------------------------------
# SHI propagation: three roles (do not conflate)
# ---------------------------------------------------------------------------
#
# All three rules propagate SHIs from higher- to lower-dimensional cells, but
# each serves a different *consumer* with incompatible correctness requirements.
# The requirements are in direct tension: role 1 needs conservative SHIs (too
# many is wrong), role 2 needs complete SHIs (too few is wrong), and role 3
# needs geometrically accurate SHIs (both over- and under-counting are wrong).
# No single rule satisfies all three.
#
# 1. **Contraction / dual graph** (:meth:`Complex.contract`, :meth:`get_dual_graph`)
#    - Rule: coface SHI intersection minus the crossing index
#      (:meth:`Complex._codim_one_face_kwargs`), then drop SHIs with no
#      same-dimension flip neighbor (:func:`filter_complex_shis_by_flip_neighbor`).
#    - Why conservative: :meth:`get_dual_graph` walks ``poly.shis`` to wire
#      adjacency for the *next* contraction level.  A spurious SHI whose flip
#      does not exist in the slice would create a phantom dual edge, pulling in
#      a cell that is not geometrically present, which breaks ``∂² = 0`` in
#      subsequent contractions.  Too-few SHIs are safe here: a missing edge
#      only means a missing adjacency, which is detectable; a phantom cell is
#      silently wrong.  See ``tests/test_meta_graph_shi_regression.py``.
#
# 2. **Meta-graph face edges** (:meth:`Complex.get_meta_graph`)
#    - Rule: try zeroing every ``ss_i != 0`` (:func:`ss_face_crossing_indices`);
#      accept only if the zeroed-SS tag exists in the chain lookup.
#    - Why complete: propagated ``_shis`` (from role 1) can be a strict subset
#      of the true active constraints when a face sits on the boundary of the
#      explored complex and has fewer cofaces than it would in the full tiling.
#      Coface intersection over-shrinks those SHIs, causing valid face incidences
#      to be silently skipped, which breaks ``∂² = 0`` in the meta-graph boundary
#      maps.  The lookup-existence check gates correctness, so trying too many
#      candidate SHIs is harmless.
#
# 3. **Meta-graph node metadata** (:func:`meta_node_shis_for_meta_node`,
#    :func:`compute_contracted_shis_top_down`)
#    - Rule: propagated ``_shis`` (from search, contraction, or the top-down
#      safety net); falls back to :func:`ss_face_crossing_indices` only when
#      ``_shis`` is missing entirely.
#    - Why accurate: the SHI count drives the 1-cell boundedness heuristic
#      (``len(_shis) >= 2`` ⟺ bounded segment, not a ray or full line).
#      Using role 2's broad candidate set would overcount and mis-classify
#      rays/lines as bounded, propagating wrong ``finite`` values upward through
#      :func:`classify_finite_ascending`.  Using role 1's intersection alone is
#      correct for interior cells but would undercount at complex boundaries for
#      the same reason as role 2.  This list is therefore a strict subset of SS
#      crossings by design and does not govern face-edge assembly.
#
# See ``tests/test_meta_graph_shi_regression.py`` for the regression that
# motivates role 1, and ``get_meta_graph`` docstring for the role 2 failure mode.


def meta_node_shis_for_meta_node(poly: Polyhedron) -> list[int]:
    """SHI list stored on meta-graph **nodes** (role 3: propagated metadata).

    Uses ``poly._shis`` when set (from search or :meth:`Complex.contract`).
    Falls back to :func:`ss_face_crossing_indices` only when ``_shis`` is
    missing.  This is **not** the rule for meta-graph face **edges**; see the
    module comment above role 2.
    """
    shis = poly._shis
    if shis is not None:
        return sorted(int(s) for s in shis)
    return list(ss_face_crossing_indices(np.asarray(poly.ss_np)))


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


def ss_face_crossing_indices(ss: np.ndarray) -> tuple[int, ...]:
    """Candidate SHI indices for meta-graph face **edges** (role 2).

    Combinatorial codimension-one crossings: every index with ``ss_i != 0``.
    Used when assembling :meth:`Complex.get_meta_graph` face incidences, **not**
    when building the contraction chain (:meth:`Complex._codim_one_face_kwargs`).
    Propagated ``_shis`` can be a strict subset after coface intersection; using
    only ``_shis`` for edge discovery omits valid faces and breaks ``∂² = 0``.
    """
    row = np.asarray(ss, dtype=np.int8).ravel()
    return tuple(int(i) for i in np.flatnonzero(row != 0))


def collect_meta_face_edges(
    cells: list[tuple[bytes, np.ndarray, tuple[int, ...]]],
    valid_face_tags: set[bytes],
) -> tuple[list[tuple[bytes, bytes, int]], list[bytes]]:
    """Return face edges (src, dst, shi) for one chunk of k-cells (role 2).

    The ``shis`` tuple on each cell should come from
    :func:`ss_face_crossing_indices`, not from propagated ``poly._shis``.
    """
    edges: list[tuple[bytes, bytes, int]] = []
    extra_tags: list[bytes] = []
    for src_tag, ss, shis in cells:
        ss_arr = np.asarray(ss)
        row = ss_arr[0]
        for shi in shis:
            shi_i = int(shi)
            old = row[shi_i]
            row[shi_i] = 0
            face_tag = ss_arr.astype(np.int8, copy=False).ravel().tobytes()
            row[shi_i] = old
            if face_tag not in valid_face_tags:
                continue
            edges.append((src_tag, face_tag, shi_i))
            extra_tags.append(face_tag)
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

    Drops SHIs on each cell with no same-dimension flip neighbor.  Called at the
    end of :meth:`Complex.contract` and :meth:`Complex.get_boundary_complex`.
    Do not replace with full SS flip-neighbor membership; see the module comment
    above role 1.

    Geometric feasibility of the induced lower-dimensional face is handled
    separately by the post-filter in :meth:`~relucent.complex.Complex.get_chain_complex`,
    which calls :func:`remove_phantom_shis` on the parent level after removing
    any infeasible child cells.

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
        if filtered != list(poly._shis):
            poly._shis = filtered
            n_changed += 1
    return n_changed


def remove_phantom_shis(
    parent_complex: Complex,
    surviving_child_tags: set[bytes],
) -> int:
    """Remove SHIs from parent cells whose induced child face was filtered out.

    Called by :meth:`~relucent.complex.Complex.get_chain_complex` after the
    uniform feasibility post-filter removes infeasible cells from a contracted
    child complex.  Ensures each parent cell's ``_shis`` list only references
    faces that exist in ``surviving_child_tags``.

    This keeps ``_shis`` accurate for two consumers:

    - **Boundedness heuristics** (e.g. the 1-cell SHI-count rule: ``len(shis)``
      determines whether a 1-cell is a segment, ray, or line).  A stale phantom
      SHI inflates this count and mis-classifies rays as bounded segments.
    - **Future** :meth:`~relucent.complex.Complex.get_dual_graph` **calls**: the
      method walks ``poly.shis`` to wire adjacency for the next contraction level.
      Phantom entries would produce spurious dual-graph edges and break ``∂² = 0``.

    Args:
        parent_complex: The contracted complex one level above the filtered child.
        surviving_child_tags: ``encode_ss`` tags of cells that survived filtering.

    Returns:
        Number of cells in ``parent_complex`` whose ``_shis`` was modified.
    """
    n_changed = 0
    for poly in parent_complex:
        if poly._shis is None:
            continue
        ss_row = np.asarray(poly.ss_np, dtype=np.int8).ravel().copy()
        new_shis: list[int] = []
        for shi_i in poly._shis:
            old = ss_row[shi_i]
            ss_row[shi_i] = 0
            if ss_row.tobytes() in surviving_child_tags:
                new_shis.append(shi_i)
            ss_row[shi_i] = old
        if new_shis != list(poly._shis):
            poly._shis = new_shis
            n_changed += 1
    return n_changed


def compute_contracted_shis_top_down(by_dim: dict[int, Complex]) -> None:
    """Fill missing ``_shis`` on chain cells (role 3).

    The chain complex from :meth:`Complex.contract` already sets ``_shis`` via
    coface intersection and flip-neighbor filtering (role 1).  This pass is a
    safety net for contracted cells that still lack ``_shis``.  It does **not**
    drive meta-graph face-edge discovery; that uses
    :func:`ss_face_crossing_indices` (role 2).  Boundedness is classified
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
                face_tag = ss_face.tobytes()

                contrib = set(coface._shis) - {shi}
                if face_tag not in face_shis:
                    face_shis[face_tag] = contrib.copy()
                else:
                    face_shis[face_tag] &= contrib

        # Apply the computed SHIs to the (k-1)-dim cells.
        face_lookup = {p.tag: p for p in by_dim[k - 1]}
        neighbor_tags = set(face_lookup.keys())
        n_set = 0
        for face_tag, shis in face_shis.items():
            face = face_lookup.get(face_tag)
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
