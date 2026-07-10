"""Meta-graph transforms: truncation, compactification, and assembly audits.

Combinatorial primitives (sign-sequence face incidence, dual-graph adjacency,
boundedness classification) live in :mod:`relucent.incidence`; certification
of a :class:`~relucent.complex.Complex` lives in :mod:`relucent.certify`. This
module only holds operations on an already-assembled meta-graph: augmenting it
for homology at infinity, and auditing that
:meth:`~relucent.complex.Complex.get_meta_graph` assembled it correctly.

Re-exports of commonly used incidence primitives and error types are provided
below for backward compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np

# Geometric check moved to certify.py
from relucent.certify import verify_arrangement_genericity as verify_arrangement_genericity

# Re-export error types (previously defined here)
from relucent.errors import CubicalAmbiguityError as CubicalAmbiguityError
from relucent.errors import CubicalConsistencyError as CubicalConsistencyError
from relucent.errors import NonGenericArrangementError as NonGenericArrangementError

# Re-export incidence primitives (previously defined here)
from relucent.incidence import META_FACE_PARALLEL_MIN_CELLS as META_FACE_PARALLEL_MIN_CELLS
from relucent.incidence import classify_finite_ascending as classify_finite_ascending
from relucent.incidence import classify_finite_combinatorial as classify_finite_combinatorial
from relucent.incidence import classify_finite_lp_fallback as classify_finite_lp_fallback
from relucent.incidence import classify_lazy_face_polys as classify_lazy_face_polys
from relucent.incidence import classify_one_cells_finite_from_face_edges as classify_one_cells_finite_from_face_edges
from relucent.incidence import collect_meta_face_edges as collect_meta_face_edges
from relucent.incidence import cubical_cell_shis as cubical_cell_shis
from relucent.incidence import dual_edges_top_dim as dual_edges_top_dim
from relucent.incidence import dual_graph_edge_top_dim as dual_graph_edge_top_dim
from relucent.incidence import face_tag as face_tag
from relucent.incidence import flip_tag as flip_tag
from relucent.incidence import geometric_infeasible_one_cells as geometric_infeasible_one_cells
from relucent.incidence import meta_node_attrs as meta_node_attrs
from relucent.incidence import propagate_infeasible_exclusion as propagate_infeasible_exclusion
from relucent.incidence import set_contracted_shis as set_contracted_shis
from relucent.incidence import ss_nonzero_indices as ss_nonzero_indices
from relucent.incidence import sync_shis_from_dual_graph as set_shis_from_dual_graph  # renamed
from relucent.incidence import sync_shis_from_dual_graph as sync_shis_from_dual_graph
from relucent.incidence import verify_contracted_shis as verify_contracted_shis
from relucent.incidence import verify_dual_graph_cubical as verify_dual_graph_cubical
from relucent.incidence import verify_flip_shi_symmetry as verify_top_cell_flip_shi_symmetry  # renamed
from relucent.incidence import verify_shi_flip_neighbors as verify_shi_flip_neighbors
from relucent.incidence import verify_shis_from_dual_graph as verify_shis_from_dual_graph
from relucent.poly import Polyhedron

# ``shi`` edge attribute for truncation incidences (not a network SHI).
TRUNCATION_META_SHI: int = -1

INFINITY_POINT_META_NODE: tuple[str] = ("infty",)
INFINITY_POINT_META_SHI: int = -2

__all__ = [
    "INFINITY_POINT_META_NODE",
    "INFINITY_POINT_META_SHI",
    "META_FACE_PARALLEL_MIN_CELLS",
    "TRUNCATION_META_SHI",
    "CubicalAmbiguityError",
    "CubicalConsistencyError",
    "NonGenericArrangementError",
    "classify_finite_ascending",
    "classify_lazy_face_polys",
    "classify_one_cells_finite_from_face_edges",
    "collect_meta_face_edges",
    "cubical_cell_shis",
    "dual_edges_top_dim",
    "dual_graph_edge_top_dim",
    "face_tag",
    "finite_cells_subgraph",
    "flip_tag",
    "geometric_infeasible_one_cells",
    "meta_node_attrs",
    "one_point_compactify_meta_graph",
    "set_contracted_shis",
    "set_shis_from_dual_graph",
    "ss_nonzero_indices",
    "sync_shis_from_dual_graph",
    "truncate_meta_graph",
    "verify_contracted_shis",
    "verify_dual_graph_cubical",
    "verify_meta_graph_incidence",
    "verify_arrangement_genericity",
    "verify_meta_graph_one_cells",
    "verify_shi_flip_neighbors",
    "verify_shis_from_dual_graph",
    "verify_top_cell_flip_shi_symmetry",
]


def finite_cells_subgraph(meta: nx.MultiDiGraph[Any]) -> nx.MultiDiGraph[Any]:
    """Return the subcomplex induced by nodes with ``finite is True``."""
    finite = [n for n, a in meta.nodes(data=True) if a.get("finite", None) is True]
    return meta.subgraph(finite).copy()


def _truncate_n_copies(orig: Any, meta: nx.MultiDiGraph[Any], unbounded: set[Any], _cache: dict[Any, int]) -> int:
    """Number of bilateral truncation copies needed for an unbounded cell.

    A cell needs 2 copies when it has no existing (dim-1)-face out-edges within the
    unbounded subgraph -- meaning both ends of the cell exit the complex. This is
    determined bottom-up (1-cells first, then 2-cells, etc.) so that higher-dimensional
    cells can inherit from their boundaries.
    """
    if orig in _cache:
        return _cache[orig]
    k = int(meta.nodes[orig].get("dim", -1))
    if k <= 0:
        _cache[orig] = 1
        return 1
    km1_dim = k - 1
    # Count existing (k-1)-face out-edges to BOUNDED cells (not in unbounded set).
    bounded_lower = sum(
        1 for _, v, _ in meta.out_edges(orig, data=True) if int(meta.nodes[v].get("dim", -1)) == km1_dim and v not in unbounded
    )
    if k == 1:
        # For 1-cells, count all existing 0-face out-edges (bounded + unbounded).
        all_lower = sum(1 for _, v, _ in meta.out_edges(orig, data=True) if int(meta.nodes[v].get("dim", -1)) == 0)
        # Need 2 truncation copies iff the 1-cell has no original 0-face endpoints at all
        # (i.e., it's an infinite line rather than a ray or segment).
        n = 2 if all_lower == 0 else 1
    else:
        # Inherit from the maximum across unbounded (k-1)-face boundaries.
        ub_lower_copies = [
            _truncate_n_copies(v, meta, unbounded, _cache)
            for _, v, _ in meta.out_edges(orig, data=True)
            if int(meta.nodes[v].get("dim", -1)) == km1_dim and v in unbounded
        ]
        n = max(ub_lower_copies) if ub_lower_copies else 1
        if bounded_lower > 0:
            # Has at least one bounded face -- only one open end from the truncation side.
            n = min(n, 1)
    _cache[orig] = n
    return n


def truncate_meta_graph(meta: nx.MultiDiGraph[Any]) -> None:
    """Augment ``meta`` in place with combinatorial truncation at infinity.

    Equivalent to intersecting the complex with a large enough bounding box: bounded
    cells are unaffected, each unbounded cell gains one or two (dim-1)-dimensional
    truncation faces.

    Every node's ``ss`` gains a trailing ``1`` (strictly inside the truncation halfspace).
    For each unbounded cell ``n``, one or two truncation duplicates are created:

    - **One copy** (``("trunc", n)``): when ``n`` has at least one existing (dim-1)-face
      boundary -- the cell is a *ray* with one open end.
    - **Two copies** (``("trunc", n)`` and ``("trunc2", n)``): when ``n`` has no existing
      (dim-1)-face boundary -- the cell is an *infinite line/plane* with two open ends.
      The second copy closes the other end, giving the cell exactly two boundary faces.

    Each duplicate has trailing ``0`` on ``ss``, dimension decremented by one, and is
    marked ``finite=True``. Face edges among duplicates mirror the induced subgraph on
    unbounded cells independently for each truncation layer.
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

    # Compute number of truncation copies bottom-up (1-cells first, then higher dims).
    n_copies_cache: dict[Any, int] = {}
    for orig in sorted(unbounded, key=lambda n: int(meta.nodes[n].get("dim", -1))):
        _truncate_n_copies(orig, meta, unbounded, n_copies_cache)

    # Create truncation duplicates (one or two per unbounded cell).
    layer0: set[Any] = set()
    layer1: set[Any] = set()

    for orig in unbounded:
        oa = meta.nodes[orig]
        k = int(oa.get("dim", -1))
        ss_in = oa.get("ss")
        if k <= 0 or ss_in is None:
            continue
        ss_on_cut = np.asarray(ss_in).copy()
        ss_on_cut[..., -1] = 0
        node_attrs = {
            "poly": oa.get("poly"),
            "dim": k - 1,
            "ss": ss_on_cut,
            "finite": True,
            "shis": list(oa.get("shis", [])),
            "truncation_duplicate": True,
        }
        dup0 = ("trunc", orig)
        layer0.add(dup0)
        meta.add_node(dup0, **node_attrs)
        meta.add_edge(orig, dup0, shi=TRUNCATION_META_SHI)
        if n_copies_cache.get(orig, 1) >= 2:
            dup1 = ("trunc2", orig)
            layer1.add(dup1)
            meta.add_node(dup1, **node_attrs)
            meta.add_edge(orig, dup1, shi=TRUNCATION_META_SHI)

    # Mirror ub_faces edges into each truncation layer independently.
    for u, v, ed in ub_faces.edges(data=True):
        tu0, tv0 = ("trunc", u), ("trunc", v)
        if tu0 in layer0 and tv0 in layer0:
            meta.add_edge(tu0, tv0, **dict(ed))
        tu1, tv1 = ("trunc2", u), ("trunc2", v)
        if tu1 in layer1 and tv1 in layer1:
            meta.add_edge(tu1, tv1, **dict(ed))


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


def _meta_one_cell_zero_face_count(meta: nx.MultiDiGraph[Any], node: Any) -> int:
    return sum(1 for _u, v, _ in meta.out_edges(node, data=True) if int(meta.nodes[v].get("dim", -1)) == 0)


def verify_meta_graph_one_cells(meta: nx.MultiDiGraph[Any]) -> None:
    """Require sound 1-cell flip-SHI counts and boundedness labels.

      Each 1-cell has **0–2** flip-SHI neighbors and **0–2** combinatorial 0-face endpoints.
      Bounded segments (``finite is True``) close at both ends; fewer than two flip SHIs
    implies ``finite is False``.
    """
    for node, attrs in meta.nodes(data=True):
        if int(attrs.get("dim", -1)) != 1:
            continue
        shis = [int(s) for s in attrs.get("shis", ())]
        n_shis = len(shis)
        if n_shis > 2:
            raise CubicalConsistencyError(f"1-cell {node!r} has {n_shis} flip SHIs {shis!r}; expected at most 2.")
        n_zero = _meta_one_cell_zero_face_count(meta, node)
        if n_zero > 2:
            raise CubicalConsistencyError(f"1-cell {node!r} has {n_zero} combinatorial 0-faces; expected at most 2.")
        finite = attrs.get("finite")
        if finite is None:
            continue
        if finite is True:
            if n_shis < 2 or n_zero < 2:
                raise CubicalConsistencyError(
                    f"Bounded 1-cell {node!r} must have two flip SHIs and two 0-faces; "
                    + f"got shis={shis!r}, n_zero={n_zero}."
                )
            continue
        if n_shis < 2:
            continue
        if n_zero >= 2:
            raise CubicalConsistencyError(
                f"1-cell {node!r} is marked unbounded but has {n_shis} flip SHIs and {n_zero} 0-faces."
            )


def verify_meta_graph_incidence(
    meta: nx.MultiDiGraph[Any],
    by_dim: Mapping[int, Any],
    lookup: dict[bytes, Polyhedron],
) -> None:
    """Check assembled meta-graph matches the stateless incidence engine.

    - Face edges equal ``ss_nonzero_indices`` zeroings kept by lookup.
    - Node ``shis`` equal :func:`~relucent.incidence.cubical_cell_shis` on each
      dimension slice (never the propagated ``poly._shis`` LP cache).
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
            # 1-cells: filter geometrically infeasible crossings, matching meta_node_attrs.
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

    verify_meta_graph_one_cells(meta)
