"""Meta-graph transforms: truncation, compactification, and assembly audits.

Combinatorial primitives (sign-sequence face incidence, dual-graph adjacency,
boundedness classification) live in :mod:`relucent.graph.incidence`; certification
of a :class:`~relucent.core.complex.Complex` lives in :mod:`relucent.verify.certify`. This
module holds operations on an already-assembled meta-graph:

- **Truncation / compactification** — :func:`truncate_meta_graph`, :func:`one_point_compactify_meta_graph`,
  :func:`finite_cells_subgraph`; called from :meth:`~relucent.core.complex.Complex.get_betti_numbers`
  and :mod:`relucent.topology.persistence`.
- **Post-assembly audits** — :func:`verify_meta_graph_incidence`, :func:`verify_meta_graph_one_cells`;
  used when :meth:`~relucent.core.complex.Complex.get_meta_graph` is called with ``verify=True``.

Re-exports of commonly used incidence primitives and error types are provided
below for backward compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np

import relucent.config as cfg

# Re-export error types (previously defined here)
from relucent.core.errors import CubicalAmbiguityError as CubicalAmbiguityError
from relucent.core.errors import CubicalConsistencyError as CubicalConsistencyError
from relucent.core.errors import NonGenericArrangementError as NonGenericArrangementError
from relucent.core.poly import Polyhedron

# Re-export incidence primitives (previously defined here)
from relucent.graph.incidence import META_FACE_PARALLEL_MIN_CELLS as META_FACE_PARALLEL_MIN_CELLS
from relucent.graph.incidence import assemble_face_edges_by_dim as assemble_face_edges_by_dim
from relucent.graph.incidence import classify_finite_ascending as classify_finite_ascending
from relucent.graph.incidence import classify_finite_combinatorial as classify_finite_combinatorial
from relucent.graph.incidence import classify_finite_lp_fallback as classify_finite_lp_fallback
from relucent.graph.incidence import classify_lazy_face_polys as classify_lazy_face_polys
from relucent.graph.incidence import classify_one_cells_finite_from_face_edges as classify_one_cells_finite_from_face_edges
from relucent.graph.incidence import collect_meta_face_edges as collect_meta_face_edges
from relucent.graph.incidence import cubical_cell_shis as cubical_cell_shis
from relucent.graph.incidence import dual_edges_top_dim as dual_edges_top_dim
from relucent.graph.incidence import dual_graph_edge_top_dim as dual_graph_edge_top_dim
from relucent.graph.incidence import face_tag as face_tag
from relucent.graph.incidence import flip_tag as flip_tag
from relucent.graph.incidence import geometric_infeasible_one_cells as geometric_infeasible_one_cells
from relucent.graph.incidence import meta_node_attrs as meta_node_attrs
from relucent.graph.incidence import propagate_infeasible_exclusion as propagate_infeasible_exclusion
from relucent.graph.incidence import set_contracted_shis as set_contracted_shis
from relucent.graph.incidence import ss_nonzero_indices as ss_nonzero_indices
from relucent.graph.incidence import sync_shis_from_dual_graph as set_shis_from_dual_graph  # renamed
from relucent.graph.incidence import sync_shis_from_dual_graph as sync_shis_from_dual_graph
from relucent.graph.incidence import verify_contracted_shis as verify_contracted_shis
from relucent.graph.incidence import verify_dual_graph_cubical as verify_dual_graph_cubical
from relucent.graph.incidence import verify_flip_shi_symmetry as verify_top_cell_flip_shi_symmetry  # renamed
from relucent.graph.incidence import verify_shi_flip_neighbors as verify_shi_flip_neighbors
from relucent.graph.incidence import verify_shis_from_dual_graph as verify_shis_from_dual_graph
from relucent.utils import encode_ss

# Geometric check moved to certify.py
from relucent.verify.certify import verify_arrangement_genericity as verify_arrangement_genericity

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
    "assemble_face_edges_by_dim",
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
    "rebuild_meta_graph_face_edges",
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
    """Return the subcomplex induced by nodes with ``finite is True``.

    Used by :meth:`~relucent.core.complex.Complex.get_betti_numbers_from_meta` when
    ``respect_finite=True`` to compute homology on bounded cells only (no truncation).
    """
    finite = [n for n, a in meta.nodes(data=True) if a.get("finite", None) is True]
    return meta.subgraph(finite).copy()


def _meta_nodes_by_dim(meta: nx.MultiDiGraph[Any]) -> dict[int, list[Any]]:
    nodes_by_dim: dict[int, list[Any]] = {}
    for n, attrs in meta.nodes(data=True):
        nodes_by_dim.setdefault(int(attrs.get("dim", -1)), []).append(n)
    return nodes_by_dim


def _truncation_bit_indices(ss: np.ndarray) -> tuple[int, int]:
    """Flattened indices of the two trailing truncation coordinates."""
    flat = np.asarray(ss, dtype=np.int8).reshape(-1)
    return int(flat.size - 2), int(flat.size - 1)


def _ss_with_truncation_bits(ss: np.ndarray, t1: int, t2: int) -> np.ndarray:
    """Append two truncation halfspace bits to a sign sequence."""
    a = np.asarray(ss)
    dt = np.int8 if np.issubdtype(a.dtype, np.integer) else a.dtype
    return np.hstack([a, np.full((a.shape[0], 2), [t1, t2], dtype=dt)])


def _facet_participates_in_truncation_sidedness(facet: Any, meta: nx.MultiDiGraph[Any]) -> bool:
    """Whether a codimension-one facet contributes to cap-sidedness on its coface."""
    dim = int(meta.nodes[facet].get("dim", -1))
    if dim != 1:
        return True
    if meta.nodes[facet].get("finite") is None:
        return False
    poly = meta.nodes[facet].get("poly")
    return not (poly is not None and poly._finite is None)


def _combinatorial_zero_faces(meta: nx.MultiDiGraph[Any], node: Any) -> int:
    """0-face count from combinatorial incidence (pre-exclusion), when recorded on the node."""
    stored = meta.nodes[node].get("comb_n_zero_faces")
    if stored is not None:
        return int(stored)
    return _meta_one_cell_zero_face_count(meta, node)


def _facet_is_anchored_for_sidedness(facet: Any, meta: nx.MultiDiGraph[Any], km1_dim: int) -> bool:
    """Whether a facet has a bounded face anchoring its truncation patch."""
    if km1_dim == 1:
        return _combinatorial_zero_faces(meta, facet) >= 1
    bounded_lower = sum(
        1
        for _, v, _ in meta.out_edges(facet, data=True)
        if int(meta.nodes[v].get("dim", -1)) == km1_dim - 1 and meta.nodes[v].get("finite") is True
    )
    return bounded_lower > 0


def _raw_unbounded_facets(
    orig: Any,
    meta: nx.MultiDiGraph[Any],
    unbounded: set[Any],
    km1_dim: int,
) -> list[Any]:
    """Participating unbounded ``(k−1)``-facets (before bi-infinite sidedness filtering)."""
    return [
        v
        for _, v, _ in meta.out_edges(orig, data=True)
        if int(meta.nodes[v].get("dim", -1)) == km1_dim
        if v in unbounded
        if _facet_participates_in_truncation_sidedness(v, meta)
    ]


def _facets_for_sidedness_propagation(
    orig: Any,
    meta: nx.MultiDiGraph[Any],
    unbounded: set[Any],
    km1_dim: int,
) -> list[Any]:
    """Unbounded facets whose open-cap counts drive sidedness on an unanchored coface.

    Bi-infinite line facets (no cubical 0-face anchor) are dropped when ``poly`` is set:
    inheriting bilateral openness from them onto a unilateral coface creates phantom
    0-faces. This filter is an *inheritance* policy only — not an existence test for
    whether the coface needs a truncation cap (see :func:`_open_cap_count`).
    """
    candidates = _raw_unbounded_facets(orig, meta, unbounded, km1_dim)
    if candidates and any(meta.nodes[v].get("poly") is not None for v in candidates):
        anchored = [v for v in candidates if _facet_is_anchored_for_sidedness(v, meta, km1_dim)]
        if anchored:
            return anchored
        # Bi-infinite lines keep bilateral caps locally; cofaces fall through to one local cut.
        return []
    if candidates and all(meta.nodes[v].get("poly") is None for v in candidates):
        return candidates
    return candidates


def _one_cell_open_cap_count(orig: Any, meta: nx.MultiDiGraph[Any]) -> int:
    """Open-cap count for an unbounded 1-cell (0, 1, or 2 bilateral caps at infinity)."""
    attrs = meta.nodes[orig]
    if attrs.get("finite") is None:
        return 0
    poly = attrs.get("poly")
    if poly is not None and poly._finite is None:
        return 0
    comb_n_zero_raw = attrs.get("comb_n_zero_faces")
    comb_n_zero = _meta_one_cell_zero_face_count(meta, orig) if comb_n_zero_raw is None else int(comb_n_zero_raw)
    zero_faces = [v for _, v, _ in meta.out_edges(orig, data=True) if int(meta.nodes[v].get("dim", -1)) == 0]
    bounded_zeros = sum(1 for z in zero_faces if meta.nodes[z].get("finite") is True)
    if comb_n_zero >= 2 or bounded_zeros >= 2:
        return 0
    if comb_n_zero == 1 or bounded_zeros == 1:
        return 1
    return 2


def _open_cap_count(
    orig: Any,
    meta: nx.MultiDiGraph[Any],
    unbounded: set[Any],
    cache: dict[Any, int],
) -> int:
    """How many truncation caps this unbounded cell needs (0, 1, or 2).

    Cap is a functor on unbounded cells: each unbounded k-cell maps to a (k-1)-cell
      sphere-cut, and face incidence is preserved among trunc-bit-compatible cells
    (a cap of a facet is a facet of the cap).

    Anchored cells (``bounded_lower > 0``) use *raw* unbounded facets to decide whether
    a sphere-cut exists; bi-infinite sidedness filtering applies only to unanchored
    inheritance.
    """
    if orig in cache:
        return cache[orig]
    k = int(meta.nodes[orig].get("dim", -1))
    if k <= 0:
        cache[orig] = 0
        return 0
    km1_dim = k - 1

    if k == 1:
        n = _one_cell_open_cap_count(orig, meta)
        cache[orig] = n
        return n

    bounded_lower = sum(
        1
        for _, v, _ in meta.out_edges(orig, data=True)
        if int(meta.nodes[v].get("dim", -1)) == km1_dim and meta.nodes[v].get("finite") is True
    )
    raw_unbounded = _raw_unbounded_facets(orig, meta, unbounded, km1_dim)
    if bounded_lower > 0:
        # Bounded facet anchors the sphere-cut to one connected patch.
        # Use raw facets: bi-infinite lines still make the cell unbounded at infinity.
        n = 1 if raw_unbounded else 0
        cache[orig] = n
        return n

    # Unanchored (k>=2)-cell: propagate sidedness from filtered unbounded facets.
    # Empty facet list (only unanchored lines) defaults to one local cap.
    sidedness_facets = _facets_for_sidedness_propagation(orig, meta, unbounded, km1_dim)
    lower_open = [_open_cap_count(v, meta, unbounded, cache) for v in sidedness_facets]
    if lower_open and min(lower_open) != max(lower_open):
        raise NonGenericArrangementError(
            f"unbounded {k}-cell {orig!r} has no bounded (k-1)-facet, but its unbounded "
            + f"facets disagree on truncation sidedness ({lower_open}); the recession cone "
            + "is likely a degenerate linear subspace (e.g. parallel bent hyperplanes with "
            + "no anchor). Generic networks should not hit this branch."
        )
    n = max(lower_open) if lower_open else 1
    cache[orig] = n
    return n


def _cap_sign_sequence(parent_ss: np.ndarray, *, cap_index: int, n_caps: int) -> np.ndarray:
    """Sign sequence of a truncation cap (zeroing the appropriate truncation bit)."""
    ss = np.asarray(parent_ss, dtype=np.int8).copy()
    if n_caps >= 2:
        if cap_index == 0:
            ss[..., -2] = 0
            ss[..., -1] = 1
        else:
            ss[..., -2] = 1
            ss[..., -1] = 0
    else:
        ss[..., -2] = 0
        ss[..., -1] = 0
    return ss


def _sync_meta_node_shis(meta: nx.MultiDiGraph[Any], *, exclude_truncation_bits: bool = False) -> None:
    """Refresh ``crossings`` / ``shis`` on every node from cubical flip-neighbor rules."""
    nodes_by_dim = _meta_nodes_by_dim(meta)
    for dim, nodes in nodes_by_dim.items():
        neighbor_tags: set[bytes] = set()
        for n in nodes:
            ss = meta.nodes[n].get("ss")
            if ss is not None:
                neighbor_tags.add(encode_ss(np.asarray(ss, dtype=np.int8)))

        for n in nodes:
            attrs = meta.nodes[n]
            ss = attrs.get("ss")
            if ss is None:
                continue
            ss_arr = np.asarray(ss, dtype=np.int8)
            attrs["crossings"] = list(ss_nonzero_indices(ss_arr))
            trunc_shis: frozenset[int] = frozenset()
            if exclude_truncation_bits:
                t1, t2 = _truncation_bit_indices(ss_arr)
                trunc_shis = frozenset({t1, t2})
            shis = cubical_cell_shis(ss_arr, neighbor_tags=neighbor_tags, exclude_shis=trunc_shis)
            poly = attrs.get("poly")
            if int(dim) == 1 and poly is not None and poly.halfspaces is not None:
                shis = [s for s in shis if poly.is_shi_face_feasible(int(s))]
            edge_shis: list[int] = []
            if int(dim) == 1:
                edge_shis = sorted(
                    {
                        int(shi)
                        for _u, v, _k, ed in meta.out_edges(n, keys=True, data=True)
                        if int(meta.nodes[v].get("dim", -1)) == 0
                        if (shi := ed.get("shi")) is not None
                        if int(shi) not in trunc_shis
                    }
                )
            if int(dim) == 1 and attrs.get("finite") is True and len(shis) < 2:
                if len(edge_shis) >= 2 or (exclude_truncation_bits and edge_shis):
                    shis = edge_shis
            elif exclude_truncation_bits and int(dim) == 1 and len(shis) > 2:
                # Prefer combinatorial 0-face incidences over excess cubical flips.
                shis = edge_shis if 1 <= len(edge_shis) <= 2 else shis[:2]
            attrs["shis"] = shis


def _reclassify_meta_finite(meta: nx.MultiDiGraph[Any]) -> None:
    """Refresh ``finite`` on meta nodes from combinatorial face-edge incidence."""
    nodes_by_dim = _meta_nodes_by_dim(meta)

    for n, attrs in meta.nodes(data=True):
        if int(attrs.get("dim", -1)) != 1:
            continue
        n_zero = _meta_one_cell_zero_face_count(meta, n)
        if n_zero == 0:
            attrs["finite"] = None
        elif n_zero >= 2:
            attrs["finite"] = True
        else:
            attrs["finite"] = False

    max_dim = max(nodes_by_dim.keys(), default=0)
    for k in range(2, max_dim + 1):
        for n in nodes_by_dim.get(k, []):
            face_tags = {v for _, v, _ in meta.out_edges(n, data=True) if int(meta.nodes[v].get("dim", -1)) == k - 1}
            if not face_tags:
                continue
            n_bounded = 0
            n_unbounded = 0
            n_infeasible = 0
            unknown = False
            for ft in face_tags:
                fv = meta.nodes[ft].get("finite")
                if fv is None:
                    n_infeasible += 1
                elif fv is False:
                    n_unbounded += 1
                elif fv is True:
                    n_bounded += 1
                else:
                    unknown = True
            if n_unbounded > 0:
                meta.nodes[n]["finite"] = False
            elif n_bounded > 0 and not unknown:
                meta.nodes[n]["finite"] = True
            elif n_infeasible > 0 and n_bounded == 0 and n_unbounded == 0 and not unknown:
                meta.nodes[n]["finite"] = None


def rebuild_meta_graph_face_edges(meta: nx.MultiDiGraph[Any]) -> None:
    """Replace all face edges using :func:`~relucent.graph.incidence.collect_meta_face_edges`.

    Called at the end of :func:`truncate_meta_graph` after cap cells are added and sign
    sequences are retagged.
    """
    nodes_by_dim = _meta_nodes_by_dim(meta)
    valid_tags = set(meta.nodes)
    cells_by_dim: dict[int, list[tuple[Any, np.ndarray, tuple[int, ...]]]] = {}
    for dim, nodes in nodes_by_dim.items():
        if dim <= 0:
            continue
        cells: list[tuple[Any, np.ndarray, tuple[int, ...]]] = []
        for n in nodes:
            ss = meta.nodes[n].get("ss")
            if ss is None:
                continue
            ss_arr = np.asarray(ss, dtype=np.int8)
            cells.append((n, ss_arr, ss_nonzero_indices(ss_arr)))
        if cells:
            cells_by_dim[int(dim)] = cells

    edges_by_dim = assemble_face_edges_by_dim(cells_by_dim, valid_tags)
    meta.remove_edges_from(list(meta.edges()))
    for edges in edges_by_dim.values():
        meta.add_edges_from((u, v, {"shi": int(shi)}) for u, v, shi in edges)


def _retag_meta_nodes_from_ss(meta: nx.MultiDiGraph[Any]) -> dict[Any, Any]:
    """Relabel nodes so keys match :func:`~relucent.utils.encode_ss` of their ``ss``."""
    tag_remap: dict[Any, Any] = {}
    for n, attrs in meta.nodes(data=True):
        ss = attrs.get("ss")
        if ss is None:
            continue
        new_tag = encode_ss(np.asarray(ss, dtype=np.int8))
        if new_tag != n:
            tag_remap[n] = new_tag
    if tag_remap:
        nx.relabel_nodes(meta, tag_remap, copy=False)
    return tag_remap


def _trunc_bits(ss: np.ndarray) -> tuple[int, int]:
    flat = np.asarray(ss).ravel()
    return int(flat[-2]), int(flat[-1])


def _assert_truncation_invariants(
    meta: nx.MultiDiGraph[Any],
    *,
    pre_trunc_edges: list[tuple[Any, Any, int]],
    cap_count_cache: dict[Any, int],
) -> None:
    """CAREFUL_MODE checks after truncation incidence is assembled.

    Only cubical face edges (``face_tag(parent, shi) == child`` on extended SS) are
    required to survive when trunc bits agree. Hand-built or non-cubical incidents are
    ignored.
    """
    for u, v, shi in pre_trunc_edges:
        if u not in meta or v not in meta:
            continue
        if meta.has_edge(u, v):
            continue
        ss_u = meta.nodes[u].get("ss")
        ss_v = meta.nodes[v].get("ss")
        if ss_u is None or ss_v is None:
            continue
        ss_u_arr = np.asarray(ss_u, dtype=np.int8)
        if shi < 0 or shi >= ss_u_arr.size or int(ss_u_arr.ravel()[shi]) == 0:
            continue
        # Only enforce survival for edges that remain cubical faces after extension.
        if face_tag(ss_u_arr, int(shi)) != v:
            continue
        if _trunc_bits(ss_u_arr) == _trunc_bits(ss_v):
            raise CubicalConsistencyError(
                f"pre-truncation face edge {u!r}->{v!r} was dropped after truncate "
                + "despite matching trunc bits; incidence rebuild is incomplete."
            )
        parent_caps = cap_count_cache.get(u, 0)
        if parent_caps <= 0 and int(meta.nodes[u].get("dim", -1)) >= 2:
            raise CubicalConsistencyError(
                f"pre-truncation face edge {u!r}->{v!r} dropped due to trunc-bit "
                + f"mismatch {_trunc_bits(ss_u_arr)} vs {_trunc_bits(ss_v)}, but parent "
                + "has no truncation caps to replace it."
            )


def truncate_meta_graph(meta: nx.MultiDiGraph[Any]) -> None:
    """Augment ``meta`` in place with combinatorial truncation at infinity.

    Extends every sign sequence with two truncation bits, materializes cap cells as
    ordinary byte-tagged nodes via :func:`~relucent.graph.incidence.face_tag`, then rebuilds
    face edges through :func:`collect_meta_face_edges`.

    Network faces between cells whose trunc bits agree are recovered by the cubical
    ``face_tag`` rebuild. Faces between cells with disagreeing openness (e.g. a
    unilateral coface and a bi-infinite line) are intentionally omitted: restoring them
    without a matching sphere-cut breaks ``∂²=0``. Those cofaces receive a trunc-cap
    instead (see :func:`_open_cap_count`).

    Called automatically by :meth:`~relucent.core.complex.Complex.get_betti_numbers` when
    ``compactify=False`` (the default), and by persistent-homology code in
    :mod:`relucent.topology.persistence` for the same link-at-infinity convention.
    """
    if meta.number_of_nodes() == 0:
        return

    unbounded = {n for n, a in meta.nodes(data=True) if a.get("finite", None) is False}

    cap_count_cache: dict[Any, int] = {}
    for orig in sorted(unbounded, key=lambda n: int(meta.nodes[n].get("dim", -1))):
        _open_cap_count(orig, meta, unbounded, cap_count_cache)

    # Snapshot for careful-mode incidence audit (node ids change at retag).
    pre_trunc_edges = [(u, v, int(d.get("shi", -1))) for u, v, d in meta.edges(data=True)]

    for n, attrs in meta.nodes(data=True):
        ss0 = attrs.get("ss")
        if ss0 is None:
            continue
        n_caps = cap_count_cache.get(n, 0) if n in unbounded else 0
        t1, t2 = (1, 1) if n_caps >= 2 else (1, 0)
        attrs["ss"] = _ss_with_truncation_bits(np.asarray(ss0), t1, t2)

    tag_remap = _retag_meta_nodes_from_ss(meta)
    if tag_remap:
        cap_count_cache = {tag_remap.get(k, k): v for k, v in cap_count_cache.items()}
        unbounded = {tag_remap.get(n, n) for n in unbounded}
        pre_trunc_edges = [
            (tag_remap.get(u, u), tag_remap.get(v, v), shi) for u, v, shi in pre_trunc_edges
        ]

    for orig in unbounded:
        oa = meta.nodes[orig]
        k = int(oa.get("dim", -1))
        ss_in = oa.get("ss")
        if k <= 0 or ss_in is None:
            continue
        n_caps = cap_count_cache.get(orig, 0)
        if n_caps <= 0:
            continue
        t1_shi, t2_shi = _truncation_bit_indices(np.asarray(ss_in))
        cap_bits = [t1_shi] if n_caps == 1 else [t1_shi, t2_shi]
        for cap_index, bit_shi in enumerate(cap_bits):
            cap_ss = _cap_sign_sequence(np.asarray(ss_in), cap_index=cap_index, n_caps=n_caps)
            cap_tag = face_tag(np.asarray(ss_in), int(bit_shi))
            if cap_tag in meta.nodes:
                continue
            meta.add_node(
                cap_tag,
                poly=oa.get("poly"),
                dim=k - 1,
                ss=cap_ss,
                finite=True,
                shis=[],
            )

    rebuild_meta_graph_face_edges(meta)
    _reclassify_meta_finite(meta)
    _sync_meta_node_shis(meta, exclude_truncation_bits=True)
    verify_meta_graph_one_cells(meta, truncated=True)
    if cfg.CAREFUL_MODE:
        _assert_truncation_invariants(
            meta,
            pre_trunc_edges=pre_trunc_edges,
            cap_count_cache=cap_count_cache,
        )


def one_point_compactify_meta_graph(meta: nx.MultiDiGraph[Any]) -> bool:
    """Augment ``meta`` in place with a single point-at-infinity 0-cell.

    Mirrors ``canonicalpoly2.0/polyhedra/topology.get_coboundary_matrices``: each
    1-cell whose boundary consists of a single 0-cell (an unbounded end) gains a
    second incidence to one new 0-cell representing infinity.  Returns whether the
    infinity node was added.

    Called by :meth:`~relucent.core.complex.Complex.get_betti_numbers` when
    ``compactify="one_point"``.
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


def verify_meta_graph_one_cells(meta: nx.MultiDiGraph[Any], *, truncated: bool = False) -> None:
    """Require sound 1-cell flip-SHI counts and boundedness labels.

    Each 1-cell has **0–2** flip-SHI neighbors and **0–2** combinatorial 0-face endpoints.
    Bounded segments (``finite is True``) close at both ends; fewer than two flip SHIs
    implies ``finite is False``. After truncation, a capped segment may have no network
    flip SHIs when its only crossings are truncation bits.

    Called at the end of :func:`truncate_meta_graph` (``truncated=True``) and from
    :func:`verify_meta_graph_incidence` during :meth:`~relucent.core.complex.Complex.get_meta_graph`
    debug verification.
    """
    min_bounded_shis = 0 if truncated else 2
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
            if n_shis < min_bounded_shis or n_zero < 2:
                raise CubicalConsistencyError(
                    f"Bounded 1-cell {node!r} must have at least {min_bounded_shis} flip SHI(s) and two 0-faces; "
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
    - Node ``shis`` equal :func:`~relucent.graph.incidence.cubical_cell_shis` on each
      dimension slice (never the propagated ``poly._shis`` LP cache).
    - ``finite`` on chain cells matches combinatorial classification from face edges.

    Invoked when :meth:`~relucent.core.complex.Complex.get_meta_graph` is called with
    ``verify=True`` (debugging only; too expensive for routine topology).
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
