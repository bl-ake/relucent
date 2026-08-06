"""Recover the chain complex directly from verified vertices' local stars.

Masden, *Algorithmic Determination of the Combinatorial Structure of the Linear
Regions of ReLU Neural Networks* (2022), Theorem 20: for a generic,
supertransversal network the sign-sequence complex ``S(F)`` is a **pure,
ambient-dimensional cubical complex**. Every vertex has exactly ``ambient_dim``
zero sign entries (Lemma 16), and once such a vertex is verified, *every*
sign assignment on those zero entries — holding all other entries fixed — is a
real, present cell of ``C(F)`` (Lemma 18's sign-product semigroup). No
independent rediscovery of neighboring top-dimensional cells, dual-graph cube
verification, or coverage heuristic is required for correctness.

This replaces the previous ``graph.covectors`` approach, which only recovered
a cell when the *complete* ``2^c`` cube of top-dimensional cofaces had been
independently discovered by BFS — strictly more than the theorem requires, and
the root cause of cells that provably exist being silently dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import combinations, product
from typing import TYPE_CHECKING

import numpy as np

from relucent.utils import encode_ss

if TYPE_CHECKING:
    import networkx as nx

    from relucent.core.poly import Polyhedron

__all__ = ["VertexRecord", "find_vertices", "expand_vertex_star", "recover_cells_from_vertices"]


@dataclass(frozen=True)
class VertexRecord:
    """A verified vertex, with everything needed to expand its local star."""

    ss: np.ndarray
    point: np.ndarray
    witness_tag: bytes
    varying_shis: tuple[int, ...]

    @property
    def tag(self) -> bytes:
        return encode_ss(self.ss)


def _coface_incident_shis(graph: nx.Graph[Polyhedron], poly: Polyhedron) -> set[int]:
    """Hyperplanes with a real (dual-graph-verified) flip-neighbor edge from ``poly``."""
    return {int(data["shi"]) for _u, _v, data in graph.edges(poly, data=True) if data.get("shi") is not None}


def find_vertices(
    top_cells: Iterable[Polyhedron],
    graph: nx.Graph[Polyhedron],
    *,
    ambient_dim: int,
    top_dim: int,
    verify_vertex: Callable[[Polyhedron, np.ndarray], np.ndarray | None],
) -> dict[bytes, VertexRecord]:
    """Seed every candidate vertex reachable from ``top_cells``, and verify it.

    A cell of dimension ``top_dim`` already has ``ambient_dim - top_dim`` zero
    entries (Lemma 16); reaching a full vertex needs exactly ``top_dim`` more,
    chosen from the cell's *incident* dual-graph directions (real flip
    neighbors — not every nonzero sign entry). Unlike the cubical-star
    approach, this never requires the ``2^top_dim`` cube of neighboring top
    cells to already exist: each candidate is checked directly via
    ``verify_vertex`` (typically :meth:`Polyhedron.verify_vertex_covector`),
    a single float64 equality solve plus a strict forward-sign check.
    """
    n_more = top_dim
    vertices: dict[bytes, VertexRecord] = {}
    tried: set[bytes] = set()
    for root in top_cells:
        root_ss = np.asarray(root.ss_np, dtype=np.int8)
        n_fixed_zeros = int(np.count_nonzero(root_ss.ravel() == 0))
        if n_fixed_zeros + n_more != ambient_dim:
            # Lemma 16: a `top_dim`-cell has exactly `ambient_dim - top_dim`
            # zero entries; reaching a full vertex needs exactly `top_dim`
            # more. A mismatch means `root` is not actually a `top_dim`-cell.
            continue
        incident = sorted(_coface_incident_shis(graph, root))
        if n_more == 0:
            combos: Iterable[tuple[int, ...]] = [()]
        elif len(incident) < n_more:
            continue
        else:
            combos = combinations(incident, n_more)
        for combo in combos:
            candidate = root_ss.copy()
            flat = candidate.reshape(-1)
            for shi in combo:
                flat[shi] = 0
            tag = encode_ss(candidate)
            if tag in tried:
                continue
            tried.add(tag)
            point = verify_vertex(root, candidate)
            if point is None:
                continue
            varying = tuple(sorted(combo))
            vertices[tag] = VertexRecord(ss=candidate, point=point, witness_tag=root.tag, varying_shis=varying)
    return vertices


def expand_vertex_star(vertex: VertexRecord) -> Iterable[np.ndarray]:
    """Every cell in a verified vertex's local star (Theorem 20).

    Varies each of ``vertex.varying_shis`` (the ``top_dim`` coordinates that
    distinguish this vertex from its witness top cell) independently over
    ``{-1, 0, 1}``, holding every other coordinate — including the witness's
    own already-zero entries — fixed. This stays within the ambient
    ``top_dim`` of the complex being recovered (e.g. a boundary sub-complex's
    always-zero coordinate is never perturbed), and produces cells of every
    dimension from 0 (the vertex itself) up to ``top_dim``.
    """
    base = np.asarray(vertex.ss, dtype=np.int8)
    shis = vertex.varying_shis
    for signs in product((-1, 0, 1), repeat=len(shis)):
        ss = base.copy()
        flat = ss.reshape(-1)
        for shi, sign in zip(shis, signs, strict=True):
            flat[shi] = sign
        yield ss


def recover_cells_from_vertices(
    top_cells: Iterable[Polyhedron],
    graph: nx.Graph[Polyhedron],
    *,
    ambient_dim: int,
    top_dim: int,
    verify_vertex: Callable[[Polyhedron, np.ndarray], np.ndarray | None],
) -> tuple[dict[int, dict[bytes, np.ndarray]], dict[bytes, VertexRecord]]:
    """Recover every cell reachable from a finite, verified vertex.

    Returns ``(cells_by_dim, vertices)``: ``cells_by_dim[k]`` maps cell tag to
    sign sequence for every recovered ``k``-cell (``0 <= k <= top_dim``);
    ``vertices`` maps vertex tag to :class:`VertexRecord` (interior point,
    witness top cell) for materialization.

    Every generated cell of dimension ``k >= 1`` has, by construction, at
    least one verified vertex among its own faces (its generating vertex,
    reached by zeroing all of ``varying_shis``) — a cell with every endpoint
    unverifiable can never be produced, so no separate "cascade drop" pass is
    needed the way the old cubical-star reconstruction required.
    """
    vertices = find_vertices(
        top_cells,
        graph,
        ambient_dim=ambient_dim,
        top_dim=top_dim,
        verify_vertex=verify_vertex,
    )
    cells_by_dim: dict[int, dict[bytes, np.ndarray]] = {k: {} for k in range(top_dim + 1)}
    for vertex in vertices.values():
        for ss in expand_vertex_star(vertex):
            zero_count = int(np.count_nonzero(ss.reshape(-1)[list(vertex.varying_shis)] == 0))
            dim = top_dim - zero_count
            cells_by_dim[dim][encode_ss(ss)] = ss
    return cells_by_dim, vertices
