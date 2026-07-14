"""Recover a generic cubical face lattice from a labeled tope graph."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np

from relucent.core.errors import CubicalAmbiguityError, CubicalConsistencyError
from relucent.core.poly import Polyhedron
from relucent.utils import encode_ss, flip_ss_at_shi

__all__ = ["CovectorCell", "enumerate_covectors", "sign_intersection"]


@dataclass(frozen=True)
class CovectorCell:
    """One cell recovered from its cubical star of top-dimensional cofaces."""

    ss: np.ndarray
    coface_tags: frozenset[bytes]
    zero_shis: tuple[int, ...]

    @property
    def tag(self) -> bytes:
        """Stable sign-sequence tag."""
        return encode_ss(self.ss)


def sign_intersection(sign_sequences: np.ndarray) -> np.ndarray:
    """Intersect tope signs: retain constants and zero every varying coordinate."""
    rows = np.asarray(sign_sequences, dtype=np.int8)
    if rows.ndim == 3 and rows.shape[1] == 1:
        rows = rows[:, 0, :]
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("sign_sequences must be a nonempty 2-D array")
    if np.any((rows != -1) & (rows != 0) & (rows != 1)):
        raise ValueError("sign sequences must contain only -1, 0, and 1")

    first = rows[0]
    common = np.all(rows == first, axis=0)
    result = np.where(common, first, 0).astype(np.int8, copy=False)
    return result.reshape(1, -1)


def _incident_shis(graph: nx.Graph[Polyhedron], root: Polyhedron) -> tuple[int, ...]:
    shis: set[int] = set()
    for _u, _v, data in graph.edges(root, data=True):
        shi = data.get("shi")
        if shi is None:
            raise CubicalConsistencyError("Dual-graph edge is missing 'shi' attribute.")
        shi_i = int(shi)
        if shi_i in shis:
            raise CubicalAmbiguityError(f"Top cell {root.tag!r} has multiple incident edges labeled shi={shi_i}.")
        shis.add(shi_i)
    return tuple(sorted(shis))


def _cube_cofaces(
    root: Polyhedron,
    zero_shis: tuple[int, ...],
    *,
    top_by_tag: dict[bytes, Polyhedron],
) -> tuple[Polyhedron, ...] | None:
    root_ss = np.asarray(root.ss_np, dtype=np.int8)
    cofaces: list[Polyhedron] = []
    for n_flips in range(len(zero_shis) + 1):
        for flipped in combinations(zero_shis, n_flips):
            ss = root_ss.copy()
            for shi in flipped:
                ss = np.asarray(flip_ss_at_shi(ss, int(shi)), dtype=np.int8).reshape(root_ss.shape)
            coface = top_by_tag.get(encode_ss(ss))
            if coface is None:
                return None
            cofaces.append(coface)
    return tuple(cofaces)


def _verify_cube_edges(
    cofaces: tuple[Polyhedron, ...],
    zero_shis: tuple[int, ...],
    *,
    graph: nx.Graph[Polyhedron],
    top_by_tag: dict[bytes, Polyhedron],
) -> None:
    coface_tags = {p.tag for p in cofaces}
    for coface in cofaces:
        ss = np.asarray(coface.ss_np, dtype=np.int8)
        for shi in zero_shis:
            neighbor_tag = encode_ss(flip_ss_at_shi(ss, int(shi)))
            neighbor = top_by_tag.get(neighbor_tag)
            if neighbor is None or neighbor_tag not in coface_tags:
                raise CubicalConsistencyError(f"Candidate cube at {coface.tag!r} is missing its shi={shi} flip.")
            if not graph.has_edge(coface, neighbor):
                raise CubicalConsistencyError(f"Candidate cube is missing dual edge shi={shi} on {coface.tag!r}.")
            edge_shi = graph.edges[coface, neighbor].get("shi")
            if edge_shi is None or int(edge_shi) != int(shi):
                raise CubicalConsistencyError(f"Candidate cube edge on {coface.tag!r} has label {edge_shi!r}, expected {shi}.")


def enumerate_covectors(
    top_cells: list[Polyhedron],
    graph: nx.Graph[Polyhedron],
    *,
    ambient_dim: int,
    top_dim: int | None = None,
) -> dict[int, dict[bytes, CovectorCell]]:
    """Enumerate every generic cell from labeled hypercubes in the tope graph.

    The returned mapping is keyed by cell dimension. A local codimension-``c``
    cell has a complete ``c``-cube of top-dimensional cofaces. Its covector
    contains the slice's fixed zeros plus the ``c`` varying cube directions.
    """
    if ambient_dim < 0:
        raise ValueError("ambient_dim must be nonnegative")
    if top_dim is None:
        top_dim = ambient_dim
    if top_dim < 0 or top_dim > ambient_dim:
        raise ValueError("top_dim must lie between zero and ambient_dim")
    top_by_tag = {p.tag: p for p in top_cells}
    if len(top_by_tag) != len(top_cells):
        raise CubicalAmbiguityError("Top-dimensional cells have duplicate sign-sequence tags.")
    if set(graph.nodes) != set(top_cells):
        raise CubicalConsistencyError("Dual graph nodes do not match the supplied top-dimensional cells.")

    by_dim: dict[int, dict[bytes, CovectorCell]] = {dim: {} for dim in range(top_dim + 1)}
    for root in top_cells:
        incident = _incident_shis(graph, root)
        fixed_zeros = set(int(i) for i in np.flatnonzero(np.asarray(root.ss_np).ravel() == 0))
        for local_codim in range(top_dim + 1):
            if len(incident) < local_codim:
                continue
            for cube_shis in combinations(incident, local_codim):
                cofaces = _cube_cofaces(root, cube_shis, top_by_tag=top_by_tag)
                if cofaces is None:
                    continue
                _verify_cube_edges(cofaces, cube_shis, graph=graph, top_by_tag=top_by_tag)
                ss = sign_intersection(np.stack([np.asarray(p.ss_np) for p in cofaces]))
                actual_zeros = tuple(int(i) for i in np.flatnonzero(ss.ravel() == 0))
                expected_zeros = tuple(sorted(fixed_zeros | set(cube_shis)))
                if actual_zeros != expected_zeros:
                    raise CubicalConsistencyError(
                        f"Cube directions {cube_shis!r} produced zero set {actual_zeros!r}, " + f"expected {expected_zeros!r}."
                    )

                cell = CovectorCell(
                    ss=ss,
                    coface_tags=frozenset(p.tag for p in cofaces),
                    zero_shis=actual_zeros,
                )
                dim = top_dim - local_codim
                previous = by_dim[dim].get(cell.tag)
                if previous is not None and previous.coface_tags != cell.coface_tags:
                    raise CubicalAmbiguityError(f"Covector {cell.tag!r} has inconsistent top-cell stars.")
                by_dim[dim][cell.tag] = cell
    return by_dim
