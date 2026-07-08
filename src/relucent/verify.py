"""Invariant checks for polyhedral complexes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np

from relucent import meta_graph as mg
from relucent.poly import Polyhedron
from relucent.utils import flip_ss_at_shi

if TYPE_CHECKING:
    from relucent.complex import Complex

__all__ = [
    "ComplexNotCompleteError",
    "ComplexNotVerifiedError",
    "DualGraphAsymmetricEdgeError",
    "ShiFlipInvariantError",
    "ShiProofError",
    "verify_boundary_cell",
    "verify_complex",
    "verify_dual_graph_edges",
    "verify_lp_flip_neighbors_in_complex",
    "verify_shi_flip_symmetry",
    "verify_shi_geometry",
]


class ComplexNotCompleteError(RuntimeError):
    """Topology routine requires a fully explored complex."""


class ComplexNotVerifiedError(RuntimeError):
    """Topology routine requires a complex that passed invariant verification."""


class DualGraphAsymmetricEdgeError(ValueError):
    """Dual-graph edge is not supported by both endpoint SHI lists."""


class ShiFlipInvariantError(ValueError):
    """A cached SHI lacks a symmetric flip neighbor in the complex."""


class ShiProofError(ValueError):
    """SHI facet proof failed under strict verification."""


def verify_shi_flip_symmetry(cplx: Complex) -> None:
    """Every top-cell SHI must flip to a same-dimension neighbor that lists the SHI too."""
    if len(cplx) == 0:
        return
    top_dim = max(int(p.dim) for p in cplx)
    top_cells = [p for p in cplx if int(p.dim) == top_dim]
    for poly in top_cells:
        shis = poly.shis
        for shi in shis:
            ss = np.asarray(poly.ss_np)
            if int(ss.ravel()[int(shi)]) == 0:
                raise ShiFlipInvariantError(f"SHI {shi} on {poly!r} has ss[{shi}]=0.")
            neighbor_ss = flip_ss_at_shi(ss, int(shi))
            try:
                neighbor = cplx[neighbor_ss]
            except KeyError as exc:
                raise ShiFlipInvariantError(f"SHI {shi} on {poly!r} has no flip neighbor in the complex.") from exc
            if int(neighbor.dim) != top_dim:
                raise ShiFlipInvariantError(
                    f"SHI {shi} on {poly!r} flips to wrong dimension {neighbor.dim} (expected {top_dim})."
                )
            if int(shi) not in neighbor.shis:
                raise ShiFlipInvariantError(f"Asymmetric SHI {shi}: listed on {poly!r} but not on flip neighbor {neighbor!r}.")


def verify_dual_graph_edges(
    graph: nx.Graph[Polyhedron],
    cplx: Complex,
    *,
    top_dim: int | None = None,
) -> None:
    """Bidirectional SHI support and cubical face-tag consistency on dual edges."""
    if graph.number_of_edges() == 0:
        return
    if top_dim is None:
        top_dim = max(int(p.dim) for p in cplx)
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
    mg.verify_dual_graph_cubical(top_cells, graph, top_dim=top_dim)


def verify_boundary_cell(poly: Polyhedron, boundary_shi: int) -> None:
    """Both ambient cofaces of a boundary top cell must be nonempty."""
    from relucent.boundary_search import _both_ambient_cofaces_feasible

    if not _both_ambient_cofaces_feasible(poly, boundary_shi):
        raise ValueError(f"Boundary cell {poly!r} fails ambient coface feasibility at shi={boundary_shi}.")


def verify_shi_geometry(poly: Polyhedron, *, bound: float | None = None) -> None:
    """Recompute SHIs and require the cached list to match."""
    from relucent.calculations import get_shis

    if poly._shis is None:
        raise ShiProofError(f"Polyhedron {poly!r} has no cached _shis.")
    if bound is None:
        bound = poly.bound
    if bound is None:
        from relucent._network_scale import default_polyhedron_bound

        if poly._net is None:
            raise ShiProofError(f"Polyhedron {poly!r} has no network for bound estimation.")
        bound = default_polyhedron_bound(poly._net)
    fresh = get_shis(poly, bound=float(bound), strict=True)
    if set(fresh) != set(poly._shis):
        raise ShiProofError(f"Cached _shis {sorted(poly._shis)} != recomputed {sorted(fresh)} on {poly!r}.")


def verify_lp_flip_neighbors_in_complex(cplx: Complex) -> None:
    """Every LP facet on a top cell must flip to a same-dimension neighbor in the complex."""
    from relucent._network_scale import default_polyhedron_bound
    from relucent.calculations import get_shis
    from relucent.complex import IncompleteDualGraphError

    if len(cplx) == 0:
        return
    top_dim = max(int(p.dim) for p in cplx)
    if top_dim != int(cplx.dim):
        return  # contracted slices skip ambient LP completeness
    missing: list[str] = []
    for poly in cplx:
        if int(poly.dim) != top_dim:
            continue
        bound = poly.bound
        if bound is None:
            bound = default_polyhedron_bound(cplx._net)
        if bound is None:
            continue  # no bound → can't run facet LP
        try:
            lp_shis = get_shis(poly, bound=float(bound), strict=False)
        except ValueError:
            continue  # infeasible/degenerate cell, not a missing-neighbor signal
        for shi in lp_shis:
            shi_i = int(shi)
            ss = np.asarray(poly.ss_np)
            if int(ss.ravel()[shi_i]) == 0:
                continue  # inactive hyperplane on this cell
            neighbor_ss = flip_ss_at_shi(ss, shi_i)
            try:
                neighbor = cplx[neighbor_ss]
            except KeyError:
                missing.append(f"LP facet shi={shi_i} on {poly!r} has no neighbor in complex")
                continue
            if int(neighbor.dim) != top_dim:
                missing.append(f"LP facet shi={shi_i} on {poly!r} flips to dim {neighbor.dim} (expected {top_dim})")
    if missing:
        raise IncompleteDualGraphError(
            "Dual graph is incomplete relative to LP facets: "
            + f"{len(missing)} missing neighbor(s). "
            + missing[0]
            + (" ..." if len(missing) > 1 else "")
        )


def verify_complex(
    cplx: Complex,
    *,
    level: Literal["fast", "full"] = "fast",
    graph: nx.Graph[Polyhedron] | None = None,
    record_state: bool = False,
) -> None:
    """Run tiered invariant checks; sets ``cplx._verified`` on success.

    When ``record_state`` is True, also updates :meth:`~relucent.complex.Complex.set_exploration_state`
    so callers do not need a separate state write.
    """
    if len(cplx) == 0:
        if record_state:
            complete = True if cplx._complete is None else bool(cplx._complete)
            cplx.set_exploration_state(complete=complete, verified=True)
        else:
            cplx._verified = True
        return

    g = (
        graph
        if graph is not None
        else cplx.get_dual_graph(
            verbose=False,
            require_complete=cplx.complete is True,
            verify=False,
            cubical=False,
        )
    )
    verify_shi_flip_symmetry(cplx)
    verify_dual_graph_edges(g, cplx)
    if level == "fast" and cplx.complete is True:
        verify_lp_flip_neighbors_in_complex(cplx)  # skip expensive LPs on partial complexes
    if level == "full":
        for poly in cplx:
            if poly._shis is not None:
                verify_shi_geometry(poly)
    if record_state:
        complete = True if cplx._complete is None else bool(cplx._complete)
        cplx.set_exploration_state(complete=complete, verified=True)
    else:
        cplx._verified = True  # legacy path for direct callers
