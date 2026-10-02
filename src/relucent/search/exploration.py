"""Exploration completion and certification for polyhedral complexes."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from relucent._internal.logging import logger
from relucent._internal.network_scale import default_polyhedron_bound
from relucent._internal.parallel import process_aware_cpu_count
from relucent.core.errors import IncompleteDualGraphError
from relucent.graph.incidence import set_contracted_shis, verify_contracted_shis
from relucent.verify.certify import CertifyLevel, certify_complex, verify_boundary_cell

if TYPE_CHECKING:
    from relucent.core.complex import Complex

__all__ = [
    "SearchResult",
    "explore_for_topology",
    "finalize_ambient_search",
    "finalize_boundary_complex",
    "generic_topology_start",
]


def finalize_ambient_search(cx: Complex, *, complete: bool, verify: bool) -> None:
    """Certify a fully explored ambient complex (dual-graph SHI sync + optional certify)."""
    if not complete:
        cx.set_exploration_state(complete=False, verified=False)
        if verify:
            # Partial complexes are opt-in (verify=False or an explicit cap).
            raise IncompleteDualGraphError(
                "Search incomplete: frontier not exhausted. Explore further or pass "
                + "max_polys to opt into a partial complex."
            )
        return
    # Build dual graph and resync top-cell _shis from it (repair=True, the default).
    graph = cx.get_dual_graph(require_complete=False)
    if verify:
        top_dim = max(int(p.dim) for p in cx)
        if top_dim == int(cx.dim):
            for poly in cx:
                if int(poly.dim) == top_dim and poly._shis is not None:
                    poly._shis_strict = True
    cx.set_exploration_state(complete=True, verified=False)
    if verify:
        logger.debug("ambient finalize: certifying %d polyhedra ...", len(cx))
        t_verify = time.perf_counter()
        certify_complex(cx, level=CertifyLevel.COMPLETE, graph=graph, record_state=True)
        logger.debug("ambient finalize: certification finished in %.1fs", time.perf_counter() - t_verify)
    else:
        cx.set_exploration_state(complete=True, verified=False)


def finalize_boundary_complex(
    cx: Complex,
    boundary_shi: int,
    *,
    bound: float | None = None,
    nworkers: int | None = None,
    verify: bool = True,
    **shis_kwargs: Any,
) -> None:
    """Ambient coface SHIs, dual graph, genericity, and invariant certification."""
    from relucent.search.boundary_search import _apply_ambient_boundary_shis

    if bound is None:
        bound = default_polyhedron_bound(cx._net)

    n_cells = len(cx)
    nw = nworkers or process_aware_cpu_count() or 1
    t0 = time.perf_counter()
    logger.debug(
        f"discover finalize: {n_cells} cells, ambient coface _shis ({nw} workers) ...",
    )
    ambient_shis_kwargs = {k: v for k, v in shis_kwargs.items() if k != "subset"}  # slice-only kwarg
    _apply_ambient_boundary_shis(
        cx,
        boundary_shi,
        bound=bound,
        nworkers=nw,
        **ambient_shis_kwargs,
    )
    logger.debug(
        "discover finalize: ambient coface _shis finished in " + f"{time.perf_counter() - t0:.1f}s",
    )
    if verify:
        for poly in cx:
            verify_boundary_cell(poly, boundary_shi)
    for poly in cx:
        poly._finite = None  # slice search may leave stale boundedness flags
        poly._finite_computed = False
    t2 = time.perf_counter()
    logger.debug("discover finalize: building dual graph ...")
    cx._dual_graph = cx.get_dual_graph(require_complete=verify)
    logger.debug(
        "discover finalize: dual graph finished in " + f"{time.perf_counter() - t2:.1f}s",
    )
    t4 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # genericity can warn on near-degenerate 1-cells
        cx.verify_arrangement_genericity()
    logger.debug(
        "discover finalize: genericity verify finished in " + f"{time.perf_counter() - t4:.1f}s",
    )
    set_contracted_shis(cx)  # boundary top cells need contracted SHIs before certify
    if verify:
        verify_contracted_shis(cx)
        certify_complex(cx, level=CertifyLevel.COMPLETE, graph=cx._dual_graph, record_state=True)
    else:
        cx.set_exploration_state(complete=True, verified=False)  # defers certification when verify=False


@dataclass(frozen=True)
class SearchResult:
    """What :meth:`~relucent.core.complex.Complex.searcher` (and ``bfs``/``dfs``/``random_walk``) report."""

    #: Largest number of hyperplane crossings from the start cell that was reached.
    depth: int
    #: Running mean of the number of facets (SHIs) per discovered cell.
    mean_facets: float
    #: Wall-clock seconds spent searching.
    search_time: float
    #: Neighbor computations that failed, as ``(polyhedron, error)`` records. True phantom
    #: neighbors (empty flip patterns) are included but don't make the search incomplete.
    bad_shi_computations: list[Any] = field(default_factory=list)
    #: Whether the search ran until no unexplored neighbors were left (not stopped by
    #: ``max_polys``, ``max_depth``, or a failed neighbor computation).
    complete: bool = False
    #: Whether certification passed; ``None`` if it didn't run.
    verified: bool | None = None


def generic_topology_start(cplx: Complex, *, seed: int = 0) -> np.ndarray:
    """Return an interior start point that does not lie on any hyperplane."""
    rng = np.random.default_rng(seed)
    for _ in range(32):
        start = rng.normal(size=(1, cplx.dim))
        if not (cplx.point2ss(start) == 0).any():
            return np.asarray(start, dtype=np.float64).reshape(-1)
    raise RuntimeError("could not find generic start for topology exploration")


def explore_for_topology(
    cplx: Complex,
    start: np.ndarray | None = None,
    *,
    seed: int = 0,
    max_polys: float = float("inf"),
    nworkers: int | None = None,
) -> None:
    """BFS from ``start`` and require a complete, verified ambient complex.

    When ``start`` is None, :func:`generic_topology_start` picks an interior point.
    """
    if start is None:
        start = generic_topology_start(cplx, seed=seed)
    cplx.bfs(np.asarray(start, dtype=np.float64).reshape(1, -1), max_polys=max_polys, nworkers=nworkers)
    if cplx.complete is not True:
        raise IncompleteDualGraphError("explore_for_topology hit max_polys before completing")
    if cplx.verified is not True:
        raise IncompleteDualGraphError("explore_for_topology expected a verified complex")
