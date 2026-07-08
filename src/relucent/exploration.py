"""Exploration completion and certification for polyhedral complexes."""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from relucent._network_scale import default_polyhedron_bound
from relucent.utils import process_aware_cpu_count

if TYPE_CHECKING:
    from relucent.complex import Complex

__all__ = [
    "explore_for_topology",
    "finalize_ambient_search",
    "finalize_boundary_complex",
    "generic_topology_start",
    "search_stats_dict",
]


def finalize_ambient_search(cx: Complex, *, complete: bool, verify: bool) -> None:
    """Certify a fully explored ambient complex (dual-graph SHI sync + optional verify)."""
    from relucent.complex import IncompleteDualGraphError  # lazy: breaks complex ↔ exploration cycle

    if not complete:
        cx.set_exploration_state(complete=False, verified=False)
        if verify:
            # Partial complexes are opt-in (verify=False or an explicit cap).
            raise IncompleteDualGraphError(
                "Search incomplete: frontier not exhausted. Explore further or pass "
                + "max_polys to opt into a partial complex."
            )
        return
    # Rebuild dual-graph edges and sync top-cell _shis from them.
    cx.get_dual_graph(verbose=False, require_complete=False, verify=False, cubical=False)
    cx.set_exploration_state(complete=True, verified=False)
    if verify:
        from relucent.verify import verify_complex

        verify_complex(cx, level="fast", record_state=True)
    else:
        cx.set_exploration_state(complete=True, verified=False)


def finalize_boundary_complex(
    cx: Complex,
    boundary_shi: int,
    *,
    bound: float | None = None,
    nworkers: int | None = None,
    verbose: bool = False,
    verify: bool = True,
    **shis_kwargs: Any,
) -> None:
    """Ambient coface SHIs, dual graph, genericity, and invariant verification."""
    from relucent import meta_graph as mg
    from relucent.boundary_search import _apply_ambient_boundary_shis, _phase_log
    from relucent.verify import verify_boundary_cell, verify_complex

    if bound is None:
        bound = default_polyhedron_bound(cx._net)

    n_cells = len(cx)
    nw = nworkers or process_aware_cpu_count() or 1
    t0 = time.perf_counter()
    _phase_log(
        f"discover finalize: {n_cells} cells, ambient coface _shis ({nw} workers) ...",
        verbose=verbose,
    )
    ambient_shis_kwargs = {k: v for k, v in shis_kwargs.items() if k != "subset"}  # slice-only kwarg
    _apply_ambient_boundary_shis(
        cx,
        boundary_shi,
        bound=bound,
        nworkers=nw,
        verbose=verbose,
        **ambient_shis_kwargs,
    )
    _phase_log(
        "discover finalize: ambient coface _shis finished in " + f"{time.perf_counter() - t0:.1f}s",
        verbose=verbose,
    )
    if verify:
        for poly in cx:
            verify_boundary_cell(poly, boundary_shi)
    for poly in cx:
        poly._finite = None  # slice search may leave stale boundedness flags
        poly._finite_computed = False
    t2 = time.perf_counter()
    _phase_log("discover finalize: building dual graph ...", verbose=verbose)
    cx._dual_graph = cx.get_dual_graph(verbose=verbose, require_complete=verify, verify=verify, cubical=False)
    _phase_log(
        "discover finalize: dual graph finished in " + f"{time.perf_counter() - t2:.1f}s",
        verbose=verbose,
    )
    t4 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # genericity can warn on near-degenerate 1-cells
        cx.verify_arrangement_genericity()
    _phase_log(
        "discover finalize: genericity verify finished in " + f"{time.perf_counter() - t4:.1f}s",
        verbose=verbose,
    )
    mg.assign_contracted_shis(cx)  # boundary top cells need contracted SHIs before verify
    if verify:
        verify_complex(cx, level="fast", graph=cx._dual_graph, record_state=True)
    else:
        cx.set_exploration_state(complete=True, verified=False)  # discoverer defers certification when verify=False


def search_stats_dict(
    *,
    depth: int,
    rolling_average: float,
    search_time: float,
    bad_shi_computations: list[Any],
    complete: bool,
    verified: bool | None = None,
) -> dict[str, Any]:
    """Standard return payload for :func:`~relucent.search.searcher` and boundary BFS."""
    out: dict[str, Any] = {
        "Search Depth": depth,
        "Avg # Facets Uncorrected": rolling_average,
        "Search Time": search_time,
        "Bad SHI Computations": bad_shi_computations,
        "Complete": complete,
    }
    if verified is not None:
        out["Verified"] = verified
    return out


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
    from relucent.complex import IncompleteDualGraphError

    if start is None:
        start = generic_topology_start(cplx, seed=seed)
    kwargs: dict[str, object] = {
        "start": np.asarray(start, dtype=np.float64).reshape(1, -1),
        "max_polys": max_polys,
    }
    if nworkers is not None:
        kwargs["nworkers"] = nworkers
    cplx.bfs(**kwargs)
    if cplx.complete is not True:
        raise IncompleteDualGraphError("explore_for_topology hit max_polys before completing")
    if cplx.verified is not True:
        raise IncompleteDualGraphError("explore_for_topology expected a verified complex")
