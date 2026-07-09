"""Invariant checks for polyhedral complexes."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from relucent import meta_graph as mg
from relucent._logging import logger
from relucent.poly import Polyhedron
from relucent.utils import encode_ss, flip_ss_at_shi, get_mp_context, process_aware_cpu_count
from relucent.worker_context import get_worker_context, set_worker_context

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


def _iter_top_dim_polys(cplx: Complex, top_dim: int) -> Iterable[Polyhedron]:
    """Yield top-dimensional cells in stable complex iteration order."""
    for poly in cplx:
        if int(poly.dim) == top_dim:
            yield poly


def _verify_lp_neighbors_for_poly(
    poly: Polyhedron,
    *,
    top_tags: frozenset[bytes],
    bound: float,
) -> tuple[str | None, list[str]]:
    """Return a per-poly SHI recompute error or missing-neighbor diagnostics."""
    from relucent.calculations import get_shis

    try:
        lp_shis = get_shis(poly, bound=float(bound), strict=True)
    except ValueError as exc:
        return str(exc), []

    missing: list[str] = []
    ss = np.asarray(poly.ss_np, dtype=np.int8)
    for shi in lp_shis:
        shi_i = int(shi)
        if int(ss.ravel()[shi_i]) == 0:
            continue  # inactive hyperplane on this cell
        neighbor_ss = flip_ss_at_shi(ss, shi_i)
        if encode_ss(neighbor_ss) not in top_tags:
            missing.append(f"LP facet shi={shi_i} on {poly!r} has no neighbor in complex")
            continue
    return None, missing


def _missing_lp_neighbors_for_shis(
    poly: Polyhedron,
    *,
    shis: Iterable[int],
    top_tags: frozenset[bytes],
) -> list[str]:
    """Return missing-neighbor diagnostics for a trusted SHI list."""
    missing: list[str] = []
    ss = np.asarray(poly.ss_np, dtype=np.int8)
    for shi in shis:
        shi_i = int(shi)
        if int(ss.ravel()[shi_i]) == 0:
            continue
        neighbor_ss = flip_ss_at_shi(ss, shi_i)
        if encode_ss(neighbor_ss) not in top_tags:
            missing.append(f"LP facet shi={shi_i} on {poly!r} has no neighbor in complex")
    return missing


def _poly_has_strict_cached_shis(poly: Polyhedron) -> bool:
    """Whether ``poly._shis`` came from a strict SHI solve and can be trusted directly."""
    return poly._shis is not None and bool(getattr(poly, "_shis_strict", False))


def _verify_lp_neighbors_worker(
    task: tuple[np.ndarray, float, frozenset[bytes]],
) -> tuple[str, str | None, list[str]]:
    """Recompute LP SHIs for one top cell inside a worker process."""
    ss, bound, top_tags = task
    ctx = get_worker_context()
    poly = Polyhedron(ctx.net, ss, bound=float(bound))
    err, missing = _verify_lp_neighbors_for_poly(poly, top_tags=top_tags, bound=float(bound))
    return repr(poly), err, missing


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


def verify_lp_flip_neighbors_in_complex(cplx: Complex, *, nworkers: int | None = None) -> None:
    """Every LP facet on a top cell must flip to a same-dimension neighbor in the complex."""
    from relucent._network_scale import default_polyhedron_bound
    from relucent.complex import IncompleteDualGraphError

    if len(cplx) == 0:
        return
    top_dim = max(int(p.dim) for p in cplx)
    if top_dim != int(cplx.dim):
        return  # contracted slices skip ambient LP completeness
    top_polys = list(_iter_top_dim_polys(cplx, top_dim))
    top_tags = frozenset(poly.tag for poly in top_polys)
    tasks: list[tuple[np.ndarray, float, frozenset[bytes]]] = []
    serial_tasks: list[tuple[Polyhedron, float]] = []
    for poly in top_polys:
        bound = poly.bound
        if bound is None:
            bound = default_polyhedron_bound(cplx._net)
        if bound is None:
            continue  # no bound -> can't run facet LP
        bound_f = float(bound)
        if _poly_has_strict_cached_shis(poly):
            serial_tasks.append((poly, bound_f))
        else:
            tasks.append((np.asarray(poly.ss_np, dtype=np.int8), bound_f, top_tags))

    total_cells = len(serial_tasks) + len(tasks)
    if total_cells == 0:
        return

    # Use one Gurobi thread per worker to avoid multiplying solver threads by process count.
    requested_workers = nworkers or process_aware_cpu_count() or 1
    worker_count = max(1, min(requested_workers, len(tasks))) if tasks else 1
    if tasks:
        logger.info(
            "verify_lp_flip_neighbors_in_complex: certifying %d top cells "
            + "(%d trusted cached SHIs, %d LP recertify, %d worker%s)",
            total_cells,
            len(serial_tasks),
            len(tasks),
            worker_count,
            "" if worker_count == 1 else "s",
        )
    else:
        logger.info(
            "verify_lp_flip_neighbors_in_complex: certifying %d top cells " + "(trusted cached SHIs only, no LP recertify)",
            total_cells,
        )
    t_verify = time.perf_counter()
    missing: list[str] = []
    pbar = tqdm(
        desc="verify_lp_flip_neighbors_in_complex",
        total=total_cells,
        mininterval=1,
    )
    if serial_tasks:
        for poly, _bound in serial_tasks:
            assert poly._shis is not None
            missing.extend(_missing_lp_neighbors_for_shis(poly, shis=poly._shis, top_tags=top_tags))
            pbar.update()
    if tasks and worker_count == 1:
        for ss, bound, _top_tags in tasks:
            poly = Polyhedron(cplx._net, ss, bound=bound)
            err, poly_missing = _verify_lp_neighbors_for_poly(poly, top_tags=top_tags, bound=bound)
            if err is not None:
                pbar.close()
                raise IncompleteDualGraphError(
                    "Dual graph completeness could not be certified from LP facets: "
                    + f"failed to recompute SHIs on {poly!r}: {err}"
                )
            missing.extend(poly_missing)
            pbar.update()
    elif tasks:
        with get_mp_context().Pool(
            worker_count,
            initializer=set_worker_context,
            initargs=(cplx._net, False, 1),
        ) as pool:
            for poly_repr, err, poly_missing in pool.imap_unordered(_verify_lp_neighbors_worker, tasks):
                if err is not None:
                    pbar.close()
                    raise IncompleteDualGraphError(
                        "Dual graph completeness could not be certified from LP facets: "
                        + f"failed to recompute SHIs on {poly_repr}: {err}"
                    )
                missing.extend(poly_missing)
                pbar.update()
    pbar.close()

    logger.info(
        "verify_lp_flip_neighbors_in_complex: certified %d top cells in %.1fs",
        total_cells,
        time.perf_counter() - t_verify,
    )

    if missing:
        missing.sort()
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
    logger.info("verify_complex: SHI flip symmetry ...")
    t_stage = time.perf_counter()
    verify_shi_flip_symmetry(cplx)
    logger.info("verify_complex: SHI flip symmetry finished in %.1fs", time.perf_counter() - t_stage)
    logger.info("verify_complex: dual-graph edge certification ...")
    t_stage = time.perf_counter()
    verify_dual_graph_edges(g, cplx)
    logger.info("verify_complex: dual-graph edge certification finished in %.1fs", time.perf_counter() - t_stage)
    if level == "fast" and cplx.complete is True:
        logger.info("verify_complex: LP facet completeness certification ...")
        t_stage = time.perf_counter()
        verify_lp_flip_neighbors_in_complex(cplx)  # skip expensive LPs on partial complexes
        logger.info("verify_complex: LP facet completeness finished in %.1fs", time.perf_counter() - t_stage)
    if level == "full":
        for poly in cplx:
            if poly._shis is not None:
                verify_shi_geometry(poly)
    if record_state:
        complete = True if cplx._complete is None else bool(cplx._complete)
        cplx.set_exploration_state(complete=complete, verified=True)
    else:
        cplx._verified = True  # legacy path for direct callers
