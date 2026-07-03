"""Slice-restricted BFS and MIP-driven boundary complex discovery."""

from __future__ import annotations

import time
import warnings
from collections import deque
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from tqdm.auto import tqdm

import relucent.config as cfg
from relucent._logging import logger
from relucent.boundary_mip import _is_top_boundary_ss, price_boundary_witness
from relucent.calculations import get_shis
from relucent.complex import IncompleteDualGraphError
from relucent.poly import Polyhedron
from relucent.search import (
    SEARCH_REQUIRED_GEOMETRY_PROPERTIES,
    _cancel_pending_neighbor,
    retain_geometry_caches,
    search_calculations,
)
from relucent.utils import BlockingQueue, encode_ss, flip_ss_at_shi, get_mp_context, process_aware_cpu_count
from relucent.worker_context import set_worker_context

if TYPE_CHECKING:
    from relucent.complex import Complex
    from relucent.model import ReLUNetwork

__all__ = [
    "BoundaryDiscoveryStats",
    "boundary_searcher",
    "discover_boundary_complex",
    "discover_boundary_component",
]


class BoundaryDiscoveryStats(dict[str, Any]):
    """Statistics returned by :func:`discover_boundary_complex`."""


def _discovery_log(msg: str, *, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


def _phase_log(msg: str, *, verbose: bool) -> None:
    """Emit progress to logger and optionally stdout (for long finalize / post steps)."""
    logger.info(msg)
    if verbose:
        print(msg, flush=True)


def _coface_intersection_shis(
    coface_a: Polyhedron,
    coface_b: Polyhedron,
    crossing_shi: int,
    *,
    boundary_shi: int,
) -> list[int]:
    """SHI list from coface intersection, matching :meth:`Complex._codim_one_face_kwargs`."""
    drop = {int(crossing_shi), int(boundary_shi)}
    return sorted(int(s) for s in (set(coface_a.shis) & set(coface_b.shis) - drop))


def _apply_boundary_coface_metadata(
    coface_a: Polyhedron,
    coface_b: Polyhedron,
    crossing_shi: int,
    *,
    boundary_shi: int,
) -> None:
    """Apply coface-intersection ``_shis`` to both slice cofaces."""
    edge_shis = _coface_intersection_shis(
        coface_a,
        coface_b,
        crossing_shi,
        boundary_shi=boundary_shi,
    )
    coface_b._shis = edge_shis


def _ambient_lift_polyhedra(
    poly: Polyhedron,
    boundary_shi: int,
) -> tuple[Polyhedron, Polyhedron]:
    ss = np.asarray(poly.ss_np, dtype=np.int8).copy()
    bshi = int(boundary_shi)
    ss_pos = ss.copy()
    ss_neg = ss.copy()
    ss_pos.ravel()[bshi] = 1
    ss_neg.ravel()[bshi] = -1
    net = poly._net
    if net is None:
        raise ValueError("boundary cell missing network reference for ambient lift")
    return Polyhedron(net, ss_pos), Polyhedron(net, ss_neg)


def _ambient_coface_shis_for_boundary_cell(
    poly: Polyhedron,
    boundary_shi: int,
    *,
    bound: float | None = None,
    **shis_kwargs: Any,
) -> list[int]:
    """``_shis`` for a boundary top cell, matching :meth:`Complex.get_boundary_cells`.

    Lifts ``poly`` off the bent hyperplane (``ss[boundary_shi] = ±1``) and takes
    the coface SHI intersection of the two ambient top cells, as in
    :meth:`Complex._codim_one_face_kwargs`.
    """
    ss = np.asarray(poly.ss_np, dtype=np.int8).copy()
    bshi = int(boundary_shi)
    if bound is None:
        bound = cfg.DEFAULT_SEARCH_BOUND
    if int(ss.ravel()[bshi]) != 0:
        raise ValueError(f"Expected ss[{bshi}]=0 on boundary cell, got {ss!r}")
    net = poly._net
    if net is None:
        raise ValueError("boundary cell missing network reference for ambient lift")
    p_pos, p_neg = _ambient_lift_polyhedra(poly, bshi)
    try:
        shis_pos = get_shis(p_pos, bound=bound, **shis_kwargs)
        shis_neg = get_shis(p_neg, bound=bound, **shis_kwargs)
    except Exception:
        saved = poly._shis
        poly._shis = None
        try:
            return [int(s) for s in get_shis(poly, bound=bound, **shis_kwargs) if int(s) != bshi]
        finally:
            poly._shis = saved
    return sorted(int(s) for s in (set(shis_pos) & set(shis_neg) - {bshi}))


def _ambient_boundary_metadata_for_cell(
    poly: Polyhedron,
    boundary_shi: int,
    *,
    bound: float | None = None,
    **shis_kwargs: Any,
) -> tuple[list[int], Any]:
    """Return ``(_shis, halfspaces)`` matching :meth:`Complex.get_boundary_cells`."""
    bshi = int(boundary_shi)
    if bound is None:
        bound = cfg.DEFAULT_SEARCH_BOUND
    shis = _ambient_coface_shis_for_boundary_cell(
        poly,
        bshi,
        bound=bound,
        **shis_kwargs,
    )
    p_pos, _ = _ambient_lift_polyhedra(poly, bshi)
    return shis, p_pos.halfspaces


def _apply_ambient_boundary_shis(
    cx: Complex,
    boundary_shi: int,
    *,
    bound: float | None = None,
    nworkers: int | None = None,
    verbose: bool = False,
    **shis_kwargs: Any,
) -> None:
    """Assign coface-intersection ``_shis`` to every top boundary cell."""
    if len(cx) == 0:
        return
    nw = nworkers or process_aware_cpu_count() or 1
    polys = list(cx)
    if nw <= 1 or len(polys) < 32:
        for poly in polys:
            shis, halfspaces = _ambient_boundary_metadata_for_cell(
                poly,
                boundary_shi,
                bound=bound,
                **shis_kwargs,
            )
            poly._shis = shis
            poly._halfspaces = halfspaces
        return

    from relucent.utils import get_mp_context
    from relucent.worker_context import set_worker_context

    tasks = [(poly.ss_np, poly.tag) for poly in polys]
    tag_to_poly = {poly.tag: poly for poly in polys}
    with get_mp_context().Pool(nw, initializer=set_worker_context, initargs=(cx._net, False)) as pool:
        results = pool.starmap(
            partial(
                _ambient_coface_shis_worker,
                boundary_shi=boundary_shi,
                bound=bound,
                shis_kwargs=shis_kwargs,
            ),
            tasks,
        )
    for tag, shis, halfspaces in results:
        tag_to_poly[tag]._shis = shis
        tag_to_poly[tag]._halfspaces = halfspaces
    if verbose:
        _phase_log(
            "discover finalize: ambient coface _shis for " + f"{len(polys)} cells ({nw} workers)",
            verbose=True,
        )


def _ambient_coface_shis_worker(
    ss: np.ndarray,
    tag: bytes,
    *,
    boundary_shi: int,
    bound: float | None,
    shis_kwargs: dict[str, Any],
) -> tuple[bytes, list[int], Any]:
    from relucent.worker_context import get_worker_context

    ctx = get_worker_context()
    poly = Polyhedron(ctx.net, ss)
    shis, halfspaces = _ambient_boundary_metadata_for_cell(
        poly,
        boundary_shi,
        bound=bound,
        **shis_kwargs,
    )
    return tag, shis, halfspaces


def _finalize_boundary_complex(
    cx: Complex,
    boundary_shi: int,
    *,
    bound: float | None = None,
    nworkers: int | None = None,
    verbose: bool = False,
    **shis_kwargs: Any,
) -> None:
    """Complete ambient SHIs, dual graph, filter SHIs, and verify (matches baseline path)."""
    from relucent import meta_graph as mg

    n_cells = len(cx)
    nw = nworkers or process_aware_cpu_count() or 1
    t0 = time.perf_counter()
    _phase_log(
        "discover finalize: " + f"{n_cells} cells, ambient coface _shis ({nw} workers) ...",
        verbose=verbose,
    )
    ambient_shis_kwargs = {k: v for k, v in shis_kwargs.items() if k != "subset"}
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
    for poly in cx:
        poly._finite = None
        poly._finite_computed = False
    t2 = time.perf_counter()
    _phase_log("discover finalize: building dual graph ...", verbose=verbose)
    cx._dual_graph = cx.get_dual_graph(verbose=verbose, require_complete=True, cubical=False)
    _phase_log(
        "discover finalize: dual graph finished in " + f"{time.perf_counter() - t2:.1f}s",
        verbose=verbose,
    )
    t3 = time.perf_counter()
    mg.filter_complex_shis_by_flip_neighbor(cx)
    _phase_log(
        "discover finalize: SHI filter finished in " + f"{time.perf_counter() - t3:.1f}s",
        verbose=verbose,
    )
    t4 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cx.verify_arrangement_genericity()
    _phase_log(
        "discover finalize: genericity verify finished in " + f"{time.perf_counter() - t4:.1f}s",
        verbose=verbose,
    )


def boundary_searcher(
    cx: Complex,
    boundary_shi: int,
    start: Polyhedron,
    max_depth: float = float("inf"),
    max_polys: float = float("inf"),
    bound: float | None = None,
    nworkers: int | None = None,
    verbose: int | None = None,
    geometry_properties: Iterable[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """BFS on a bent hyperplane slice with ``ss[boundary_shi] = 0`` fixed.

    Args:
        cx: Boundary complex under construction (typically empty).
        boundary_shi: Global SHI index pinned to zero on every discovered cell.
        start: Witness polyhedron with exactly one zero at ``boundary_shi``.
        max_depth: Maximum flip depth from the start cell.
        max_polys: Maximum cells to discover in this component.
        bound: Gurobi box bound for SHI LPs.
        nworkers: Worker process count.
        verbose: Progress verbosity.
        geometry_properties: Optional geometry caches (default topology-only).
        **kwargs: Forwarded to :func:`~relucent.calculations.get_shis`.

    Returns:
        Search statistics dict (same keys as :func:`~relucent.search.searcher`).

    Raises:
        ValueError: If ``start`` is not a top-dimensional boundary cell.
        IncompleteDualGraphError: If exploration stops before the dual graph is complete.
    """
    if verbose is None:
        verbose = cfg.VERBOSE
    if bound is None:
        bound = cfg.DEFAULT_SEARCH_BOUND
    if not _is_top_boundary_ss(start.ss_np, boundary_shi):
        raise ValueError(f"Start sign sequence must have ss[{boundary_shi}]=0 as its only zero entry; got {start.ss_np!r}")

    search_props = (
        SEARCH_REQUIRED_GEOMETRY_PROPERTIES
        if geometry_properties is None
        else tuple(dict.fromkeys((*SEARCH_REQUIRED_GEOMETRY_PROPERTIES, *geometry_properties)))
    )
    shi_subset = [j for j in range(cx.n) if j != boundary_shi]
    shis_kwargs = dict(kwargs)
    shis_kwargs.setdefault("subset", shi_subset)

    pending_neighbors: dict[bytes, list[tuple[int, int]]] = {}
    nworkers = nworkers or process_aware_cpu_count()
    if verbose:
        logger.info("boundary_searcher running on %d workers (boundary_shi=%d)", nworkers, boundary_shi)

    queue = BlockingQueue(
        queue_class=deque,
        pop=cast(Any, deque.popleft),
        push=cast(Any, deque.append),
    )

    start = cx.add_polyhedron(start, check_exists=False)
    result = get_shis(start, bound=bound, **shis_kwargs)
    assert isinstance(result, list)
    start._shis = [s for s in result if int(s) != boundary_shi]
    retain_geometry_caches(start, search_props)
    start_index = cx.ssm[start.ss_np]

    for shi in start.shis:
        if shi == boundary_shi:
            continue
        ss = flip_ss_at_shi(start.ss_np, shi)
        if not _is_top_boundary_ss(ss, boundary_shi):
            continue
        pending_neighbors[encode_ss(ss)] = []
        queue.push((ss, shi, 1, start_index))

    rolling_average = len(start.shis)
    bad_shi_computations: list[Any] = []
    pbar = tqdm(
        desc="Boundary Search",
        mininterval=5,
        total=max_polys if max_polys != float("inf") else None,
        disable=not verbose,
    )
    pbar.update(n=1)
    pbar.get_lock().locks = []

    unprocessed = len(queue)
    depth = 0
    search_time = 0.0
    t_search = time.perf_counter()

    if unprocessed > 0:
        with get_mp_context().Pool(nworkers, initializer=set_worker_context, initargs=(cx._net, False)) as pool:
            try:
                for p, shi, depth, node_index in pool.imap_unordered(
                    partial(
                        search_calculations,
                        bound=bound,
                        geometry_properties=search_props,
                        **shis_kwargs,
                    ),
                    queue,
                ):
                    unprocessed -= 1
                    node = cast(Polyhedron, cx.index2poly[node_index])
                    if not isinstance(p, Polyhedron):
                        bad_shi_computations.append((node, shi, depth, str(p)))
                        _cancel_pending_neighbor(cx, pending_neighbors, node, shi)
                        if unprocessed == 0 or len(cx) >= max_polys:
                            break
                        continue

                    if not _is_top_boundary_ss(p.ss_np, boundary_shi):
                        _cancel_pending_neighbor(cx, pending_neighbors, node, shi)
                        if unprocessed == 0 or len(cx) >= max_polys:
                            break
                        continue

                    if p._net is None:
                        p._net = cx._net

                    p = cx.add_polyhedron(p)
                    pending_neighbors.pop(encode_ss(p.ss_np), None)

                    if depth < max_depth:
                        poly_index = cx.ssm[p.ss_np]
                        for new_shi in p.shis:
                            if new_shi in (boundary_shi, shi) or len(cx) >= max_polys:
                                continue
                            ss = flip_ss_at_shi(p.ss_np, new_shi)
                            if not _is_top_boundary_ss(ss, boundary_shi):
                                continue
                            tag = encode_ss(ss)
                            if tag in cx.tag2poly:
                                continue
                            if tag not in pending_neighbors:
                                pending_neighbors[tag] = []
                                queue.push((ss, new_shi, depth + 1, poly_index))
                                unprocessed += 1
                            else:
                                pending_neighbors[tag].append((poly_index, new_shi))

                    pbar.update(n=len(cx) - pbar.n)
                    rolling_average = (rolling_average * (len(cx) - 1) + len(p.shis)) / len(cx)
                    pbar.set_postfix_str(
                        f"Depth: {depth}  Unprocessed: {unprocessed} Mistakes: {len(bad_shi_computations)}"
                        + f" Mean Facets: {rolling_average:.2f}",
                        refresh=False,
                    )
                    if unprocessed == 0 or len(cx) >= max_polys:
                        break
            finally:
                queue.close()
                search_time = time.perf_counter() - t_search
                pbar.close()
                pool.terminate()
                pool.join()
    else:
        queue.close()
        search_time = time.perf_counter() - t_search
        pbar.close()

    complete = unprocessed == 0 and len(cx) < max_polys
    if not complete:
        raise IncompleteDualGraphError(
            f"Boundary slice search incomplete for shi={boundary_shi}: "
            + f"unprocessed={unprocessed}, cells={len(cx)}, max_polys={max_polys}"
        )

    for poly in cx:
        poly._finite = None
        poly._finite_computed = False
    cx.get_dual_graph(verbose=bool(verbose), require_complete=True, cubical=False)
    return {
        "Search Depth": depth,
        "Avg # Facets Uncorrected": rolling_average,
        "Search Time": search_time,
        "Bad SHI Computations": bad_shi_computations,
        "Complete": complete,
    }


def discover_boundary_component(
    net: ReLUNetwork,
    boundary_shi: int,
    exclude_tags: set[bytes],
    *,
    bound: float | None = None,
    nworkers: int | None = None,
    verbose: int | None = None,
    **kwargs: Any,
) -> tuple[Complex, dict[str, Any]] | None:
    """Discover one connected boundary component not represented in ``exclude_tags``."""
    from relucent.complex import Complex

    witness = price_boundary_witness(
        net,
        boundary_shi,
        exclude_tags,
        bound=bound,
        verbose=bool(verbose),
    )
    if witness is None:
        return None

    cx = Complex(net)
    search_info = boundary_searcher(
        cx,
        boundary_shi,
        witness,
        bound=bound,
        nworkers=nworkers,
        verbose=verbose,
        **kwargs,
    )
    return cx, search_info


def discover_boundary_complex(
    net: ReLUNetwork,
    boundary_shi: int,
    *,
    bound: float | None = None,
    nworkers: int | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> tuple[Complex, BoundaryDiscoveryStats]:
    """Discover the full boundary complex via MIP pricing + slice BFS per component."""
    from relucent.complex import Complex

    merged = Complex(net)
    visited: set[bytes] = set()
    n_components = 0
    n_pricing_calls = 0
    pricing_time = 0.0
    search_time = 0.0

    while True:
        n_pricing_calls += 1
        _discovery_log(
            "discover_boundary_complex: pricing call "
            + f"{n_pricing_calls} (excluded_tags={len(visited)}, boundary_shi={boundary_shi}) ...",
            verbose=verbose,
        )
        t0 = time.perf_counter()
        witness = price_boundary_witness(
            net,
            boundary_shi,
            visited,
            bound=bound,
            verbose=verbose,
            pricing_call=n_pricing_calls,
        )
        call_seconds = time.perf_counter() - t0
        pricing_time += call_seconds
        if witness is None:
            _discovery_log(
                "discover_boundary_complex: pricing call "
                + f"{n_pricing_calls} proven infeasible after {call_seconds:.3f}s "
                + "(no more boundary components)",
                verbose=verbose,
            )
            break

        _discovery_log(
            "discover_boundary_complex: pricing call "
            + f"{n_pricing_calls} found witness after {call_seconds:.3f}s, "
            + "starting boundary search ...",
            verbose=verbose,
        )

        cx = Complex(net)
        search_info = boundary_searcher(
            cx,
            boundary_shi,
            witness,
            bound=bound,
            nworkers=nworkers,
            verbose=1 if verbose else 0,
            **kwargs,
        )
        n_components += 1
        search_time += float(search_info.get("Search Time", 0.0))
        for poly in cx:
            merged.add_polyhedron(poly, check_exists=True)
            visited.add(poly.tag)
        _discovery_log(
            "discover_boundary_complex: component " + f"{n_components} finished, n_cells={len(cx)}, total_cells={len(merged)}",
            verbose=verbose,
        )

    if len(merged) == 0:
        stats = BoundaryDiscoveryStats(
            n_components=0,
            n_pricing_calls=n_pricing_calls,
            n_cells=0,
            pricing_time=pricing_time,
            search_time=search_time,
            post_halfspaces_s=0.0,
            post_classify_s=0.0,
            post_verify_s=0.0,
            post_total_s=0.0,
            mip_failures=0,
        )
        return merged, stats

    _discovery_log(
        "discover_boundary_complex: post-processing " + f"{len(merged)} cells (SHI filter, genericity) ...",
        verbose=verbose,
    )
    t_post = time.perf_counter()
    t_verify = time.perf_counter()
    _finalize_boundary_complex(
        merged,
        boundary_shi,
        bound=bound,
        nworkers=nworkers,
        verbose=verbose,
        **kwargs,
    )
    post_verify_s = time.perf_counter() - t_verify
    post_halfspaces_s = post_verify_s  # halfspaces computed inside finalize
    post_classify_s = 0.0
    post_total_s = time.perf_counter() - t_post
    _discovery_log(
        "discover_boundary_complex: post-processing finished in " + f"{post_total_s:.3f}s",
        verbose=verbose,
    )

    stats = BoundaryDiscoveryStats(
        n_components=n_components,
        n_pricing_calls=n_pricing_calls,
        n_cells=len(merged),
        pricing_time=pricing_time,
        search_time=search_time,
        post_halfspaces_s=post_halfspaces_s,
        post_classify_s=post_classify_s,
        post_verify_s=post_verify_s,
        post_total_s=post_total_s,
        mip_failures=0,
    )
    return merged, stats
