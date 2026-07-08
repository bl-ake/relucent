"""Slice-restricted BFS and MIP-driven boundary complex discovery."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from tqdm.auto import tqdm

import relucent.config as cfg
from relucent import meta_graph as mg
from relucent._logging import logger
from relucent._network_scale import default_polyhedron_bound
from relucent.boundary_mip import _is_top_boundary_ss, price_boundary_witness
from relucent.calculations import get_shis
from relucent.exploration import finalize_boundary_complex, search_stats_dict
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


def _both_ambient_cofaces_feasible(poly: Polyhedron, boundary_shi: int) -> bool:
    """True when lifts to ``ss[boundary_shi]=±1`` are both nonempty."""
    p_pos, p_neg = _ambient_lift_polyhedra(poly, boundary_shi)
    return p_pos.feasible and p_neg.feasible


def _ambient_coface_shis_for_boundary_cell(
    poly: Polyhedron,
    boundary_shi: int,
    *,
    bound: float | None = None,
    **shis_kwargs: Any,
) -> list[int]:
    """``_shis`` for a boundary top cell, matching :meth:`Complex.get_boundary_cells`.

    When :data:`~relucent.config.CUBICAL_DUAL_GRAPH` is True, returns all nonzero
    sign-sequence crossings on the slice (finalized by
    :func:`~relucent.meta_graph.assign_contracted_shis`).  Otherwise lifts
    ``poly`` off the bent hyperplane and intersects ambient coface SHI sets.
    """
    ss = np.asarray(poly.ss_np, dtype=np.int8).copy()
    bshi = int(boundary_shi)
    if int(ss.ravel()[bshi]) != 0:
        raise ValueError(f"Expected ss[{bshi}]=0 on boundary cell, got {ss!r}")
    if cfg.CUBICAL_DUAL_GRAPH:
        return sorted(int(s) for s in mg.ss_nonzero_indices(ss))
    net = poly._net
    if net is None:
        raise ValueError("boundary cell missing network reference for ambient lift")
    if bound is None:
        bound = default_polyhedron_bound(net)
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
    net = poly._net
    if bound is None:
        bound = cfg.DEFAULT_SEARCH_BOUND if net is None else default_polyhedron_bound(net)
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
    """Assign slice ``_shis`` to every top boundary cell before dual-graph finalize."""
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
    verify: bool = True,
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
        verify: When True (default), require complete exploration and run invariant
            checks at the end. Skipped when exploration hits ``max_polys`` before the
            frontier is exhausted. Sets ``strict=True`` on SHI LPs.
        **kwargs: Forwarded to :func:`~relucent.calculations.get_shis`.

    Returns:
        Search statistics dict (same keys as :func:`~relucent.search.searcher`,
        including ``"Complete"``; ``"Verified"`` is omitted here because certification
        runs later in :func:`~relucent.exploration.finalize_boundary_complex`).

    Raises:
        ValueError: If ``start`` is not a top-dimensional boundary cell.
    """
    if verbose is None:
        verbose = cfg.VERBOSE
    if bound is None:
        bound = default_polyhedron_bound(cx._net)
    shis_kwargs = dict(kwargs)
    if verify:
        shis_kwargs["strict"] = True
    if not _is_top_boundary_ss(start.ss_np, boundary_shi):
        raise ValueError(f"Start sign sequence must have ss[{boundary_shi}]=0 as its only zero entry; got {start.ss_np!r}")

    search_props = (
        SEARCH_REQUIRED_GEOMETRY_PROPERTIES
        if geometry_properties is None
        else tuple(dict.fromkeys((*SEARCH_REQUIRED_GEOMETRY_PROPERTIES, *geometry_properties)))
    )
    shi_subset = [j for j in range(cx.n) if j != boundary_shi]  # pinned zero, not a flip facet
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
    try:
        result = get_shis(start, bound=bound, **shis_kwargs)
    except ValueError as exc:
        if "Initial Solve Failed" not in str(exc):
            raise
        result = _ambient_coface_shis_for_boundary_cell(start, boundary_shi, bound=bound, **shis_kwargs)
    assert isinstance(result, list)
    start._shis = [s for s in result if int(s) != boundary_shi]  # boundary hyperplane is not a facet
    retain_geometry_caches(start, search_props)
    start_index = cx.ssm[start.ss_np]
    failed_flips: set[tuple[bytes, int]] = set()

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
    depth_limited = False
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
                        _cancel_pending_neighbor(cx, pending_neighbors, node, shi, failed_flips)
                        if unprocessed == 0 or len(cx) >= max_polys:
                            break
                        continue

                    if not _is_top_boundary_ss(p.ss_np, boundary_shi):
                        _cancel_pending_neighbor(cx, pending_neighbors, node, shi, failed_flips)
                        if unprocessed == 0 or len(cx) >= max_polys:
                            break
                        continue

                    if p._net is None:
                        p._net = cx._net

                    if not _both_ambient_cofaces_feasible(p, boundary_shi):
                        _cancel_pending_neighbor(cx, pending_neighbors, node, shi, failed_flips)
                        if unprocessed == 0 or len(cx) >= max_polys:
                            break
                        continue

                    p = cx.add_polyhedron(p)
                    pending_neighbors.pop(encode_ss(p.ss_np), None)

                    if depth < max_depth:
                        poly_index = cx.ssm[p.ss_np]
                        for new_shi in p.shis:
                            if new_shi in (boundary_shi, shi) or len(cx) >= max_polys:
                                continue
                            if (p.tag, int(new_shi)) in failed_flips:
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
                    elif max_depth != float("inf"):
                        # Same depth_limited probe as ambient searcher.
                        for new_shi in p.shis:
                            if new_shi in (boundary_shi, shi):
                                continue
                            if (p.tag, int(new_shi)) in failed_flips:
                                continue
                            ss = flip_ss_at_shi(p.ss_np, new_shi)
                            if not _is_top_boundary_ss(ss, boundary_shi):
                                continue
                            tag = encode_ss(ss)
                            if tag not in cx.tag2poly and tag not in pending_neighbors:
                                depth_limited = True
                                break

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

    hit_cap = max_polys != float("inf") and len(cx) >= max_polys
    complete = unprocessed == 0 and not hit_cap and not depth_limited
    if not complete:
        # Certification deferred to finalize_boundary_complex in discover_boundary_complex.
        cx.set_exploration_state(complete=False, verified=False)
        return search_stats_dict(
            depth=depth,
            rolling_average=rolling_average,
            search_time=search_time,
            bad_shi_computations=bad_shi_computations,
            complete=False,
            verified=False,
        )

    for poly in cx:
        poly._finite = None
        poly._finite_computed = False
    return search_stats_dict(
        depth=depth,
        rolling_average=rolling_average,
        search_time=search_time,
        bad_shi_computations=bad_shi_computations,
        complete=True,
        verified=None,
    )


def discover_boundary_complex(
    net: ReLUNetwork,
    boundary_shi: int,
    *,
    bound: float | None = None,
    nworkers: int | None = None,
    verbose: bool = False,
    verify: bool = True,
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
            verify=verify,
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
    finalize_boundary_complex(
        merged,
        boundary_shi,
        bound=bound,
        nworkers=nworkers,
        verbose=verbose,
        verify=verify,
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
