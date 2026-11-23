"""Slice-restricted BFS: one connected component of a neuron's bent hyperplane.

:func:`boundary_component` explores only cells on the hyperplane ``ss[i] = 0``, starting from a
cell the caller supplies, and returns the certified component that contains it. It finds no other
component and says nothing about whether others exist. For the whole hyperplane, explore the
input space (:meth:`~relucent.core.complex.Complex.bfs`) and then call
:meth:`~relucent.core.complex.Complex.boundary_complex`.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np

import relucent.config as cfg
from relucent._internal.cache import UNSET
from relucent._internal.logging import logger, progress, with_verbosity
from relucent._internal.network_scale import default_polyhedron_bound
from relucent._internal.parallel import BlockingQueue, process_aware_cpu_count, worker_pool
from relucent.core.errors import AmbiguousGeometryError, IncompleteDualGraphError, NonGenericArrangementError
from relucent.core.poly import Polyhedron
from relucent.core.ss import encode_ss, flip_ss_at_shi
from relucent.geometry import calculations
from relucent.geometry.calculations import shis_are_certified
from relucent.graph.incidence import ss_nonzero_indices
from relucent.search.engine import (
    SEARCH_REQUIRED_GEOMETRY_PROPERTIES,
    _cancel_pending_neighbor,
    blocking_bad_shi_computations,
    retain_geometry_caches,
    search_calculations,
)
from relucent.search.exploration import SearchResult, finalize_boundary_complex
from relucent.search.worker_context import set_worker_context

if TYPE_CHECKING:
    from relucent.core.complex import Complex
    from relucent.model.model import ReLUNetwork

__all__ = [
    "boundary_component",
    "boundary_searcher",
]


def _is_top_boundary_ss(ss: np.ndarray, boundary_shi: int) -> bool:
    """True when ``ss[boundary_shi]`` is the only zero: a top cell of that neuron's bent hyperplane."""
    row = np.asarray(ss, dtype=np.int8).ravel()
    if row[int(boundary_shi)] != 0:
        return False
    return int(np.count_nonzero(row == 0)) == 1


def _ambient_lift_polyhedra(
    poly: Polyhedron,
    boundary_shi: int,
) -> tuple[Polyhedron, Polyhedron]:
    ss = np.asarray(poly.ss_np, dtype=np.int8).copy()
    bshi = int(boundary_shi)
    ss_pos = ss.copy()
    ss_neg = ss.copy()
    ss_pos[bshi] = 1
    ss_neg[bshi] = -1
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
    _bound: float | None = None,
    **_shis_kwargs: Any,
) -> list[int]:
    """``_shis`` for a boundary top cell, matching :meth:`Complex.boundary_cells`.

    Returns all nonzero sign-sequence crossings on the slice (finalized by
    :func:`~relucent.graph.incidence.set_contracted_shis`).
    """
    ss = np.asarray(poly.ss_np, dtype=np.int8).copy()
    bshi = int(boundary_shi)
    if int(ss[bshi]) != 0:
        raise ValueError(f"Expected ss[{bshi}]=0 on boundary cell, got {ss!r}")
    return sorted(int(s) for s in ss_nonzero_indices(ss))


def _ambient_boundary_metadata_for_cell(
    poly: Polyhedron,
    boundary_shi: int,
    *,
    bound: float | None = None,
    **shis_kwargs: Any,
) -> tuple[list[int], Any, np.ndarray, np.ndarray | None]:
    """Return ``(_shis, halfspaces, halfspaces_err, halfspaces_ss)`` matching :meth:`Complex.boundary_cells`."""
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
    return shis, p_pos.halfspaces, p_pos.halfspaces_err_np, p_pos.halfspaces_rows_ss


def _set_coface_rows(poly: Polyhedron, halfspaces: Any, halfspaces_err: np.ndarray, halfspaces_ss: np.ndarray | None) -> None:
    """Give a boundary cell its ambient coface's rows, with their float64 error scale."""
    poly._halfspaces = halfspaces
    poly._halfspaces_np = None
    poly._halfspaces_err = halfspaces_err
    poly._halfspaces_ss = halfspaces_ss
    poly._halfspaces_own = False


def _apply_ambient_boundary_shis(
    cx: Complex,
    boundary_shi: int,
    *,
    bound: float | None = None,
    nworkers: int | None = None,
    **shis_kwargs: Any,
) -> None:
    """Assign slice ``_shis`` to every top boundary cell before dual-graph finalize."""
    if len(cx) == 0:
        return
    nw = nworkers or process_aware_cpu_count() or 1
    polys = list(cx)
    if nw <= 1 or len(polys) < 32:
        for poly in polys:
            shis, halfspaces, halfspaces_err, halfspaces_ss = _ambient_boundary_metadata_for_cell(
                poly,
                boundary_shi,
                bound=bound,
                **shis_kwargs,
            )
            poly._shis = shis
            poly._shis_strict = False  # slice crossings, not this cell's certified LP facets
            _set_coface_rows(poly, halfspaces, halfspaces_err, halfspaces_ss)
        return

    from relucent.search.worker_context import set_worker_context

    tasks = [(poly.ss_np, poly.tag) for poly in polys]
    tag_to_poly = {poly.tag: poly for poly in polys}
    with worker_pool(nw, initializer=set_worker_context, initargs=(cx._net, False)) as pool:
        results = pool.starmap(
            partial(
                _ambient_coface_shis_worker,
                boundary_shi=boundary_shi,
                bound=bound,
                shis_kwargs=shis_kwargs,
            ),
            tasks,
        )
    for tag, shis, halfspaces, halfspaces_err, halfspaces_ss in results:
        tag_to_poly[tag]._shis = shis
        tag_to_poly[tag]._shis_strict = False  # slice crossings, not this cell's certified LP facets
        _set_coface_rows(tag_to_poly[tag], halfspaces, halfspaces_err, halfspaces_ss)
    logger.debug(
        "boundary finalize: ambient coface _shis for " + f"{len(polys)} cells ({nw} workers)",
    )


def _ambient_coface_shis_worker(
    ss: np.ndarray,
    tag: bytes,
    *,
    boundary_shi: int,
    bound: float | None,
    shis_kwargs: dict[str, Any],
) -> tuple[bytes, list[int], Any, np.ndarray, np.ndarray | None]:
    from relucent.search.worker_context import get_worker_context

    ctx = get_worker_context()
    poly = Polyhedron(ctx.net, ss)
    shis, halfspaces, halfspaces_err, halfspaces_ss = _ambient_boundary_metadata_for_cell(
        poly,
        boundary_shi,
        bound=bound,
        **shis_kwargs,
    )
    return tag, shis, halfspaces, halfspaces_err, halfspaces_ss


@with_verbosity
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
) -> SearchResult:
    """BFS on a bent hyperplane slice with ``ss[boundary_shi] = 0`` fixed.

    Args:
        cx: Boundary complex under construction (typically empty).
        boundary_shi: Global SHI index pinned to zero on every discovered cell.
        start: Witness polyhedron with exactly one zero at ``boundary_shi``.
        max_depth: Maximum flip depth from the start cell.
        max_polys: Maximum cells to discover in this component.
        bound: Gurobi box bound for SHI LPs.
        nworkers: Worker process count.
        verbose: Output level: ``0`` quiet, ``1`` progress bars and summaries, ``2`` search
            detail. ``None`` uses :data:`relucent.config.VERBOSE`.
        geometry_properties: Optional geometry caches (default topology-only).
        verify: When True (default), require complete exploration. Certification runs
            later in :func:`~relucent.search.exploration.finalize_boundary_complex`.
        **kwargs: Forwarded to :func:`~relucent.geometry.calculations.shis`.

    Returns:
        :class:`~relucent.search.exploration.SearchResult`. ``verified`` is ``None`` (or
        ``False`` if the search stopped early) because certification runs later, in
        :func:`~relucent.search.exploration.finalize_boundary_complex`.

    Raises:
        ValueError: If ``start`` is not a top-dimensional boundary cell.
    """
    del verbose  # applied by @with_verbosity
    if bound is None:
        bound = default_polyhedron_bound(cx._net)
    shis_kwargs = dict(kwargs)
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
    logger.info(
        "boundary_searcher running on %d workers (boundary_shi=%d, verify=%s)",
        nworkers,
        boundary_shi,
        verify,
    )

    queue = BlockingQueue(
        queue_class=deque,
        pop=cast(Any, deque.popleft),
        push=cast(Any, deque.append),
    )

    start = cx.add_polyhedron(start, check_exists=False)
    certified = shis_are_certified(shis_kwargs)
    try:
        result = calculations.shis(start, bound=bound, **shis_kwargs)
    except ValueError as exc:
        if "Initial Solve Failed" not in str(exc):
            raise
        result = _ambient_coface_shis_for_boundary_cell(start, boundary_shi, bound=bound, **shis_kwargs)
        certified = False
    assert isinstance(result, list)
    start._shis = [s for s in result if int(s) != boundary_shi]  # boundary hyperplane is not a facet
    start._shis_strict = certified
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
    pbar = progress(
        desc="Boundary Search",
        mininterval=5,
        total=max_polys if max_polys != float("inf") else None,
    )
    pbar.update(n=1)
    pbar.get_lock().locks = []

    unprocessed = len(queue)
    depth = 0
    search_time = 0.0
    depth_limited = False
    t_search = time.perf_counter()

    if unprocessed > 0:
        with worker_pool(nworkers, initializer=set_worker_context, initargs=(cx._net, False)) as pool:
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
                    if isinstance(p, (AmbiguousGeometryError, NonGenericArrangementError)):
                        # Not a missing neighbor: the geometry itself cannot be decided, or the
                        # arrangement is not simple. Surface it rather than let it degrade into an
                        # incomplete-complex error.
                        raise p
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
    else:
        queue.close()
        search_time = time.perf_counter() - t_search
        pbar.close()

    hit_cap = max_polys != float("inf") and len(cx) >= max_polys
    blocking_mistakes = blocking_bad_shi_computations(bad_shi_computations)
    # As in ambient search, a failed neighbor task means we pruned the frontier and
    # cannot treat an empty queue as proof that the boundary complex is complete —
    # except for true phantoms (empty flip-neighbor sign patterns).
    complete = unprocessed == 0 and not hit_cap and not depth_limited and not blocking_mistakes
    n_phantom = len(bad_shi_computations) - len(blocking_mistakes)
    if n_phantom:
        logger.info("boundary search: ignored %d true phantom flip-neighbor(s)", n_phantom)
    if not complete:
        # Certification deferred to finalize_boundary_complex in boundary_component.
        cx.set_exploration_state(complete=False, verified=False)
        return SearchResult(
            depth=depth,
            mean_facets=rolling_average,
            search_time=search_time,
            bad_shi_computations=bad_shi_computations,
            complete=False,
            verified=False,
        )

    for poly in cx:
        poly._finite = UNSET
    return SearchResult(
        depth=depth,
        mean_facets=rolling_average,
        search_time=search_time,
        bad_shi_computations=bad_shi_computations,
        complete=True,
        verified=None,
    )


@with_verbosity
def boundary_component(
    net: ReLUNetwork,
    boundary_shi: int,
    start: Polyhedron | np.ndarray,
    *,
    bound: float | None = None,
    nworkers: int | None = None,
    verify: bool = True,
    verbose: int | None = None,
    **kwargs: Any,
) -> Complex:
    """The connected component of neuron ``boundary_shi``'s bent hyperplane that contains ``start``.

    Explores the slice ``ss[boundary_shi] = 0`` by :func:`boundary_searcher` and certifies the result
    with :func:`~relucent.search.exploration.finalize_boundary_complex`. Only the component holding
    ``start`` is found; nothing is implied about other components.

    Args:
        net: Canonical :class:`~relucent.model.model.ReLUNetwork`.
        boundary_shi: Global supporting-hyperplane index of the neuron whose bent hyperplane to explore.
        start: A top cell of that hyperplane: a :class:`~relucent.core.poly.Polyhedron` or a sign
            sequence whose only zero is at ``boundary_shi``, with both ambient cofaces nonempty.
        bound: Gurobi box bound for SHI LPs; defaults to ``default_polyhedron_bound()``.
        nworkers: Worker process count.
        verify: When True (default), require a complete search and certify the component.
        verbose: Output level: ``0`` quiet, ``1`` progress bars and summaries, ``2`` debug detail.
            ``None`` uses :data:`relucent.config.VERBOSE`.
        **kwargs: Forwarded to the slice search's SHI routines.

    Returns:
        A new :class:`Complex` of the component's top cells.

    Raises:
        ValueError: If ``start`` is not a top cell of the hyperplane.
        IncompleteDualGraphError: If ``verify`` and the search stopped before exhausting the component.
    """
    del verbose  # applied by @with_verbosity
    from relucent.core.complex import Complex

    poly = start if isinstance(start, Polyhedron) else Polyhedron(net, np.asarray(start, dtype=np.int8).reshape(-1))
    if poly._net is None:
        poly._net = net
    if not _is_top_boundary_ss(poly.ss_np, boundary_shi):
        raise ValueError(f"start must have ss[{boundary_shi}]=0 as its only zero entry; got {poly.ss_np!r}")
    if not poly.feasible or not _both_ambient_cofaces_feasible(poly, boundary_shi):
        raise ValueError(f"start {poly.ss_np!r} is not a cell of neuron {boundary_shi}'s bent hyperplane")

    found = Complex(net)
    info = boundary_searcher(found, boundary_shi, poly, bound=bound, nworkers=nworkers, verify=verify, **kwargs)
    if verify and not info.complete:
        raise IncompleteDualGraphError("boundary component search stopped before exhausting the component")
    component = Complex(net)  # a fresh complex, so finalize starts from no exploration state
    for p in found:
        component.add_polyhedron(p, check_exists=True)
    finalize_boundary_complex(component, boundary_shi, bound=bound, nworkers=nworkers, verify=verify, **kwargs)
    return component
