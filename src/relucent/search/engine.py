"""Search and pathfinding over a polyhedral complex."""

from __future__ import annotations

import contextlib
import os
import random
import warnings
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from tqdm.auto import tqdm

import relucent.config as cfg
from relucent._internal.cache import UNSET
from relucent._internal.logging import logger, progress, with_verbosity
from relucent._internal.network_scale import default_polyhedron_bound
from relucent._internal.parallel import (
    BlockingQueue,
    NonBlockingQueue,
    UpdatablePriorityQueue,
    process_aware_cpu_count,
    worker_pool,
)
from relucent._internal.torch_compat import is_torch_tensor, torch
from relucent.core.errors import AmbiguousGeometryError, NonGenericArrangementError
from relucent.core.poly import Polyhedron
from relucent.core.ss import encode_ss, flip_ss_at_shi
from relucent.geometry import calculations
from relucent.geometry.calculations import shis_are_certified
from relucent.search.exploration import (
    SearchResult,
    finalize_ambient_search,
)
from relucent.search.worker_context import get_worker_context, set_worker_context, worker_context_scope

if TYPE_CHECKING:
    from relucent.core.complex import Complex

__all__ = [
    "ALL_GEOMETRY_PROPERTIES",
    "SEARCH_REQUIRED_GEOMETRY_PROPERTIES",
    "astar_calculations",
    "blocking_bad_shi_computations",
    "get_ip",
    "greedy_path",
    "hamming_astar",
    "parallel_compute_geometric_properties",
    "parallel_add",
    "retain_geometry_caches",
    "search_calculations",
    "searcher",
    "true_phantom_neighbor_error",
]

# Chebyshev center/inradius are always computed during search workers because SHI
# reliability checks depend on them. Boundedness (``finite``) is not needed to search.
SEARCH_REQUIRED_GEOMETRY_PROPERTIES: tuple[str, ...] = ("center", "inradius")

# Every property name :meth:`~relucent.core.poly.Polyhedron.compute_geometric_properties` accepts.
ALL_GEOMETRY_PROPERTIES: tuple[str, ...] = Polyhedron.GEOMETRY_PROPERTIES


def true_phantom_neighbor_error(error: object) -> bool:
    """Always False: no queued flip-neighbor is legitimately empty.

    Neighbors are queued only across facets that :func:`~relucent.geometry.calculations.shis`
    certified, i.e. with a witness point just past the facet and inside every other row. The
    region across such a facet is nonempty, so an empty neighbor means something upstream is
    wrong and must block completeness. (This used to excuse neighbors whose Chebyshev radius fell
    in a tolerance band; those were artefacts of misjudged facets.) Kept for API compatibility.
    """
    del error
    return False


def blocking_bad_shi_computations(bad_shi_computations: list[Any]) -> list[Any]:
    """Failed neighbor tasks that still prevent certifying exploration complete."""
    blocking: list[Any] = []
    for item in bad_shi_computations:
        err = str(item[3]) if isinstance(item, tuple) and len(item) > 3 else str(item)
        if not true_phantom_neighbor_error(err):
            blocking.append(item)
    return blocking


def _cancel_pending_neighbor(
    cx: Complex,
    pending_neighbors: dict[bytes, list[tuple[int, int]]],
    node: Polyhedron,
    shi: int,
    failed_flips: set[tuple[bytes, int]],
) -> None:
    """Drop a failed in-flight neighbor; record flips not to retry from each endpoint."""
    failed_tag = encode_ss(flip_ss_at_shi(node.ss_np, shi))
    failed_flips.add((node.tag, int(shi)))
    for waiter_index, waiter_shi in pending_neighbors.pop(failed_tag, []):
        waiter = cx.index2poly[waiter_index]
        failed_flips.add((waiter.tag, int(waiter_shi)))


def retain_geometry_caches(p: Polyhedron, properties: Iterable[str]) -> None:
    """Retain geometry caches listed in *properties*; drop other heavy caches."""
    # Workers only need what search asked for; drop the rest to keep IPC payloads small.
    requested = {str(name).strip() for name in properties if str(name).strip()}
    for name, attrs in (
        ("halfspaces", ("_halfspaces", "_halfspaces_np", "_halfspaces_err")),
        ("W", ("_w",)),
        ("b", ("_b",)),
    ):
        if name not in requested:
            for attr in attrs:
                setattr(p, attr, None)
    # Qhull objects are the heaviest; drop them unless something needs them.
    if not (requested & Polyhedron.QHULL_PROPERTIES):
        p._qhull = UNSET


def _worker_prepare_poly(
    p: Polyhedron,
    props: tuple[str, ...],
    *,
    env: Any,
    shis_kwargs: dict[str, Any] | None = None,
    need_interior: bool = False,
    shuffle_shis: bool = False,
) -> Exception | None:
    """Compute geometry (and optionally SHIs) on *p*. Return an error, or None on success."""
    try:
        p.compute_geometric_properties(props, env=env)
    except (ValueError, AmbiguousGeometryError) as error:
        return error
    if not p.feasible:
        # Thin cells need no special case: solve_radius only reports a cell nonempty with a
        # center verified strictly inside every row, and raises when it cannot decide.
        return AmbiguousGeometryError(f"Polyhedron {p!r} is infeasible (empty), but it was reached across a certified facet")
    if need_interior:
        p._ensure_interior_point(env)
    if shis_kwargs is not None:
        try:
            if p._shis is None:
                result = calculations.shis(p, env=env, **shis_kwargs)
                p._shis = result[0] if isinstance(result, tuple) else result
                p._shis_strict = shis_are_certified(shis_kwargs)
        except Exception as error:
            return error
    retain_geometry_caches(p, props)
    if shuffle_shis and isinstance(p._shis, list):
        random.shuffle(p._shis)  # spreads load when many workers race on the same parent
    return None


def _start_shis_for_search(
    start: Polyhedron,
    *,
    bound: float,
    shis_kwargs: dict[str, Any],
) -> list[int]:
    """SHIs for the search seed cell (certified like every other cell's; no fallback)."""
    result = calculations.shis(start, bound=bound, **shis_kwargs)
    assert isinstance(result, list)
    return result


def _apply_cube_filter(p: Polyhedron, cube_mode: str, cube_radius: float) -> bool:
    """Return whether *p* should be kept under the cube filter (possibly clipping it)."""
    try:
        bounded, _, bounded_err = p._halfspaces_with_bounding_box_err(cube_radius)
    except ValueError:
        # Outside the box entirely — keep only if we're explicitly hunting exterior cells.
        return cube_mode == "exclude"
    if cube_mode == "intersect":
        return True  # tag as intersecting the box but don't rewrite halfspaces
    if cube_mode == "exclude":
        return False  # drop anything that touches the box
    p._halfspaces = bounded  # type: ignore[assignment]  # clipped: replace with bounded form
    p._halfspaces_np = bounded
    p._halfspaces_err = bounded_err
    p._halfspaces_own = False
    return p.feasible


def search_calculations(
    task: np.ndarray | torch.Tensor | tuple[Any, ...],
    geometry_properties: tuple[str, ...] = SEARCH_REQUIRED_GEOMETRY_PROPERTIES,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Worker for neighbor enumeration: Chebyshev geometry, SHIs, and optional caches."""
    ctx = get_worker_context()
    ss = task[0] if isinstance(task, tuple) else task
    rest = task[1:] if isinstance(task, tuple) else ()
    bound = kwargs.get("bound")
    if bound is None and ctx.net is not None:
        bound = default_polyhedron_bound(ctx.net)
    p = Polyhedron(ctx.net, ss, bound=bound)
    if err := _worker_prepare_poly(p, geometry_properties, env=ctx.env, shis_kwargs=kwargs, shuffle_shis=True):
        return err, *rest  # parent treats non-Polyhedron first slot as failure
    return (p, *rest)


def geometric_calculations(
    task: tuple[np.ndarray, list[int] | None, int, *tuple[Any, ...]]
    | tuple[np.ndarray, list[int] | None, bool, int, *tuple[Any, ...]],
    geometry_properties: Iterable[str] = ALL_GEOMETRY_PROPERTIES,
    bound: float | None = None,
) -> tuple[Any, ...]:
    """Worker function for geometric-property computation on known polyhedra."""
    ctx = get_worker_context()
    ss: np.ndarray
    shis: list[int] | None
    poly_index: int
    shis_strict: bool
    rest: tuple[Any, ...]
    if len(task) >= 4:
        ss = cast(np.ndarray, task[0])
        shis = cast(list[int] | None, task[1])
        shis_strict = bool(task[2])
        poly_index = int(task[3])
        rest = task[4:]
    else:
        ss = cast(np.ndarray, task[0])
        shis = cast(list[int] | None, task[1])
        poly_index = int(task[2])
        rest = task[3:]
        shis_strict = False
    p = Polyhedron(ctx.net, ss, shis=shis, bound=bound, shis_strict=bool(shis_strict))
    if err := _worker_prepare_poly(p, tuple(geometry_properties), env=ctx.env):
        return err, poly_index, *rest
    return (p, poly_index, *rest)


@with_verbosity
def parallel_compute_geometric_properties(
    cx: Complex,
    nworkers: int | None = None,
    geometry_properties: Iterable[str] = ALL_GEOMETRY_PROPERTIES,
    verbose: int | None = None,
) -> dict[str, Any]:
    """Compute selected properties for all existing polyhedra in parallel.

    Args:
        cx: Complex whose polyhedra should be updated.
        nworkers: Number of worker processes to use.
        geometry_properties: Iterable of cache/property names to compute and
            retain on each polyhedron. Defaults to
            :data:`ALL_GEOMETRY_PROPERTIES`.
        verbose: ``0`` silences output; ``1`` shows worker counts and progress bars;
            ``2`` adds debug detail. ``None`` uses :data:`relucent.config.VERBOSE`.
    """
    del verbose  # applied by @with_verbosity
    if len(cx) == 0:
        return {"Computed": 0, "Failed": []}

    nworkers = nworkers or process_aware_cpu_count()
    logger.info("Computing geometric properties on %d workers", nworkers)

    tasks = [(poly.ss_np, poly._shis, bool(getattr(poly, "_shis_strict", False)), i) for i, poly in enumerate(cx)]
    failed: list[tuple[int, str]] = []
    computed = 0
    with worker_pool(nworkers, initializer=set_worker_context, initargs=(cx._net, False)) as pool:
        for result in progress(
            pool.imap_unordered(
                partial(
                    geometric_calculations,
                    geometry_properties=geometry_properties,
                ),
                tasks,
            ),
            total=len(tasks),
            desc="Computing Geometry",
            mininterval=5,
        ):
            poly_or_error, poly_index, *_ = result
            if isinstance(poly_or_error, Polyhedron):
                if poly_or_error._net is None:
                    poly_or_error._net = cx._net
                cx.index2poly[poly_index] = poly_or_error
                computed += 1
            else:
                failed.append((poly_index, str(poly_or_error)))
    return {"Computed": computed, "Failed": failed}


@with_verbosity
def parallel_add(
    cx: Complex,
    points: Iterable[torch.Tensor | np.ndarray],
    nworkers: int | None = None,
    bound: float | None = None,
    geometry_properties: Iterable[str] = ALL_GEOMETRY_PROPERTIES,
    verbose: int | None = None,
) -> list[Polyhedron | None]:
    """Add multiple polyhedra from data points using parallel processing.

    Processes a batch of data points in parallel, computing their corresponding
    polyhedra and adding them to the complex.

    Args:
        cx: The polyhedral complex.
        points: A list or iterable of data points (each as torch.Tensor or np.ndarray).
        nworkers: Number of worker processes to use. If None, uses the number
            of CPU cores. Defaults to None.
        bound: Constraint radius for numerical stability when computing halfspaces.
            Defaults to config.DEFAULT_PARALLEL_ADD_BOUND.
        geometry_properties: Iterable of cache/property names to compute and retain
            on each polyhedron. Defaults to :data:`ALL_GEOMETRY_PROPERTIES`.
        verbose: ``0`` silences output; ``1`` shows worker counts and progress bars;
            ``2`` adds debug detail. ``None`` uses :data:`relucent.config.VERBOSE`.

    Returns:
        list: A list of Polyhedron objects (or None for failed computations)
            corresponding to the input points.
    """
    del verbose  # applied by @with_verbosity
    if bound is None:
        bound = cfg.DEFAULT_PARALLEL_ADD_BOUND

    nworkers = nworkers or process_aware_cpu_count()
    logger.info("parallel_add using %d workers", nworkers)
    sss = [
        (s.detach().cpu().numpy() if is_torch_tensor(s := cx.point2ss(p)) else s)
        for p in progress(points, desc="Getting SSs", mininterval=5)
    ]  # materialize SSs up front so pool tasks are plain numpy arrays

    tasks = [(ss, None, i) for i, ss in enumerate(sss)]
    with worker_pool(nworkers, initializer=set_worker_context, initargs=(cx._net,)) as pool:
        results = pool.map(
            partial(
                geometric_calculations,
                geometry_properties=geometry_properties,
                bound=bound,
            ),
            progress(tasks, desc="Adding Polys", mininterval=5),
        )
    # results are (poly_or_error, poly_index); restore original order
    ordered: list[Polyhedron | None] = [None] * len(sss)
    for poly_or_error, poly_index, *_ in results:
        if isinstance(poly_or_error, Polyhedron):
            poly_or_error._net = cx._net
            cx.add_polyhedron(poly_or_error)
            ordered[poly_index] = poly_or_error
    return ordered


def get_ip(
    p: Polyhedron,
    shi: int,
) -> tuple[Polyhedron, int] | tuple[ValueError, int]:
    """Flip one SHI in *p*'s sign sequence and find an interior point for the neighbor."""
    ctx = get_worker_context()
    try:
        n = Polyhedron(ctx.net, flip_ss_at_shi(p.ss_np, shi))
        # Neighbor may be skinny; try progressively larger bounding boxes until one works.
        for max_radius in cfg.advanced.INTERIOR_POINT_RADIUS_SEQUENCE:
            with contextlib.suppress(ValueError):
                n._interior_point = n.find_interior_point(env=ctx.env, max_radius=max_radius)
        return n, shi
    except ValueError as e:
        return e, shi


def astar_calculations(
    task: Polyhedron | tuple[Any, ...],
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Worker for A* search: geometry, interior point, and SHIs on a polyhedron."""
    ctx = get_worker_context()
    p = task[0] if isinstance(task, tuple) else task
    rest = task[1:] if isinstance(task, tuple) else ()
    if p._net is None:
        p._net = ctx.net
    if err := _worker_prepare_poly(
        p,
        SEARCH_REQUIRED_GEOMETRY_PROPERTIES,
        env=ctx.env,
        shis_kwargs=kwargs,
        need_interior=True,  # heuristic needs interior points for tie-breaking
        shuffle_shis=True,
    ):
        return p, err, *rest  # keep the poly so the parent can log which node failed
    assert isinstance(p._shis, list), "calculations.shis() returns a list"
    return p, *rest


# How the searcher treats cells against the cube ``[-cube_radius, cube_radius]^d``.
CubeMode = Literal["unrestricted", "intersect", "clipped", "exclude"]


@with_verbosity
def searcher(
    cx: Complex,
    start: torch.Tensor | np.ndarray | Polyhedron | None = None,
    max_depth: float = float("inf"),
    max_polys: float = float("inf"),
    queue: Any = None,
    bound: float | None = None,
    nworkers: int | None = None,
    verbose: int | None = None,
    cube_radius: float | None = None,
    cube_mode: CubeMode = "unrestricted",
    geometry_properties: Iterable[str] | None = None,
    verify: bool = True,
    **kwargs: Any,
) -> SearchResult:
    """Search the complex by crossing supporting hyperplanes to find neighbors.

    Generic over traversal strategy (BFS, DFS, random walk): the queue decides the order.
    See Complex.bfs(), Complex.dfs(), and Complex.random_walk() for examples.

    Args:
        cx: The :class:`~relucent.core.complex.Complex` (not the module; workers use
            :func:`~relucent.search.worker_context.get_worker_context`).
        start: Starting point (tensor / array-like) or a Polyhedron. Defaults to the origin.
        max_depth: Max number of hyperplane crossings. Defaults to infinity.
        max_polys: Max number of polyhedra to discover. Defaults to infinity.
        queue: Anything with push() and pop() that sets the search order.
            Defaults to a FIFO BlockingQueue.
        bound: Constraint radius when computing halfspaces. ``None`` uses
            :func:`~relucent._internal.network_scale.default_polyhedron_bound`.
        nworkers: Worker processes. ``None`` uses the CPU count.
        verbose: ``0`` silences output; ``1`` shows worker counts and progress bars;
            ``2`` adds debug detail. ``None`` uses :data:`relucent.config.VERBOSE`.
        geometry_properties: Polyhedron properties to compute and keep. ``None``
            (default) is topology-only; pass :data:`ALL_GEOMETRY_PROPERTIES` or a subset
            for more. ``finite``, ``center``, and ``inradius`` are always computed.
        verify: If True (default), require complete exploration and run
            :func:`~relucent.verify.certify.certify_complex` at the end. Skipped if
            ``max_polys`` stops the search early. A finite ``max_depth`` can leave
            ``complete=False``, which raises unless the cap was hit.
        **kwargs: Additional arguments passed to :func:`~relucent.geometry.calculations.shis`.

    Returns:
        :class:`~relucent.search.exploration.SearchResult`.

    Raises:
        ValueError: If the start point lies on a hyperplane (has zero in SS).
        IncompleteDualGraphError: If ``verify`` is True and exploration stops early
            for reasons other than hitting ``max_polys``.
    """
    del verbose  # applied by @with_verbosity
    if bound is None:
        bound = default_polyhedron_bound(cx._net)

    shis_kwargs = dict(kwargs)

    if cube_mode not in {"unrestricted", "intersect", "clipped", "exclude"}:
        raise ValueError("cube_mode must be one of {'unrestricted', 'intersect', 'clipped', 'exclude'}")
    elif cube_mode != "unrestricted":
        assert cube_radius is not None, "cube_radius must be provided when cube_mode is not 'unrestricted'"
        logger.info("Applying cube filter with mode '%s' and radius %s", cube_mode, cube_radius)
    elif cube_radius is not None:
        warnings.warn("cube_radius is provided but cube_mode is 'unrestricted'. Ignoring cube_radius.", stacklevel=2)
        cube_radius = None

    search_props = (
        SEARCH_REQUIRED_GEOMETRY_PROPERTIES
        if geometry_properties is None
        else tuple(dict.fromkeys((*SEARCH_REQUIRED_GEOMETRY_PROPERTIES, *geometry_properties)))
    )

    # Keys are neighbor tags with an in-flight discovery task; values list polys
    # that deferred queueing because discovery was already pending.
    pending_neighbors: dict[bytes, list[tuple[int, int]]] = {}
    nworkers = nworkers or process_aware_cpu_count()
    logger.info("searcher running on %d workers", nworkers)
    if queue is None:
        queue = BlockingQueue(
            queue_class=deque,
            pop=cast(Callable[[deque[object]], object], deque.popleft),
            push=cast(Callable[[deque[object], object], None], deque.append),
        )
    if start is None:
        start = cx.add_point(np.zeros(cx._net.input_shape, dtype=np.float64))
    elif isinstance(start, Polyhedron):
        start = cx.add_polyhedron(start)
    else:
        start = cx.add_point(start)
    if start.bound is None:
        start.bound = bound
    if cube_mode != "unrestricted" and not _apply_cube_filter(start, cube_mode, cast(float, cube_radius)):
        queue.close()
        cx.set_exploration_state(complete=True, verified=False)
        return SearchResult(
            depth=0,
            mean_facets=0.0,
            search_time=0.0,
            bad_shi_computations=[],
            complete=True,
            verified=False,
        )
    if (start.ss_np == 0).any():
        raise ValueError("Start point must not be on a hyperplane")
    start._shis = _start_shis_for_search(start, bound=bound, shis_kwargs=shis_kwargs)
    start._shis_strict = shis_are_certified(shis_kwargs)
    retain_geometry_caches(start, search_props)
    start_index = cx.ssm[start.ss_np]
    # Seed the frontier: each task is (neighbor ss, shi crossed, depth, parent index).
    failed_flips: set[tuple[bytes, int]] = set()
    for shi in start.shis:
        if (start.tag, int(shi)) in failed_flips:
            continue
        ss = flip_ss_at_shi(start.ss_np, shi)
        pending_neighbors[encode_ss(ss)] = []
        queue.push((ss, shi, 1, start_index))

    rolling_average = len(start.shis)
    bad_shi_computations: list[Any] = []
    invalid_proof_suppressed = 0
    pbar = progress(
        desc="Search Progress",
        mininterval=5,
        total=max_polys if max_polys != float("inf") else None,
    )
    pbar.update(n=1)
    # Clear stale multiprocessing locks from an earlier pool. tqdm takes a process-wide
    # lock during imap_unordered; if a pool was terminated without joining, leftovers
    # can deadlock the next one.
    pbar.get_lock().locks = []

    unprocessed = len(queue)
    depth = 0
    search_time = 0.0
    depth_limited = False

    if unprocessed > 0:
        with worker_pool(nworkers, initializer=set_worker_context, initargs=(cx._net, False)) as pool:
            try:
                for p, shi, depth, node_index in pool.imap_unordered(  # type: ignore[assignment]
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
                        if node.num_shis < min(cx.dim, cx.n):
                            warnings.warn(
                                RuntimeWarning(f"Polyhedron {node} has less than {min(cx.dim, cx.n)} SHIs"),
                                stacklevel=2,
                            )
                        if unprocessed == 0 or len(cx) >= max_polys:
                            break
                        continue

                    if p._net is None:
                        p._net = cx._net

                    if cube_mode != "unrestricted" and not _apply_cube_filter(p, cube_mode, cast(float, cube_radius)):
                        _cancel_pending_neighbor(cx, pending_neighbors, node, shi, failed_flips)
                        if unprocessed == 0 or len(cx) >= max_polys:
                            break
                        continue

                    p = cx.add_polyhedron(p)
                    pending_neighbors.pop(encode_ss(p.ss_np), None)  # this tag is no longer "in flight"

                    for warning in getattr(p, "warnings", ()) or ():
                        if "Invalid Proof for SHI" in str(warning):
                            invalid_proof_suppressed += 1

                    if depth < max_depth:
                        poly_index = cx.ssm[p.ss_np]
                        for new_shi in p.shis:
                            if new_shi != shi and len(cx) < max_polys:
                                if (p.tag, int(new_shi)) in failed_flips:
                                    continue
                                ss = flip_ss_at_shi(p.ss_np, new_shi)
                                tag = encode_ss(ss)
                                if tag in cx.tag2poly:
                                    continue
                                if tag not in pending_neighbors:
                                    pending_neighbors[tag] = []
                                    queue.push((ss, new_shi, depth + 1, poly_index))
                                    unprocessed += 1
                                else:
                                    # Same neighbor already queued — remember who was waiting.
                                    pending_neighbors[tag].append((poly_index, new_shi))
                    elif max_depth != float("inf"):
                        # Queue may drain while neighbors beyond max_depth still exist.
                        for new_shi in p.shis:
                            if new_shi == shi:
                                continue
                            if (p.tag, int(new_shi)) in failed_flips:
                                continue
                            ss = flip_ss_at_shi(p.ss_np, new_shi)
                            tag = encode_ss(ss)
                            if tag not in cx.tag2poly and tag not in pending_neighbors:
                                depth_limited = True
                                break

                    pbar.update(n=len(cx) - pbar.n)
                    rolling_average = (rolling_average * (len(cx) - 1) + len(p.shis)) / len(cx)

                    assert isinstance(p._shis, list)
                    pbar.set_postfix_str(
                        f"Depth: {depth}  Unprocessed: {unprocessed} Mistakes: {len(bad_shi_computations)}"
                        + f" Mean Facets: {rolling_average:.2f}",
                        refresh=False,
                    )

                    if unprocessed == 0 or len(cx) >= max_polys:
                        break
            finally:
                # Closing the queue lets the pool stop handing out tasks; leaving the ``with``
                # block then waits for the tasks in flight instead of killing their workers.
                queue.close()
                search_time = pbar.format_dict["elapsed"]
                pbar.close()
    else:
        queue.close()
        search_time = pbar.format_dict["elapsed"]
        pbar.close()

    hit_cap = max_polys != float("inf") and len(cx) >= max_polys
    blocking_mistakes = blocking_bad_shi_computations(bad_shi_computations)
    # Failed neighbor discoveries mean the frontier was pruned, so queue exhaustion
    # alone is not enough to certify that the ambient complex is complete — except
    # for true phantoms (empty flip-neighbor sign patterns).
    complete = unprocessed == 0 and not hit_cap and not depth_limited and not blocking_mistakes
    # Skip verify when max_polys hit — frontier may still have neighbors (LP false-fail).
    do_verify = verify and complete and not hit_cap
    finalize_ambient_search(cx, verify=do_verify, complete=complete)  # sync SHIs + optional certify_complex

    n_phantom = len(bad_shi_computations) - len(blocking_mistakes)
    if n_phantom:
        logger.info("searcher: ignored %d true phantom flip-neighbor(s)", n_phantom)

    if invalid_proof_suppressed:
        logger.info(
            "searcher: suppressed %d invalid SHI proof(s) during frontier exploration",
            invalid_proof_suppressed,
        )

    return SearchResult(
        depth=depth,
        mean_facets=rolling_average,
        search_time=search_time,
        bad_shi_computations=bad_shi_computations,
        complete=complete,
        verified=cx.verified,
    )


def _greedy_path_helper(cx: Complex, start: Polyhedron, end: Polyhedron, diffs: set[int] | None = None) -> list[Polyhedron]:
    if start == end:
        return [start]

    if (start.ss_np == 0).any():
        raise ValueError("Start point must not be on a hyperplane")

    # Indices where start and end disagree on sign — must flip an even number to match.
    diffs = diffs or set(np.argwhere((start.ss_np != end.ss_np).ravel()).ravel().tolist())

    if not start._shis:
        calculations.shis(start)
    shis_set = set(start.shis)
    groupa = shis_set & diffs  # flip toward the goal (removes one diff)
    groupb = shis_set - diffs  # detour: flip away, then recurse with that diff added back
    for shi, next_diffs in [(s, diffs - {s}) for s in groupa] + [(s, diffs | {s}) for s in groupb]:
        rest = _greedy_path_helper(cx, cx.ss2poly(flip_ss_at_shi(start.ss_np, shi)), end, next_diffs)
        if rest:
            return [start] + rest
    return []


def greedy_path(
    cx: Complex,
    start: torch.Tensor | np.ndarray | Polyhedron,
    end: torch.Tensor | np.ndarray | Polyhedron,
) -> list[Polyhedron] | None:
    """Greedily find a path between two data points.

    Attempts to find a path through adjacent polyhedra from start to end
    using a greedy strategy. This method can be slow for large complexes
    as it explores many paths.

    Args:
        cx: The polyhedral complex.
        start: Starting data point as torch.Tensor or np.ndarray.
        end: Ending data point as torch.Tensor or np.ndarray.

    Returns:
        list or None: A list of Polyhedron objects representing the path
            from start to end, or None if no path is found.
    """
    start_poly = cx.add_polyhedron(start) if isinstance(start, Polyhedron) else cx.add_point(start)
    end_poly = cx.add_polyhedron(end) if isinstance(end, Polyhedron) else cx.add_point(end)
    return _greedy_path_helper(cx, start_poly, end_poly)


@with_verbosity
def hamming_astar(
    cx: Complex,
    start: torch.Tensor | np.ndarray | Polyhedron,
    end: torch.Tensor | np.ndarray | Polyhedron,
    nworkers: int | None = None,
    bound: float | None = None,
    max_polys: float = float("inf"),
    verbose: int | None = None,
    num_threads: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Find a path between two data polyhedra using the A* search algorithm.

    A* with Hamming distance between sign sequences as the heuristic, plus Euclidean
    distance between interior points to break ties.

    Args:
        cx: The polyhedral complex.
        start: Starting data point as torch.Tensor or np.ndarray.
        end: Ending data point as torch.Tensor or np.ndarray.
        nworkers: Number of worker processes for parallel neighbor evaluation.
            ``None`` (default) uses ``min(CPU count, number of ReLU units)``.
        bound: Constraint radius when computing halfspaces.
            Defaults to config.DEFAULT_SEARCH_BOUND.
        max_polys: Maximum number of polyhedra to explore during search.
            Defaults to infinity.
        verbose: ``0`` silences output; ``1`` shows a progress bar. ``None`` uses
            :data:`relucent.config.VERBOSE`.
        **kwargs: Additional arguments passed to :func:`~relucent.geometry.calculations.shis`.

    Returns:
        dict[str, Any]: A dictionary with the following keys:

            - ``"path"`` (``list[Polyhedron] | None``): The found path from ``start`` to
              ``end`` as a list of adjacent polyhedra (inclusive endpoints), or ``None``
              if no path was found before termination.
            - ``"succeeded"`` (``bool``): True iff a path to ``end`` was found.
            - ``"termination"`` (``str``): Why the search ended. One of:
              ``"found"``, ``"same_region"``, ``"max_polys"``, ``"exhausted"``, ``"stopped"``.

            - ``"start"`` (``Polyhedron``): The start polyhedron used for the search.
            - ``"goal"`` (``Polyhedron``): The goal polyhedron used for the search.

            - ``"explored"`` (``int``): Number of unique polyhedra recorded in the
              predecessor map (approximately ``len(cameFrom) + 1``).
            - ``"expanded"`` (``int``): Number of times a node was popped and expanded
              (i.e., processed to generate neighbors).
            - ``"generated"`` (``int``): Number of neighbor candidates generated.
            - ``"bad_shi_computations"`` (``int``): Count of SHI/neighbor computations that
              raised/returned an error and were skipped.

            - ``"hamming_start_goal"`` (``int``): Hamming distance between the start and
              goal sign sequences.
            - ``"best_seen_hamming"`` (``int``): Minimum Hamming distance to the goal seen
              among expanded/considered nodes.
            - ``"best_seen_heuristic"`` (``float``): Minimum heuristic value
              ``ham(p, goal) + bias(p, goal)`` seen during search.

            - ``"lower_bound_optimal_length"`` (``int``): A running lower bound on the
              optimal path length, computed as ``min_p (g(p) + ham(p, goal))``.
            - ``"best_path_length"`` (``int | None``): If ``path`` is not ``None``,
              equals ``len(path) - 1``; otherwise ``None``.

    Raises:
        ValueError: If the start point lies exactly on a neuron's boundary.
    """
    del verbose  # applied by @with_verbosity
    if bound is None:
        bound = cfg.DEFAULT_SEARCH_BOUND

    start_poly = cx.add_polyhedron(start) if isinstance(start, Polyhedron) else cx.add_point(start)
    end_poly = cx.add_polyhedron(end) if isinstance(end, Polyhedron) else cx.add_point(end)
    hamming_start_goal = int((start_poly.ss_np != end_poly.ss_np).sum())
    if start_poly == end_poly:
        _ = start_poly.finite

        start_poly._interior_point = start_poly.find_interior_point()
        start_poly._shis = cast(list[int], calculations.shis(start_poly, bound=bound, collect_info=False))
        start_poly._shis_strict = True
        return {
            "path": [start_poly],
            "succeeded": True,
            "termination": "same_region",
            "start": start_poly,
            "goal": end_poly,
            "explored": 1,
            "expanded": 0,
            "generated": 0,
            "bad_shi_computations": 0,
            "hamming_start_goal": hamming_start_goal,
            "best_seen_hamming": 0,
            "best_seen_heuristic": 0.0,
            "lower_bound_optimal_length": 0,
            "best_path_length": 0,
        }

    if (start_poly.ss_np == 0).any():
        raise ValueError("Start point must not be on a hyperplane")

    # Can't use more workers than there are SHIs to hand out per expansion step.
    if nworkers is None:
        nworkers = process_aware_cpu_count() or 1
    nworkers = min(int(nworkers), cx.n)

    cameFrom: dict[Polyhedron, Polyhedron] = {}
    gScore = defaultdict(lambda: float("inf"))
    fScore = defaultdict(lambda: float("inf"))

    # Iterable queue: map(astar_calculations, openSet) blocks on pop until neighbors are pushed.
    openSet = NonBlockingQueue(
        queue_class=UpdatablePriorityQueue,
        pop=lambda pq: pq.pop(),
        push=lambda pq, task: pq.push(task, priority=0.0),
        push_with_priority=lambda pq, task, priority: pq.push(task, priority=priority),
    )

    gScore[start_poly] = 0
    fScore[start_poly] = hamming_start_goal

    result = calculations.shis(start_poly, bound=bound, **kwargs)
    assert isinstance(result, list)
    start_poly._shis = result
    start_poly._shis_strict = shis_are_certified(kwargs)

    openSet.push((start_poly,), fScore[start_poly])

    bad_shi_computations = []
    pbar = progress(
        desc="Search Progress",
        mininterval=1,
        leave=True,
        total=max_polys if max_polys != float("inf") else None,
    )
    pbar.update(n=1)

    depth = 0
    goal_reached: Polyhedron | None = None
    min_dist = float("inf")
    best_seen_hamming = hamming_start_goal
    lower_bound_optimal_length = hamming_start_goal
    expanded = 0
    generated = 0
    failed_astar_flips: set[tuple[bytes, int]] = set()

    def heuristic(p: Polyhedron) -> float:
        hamming = (p.ss_np != end_poly.ss_np).sum()
        if p.interior_point is None or end_poly.interior_point is None:
            raise ValueError("Interior point not found")
        dist = np.linalg.norm(p.interior_point - end_poly.interior_point).item()
        bias = -1 / (1 + dist)  # small nudge toward geometrically closer regions at equal Hamming
        return hamming + cfg.advanced.ASTAR_BIAS_WEIGHT * bias

    pool = None
    with worker_context_scope(cx._net, get_volumes=False, num_threads=num_threads):
        try:
            if nworkers > 1:
                pool = worker_pool(
                    nworkers,
                    initializer=set_worker_context,
                    initargs=(cx._net, False, num_threads),
                )
            # Each pop runs geometry+SHIs on a candidate before we expand its neighbors.
            for item in map(partial(astar_calculations, bound=bound, **kwargs), openSet):
                if len(item) >= 2 and isinstance(item[1], Exception):
                    # Useful when debugging spawn-related issues on macOS/Windows.
                    if os.environ.get("RELUCENT_DEBUG_ASTAR"):
                        err = item[1]
                        logger.debug("A* SHI computation failed: %s: %s", type(err).__name__, err)
                    bad_shi_computations.append(item)
                    continue

                assert len(item) > 0
                p = cast(Polyhedron, item[0])
                expanded += 1
                depth = int(gScore[p])
                p_hamming = int((p.ss_np != end_poly.ss_np).sum())
                best_seen_hamming = min(best_seen_hamming, p_hamming)
                lower_bound_optimal_length = min(lower_bound_optimal_length, int(gScore[p]) + p_hamming)

                if p == end_poly:
                    goal_reached = end_poly
                    break

                shis_to_try = [s for s in p.shis if (p.tag, int(s)) not in failed_astar_flips]
                if nworkers == 1:
                    neighbor_iter = map(partial(get_ip, p), shis_to_try)
                else:
                    assert pool is not None
                    neighbor_iter = pool.imap_unordered(
                        partial(get_ip, p),
                        shis_to_try,
                        chunksize=max(cx.n // nworkers, 1),
                    )

                for candidate, neighbor_shi in neighbor_iter:
                    if not isinstance(candidate, Polyhedron):
                        failed_astar_flips.add((p.tag, int(neighbor_shi)))
                    else:
                        generated += 1
                        tentative_gScore = gScore[p] + 1  # adjacent cells differ in exactly one nonzero ss entry
                        if candidate._net is None:
                            candidate._net = cx._net
                        if tentative_gScore < gScore[candidate]:
                            cameFrom[candidate] = p
                            gScore[candidate] = tentative_gScore
                            dist = heuristic(candidate)
                            fScore[candidate] = tentative_gScore + dist
                            if dist < min_dist:
                                min_dist = dist
                            neighbor_hamming = int((candidate.ss_np != end_poly.ss_np).sum())
                            lower_bound_optimal_length = min(
                                lower_bound_optimal_length, int(tentative_gScore) + neighbor_hamming
                            )
                            openSet.push((candidate,), fScore[candidate])

                pbar.update(n=len(cameFrom) - pbar.n)
                open_len = len(openSet)
                pbar.set_postfix_str(
                    f"LB*: {lower_bound_optimal_length} MinHeuristic: {min_dist:.3f} BestHam: {best_seen_hamming} "
                    + f"Depth: {depth} Open Set: {open_len} "
                    + f"Mistakes: {len(bad_shi_computations)} | Finite: {p.finite} # SHIs: {len(p.shis)}",
                    refresh=False,
                )

                if open_len == 0 or len(cameFrom) >= max_polys:
                    break
        finally:
            if pool is not None:
                pool.shutdown()
            openSet.close()
            # Same stale-lock guard as in searcher().
            tqdm.get_lock().locks = []
            pbar.close()
    succeeded = goal_reached == end_poly
    path: list[Polyhedron] | None
    termination: str
    if succeeded:
        termination = "found"
        path = [end_poly]
        # Walk cameFrom backward from goal to start.
        while path[-1] != start_poly:
            if cfg.CAREFUL_MODE:
                assert cameFrom[path[-1]] not in path, path
            path.append(cameFrom[path[-1]])
        path.reverse()
    else:
        path = None
        if len(cameFrom) >= max_polys:
            termination = "max_polys"
        elif len(openSet) == 0:
            termination = "exhausted"
        else:
            termination = "stopped"

    best_path_length = (len(path) - 1) if path is not None else None
    return {
        "path": path,
        "succeeded": succeeded,
        "termination": termination,
        "start": start_poly,
        "goal": end_poly,
        "explored": len(cameFrom) + 1,
        "expanded": expanded,
        "generated": generated,
        "bad_shi_computations": len(bad_shi_computations),
        "hamming_start_goal": hamming_start_goal,
        "best_seen_hamming": best_seen_hamming,
        "best_seen_heuristic": min_dist,
        "lower_bound_optimal_length": int(lower_bound_optimal_length),
        "best_path_length": best_path_length,
    }
