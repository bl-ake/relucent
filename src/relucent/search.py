"""Search and pathfinding over a polyhedral complex."""

import os
import random
import warnings
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from tqdm.auto import tqdm

import relucent.config as cfg
from relucent.calculations import get_shis
from relucent.poly import Polyhedron
from relucent.ss import SSManager
from relucent.utils import (
    BlockingQueue,
    NonBlockingQueue,
    UpdatablePriorityQueue,
    get_mp_context,
    process_aware_cpu_count,
)

if TYPE_CHECKING:
    from relucent.complex import Complex

__all__ = [
    "astar_calculations",
    "get_ip",
    "greedy_path",
    "hamming_astar",
    "parallel_compute_geometric_properties",
    "parallel_add",
    "search_calculations",
    "searcher",
]

DEFAULT_GEOMETRY_PROPERTIES: tuple[str, ...] = (
    "halfspaces",
    "W",
    "b",
    "num_dead_relus",
    "finite",
    "center",
    "inradius",
    "interior_point",
    "interior_point_norm",
    "Wl2",
)


def _normalize_geometry_properties(properties: Iterable[str] | None) -> tuple[str, ...]:
    if properties is None:
        return DEFAULT_GEOMETRY_PROPERTIES
    return tuple(dict.fromkeys(str(p).strip() for p in properties if str(p).strip()))


def search_calculations(
    task: np.ndarray | torch.Tensor | tuple[Any, ...],
    geometry_properties: Iterable[str] | None = None,
    keep_caches: bool = False,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Worker function for neighbor enumeration in search.

    This intentionally computes only what the traversal needs:
    feasibility and supporting halfspace indices (SHIs).
    """
    from relucent import complex as cx

    assert cx._net is not None, "set_globals must be used as pool initializer"
    ss = task[0] if isinstance(task, tuple) else task
    rest = task[1:] if isinstance(task, tuple) else ()
    p = Polyhedron(cx._net, ss)

    if not p.feasible:
        return "Polyhedron is infeasible (empty).", *rest

    try:
        if p._shis is None:
            result = get_shis(p, env=cx.env, **kwargs)
            p._shis = result[0] if isinstance(result, tuple) else result
    except ValueError as error:
        return error, *rest

    props = _normalize_geometry_properties(geometry_properties)
    if props:
        try:
            p.get_geometry(props, env=cx.env)
        except ValueError as error:
            return error, *rest

    if keep_caches:
        p._preserve_cache_on_pickle = True
    else:
        p.clean_data()
    if isinstance(p._shis, list):
        random.shuffle(p._shis)
    return (p, *rest)


def geometric_calculations(
    task: tuple[np.ndarray, list[int] | None, int] | tuple[Any, ...],
    geometry_properties: Iterable[str] | None = None,
    keep_caches: bool = False,
) -> tuple[Any, ...]:
    """Worker function for geometric-property computation on known polyhedra."""
    from relucent import complex as cx

    assert cx._net is not None, "set_globals must be used as pool initializer"
    ss, shis, poly_index, *rest = task
    p = Polyhedron(cx._net, ss, shis=shis)
    props = _normalize_geometry_properties(geometry_properties)
    try:
        p.get_geometry(props, env=cx.env)
    except ValueError as error:
        return error, poly_index, *rest

    if keep_caches:
        p._preserve_cache_on_pickle = True
    else:
        p.clean_data()
    return (p, poly_index, *rest)


def parallel_compute_geometric_properties(
    cx: "Complex",
    nworkers: int | None = None,
    geometry_properties: Iterable[str] | None = None,
    keep_caches: bool = False,
    verbose: int = 1,
) -> dict[str, Any]:
    """Compute selected properties for all existing polyhedra in parallel.

    Args:
        cx: Complex whose polyhedra should be updated.
        nworkers: Number of worker processes to use.
        geometry_properties: Iterable of cache/property names to compute.
            If None, defaults to :data:`DEFAULT_GEOMETRY_PROPERTIES`.
        keep_caches: If True, keep heavy caches during worker serialization.
        verbose: Whether to print progress.
    """
    from relucent.complex import set_globals

    if len(cx) == 0:
        return {"Computed": 0, "Failed": []}

    nworkers = nworkers or process_aware_cpu_count()
    if verbose:
        print(f"Computing geometric properties on {nworkers} workers")

    tasks = [(poly.ss_np, poly._shis, i) for i, poly in enumerate(cx)]
    failed: list[tuple[int, str]] = []
    computed = 0
    with get_mp_context().Pool(nworkers, initializer=set_globals, initargs=(cx._net, False)) as pool:
        for result in tqdm(
            pool.imap_unordered(
                partial(
                    geometric_calculations,
                    geometry_properties=geometry_properties,
                    keep_caches=keep_caches,
                ),
                tasks,
            ),
            total=len(tasks),
            desc="Computing Geometry",
            mininterval=5,
            disable=not verbose,
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


def parallel_add(
    cx: "Complex",
    points: Iterable[torch.Tensor | np.ndarray],
    nworkers: int | None = None,
    bound: float | None = None,
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

    Returns:
        list: A list of Polyhedron objects (or None for failed computations)
            corresponding to the input points.
    """
    if bound is None:
        bound = cfg.DEFAULT_PARALLEL_ADD_BOUND
    from relucent.complex import set_globals

    nworkers = nworkers or process_aware_cpu_count()
    print(f"Running on {nworkers} workers")
    sss = []
    for p in tqdm(points, desc="Getting SSs", mininterval=5):
        s = cx.point2ss(p)
        sss.append(s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s)

    with get_mp_context().Pool(nworkers, initializer=set_globals, initargs=(cx._net,)) as pool:
        ps = pool.map(
            partial(
                geometric_calculations,
                geometry_properties=DEFAULT_GEOMETRY_PROPERTIES,
                keep_caches=False,
            ),
            tqdm(sss, desc="Adding Polys", mininterval=5),
        )
        ps = [p[0] if isinstance(p[0], Polyhedron) else None for p in ps]
        for p in ps:
            if p is not None:
                cx.add_polyhedron(p)
        return ps


def get_ip(
    p: Polyhedron,
    shi: int,
) -> tuple[Polyhedron, int] | tuple[ValueError, int]:
    """Get an interior point for a neighbor polyhedron across a supporting hyperplane.

    This function is used by the A* search algorithm to find interior points for
    neighboring polyhedra. It flips the element of the sign sequence at the specified
    supporting hyperplane index (SHI) and attempts to find an interior point,
    increasing the search radius if necessary.

    Args:
        p: The source Polyhedron object.
        shi: The index of the supporting hyperplane to cross.

    Returns:
        tuple: If successful, returns (neighbor_polyhedron, shi). If a ValueError
            occurs, returns (error, None).
    """
    from relucent import complex as cx

    try:
        ss = p.ss_np.copy()
        ss[0, shi] = -ss[0, shi]
        assert cx._net is not None, "set_globals must be used as pool initializer"
        n = Polyhedron(cx._net, ss)
        for max_radius in cfg.INTERIOR_POINT_RADIUS_SEQUENCE:
            try:
                n._interior_point = n.get_interior_point(env=cx.env, max_radius=max_radius)
            except ValueError:
                print("Increasing max radius to find interior point")
        return n, shi
    except ValueError as e:
        return e, shi


def astar_calculations(
    task: Polyhedron | tuple[Any, ...],
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Worker function for computing polyhedron properties in A* search.

    Similar to search_calculations, but specifically designed for A* search algorithm.
    Computes center, inradius, interior point, and supporting hyperplane indices
    for polyhedra during pathfinding.

    Args:
        task: Either a Polyhedron object or a tuple containing (Polyhedron, ...)
            with additional data to be passed through.
        **kwargs: Additional arguments passed to :func:`~relucent.poly.get_shis`, such as
            'bound'.

    Returns:
        tuple: If successful, returns (Polyhedron, *rest). If an exception occurs
            during SHI computation, returns (Polyhedron, error, *rest).
    """
    from relucent import complex as cx

    p = task[0] if isinstance(task, tuple) else task
    rest = task[1:] if isinstance(task, tuple) else ()
    if p._net is None:
        assert cx._net is not None, "set_globals must be used as pool initializer"
        p._net = cx._net

    if p.finite is None:
        raise ValueError("Polyhedron is infeasible (empty).")
    if p._interior_point is None:
        p._interior_point = p.get_interior_point(env=cx.env)

    try:
        if p._shis is None:
            result = get_shis(p, env=cx.env, **kwargs)
            p._shis = result[0] if isinstance(result, tuple) else result
    except Exception as error:
        return p, error, *rest
    p.clean_data()
    assert isinstance(p._shis, list), "get_shis() returns a list"
    random.shuffle(p._shis)
    return p, *rest


def searcher(
    cx: "Complex",
    start=None,
    max_depth=float("inf"),
    max_polys=float("inf"),
    queue=None,
    bound=None,
    nworkers=None,
    verbose=1,
    cube_radius: float | None = None,
    cube_mode: str = "unrestricted",
    geometry_properties: Iterable[str] | None = None,
    keep_caches: bool = False,
    **kwargs,
):
    """Search for polyhedra in the complex by discovering neighbors.

    This is a generic search method that can be configured for different
    traversal strategies (BFS, DFS, random walk) by providing different
    queue types. It starts from a given point and explores the complex by
    crossing supporting hyperplanes to discover adjacent polyhedra.

    See Complex.bfs(), Complex.dfs(), and Complex.random_walk() for examples.

    Args:
        cx: The polyhedral complex.
        start: Starting point (torch.Tensor / np.ndarray / array-like) or a
            Polyhedron, or None (defaults to origin). Defaults to None.
        max_depth: Maximum search depth (number of hyperplane crossings).
            Defaults to infinity.
        max_polys: Maximum number of polyhedra to discover. Defaults to infinity.
        queue: Queue object that defines the order in which polyhedra are
            searched. Must have push() and pop() methods. If None, uses
            BlockingQueue (FIFO). Defaults to None.
        bound: Constraint radius for numerical stability when computing halfspaces.
            Important for numerical stability. Defaults to config.DEFAULT_SEARCH_BOUND.
        nworkers: Number of worker processes for parallel computation. If None,
            uses the number of CPU cores. Defaults to None.
        verbose: Whether to print progress information. Defaults to 1.
        geometry_properties: Iterable of polyhedron cache/property names to
            compute for each discovered polyhedron. Defaults to
            :data:`DEFAULT_GEOMETRY_PROPERTIES` (all non-SciPy geometry caches).
            Pass an empty iterable for topology-only search.
        keep_caches: If True, retain heavy caches (such as
            ``halfspaces``, ``W``, and ``b``) when worker polyhedra are sent
            back to the parent process. Defaults to False.
        **kwargs: Additional arguments passed to :func:`~relucent.poly.get_shis`.

    Returns:
        dict: Search information dictionary containing:
            - "Search Depth": Maximum depth reached
            - "Avg # Facets Uncorrected": Average number of facets per polyhedron
            - "Search Time": Elapsed time in seconds
            - "Bad SHI Computations": List of failed computations
            - "Complete": Whether search completed (no unprocessed items)

    Raises:
        ValueError: If the start point lies on a hyperplane (has zero in SS).
    """
    from relucent.complex import set_globals

    if bound is None:
        bound = cfg.DEFAULT_SEARCH_BOUND

    if cube_mode not in {"unrestricted", "intersect", "clipped", "exclude"}:
        raise ValueError("cube_mode must be one of {'unrestricted', 'intersect', 'clipped', 'exclude'}")
    elif cube_mode != "unrestricted":
        assert cube_radius is not None, "cube_radius must be provided when cube_mode is not 'unrestricted'"
        if verbose:
            print(f"Applying cube filter with mode '{cube_mode}' and radius {cube_radius}")
    elif cube_radius is not None:
        warnings.warn("cube_radius is provided but cube_mode is 'unrestricted'. Ignoring cube_radius.", stacklevel=2)
        cube_radius = None

    def _poly_intersects_cube(p: Polyhedron) -> bool:
        try:
            p.get_bounded_halfspaces(cast(float, cube_radius))
            return True
        except ValueError:
            return False

    def _apply_cube_filter(p: Polyhedron) -> bool:
        """Apply the cube filter to a polyhedron.

        Returns True if the polyhedron should be kept (and possibly clipped),
        or False if it should be discarded from the search.
        """
        intersects = _poly_intersects_cube(p)

        if cube_mode == "intersect":
            return intersects
        if cube_mode == "exclude":
            return not intersects

        if not intersects:
            return False
        bounded_halfspaces = p.get_bounded_halfspaces(cast(float, cube_radius))
        p._halfspaces = bounded_halfspaces  # type: ignore[assignment]
        p._halfspaces_np = bounded_halfspaces
        return p.feasible

    found_sss = SSManager()
    nworkers = nworkers or process_aware_cpu_count()
    if verbose:
        print(f"Running on {nworkers} workers")
    if queue is None:
        queue = BlockingQueue(
            queue_class=deque,
            pop=cast(Callable[[deque[object]], object], deque.pop),
            push=cast(Callable[[deque[object], object], None], deque.append),
        )
    if start is None:
        start = cx.add_point(torch.zeros(cx._net.input_shape, device=cx._net.device, dtype=cx._net.dtype))
    elif isinstance(start, Polyhedron):
        start = cx.add_polyhedron(start)
    else:
        start = cx.add_point(start)
    if cube_mode != "unrestricted" and not _apply_cube_filter(start):
        queue.close()
        return {
            "Search Depth": 0,
            "Avg # Facets Uncorrected": 0.0,
            "Search Time": 0.0,
            "Bad SHI Computations": [],
            "Complete": True,
        }
    found_sss.add(start.ss_np)
    if (start.ss_np == 0).any():
        raise ValueError("Start point must not be on a hyperplane")
    result = get_shis(start, bound=bound, **kwargs)
    assert isinstance(result, list)
    start._shis = result
    for shi in start.shis:
        new_ss = start.ss_np.copy()
        new_ss[0, shi] *= -1
        found_sss.add(new_ss)
        queue.push((new_ss, shi, 1, cx.ssm[start.ss_np]))
        assert new_ss in found_sss

    rolling_average = len(start.shis)
    bad_shi_computations = []
    pbar = tqdm(
        desc="Search Progress",
        mininterval=5,
        total=max_polys if max_polys != float("inf") else None,
        disable=not verbose,
    )
    pbar.update(n=1)
    pbar.get_lock().locks = []

    unprocessed = len(queue)
    depth = 0

    with get_mp_context().Pool(nworkers, initializer=set_globals, initargs=(cx._net, False)) as pool:
        try:
            for p, shi, depth, node_index in pool.imap_unordered(
                partial(
                    search_calculations,
                    bound=bound,
                    geometry_properties=geometry_properties,
                    keep_caches=keep_caches,
                    **kwargs,
                ),
                queue,
            ):
                unprocessed -= 1
                node = cast(Polyhedron, cx.index2poly[node_index])
                if not isinstance(p, Polyhedron):
                    bad_shi_computations.append((node, shi, depth, str(p)))
                    node.remove_shi(shi)
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

                if cube_mode != "unrestricted" and not _apply_cube_filter(p):
                    if unprocessed == 0 or len(cx) >= max_polys:
                        break
                    continue

                p = cx.add_polyhedron(p)

                if getattr(p, "warnings", None):
                    for warning in p.warnings:
                        try:
                            warnings.warn(warning, stacklevel=2)
                        except Exception as e:
                            print(warning, type(warning), e, end="\n\n")

                if depth < max_depth:
                    for new_shi in p.shis:
                        if new_shi != shi and len(cx) < max_polys:
                            ss = p.ss_np.copy()
                            ss[0, new_shi] *= -1
                            if ss not in found_sss:
                                queue.push((ss, new_shi, depth + 1, cx.ssm[p.ss_np]))
                                found_sss.add(ss)
                                unprocessed += 1

                pbar.update(n=len(cx) - pbar.n)
                rolling_average = (rolling_average * (len(cx) - 1) + len(p.shis)) / len(cx)

                assert isinstance(p._shis, list)
                pbar.set_postfix_str(
                    f"Depth: {depth}  Unprocessed: {unprocessed}  Faces: {len(p._shis)}  Avg: {rolling_average:.2f} "
                    + f"IP Norm: {p._interior_point_norm or -1:.2f}  Finite: {p._finite} "
                    + f"Mistakes: {len(bad_shi_computations)}",
                    refresh=False,
                )

                if unprocessed == 0 or len(cx) >= max_polys:
                    break
        except Exception:
            raise
        finally:
            queue.close()
            pbar.close()

            pool.close()
            pool.terminate()
            pool.join()

    search_info = {
        "Search Depth": depth,
        "Avg # Facets Uncorrected": rolling_average,
        "Search Time": pbar.format_dict["elapsed"],
        "Bad SHI Computations": bad_shi_computations,
        "Complete": unprocessed == 0,
    }

    return search_info


def _greedy_path_helper(cx: "Complex", start: Polyhedron, end: Polyhedron, diffs: set[int] | None = None) -> list[Polyhedron]:
    if start == end:
        return [start]

    if (start.ss_np == 0).any():
        raise ValueError("Start point must not be on a hyperplane")

    diffs = diffs or set(np.argwhere((start.ss_np != end.ss_np).ravel()).ravel().tolist())

    print("Diffs:", diffs)

    if not start._shis:
        get_shis(start)
    shis_set = set(start.shis)
    groupa = shis_set & diffs
    groupb = shis_set - diffs
    for shi in list(groupa):
        print("Crossing", shi)
        new_ss = start.ss_np.copy()
        new_ss[0, shi] *= -1
        next_poly = cx.ss2poly(new_ss)
        rest = _greedy_path_helper(cx, next_poly, end, diffs - {shi})
        if rest is not None:
            return [start] + rest
    for shi in list(groupb):
        print("Crossing", shi)
        new_ss = start.ss_np.copy()
        new_ss[0, shi] *= -1
        next_poly = cx.ss2poly(new_ss)
        rest = _greedy_path_helper(cx, next_poly, end, diffs | {shi})
        if rest is not None:
            return [start] + rest
    return []


def greedy_path(
    cx: "Complex",
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


def hamming_astar(
    cx: "Complex",
    start: torch.Tensor | np.ndarray | Polyhedron,
    end: torch.Tensor | np.ndarray | Polyhedron,
    nworkers: int | None = None,
    bound: float | None = None,
    max_polys: float = float("inf"),
    show_pbar: bool = True,
    num_threads: int | None = None,  ## TODO: Any benefits from using multiple threads here?
    **kwargs: Any,
) -> dict[str, Any]:
    """Find a path between two data polyhedra using the A* search algorithm.

    Uses the A* pathfinding algorithm with a heuristic based on Hamming
    distance between sign sequences, plus Euclidean distance between interior
    points to break ties. The heuristic should be admissible for optimal paths.

    Args:
        cx: The polyhedral complex.
        start: Starting data point as torch.Tensor or np.ndarray.
        end: Ending data point as torch.Tensor or np.ndarray.
        nworkers: Number of worker processes for parallel computation.
            Defaults to 1.
        bound: Constraint radius for numerical stability when computing halfspaces.
            Important for numerical stability. Defaults to config.DEFAULT_SEARCH_BOUND.
        max_polys: Maximum number of polyhedra to explore during search.
            Defaults to infinity.
        show_pbar: Whether to display a progress bar. Defaults to True.
        **kwargs: Additional arguments passed to get_shis().

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
    from relucent.complex import set_globals

    if bound is None:
        bound = cfg.DEFAULT_SEARCH_BOUND

    start_poly = cx.add_polyhedron(start) if isinstance(start, Polyhedron) else cx.add_point(start)
    end_poly = cx.add_polyhedron(end) if isinstance(end, Polyhedron) else cx.add_point(end)
    hamming_start_goal = int((start_poly.ss_np != end_poly.ss_np).sum())
    if start_poly == end_poly:
        print("Start and end points are in the same region")
        _ = start_poly.finite

        start_poly._interior_point = start_poly.get_interior_point()
        start_poly._shis = cast(list[int], get_shis(start_poly, bound=bound, collect_info=False))
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

    nhs = len(start_poly.halfspaces)
    nworkers = min(cast(int, nworkers if isinstance(nworkers, int) else (process_aware_cpu_count() or 1)), nhs)
    print(f"Using {nworkers} workers")

    cameFrom: dict[Polyhedron, Polyhedron] = {}
    gScore = defaultdict(lambda: float("inf"))
    fScore = defaultdict(lambda: float("inf"))

    openSet = NonBlockingQueue(
        queue_class=UpdatablePriorityQueue,
        pop=lambda pq: pq.pop(),
        push=lambda pq, task: pq.push(task, priority=0.0),
        push_with_priority=lambda pq, task, priority: pq.push(task, priority=priority),
    )

    gScore[start_poly] = 0
    fScore[start_poly] = hamming_start_goal

    result = get_shis(start_poly, bound=bound, **kwargs)
    assert isinstance(result, list)
    start_poly._shis = result

    openSet.push((start_poly,), fScore[start_poly])

    bad_shi_computations = []
    pbar = tqdm(
        desc="Search Progress" + (str(show_pbar) if show_pbar is not True else ""),
        mininterval=1,
        leave=True,
        total=max_polys if max_polys != float("inf") else None,
        disable=not show_pbar,
    )
    pbar.update(n=1)

    depth = 0
    neighbor: Polyhedron | ValueError | None = None
    min_dist = float("inf")
    best_seen_hamming = hamming_start_goal
    lower_bound_optimal_length = hamming_start_goal
    expanded = 0
    generated = 0

    def heuristic(p: Polyhedron) -> float:
        hamming = (p.ss_np != end_poly.ss_np).sum()
        if p.interior_point is None or end_poly.interior_point is None:
            raise ValueError("Interior point not found")
        dist = np.linalg.norm(p.interior_point - end_poly.interior_point).item()
        bias = -1 / (1 + dist)
        return hamming + cfg.ASTAR_BIAS_WEIGHT * bias

    pool = None
    try:
        set_globals(cx._net)
        if nworkers > 1:
            pool = get_mp_context().Pool(nworkers, initializer=set_globals, initargs=(cx._net, False, num_threads))
        for item in map(partial(astar_calculations, bound=bound, **kwargs), openSet):
            if len(item) >= 2 and isinstance(item[1], Exception):
                # Useful when debugging spawn-related issues on macOS/Windows.
                if os.environ.get("RELUCENT_DEBUG_ASTAR"):
                    err = item[1]
                    print(f"A* SHI computation failed: {type(err).__name__}: {err}")
                bad_shi_computations.append(item)
                continue

            assert len(item) > 0  ## TODO: Simplify
            p = cast(Polyhedron, item[0])
            expanded += 1
            depth = int(gScore[p])
            p_hamming = int((p.ss_np != end_poly.ss_np).sum())
            best_seen_hamming = min(best_seen_hamming, p_hamming)
            lower_bound_optimal_length = min(lower_bound_optimal_length, int(gScore[p]) + p_hamming)

            if p == end_poly:
                neighbor = end_poly
                break

            if nworkers == 1:
                neighbor_iter = map(partial(get_ip, p), p.shis)
            else:
                assert pool is not None
                neighbor_iter = pool.imap_unordered(
                    partial(get_ip, p),
                    p.shis,
                    chunksize=max(nhs // nworkers, 1),
                )

            for neighbor, neighbor_shi in neighbor_iter:
                if not isinstance(neighbor, Polyhedron):
                    p.remove_shi(neighbor_shi)
                else:
                    generated += 1
                    tentative_gScore = gScore[p] + 1  ## The hamming distance between two adjacent polyhedra is always 1
                    if neighbor._net is None:
                        neighbor._net = cx._net
                    if tentative_gScore < gScore[neighbor]:
                        cameFrom[neighbor] = p
                        gScore[neighbor] = tentative_gScore
                        dist = heuristic(neighbor)
                        fScore[neighbor] = tentative_gScore + dist
                        if dist < min_dist:
                            min_dist = dist
                        neighbor_hamming = int((neighbor.ss_np != end_poly.ss_np).sum())
                        lower_bound_optimal_length = min(lower_bound_optimal_length, int(tentative_gScore) + neighbor_hamming)
                        openSet.push((neighbor,), fScore[neighbor])

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
    except Exception:
        raise
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
            pool.close()

        openSet.close()
        tqdm.get_lock().locks = []
        pbar.close()
    succeeded = neighbor == end_poly
    path: list[Polyhedron] | None
    termination: str
    if succeeded:
        termination = "found"
        path = [end_poly]
        while path[-1] != start_poly:
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
