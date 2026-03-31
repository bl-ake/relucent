"""Search and pathfinding over a polyhedral complex."""

import multiprocessing as mp
import random
import warnings
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from tqdm.auto import tqdm

from relucent.config import (
    ASTAR_BIAS_WEIGHT,
    DEFAULT_PARALLEL_ADD_BOUND,
    DEFAULT_SEARCH_BOUND,
    INTERIOR_POINT_RADIUS_SEQUENCE,
)
from relucent.poly import Polyhedron, get_hs, get_shis
from relucent.ss import SSManager
from relucent.utils import (
    BlockingQueue,
    NonBlockingQueue,
    UpdatablePriorityQueue,
    process_aware_cpu_count,
)

if TYPE_CHECKING:
    from relucent.complex import Complex

__all__ = [
    "astar_calculations",
    "get_ip",
    "greedy_path",
    "hamming_astar",
    "parallel_add",
    "poly_calculations",
    "searcher",
]


def poly_calculations(
    task: np.ndarray | torch.Tensor | tuple[Any, ...],
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Worker function for computing polyhedron properties in parallel.

    Used by parallel_add() and searcher() to compute halfspaces, center, inradius,
    interior point, and supporting hyperplane indices (SHIs) for a given sign sequence.

    Args:
        task: Either a sign sequence or a tuple containing (sign sequence, ...) with
            additional data to be passed through.
        **kwargs: Additional arguments passed to :func:`~relucent.poly.get_shis`, such as
            'collect_info' or 'bound'.

    Returns:
        tuple: If successful, returns (Polyhedron, *rest) where rest contains
            any additional data from the input task. If a ValueError occurs
            during computation, returns (error, *rest).
    """
    from relucent import complex as cx

    assert cx.net is not None, "set_globals must be used as pool initializer"
    ss = task[0] if isinstance(task, tuple) else task
    rest = task[1:] if isinstance(task, tuple) else ()
    p = Polyhedron(cx.net, ss)

    try:
        halfspaces, W, b, num_dead_relus = cast(tuple[Any, Any, Any, int], get_hs(p, get_all_Ab=False))
        assert isinstance(halfspaces, (torch.Tensor, np.ndarray))
        assert isinstance(W, (torch.Tensor, np.ndarray))
        assert isinstance(b, (torch.Tensor, np.ndarray))
        p._halfspaces = halfspaces
        p._W = W
        p._b = b
        p._num_dead_relus = cast(int, num_dead_relus)

        center, inradius = p.get_center_inradius(env=cx.env)
        p._center = center
        p._inradius = inradius
        p._finite = center is not None

        p._interior_point = p.get_interior_point(env=cx.env)
        if p.interior_point is not None:
            p._interior_point_norm = np.linalg.norm(p.interior_point).item()
        else:
            p._interior_point_norm = float("inf")
        if isinstance(p.W, torch.Tensor):
            p._Wl2 = torch.linalg.norm(p.W).item()
        else:
            p._Wl2 = np.linalg.norm(p.W).item()

        if cx.dim <= 6 and cx.get_vol_calc:
            _ = p.volume
        if p._shis is None:
            if "collect_info" in kwargs:
                shis, shi_info = get_shis(p, env=cx.env, **kwargs)
                assert isinstance(shis, list)
                p._shis = shis
            else:
                result = get_shis(p, env=cx.env, **kwargs)
                p._shis = result[0] if isinstance(result, tuple) else result
    except ValueError as error:
        return error, *rest
    p.clean_data()

    if isinstance(p._shis, list):
        random.shuffle(p._shis)
    return (p, *rest)


def parallel_add(
    cx: "Complex",
    points: Iterable[torch.Tensor | np.ndarray],
    nworkers: int | None = None,
    bound: float = DEFAULT_PARALLEL_ADD_BOUND,
    **kwargs: Any,
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
        **kwargs: Additional arguments passed to poly_calculations() and get_shis().

    Returns:
        list: A list of Polyhedron objects (or None for failed computations)
            corresponding to the input points.
    """
    from relucent.complex import set_globals

    nworkers = nworkers or process_aware_cpu_count()
    print(f"Running on {nworkers} workers")
    sss = []
    for p in tqdm(points, desc="Getting SSs", mininterval=5):
        s = cx.point2ss(p)
        sss.append(s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s)

    with mp.Pool(nworkers, initializer=set_globals, initargs=(cx.net,)) as pool:
        ps = pool.map(partial(poly_calculations, bound=bound, **kwargs), tqdm(sss, desc="Adding Polys", mininterval=5))
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
        assert cx.net is not None, "set_globals must be used as pool initializer"
        n = Polyhedron(cx.net, ss)
        for max_radius in INTERIOR_POINT_RADIUS_SEQUENCE:
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

    Similar to poly_calculations, but specifically designed for A* search algorithm.
    Computes center, inradius, interior point, and supporting hyperplane indices
    for polyhedra during pathfinding.

    Args:
        task: Either a Polyhedron object or a tuple containing (Polyhedron, ...)
            with additional data to be passed through.
        **kwargs: Additional arguments passed to :func:`~relucent.poly.get_shis`, such as
            'collect_info' or 'bound'.

    Returns:
        tuple: If successful, returns (Polyhedron, *rest). If an exception occurs
            during SHI computation, returns (Polyhedron, error, *rest).
    """
    from relucent import complex as cx

    p = task[0] if isinstance(task, tuple) else task
    rest = task[1:] if isinstance(task, tuple) else ()
    if p.net is None:
        assert cx.net is not None, "set_globals must be used as pool initializer"
        p.net = cx.net

    if p._inradius is None:
        center, inradius = p.get_center_inradius(env=cx.env)
        p._center = center
        p._inradius = inradius
        p._finite = center is not None
    if p._interior_point is None:
        p._interior_point = p.get_interior_point(env=cx.env)

    try:
        if p._shis is None:
            if "collect_info" in kwargs:
                shis, shi_info = get_shis(p, env=cx.env, **kwargs)
                assert isinstance(shis, list)
                p._shis = shis
            else:
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
    bound=DEFAULT_SEARCH_BOUND,
    nworkers=None,
    get_volumes=True,
    verbose=1,
    cube_radius: float | None = None,
    cube_mode: str = "unrestricted",
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
        get_volumes: Whether to compute volumes for polyhedra when input
            dimension <= 6. Defaults to True.
        verbose: Whether to print progress information. Defaults to 1.
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

    if cube_mode not in {"unrestricted", "intersect", "clipped", "exclude"}:
        raise ValueError("cube_mode must be one of {'unrestricted', 'intersect', 'clipped', 'exclude'}")
    elif cube_mode != "unrestricted":
        assert cube_radius is not None, "cube_radius must be provided when cube_mode is not 'unrestricted'"
        if verbose:
            print(f"Applying cube filter with mode '{cube_mode}' and radius {cube_radius}")
    elif cube_radius is not None:
        warnings.warn("cube_radius is provided but cube_mode is 'unrestricted'. Ignoring cube_radius.")
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
        p._center = None
        p._inradius = None
        p._finite = True
        p._vertices = None
        p._volume = None
        p._hs = None
        return True

    found_sss = SSManager()
    nworkers = nworkers or process_aware_cpu_count()
    if verbose:
        print(f"Running on {nworkers} workers")
    if queue is None:
        queue = BlockingQueue()
    if start is None:
        start = cx.add_point(torch.zeros(cx.net.input_shape, device=cx.net.device, dtype=cx.net.dtype))
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

    with mp.Pool(nworkers, initializer=set_globals, initargs=(cx.net, get_volumes)) as pool:
        try:
            for p, shi, depth, node_index in pool.imap_unordered(
                partial(poly_calculations, bound=bound, **kwargs), queue
            ):
                unprocessed -= 1
                node = cx.index2poly[node_index]
                if not isinstance(p, Polyhedron):
                    bad_shi_computations.append((node, shi, depth, str(p)))
                    node._shis.remove(shi)
                    if len(node._shis) < min(cx.dim, cx.n):
                        warnings.warn(RuntimeWarning(f"Polyhedron {node} has less than {min(cx.dim, cx.n)} SHIs"))
                    if unprocessed == 0 or len(cx) >= max_polys:
                        break
                    continue

                if p.net is None:
                    p.net = cx.net

                if cube_mode != "unrestricted" and not _apply_cube_filter(p):
                    if unprocessed == 0 or len(cx) >= max_polys:
                        break
                    continue

                p = cx.add_polyhedron(p)

                if getattr(p, "warnings", None):
                    for warning in p.warnings:
                        try:
                            warnings.warn(warning)
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
                    f"Depth: {depth}  Unprocessed: {unprocessed}  Faces: {len(p._shis)}  Avg: {rolling_average:.2f} IP Norm: {p._interior_point_norm or -1:.2f}  Finite: {p._finite} Mistakes: {len(bad_shi_computations)}",
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


def _greedy_path_helper(
    cx: "Complex", start: Polyhedron, end: Polyhedron, diffs: set[int] | None = None
) -> list[Polyhedron]:
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
    bound: float = DEFAULT_SEARCH_BOUND,
    max_polys: float = float("inf"),
    show_pbar: bool = True,
    num_threads: int = 1,
    **kwargs: Any,
) -> list[Polyhedron] | None:
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
        list or None: A list of Polyhedron objects representing the path
            from start to end, or None if no path is found.

    Raises:
        ValueError: If the start point lies exactly on a neuron's boundary.
    """
    from relucent.complex import set_globals

    start_poly = cx.add_polyhedron(start) if isinstance(start, Polyhedron) else cx.add_point(start)
    end_poly = cx.add_polyhedron(end) if isinstance(end, Polyhedron) else cx.add_point(end)
    if start_poly == end_poly:
        print("Start and end points are in the same region")
        center, inradius = start_poly.get_center_inradius()
        start_poly._center = center
        start_poly._inradius = inradius
        start_poly._finite = center is not None

        start_poly._interior_point = start_poly.get_interior_point()
        start_poly._shis = cast(list[int], get_shis(start_poly, bound=bound, collect_info=False))
        return [start_poly]

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
        push=lambda pq, task, priority: pq.push(task, priority=priority),
        pop=lambda pq: pq.pop(),
    )

    gScore[start_poly] = 0
    fScore[start_poly] = (start_poly.ss_np != end_poly.ss_np).sum()

    result = get_shis(start_poly, bound=bound, **kwargs)
    assert isinstance(result, list)
    start_poly._shis = result

    openSet.push((start_poly, None, 0), 0)

    bad_shi_computations = []
    pbar = tqdm(
        desc="Search Progress" + (str(show_pbar) if show_pbar is not True else ""),
        mininterval=1,
        leave=True,
        total=max_polys if max_polys != float("inf") else None,
        disable=not show_pbar,
    )
    pbar.update(n=1)

    unprocessed = len(openSet)
    depth = 0
    neighbor: Polyhedron | ValueError | None = None
    min_dist = float("inf")
    min_p = start_poly

    def heuristic(p: Polyhedron, depth: int, shi: int) -> float:
        hamming = (p.ss_np != end_poly.ss_np).sum()
        if p.interior_point is None or end_poly.interior_point is None:
            raise ValueError("Interior point not found")
        dist = np.linalg.norm(p.interior_point - end_poly.interior_point).item()
        bias = -1 / (1 + dist)
        return hamming + ASTAR_BIAS_WEIGHT * bias

    def d(p1: Polyhedron, p2: Polyhedron) -> int:
        return 1

    pool = mp.Pool(nworkers, initializer=set_globals, initargs=(cx.net, False, num_threads)) if nworkers > 1 else None
    try:
        set_globals(cx.net)
        for item in map(partial(astar_calculations, bound=bound, **kwargs), openSet):
            if isinstance(item[1], Exception):
                bad_shi_computations.append(item)
                continue
            unprocessed -= 1

            p, shi, depth = item

            if nworkers == 1:
                neighbor_iter = map(partial(get_ip, p), (i for i in p.shis if i != shi))
            elif pool is not None:
                neighbor_iter = pool.imap_unordered(
                    partial(get_ip, p),
                    (i for i in p.shis if i != shi),
                    chunksize=max(nhs // nworkers, 1),
                )

            for neighbor, neighbor_shi in neighbor_iter:
                if not isinstance(neighbor, Polyhedron):
                    p._shis.remove(neighbor_shi)
                else:
                    tentative_gScore = gScore[p] + d(p, neighbor)
                    if neighbor.net is None:
                        neighbor.net = cx.net
                    if tentative_gScore < gScore[neighbor]:
                        cameFrom[neighbor] = p
                        gScore[neighbor] = tentative_gScore
                        dist = heuristic(neighbor, depth, shi)
                        fScore[neighbor] = tentative_gScore + dist
                        if dist < min_dist:
                            min_dist = dist
                            min_p = neighbor
                        openSet.push((neighbor, neighbor_shi, depth + 1), fScore[neighbor])
                        unprocessed += 1
                        if neighbor == end_poly:
                            break
                if neighbor == end_poly:
                    break

            pbar.update(n=len(cameFrom) - pbar.n)
            pbar.set_postfix_str(
                f"Min Distance: {min_dist:.3f} Depth: {depth} Open Set: {unprocessed} Mistakes: {len(bad_shi_computations)} | Finite: {p.finite} # SHIs: {len(p.shis)}",
                refresh=False,
            )

            if min_dist < 1:
                if min_dist > 0:
                    last_shi = np.argwhere((min_p.ss_np != end_poly.ss_np).ravel()).item()
                    if last_shi in min_p.shis:
                        cameFrom[end_poly] = min_p
                        neighbor = end_poly
                        break
                else:
                    neighbor = end_poly
                    break

            if unprocessed == 0 or len(cameFrom) >= max_polys:
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
    if neighbor == end_poly:
        path = [end_poly]
        while path[-1] != start_poly:
            assert cameFrom[path[-1]] not in path, path
            path.append(cameFrom[path[-1]])
        path.reverse()
        return path
    else:
        return None
