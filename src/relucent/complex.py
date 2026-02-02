import multiprocessing as mp
import os
import pickle
import random
import warnings
from collections import defaultdict
from functools import partial

import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from relucent.config import (
    ASTAR_BIAS_WEIGHT,
    DEFAULT_COMPLEX_PLOT_BOUND,
    DEFAULT_PARALLEL_ADD_BOUND,
    DEFAULT_SEARCH_BOUND,
    INTERIOR_POINT_RADIUS_SEQUENCE,
    PLOT_DEFAULT_MAXCOORD,
    PLOT_MARGIN_FACTOR,
)
from relucent.poly import Polyhedron
from relucent.ss import SSManager
from relucent.utils import BlockingQueue, NonBlockingQueue, UpdatablePriorityQueue, close_env, get_colors, get_env


def set_globals(get_net, get_volumes=True):
    """Initialize global variables for worker processes in multiprocessing.

    This function should only be used as an initializer for multiprocessing pools,
    never called directly by the main process. It sets up the network, environment,
    and volume calculation settings that worker processes need.

    Args:
        get_net: The neural network object to be used by worker processes.
        get_volumes: Whether to compute volumes for polyhedra when input dimension <= 6.
            Defaults to True.

    Example:
        >>> with mp.Pool(nworkers, initializer=set_globals, initargs=(net,)) as pool:
        ...     # Use pool for parallel processing
        ...     pass
    """
    global env
    env = get_env()
    global net
    net = get_net
    net.save_numpy_weights()  # Refresh weight_cpu after pickle; pickled net can have stale/corrupt weight_cpu
    global dim
    dim = np.prod(net.input_shape)
    global get_vol_calc
    get_vol_calc = get_volumes


def poly_calculations(task, **kwargs):
    """Worker function for computing polyhedron properties in parallel.

    This function is used by parallel_add() and all search methods to compute
    halfspaces, center, inradius, interior point, and supporting hyperplane
    indices (SHIs) for a given sign sequence.

    Args:
        task: Either a sign sequence or a tuple containing (sign sequence, ...) with
            additional data to be passed through.
        **kwargs: Additional arguments passed to get_shis() method, such as
            'collect_info' or 'bound'.

    Returns:
        tuple: If successful, returns (Polyhedron, *rest) where rest contains
            any additional data from the input task. If a ValueError occurs
            during computation, returns (error, *rest).
    """
    ss = task[0] if isinstance(task, tuple) else task
    rest = task[1:] if isinstance(task, tuple) else ()
    p = Polyhedron(net, ss)

    try:
        p._halfspaces, p._W, p._b = p.get_hs()

        p.get_center_inradius(env=env)
        p.get_interior_point(env=env)
        p._interior_point_norm = np.linalg.norm(p.interior_point).item()
        p._Wl2 = np.linalg.norm(p.W).item()

        if dim <= 6 and get_vol_calc:
            p.volume
        if p._shis is None:
            if "collect_info" in kwargs:
                p._shis, shi_info = p.get_shis(env=env, **kwargs)
            else:
                p._shis = p.get_shis(env=env, **kwargs)
    except ValueError as error:
        return error, *rest
    p.clean_data()
    random.shuffle(p._shis)
    return (p, *rest)


def get_ip(p, shi):
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
    try:
        ss = p.ss_np.copy()
        ss[0, shi] = -ss[0, shi]
        n = Polyhedron(net, ss)
        for max_radius in INTERIOR_POINT_RADIUS_SEQUENCE:
            try:
                n.get_interior_point(env=env, max_radius=max_radius)
            except ValueError:
                print("Increasing max radius to find interior point")
        return n, shi
    except ValueError as e:
        return e, None


def astar_calculations(task, **kwargs):
    """Worker function for computing polyhedron properties in A* search.

    Similar to poly_calculations, but specifically designed for A* search algorithm.
    Computes center, inradius, interior point, and supporting hyperplane indices
    for polyhedra during pathfinding.

    Args:
        task: Either a Polyhedron object or a tuple containing (Polyhedron, ...)
            with additional data to be passed through.
        **kwargs: Additional arguments passed to get_shis() method, such as
            'collect_info' or 'bound'.

    Returns:
        tuple: If successful, returns (Polyhedron, *rest). If an exception occurs
            during SHI computation, returns (Polyhedron, error, *rest).
    """
    p = task[0] if isinstance(task, tuple) else task
    rest = task[1:] if isinstance(task, tuple) else ()
    if p.net is None:
        p.net = net

    if p._inradius is None:
        p.get_center_inradius(env=env)
    if p._interior_point is None:
        p.get_interior_point(env=env)

    try:
        if p._shis is None:
            if "collect_info" in kwargs:
                p._shis, shi_info = p.get_shis(env=env, **kwargs)
            else:
                p._shis = p.get_shis(env=env, **kwargs)
    except Exception as error:
        return p, error, *rest
    p.clean_data()
    random.shuffle(p._shis)
    # p, neighbors = get_neighbors(p, (shi for shi in p._shis if shi != task[1]))
    return p, *rest


class Complex:
    """Manages the polyhedral complex of a neural network.

    This class provides methods for calculating, storing, and searching the h-representations
    (halfspace representations) of polyhedra in the complex.
    """

    def __init__(self, net):
        self._net = net
        self.net.save_numpy_weights()

        ## TODO: Try replacing with just a dictionary that incremements by 1
        self.ssm = SSManager()
        self.index2poly = list()

        net_layers = list(net.layers.values())
        self.ss_layers = [
            i
            for i, (layer, next_layer) in enumerate(zip(net_layers[:-1], net_layers[1:]))
            if isinstance(next_layer, nn.ReLU)
        ]

        # Build mapping from global sign-sequence indices to (layer_index, neuron_index)
        self.ssi2maski = []
        for i, layer in enumerate(self.net.layers.values()):
            if i in self.ss_layers:
                assert isinstance(layer, nn.Linear), "Only linear layers should be before ReLU layers"
                for neuron_idx in range(layer.out_features):
                    self.ssi2maski.append((i, (0, neuron_idx)))

    def __del__(self):
        close_env()

    def __getitem__(self, key):
        """Retrieve a Polyhedron from the complex by its key.

        Args:
            key: Can be either:
                - A sign sequence as np.ndarray or torch.Tensor
                - A Polyhedron object (returns the stored version)

        Returns:
            Polyhedron: The polyhedron associated with the given key.

        Raises:
            KeyError: If the polyhedron with the given key is not in the complex.
        """
        if isinstance(key, Polyhedron):
            return self.index2poly[self.ssm[key.ss_np]]
        elif isinstance(key, (np.ndarray, torch.Tensor)):
            return self.index2poly[self.ssm[key]]
        else:
            raise KeyError(f"Polyhedron with key {key} not in Complex")

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __iter__(self):
        """Iterate over all Polyhedra in the complex.

        Yields:
            Polyhedron: Polyhedra in the order they were added to the complex.
        """
        for p in self.index2poly:
            yield p

    def __len__(self):
        return len(self.index2poly)

    def save(self, filename, save_ssm=True):
        """Save the complex to a pickle file.

        Args:
            filename: Path to the output file.
            save_ssm: If True, include the SSManager in the saved state so that
                sign-sequence lookups are preserved. Defaults to True.
        """
        state = self.__getstate__()
        if save_ssm:
            state["ssm"] = self.ssm
        with open(filename, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(filename):
        """Load a Complex from a pickle file.

        Intended to be called as Complex.load(filename). The file must have been
        created by save().

        Args:
            filename: Path to the pickle file.

        Returns:
            Complex: The restored complex.
        """
        with open(filename, "rb") as f:
            state = pickle.load(f)
        cplx = Complex(state["net"])
        cplx.__setstate__(state)
        return cplx

    def __getstate__(self):
        return {
            "index2poly": self.index2poly,
            "net": self.net,
        }

    def __setstate__(self, state):
        self.__init__(state["net"])
        self.index2poly = state["index2poly"]
        if "ssm" in state:
            self.ssm = state["ssm"]
        else:
            for p in self.index2poly:
                self.ssm.add(p.ss_np)
        for p in self.index2poly:
            p.net = self.net

    @property
    def net(self):
        """The neural network. May only be set once."""
        return self._net

    @net.setter
    def net(self, value):
        if self._net is not None:
            raise ValueError("The net attribute cannot be set when it already has a non-None value")
        self._net = value

    @property
    def dim(self):
        """The input dimension of the network."""
        return np.prod(self.net.input_shape)

    @property
    def n(self):
        """The number of bent hyperplanes/neurons in the network."""
        return len(self.ssi2maski)

    @torch.no_grad()
    def ss_iterator(self, batch):
        """Generate sign sequences for each ReLU layer from a batch of data points.

        Args:
            batch: A batch of input data points as a torch.Tensor, np.ndarray, or array-like.
                Will be reshaped to match the network's input shape.

        Yields:
            torch.Tensor: Sign sequences for each ReLU layer in
                the network, indicating the activation pattern of that layer.
        """
        x = batch if isinstance(batch, torch.Tensor) else torch.as_tensor(batch)
        x = x.to(device=self.net.device, dtype=self.net.dtype).reshape((-1, *self.net.input_shape))
        for i, layer in enumerate(self.net.layers.values()):
            x = layer(x)
            if i in self.ss_layers:
                yield torch.sign(x)  # * (torch.abs(x) < 1e-12)
                if i == self.ss_layers[-1]:
                    break

    def point2ss(self, batch):
        """Convert a batch of data points to sign sequences.

        Computes the combined sign sequence across all ReLU layers for the given
        data points. Does not add the resulting polyhedra to the complex.

        Args:
            batch: A batch of input data points as a torch.Tensor, np.ndarray, or array-like.

        Returns:
            torch.Tensor or np.ndarray: The sign sequences for the input batch, with shape
                (batch_size, total_ReLU_neurons). Returns a torch.Tensor if batch is a
                torch.Tensor, otherwise a np.ndarray.
        """
        is_tensor = isinstance(batch, torch.Tensor)
        result = torch.hstack(list(self.ss_iterator(batch)))
        if is_tensor:
            return result
        return result.detach().cpu().numpy()

    def point2poly(self, point, check_exists=True):
        """Convert a data point to its corresponding Polyhedron.

        Finds the polyhedron that contains the given data point. Does not add
        the polyhedron to the complex.

        Args:
            point: A single data point as a torch.Tensor or np.ndarray.
            check_exists: If True, return the existing polyhedron from the complex
                if it already exists. Defaults to True.

        Returns:
            Polyhedron: The polyhedron containing the given point.
        """
        return self.ss2poly(self.point2ss(point), check_exists=check_exists)

    def ss2poly(self, ss, check_exists=True):
        """Convert a sign sequence to a Polyhedron.

        Creates a Polyhedron object from the given sign sequence. Does not add
        it to the complex.

        Args:
            ss: A sign sequence as a torch.Tensor or np.ndarray.
            check_exists: If True, return the existing polyhedron from the complex
                if it already exists. Defaults to True.

        Returns:
            Polyhedron: The polyhedron corresponding to the given sign sequence.
        """
        if check_exists and ss in self:
            return self[ss]
        else:
            return Polyhedron(self.net, ss)

    def add_ss(self, ss, check_exists=True):
        """Convert a sign sequence to a Polyhedron and add it to the complex.

        Args:
            ss: A sign sequence as a torch.Tensor or np.ndarray.
            check_exists: If True, return the existing polyhedron from the complex
                if it already exists. Defaults to True.

        Returns:
            Polyhedron: The polyhedron that was added (or already existed) in the complex.
        """
        return self.add_polyhedron(Polyhedron(self.net, ss), check_exists=check_exists)

    def add_polyhedron(self, p, overwrite=False, check_exists=True):
        """Add a Polyhedron to the complex.

        Args:
            p: The Polyhedron object to add.
            overwrite: If True and the polyhedron already exists, replace it with
                the new one. Defaults to False.
            check_exists: If True, check whether the polyhedron already exists in
                the complex and return the existing one if so. If False, assume
                the polyhedron is new (skip the check). Defaults to True.

        Returns:
            Polyhedron: The polyhedron that was added (or already existed) in the complex.
        """

        assert check_exists or not overwrite, "Cannot overwrite polyhedron if check_exists is False"

        if not check_exists:
            self.index2poly.append(p)
            self.ssm.add(p.ss_np)
            return p

        p_exists = p in self

        if p_exists and overwrite:
            self.index2poly[self.ssm[p.ss_np]] = p
            return p
        elif p_exists:
            return self[p]
        else:
            self.index2poly.append(p)
            self.ssm.add(p.ss_np)
            return p

    def add_point(self, data, check_exists=True):
        """Find the polyhedron containing a data point and add it to the complex.

        Args:
            data: A single data point as a torch.Tensor, np.ndarray, or array-like.
            check_exists: If True, check whether the polyhedron already exists in
                the complex and return it if so. Only set to false if you know it does not.
                Defaults to True.

        Returns:
            Polyhedron: The polyhedron containing the given point, now stored in the complex.
        """
        return self.add_ss(self.point2ss(data), check_exists=check_exists)

    def clean_data(self):
        """Clean cached data from all polyhedra in the complex.

        This method calls clean_data() on each polyhedron, which removes most of their
        computed data.
        """
        for poly in self:
            poly.clean_data()

    @torch.no_grad()
    def adjacent_polyhedra(self, poly):
        """Get the Polyhedra that are adjacent (across one BH) from the given Polyhedron.

        Also works on lower-dimensional polyhedra.
        """
        ps = set()
        shis = poly.shis
        for shi in shis:
            if poly.ss_np[0, shi] == 0:
                continue
            ss = poly.ss_np.copy()
            ss[0, shi] = -ss[0, shi]
            ps.add(self.ss2poly(ss))
        return ps

    def parallel_add(self, points, nworkers=None, bound=DEFAULT_PARALLEL_ADD_BOUND, **kwargs):
        """Add multiple polyhedra from data points using parallel processing.

        Processes a batch of data points in parallel, computing their corresponding
        polyhedra and adding them to the complex.

        Args:
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
        nworkers = nworkers or os.process_cpu_count()
        print(f"Running on {nworkers} workers")
        sss = []
        for p in tqdm(points, desc="Getting SSs", mininterval=5):
            s = self.point2ss(p)
            sss.append(s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s)

        with mp.Pool(nworkers, initializer=set_globals, initargs=(self.net,)) as pool:
            ps = pool.map(
                partial(poly_calculations, bound=bound, **kwargs), tqdm(sss, desc="Adding Polys", mininterval=5)
            )
            ps = [p[0] if isinstance(p[0], Polyhedron) else None for p in ps]
            for p in ps:
                if p is not None:
                    self.add_polyhedron(p)
            return ps

    def searcher(
        self,
        start=None,
        max_depth=float("inf"),
        max_polys=float("inf"),
        queue=None,
        bound=DEFAULT_SEARCH_BOUND,
        nworkers=None,
        get_volumes=True,
        verbose=1,
        **kwargs,
    ):
        """Search for polyhedra in the complex by discovering neighbors.

        This is a generic search method that can be configured for different
        traversal strategies (BFS, DFS, random walk) by providing different
        queue types. It starts from a given point and explores the complex by
        crossing supporting hyperplanes to discover adjacent polyhedra.

        See bfs(), dfs(), and random_walk() for examples of how to use this
        function to define specific search strategies.

        Args:
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
            **kwargs: Additional arguments passed to Polyhedron.get_shis().

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

        found_sss = SSManager()
        nworkers = nworkers or os.process_cpu_count()
        ## NOTE: If nworkers>1, the traversal order may not be correct
        if verbose:
            print(f"Running on {nworkers} workers")
        if queue is None:
            queue = BlockingQueue()
        if start is None:
            start = self.add_point(torch.zeros(self.net.input_shape, device=self.net.device, dtype=self.net.dtype))
        elif isinstance(start, Polyhedron):
            start = self.add_polyhedron(start)
        else:
            # Treat as a point.
            start = self.add_point(start)
        found_sss.add(start.ss_np)
        if (start.ss_np == 0).any():
            raise ValueError("Start point must not be on a hyperplane")
        start._shis = start.get_shis(bound=bound, **kwargs)
        ##TODO: replace with something like queue.push((start.ss_np, None, 0, self.ssm[start.ss_np]))
        for shi in start.shis:
            new_ss = start.ss_np.copy()
            new_ss[0, shi] *= -1
            found_sss.add(new_ss)
            queue.push((new_ss, shi, 1, self.ssm[start.ss_np]))
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

        with mp.Pool(nworkers, initializer=set_globals, initargs=(self.net, get_volumes)) as pool:
            try:
                for p, shi, depth, node_index in pool.imap_unordered(
                    partial(poly_calculations, bound=bound, **kwargs), queue
                ):
                    unprocessed -= 1
                    node = self.index2poly[node_index]
                    if not isinstance(p, Polyhedron):
                        bad_shi_computations.append((node, shi, depth, str(p)))
                        node._shis.remove(shi)
                        if len(node._shis) < min(self.dim, self.n):
                            warnings.warn(f"Polyhedron {node} has less than {min(self.dim, self.n)} SHIs")
                        if unprocessed == 0 or len(self) >= max_polys:
                            break
                        continue

                    ## If not using multiprocessing, you would need to check if the polyhedron's net is None first
                    p.net = self.net

                    p = self.add_polyhedron(p)

                    if depth < max_depth:
                        for new_shi in p.shis:
                            if new_shi != shi and len(self) < max_polys:
                                ss = p.ss_np.copy()
                                ss[0, new_shi] *= -1
                                if ss not in found_sss:
                                    queue.push((ss, new_shi, depth + 1, self.ssm[p.ss_np]))
                                    found_sss.add(ss)
                                    unprocessed += 1

                    pbar.update(n=len(self) - pbar.n)
                    rolling_average = (rolling_average * (len(self) - 1) + len(p.shis)) / len(self)
                    pbar.set_postfix_str(
                        f"Depth: {depth}  Unprocessed: {unprocessed}  Faces: {len(p._shis)}  Avg: {rolling_average:.2f} IP Norm: {p._interior_point_norm or -1:.2f}  Finite: {p._finite} Mistakes: {len(bad_shi_computations)}",
                        refresh=False,
                    )

                    if unprocessed == 0 or len(self) >= max_polys:
                        break
            except Exception:
                raise
            finally:
                queue.close()
                pbar.close()

                pool.close()
                pool.terminate()
                pool.join()
                pool.close()

        search_info = {
            "Search Depth": depth,
            "Avg # Facets Uncorrected": rolling_average,
            "Search Time": pbar.format_dict["elapsed"],
            "Bad SHI Computations": bad_shi_computations,
            "Complete": unprocessed == 0,
        }

        return search_info

    def bfs(self, **kwargs):
        """Perform breadth-first search of the complex.

        Explores the complex using a breadth-first strategy, discovering all
        polyhedra at depth d before moving to depth d+1. Uses a FIFO queue.

        Args:
            **kwargs: All arguments accepted by searcher().

        Returns:
            dict: Search information dictionary (see searcher() documentation).
        """
        return self.searcher(**kwargs)

    def dfs(self, **kwargs):
        """Perform depth-first search of the complex.

        Explores the complex using a depth-first strategy, following paths as
        deeply as possible before backtracking. Uses a LIFO queue.

        Args:
            **kwargs: All arguments accepted by searcher().

        Returns:
            dict: Search information dictionary (see searcher() documentation).
        """
        return self.searcher(queue=BlockingQueue(pop=lambda x: x.popleft()), **kwargs)

    def random_walk(self, **kwargs):
        """Perform random walk search of the complex.

        Explores the complex by randomly selecting which polyhedron to explore
        next from the queue.

        Args:
            **kwargs: All arguments accepted by searcher().

        Returns:
            dict: Search information dictionary (see searcher() documentation).
        """
        return self.searcher(
            queue=BlockingQueue(
                queue_class=list,
                pop=lambda x: x.pop(random.randrange(0, len(x))),
                push=lambda x, y: x.append(y),
            ),
            **kwargs,
        )

    def _greedy_path_helper(self, start, end, diffs=None):
        if start == end:
            # print("Start and end points are the same")
            return [start]

        if (start.ss_np == 0).any():
            raise ValueError("Start point must not be on a hyperplane")

        diffs = diffs or set(np.argwhere((start.ss_np != end.ss_np).flatten()).flatten().tolist())

        print("Diffs:", diffs)

        if not start._shis:
            try:
                start.get_shis()
            except ValueError:
                return None
        groupa = set(start.shis) & diffs
        groupb = set(start.shis) - diffs
        for shi in list(groupa):
            print("Crossing", shi)
            new_ss = start.ss_np.copy()
            new_ss[0, shi] *= -1
            next_poly = self.ss2poly(new_ss)
            rest = self._greedy_path_helper(next_poly, end, diffs - {shi})
            if rest is not None:
                return [start] + rest
        for shi in list(groupb):
            print("Crossing", shi)
            new_ss = start.ss_np.copy()
            new_ss[0, shi] *= -1
            next_poly = self.ss2poly(new_ss)
            rest = self._greedy_path_helper(next_poly, end, diffs + {shi})
            if rest is not None:
                return [start] + rest
        return None

    def greedy_path(self, start, end):
        """Greedily find a path between two data points.

        Attempts to find a path through adjacent polyhedra from start to end
        using a greedy strategy. This method can be slow for large complexes
        as it explores many paths.

        Args:
            start: Starting data point as torch.Tensor or np.ndarray.
            end: Ending data point as torch.Tensor or np.ndarray.

        Returns:
            list or None: A list of Polyhedron objects representing the path
                from start to end, or None if no path is found.
        """
        start = self.add_point(start)
        end = self.add_point(end)
        return self._greedy_path_helper(start, end)

    def hamming_astar(self, start, end, nworkers=None, bound=DEFAULT_SEARCH_BOUND, max_polys=float("inf"), show_pbar=True, **kwargs):
        """Find a path between two data points using A* search algorithm.

        Uses the A* pathfinding algorithm with a heuristic based on Hamming
        distance between sign sequences, plus Euclidean distance between interior
        points to break ties. The heuristic should be admissible for optimal paths.

        Args:
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

        start = self.add_point(start)
        end = self.add_point(end)
        if start == end:
            print("Start and end points are in the same region")
            start.get_center_inradius()
            start.get_interior_point()
            start.get_shis(bound=bound)
            return [start]

        if (start.ss_np == 0).any():
            raise ValueError("Start point must not be on a hyperplane")

        nhs = len(start.halfspaces)
        nworkers = min(nworkers or os.process_cpu_count(), nhs)
        print(f"Using {nworkers} workers")

        cameFrom = dict()
        gScore = defaultdict(lambda: float("inf"))
        fScore = defaultdict(lambda: float("inf"))

        openSet = NonBlockingQueue(
            queue_class=UpdatablePriorityQueue,
            push=lambda pq, task, priority: pq.push(task, priority=priority),
            pop=lambda pq: pq.pop(),
        )
        # found_sss = SSManager()
        # nworkers = nworkers or os.process_cpu_count()

        gScore[start] = 0
        fScore[start] = (start.ss_np != end.ss_np).sum()
        # found_sss.add(start.ss)
        # found = set(start)

        start._shis = start.get_shis(bound=bound, **kwargs)

        openSet.push((start, None, 0), 0)

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
        neighbor = None
        min_dist = float("inf")
        min_p = None

        def heuristic(p, depth, shi):
            hamming = (p.ss_np != end.ss_np).sum()
            dist = np.linalg.norm(p.interior_point - end.interior_point).item()
            # bias = -1 / ((1 + dist) ** 0.1)  ## TODO: Test if this is faster
            # bias = -1 / (1 + np.log(dist + 10))
            bias = -1 / (1 + dist)
            # bias = 0
            # bias = 1 / (1 + depth) - 1
            return hamming + ASTAR_BIAS_WEIGHT * bias

        def d(p1, p2):
            return 1

        # with mp.Pool(nworkers, initializer=set_globals, initargs=(self.net,)) as pool:
        pool = mp.Pool(nworkers, initializer=set_globals, initargs=(self.net,))
        try:
            # for p, neighbors, shi, depth, node_index in pool.imap(
            #     partial(astar_calculations, bound=bound, **kwargs), openSet
            # ):
            set_globals(self.net)
            for item in map(partial(astar_calculations, bound=bound, **kwargs), openSet):
                # found.remove(item[0])
                if isinstance(item[1], Exception):
                    bad_shi_computations.append(item)
                    continue
                unprocessed -= 1

                p, shi, depth = item

                for neighbor, neighbor_shi in pool.imap_unordered(
                    partial(get_ip, p), (i for i in p.shis if i != shi), chunksize=max(nhs // nworkers, 1)
                ):
                    # for neighbor, neighbor_shi in map(
                    #     partial(get_ip, p),
                    #     (i for i in p.shis if i != shi),
                    # ):
                    if not isinstance(neighbor, Polyhedron):
                        p._shis.remove(neighbor)

                    tentative_gScore = gScore[p] + d(p, neighbor)
                    ## If not using multiprocessing, you would need to check if the neighbor's net is None first
                    neighbor.net = self.net
                    if tentative_gScore < gScore[neighbor]:  ## Only needed with an inconsistent heuristic
                        cameFrom[neighbor] = p
                        gScore[neighbor] = tentative_gScore
                        dist = heuristic(neighbor, depth, shi)
                        fScore[neighbor] = tentative_gScore + dist
                        # options.append(
                        #     {
                        #         "neighbor": neighbor,
                        #         "neighbor_shi": neighbor_shi,
                        #         "fScore": fScore[neighbor],
                        #         "improvement": neighbor.ss[0, neighbor_shi] == end.ss[0, neighbor_shi],
                        #     }
                        # )
                        if dist < min_dist:
                            min_dist = dist
                            min_p = neighbor
                        openSet.push((neighbor, neighbor_shi, depth + 1), fScore[neighbor])
                        unprocessed += 1
                        if neighbor == end:
                            break
                    if neighbor == end:
                        break

                # next_one = openSet.deque.pq[0][2]
                # new_hamming = (next_one.ss != end.ss).sum()
                # if new_hamming >= (p.ss != end.ss).sum():
                #     p2 = next_one
                #     p1 = pd.DataFrame(options).sort_values("fScore", ascending=True).iloc[0]["neighbor"]

                #     print(
                #         f"The next polyhedron to be searched is {p2}: fScore: {fScore[p2]} gScore: {gScore[p2]} Hamming: {(p2.ss != end.ss).sum()} Heuristic: {heuristic(p2, 0, 0)} | Distance: {np.linalg.norm(p2.interior_point - end.interior_point).item()}"
                #     )
                #     print(
                #         f"A better option could be: {p1}: fScore: {fScore[p1]} gScore: {gScore[p1]} Hamming: {(p1.ss != end.ss).sum()} Heuristic: {heuristic(p1, 0, 0)} | Distance: {np.linalg.norm(p1.interior_point - end.interior_point).item()}"
                #     )

                pbar.update(n=len(cameFrom) - pbar.n)
                pbar.set_postfix_str(
                    f"Min Distance: {min_dist:.3f} Depth: {depth} Open Set: {unprocessed} Mistakes: {len(bad_shi_computations)} | Finite: {p.finite} # SHIs: {len(p.shis)}",
                    refresh=False,
                )

                if min_dist < 1:
                    if 0 < min_dist:
                        ## TODO: What if there are no/multiple differences?
                        last_shi = np.argwhere((min_p.ss_np != end.ss_np).flatten()).item()
                        if last_shi in min_p.shis:
                            cameFrom[end] = min_p
                            neighbor = end
                            break
                    else:
                        # raise ValueError("what in tarnation???")
                        neighbor = end
                        break

                if unprocessed == 0 or len(cameFrom) >= max_polys:
                    break
        except Exception:
            raise
        finally:
            # print(f"Closing out after {pbar.n} iterations")

            pool.terminate()
            pool.join()
            pool.close()

            openSet.close()
            tqdm.get_lock().locks = []
            pbar.close()
        #     print("Closed out")
        # print("Finished A* Search")
        if neighbor == end:
            path = [end]
            while path[-1] != start:
                assert cameFrom[path[-1]] not in path, path
                path.append(cameFrom[path[-1]])
            path.reverse()
            [(p.Wl2, p.inradius) for p in path]
            # print(f"Path found with length {len(path) - 1}:")
            # if len(path) < 100:
            #     for p1, p2 in zip(path[:-1], path[1:]):
            #         print(f"    {p1} - {np.argwhere((p1.ss != p2.ss).flatten()).item()} -> {p2}")
            return path
        else:
            # print(f"No Path Found - Final Distance: {min_dist - 1}")
            return None

    def get_poly_attrs(self, attrs):
        """Extract specified attributes from all polyhedra in the complex.

        Useful for building tabular representations or dataframes of polyhedra
        properties. Attributes are returned in the same order as polyhedra were
        added to the complex.

        Args:
            attrs: A list of attribute names to extract (e.g., ["finite", "Wl2"]).

        Returns:
            dict: A dictionary mapping attribute names to lists of attribute
                values, with one value per polyhedron in the complex based on the order
                they were added.
        """
        return {attr: [getattr(poly, attr) for poly in self] for attr in attrs}

    def get_dual_graph(
        self, relabel=False, plot=False, node_color=None, node_size=None, cmap="viridis", match_locations=False
    ):
        """Construct the dual graph of the complex.

        The dual graph represents the connectivity structure of the complex,
        where nodes are polyhedra and edges connect adjacent polyhedra (those
        sharing a supporting hyperplane).

        Args:
            relabel: If True, nodes are indexed by integers matching self.index2poly
                indices. If False, nodes are Polyhedron objects. Defaults to False.
            plot: If True, prepare the graph for visualization with pyvis by
                adding layout and styling attributes. Defaults to False.
            node_color: If "Wl2", color nodes by their Wl2 (weight norm) value.
                If "volume", color by volume. If None, no special coloring.
                Defaults to None.
            node_size: If "volume", size nodes proportionally to their volume.
                If None, use default size. Defaults to None.
            cmap: Colormap to use when node_color is specified. Defaults to "viridis".
            match_locations: If True, position graph nodes at the center points
                of their polyhedra (only works for 2D complexes). Defaults to False.

        Returns:
            networkx.Graph: The dual graph of the complex. Nodes are polyhedra
                (or integers if relabel=True), edges connect adjacent polyhedra
                and have a "shi" attribute indicating which supporting hyperplane
                they cross.

        Raises:
            ValueError: If match_locations is True and the complex is not 2D.
        """
        G = nx.Graph()
        for poly in self:
            G.add_node(poly, label=str(poly))
        for poly in tqdm(self, desc="Creating Dual Graph", leave=False):
            ss = poly.ss_np.copy()
            for shi in poly.shis:
                ss[0, shi] *= -1
                if ss in self:
                    G.add_edge(poly, self[ss], shi=shi)
                ss[0, shi] *= -1
        if plot:
            if match_locations:
                if self.dim != 2:
                    raise ValueError("Polyhedra must be 2D to match locations")

                nx.set_node_attributes(G, False, "physics")
                nx.set_node_attributes(
                    G,
                    {poly: poly.interior_point[0].item() * 10 for poly in G.nodes},
                    "x",
                )
                nx.set_node_attributes(
                    G,
                    {poly: poly.interior_point[1].item() * 10 for poly in G.nodes},
                    "y",
                )

            if node_color == "Wl2":
                colors = get_colors([poly.Wl2 for poly in G.nodes], cmap=cmap)
                for c, poly in zip(colors, G.nodes):
                    G.nodes[poly]["color"] = c
            elif node_color == "volume":
                colors = get_colors([poly.ch.volume for poly in G.nodes], cmap=cmap)
                for c, poly in zip(colors, G.nodes):
                    G.nodes[poly]["color"] = c

            if node_size == "volume":
                sizes = [poly.ch.volume for poly in G.nodes]
                maxsize = max(sizes)
                for size, poly in zip(sizes, G.nodes):
                    G.nodes[poly]["size"] = (10 + 1000 * size / maxsize) ** 1
            else:
                nx.set_node_attributes(G, 4, "size")

            for node in G.nodes:
                G.nodes[node]["label"] = " "
                G.nodes[node]["title"] = str(node)
            for edge in G.edges:
                G.edges[edge]["label"] = " "
                G.edges[edge]["title"] = str(G.edges[edge]["shi"])
        if plot or relabel:
            G = nx.relabel_nodes(G, {poly: i for i, poly in enumerate(self)})
        return G

    def recover_from_dual_graph(self, G, initial_ss, source, copy=False):
        """Recover a complex from its connectivity graph.

        Reconstructs polyhedra in the complex by traversing the adjacency graph
        of top-dimensional cells, using the supporting hyperplane indices stored
        on edges to determine how to flip sign sequence elements. This is useful
        for storing large complexes efficiently, as you only need to store the
        graph structure and SHI indices on edges rather than full polyhedron data.

        Args:
            G: A networkx.Graph representing the dual graph. Edges must have
                a "shi" attribute indicating the supporting hyperplane index.
            initial_ss: The sign sequence of a starting polyhedron
                as torch.Tensor or np.ndarray.
            source: The node key in G representing the initial polyhedron.

        Returns:
            networkx.Graph: The graph with polyhedron objects stored in node
                attributes under the "poly" key.
        """
        if copy:
            G = G.copy()
        initial_p = self.add_ss(initial_ss)
        G.nodes[source]["poly"] = initial_p
        for edge in tqdm(nx.bfs_edges(G, source=source), desc="Recovering Polyhedra", total=G.number_of_edges()):
            poly1, shi = G.nodes[edge[0]]["poly"], G.edges[edge]["shi"]
            poly2_ss = poly1.ss_np.copy()
            assert poly2_ss[0, shi] != 0
            poly2_ss[0, shi] *= -1
            poly2 = self.add_ss(poly2_ss, check_exists=False)

            G.nodes[edge[1]]["poly"] = poly2

        for node in G:
            self[G.nodes[node]["poly"]]._shis = [G.edges[edge]["shi"] for edge in G.edges(node)]

        return G

    def plot(self, label_regions=False, color=None, highlight_regions=None, ss_name=False, bound=DEFAULT_COMPLEX_PLOT_BOUND, **kwargs):
        """Plot the complex in 2D using plotly.

        Creates a 2D visualization of the complex, showing all polyhedra as
        regions in the input space. Only works for 2D input spaces.

        Args:
            label_regions: If True, add text labels showing each polyhedron's
                string representation at its center. Defaults to False.
            color: If "Wl2", color polyhedra by their Wl2 (weight norm) value.
                If None, use an equitable graph coloring. Defaults to None.
            highlight_regions: If provided, highlight these polyhedra in red.
                Can be a list of Polyhedron objects or their string representations.
                Defaults to None.
            ss_name: If True, use the full sign sequence as
                the legend label for each polyhedron. Defaults to False.
            bound: Constraint radius for plotting bounds. Passed to each
                polyhedron's plot2d() method. Defaults to None.
            **kwargs: Additional arguments passed to each polyhedron's plot2d() method.

        Returns:
            plotly.graph_objects.Figure: A plotly figure containing the 2D plot
                of the complex.
        """
        fig = go.Figure()
        polys = list(self)
        if color == "Wl2":
            colors = get_colors([poly.Wl2 for poly in polys])
        else:
            color_scheme = px.colors.qualitative.Plotly
            try:
                coloring = nx.algorithms.coloring.equitable_color(
                    self.get_dual_graph(), min(len(color_scheme), len(polys))
                )
                remap, idx = dict(), 0
                for p in polys:
                    if coloring[p] not in remap:
                        remap[coloring[p]] = idx
                        idx += 1
                colors = [color_scheme[remap[coloring[i]]] for i in polys]
            except Exception:
                print("Could not find equitable coloring, using random colors")
                colors = [color_scheme[i % len(color_scheme)] for i in range(len(polys))]
        for c, poly in tqdm(zip(colors, polys), desc="Plotting Polyhedra", total=len(polys)):
            if (highlight_regions is not None) and ((poly in highlight_regions) or (str(poly) in highlight_regions)):
                c = "red"
            if ss_name:
                name = f"{poly.ss_np.flatten().astype(int).tolist()}"
            else:
                name = f"{poly}"
            p_plot = poly.plot2d(
                name=name,
                fillcolor=c,
                line_color="black",
                mode="lines",  ## Comment out to mouse over intersections
                bound=bound,
                **kwargs,
            )
            for trace in p_plot:
                fig.add_trace(trace)
            if label_regions and poly.center is not None:
                fig.add_trace(
                    go.Scatter(x=[poly.center[0]], y=[poly.center[1]], mode="text", text=str(poly), showlegend=False)
                )
        interior_points = [np.max(np.abs(p.interior_point)) for p in self if p.finite]
        maxcoord = (np.max(interior_points) * PLOT_MARGIN_FACTOR) if len(interior_points) > 0 else min(PLOT_DEFAULT_MAXCOORD, bound if bound else PLOT_DEFAULT_MAXCOORD)
        # maxcoord = 10
        fig.update_layout(
            showlegend=True,
            # xaxis = dict(visible=False),
            # yaxis = dict(visible=False),
            plot_bgcolor="white",
            xaxis=dict(range=(-maxcoord, maxcoord)),
            yaxis_scaleanchor="x",
            yaxis=dict(range=(-maxcoord, maxcoord)),
        )
        return fig

    def plot3d(
        self,
        label_regions=False,
        color=None,
        highlight_regions=None,
        show_axes=False,
        project=True,
        **kwargs,
    ):
        """Plot the complex in 3D using plotly.

        Creates a 3D visualization of the complex, showing all polyhedra as
        regions in the input space. Only works for 3D input spaces.

        Args:
            label_regions: If True, add text labels showing each polyhedron's
                string representation. Defaults to False.
            color: If "Wl2", color polyhedra by their Wl2 (weight norm) value.
                If None, use an equitable graph coloring. Defaults to None.
            highlight_regions: If provided, highlight these polyhedra in red.
                Can be a list of Polyhedron objects or their string representations.
                Defaults to None.
            show_axes: If True, display the coordinate axes. Defaults to False.
            project: If True, also show a projection of the complex onto the
                xy plane. Defaults to True.
            **kwargs: Additional arguments passed to each polyhedron's plot3d() method.

        Returns:
            plotly.graph_objects.Figure: A plotly figure containing the 3D plot
                of the complex.
        """
        fig = go.Figure()
        polys = list(self)
        if color == "Wl2":
            colors = get_colors([poly.Wl2 for poly in polys])
        else:
            color_scheme = px.colors.qualitative.Plotly
            try:
                coloring = nx.algorithms.coloring.equitable_color(
                    self.get_dual_graph(), min(len(color_scheme), len(polys))
                )
                colors = [color_scheme[coloring[i]] for i in polys]
            except Exception:
                print("Could not find equitable coloring, using random colors")
                colors = [color_scheme[i % len(color_scheme)] for i in range(len(polys))]
        outlines, meshes = [], []
        for c, poly in tqdm(zip(colors, polys), desc="Plotting Polyhedra", total=len(polys)):
            if (highlight_regions is not None) and ((poly in highlight_regions) or (str(poly) in highlight_regions)):
                c = "red"
            p_plot = poly.plot3d(
                name=f"{poly}",
                color=c,
                # outlinecolor="black",
                **kwargs,
            )
            if p_plot is not None:
                if isinstance(p_plot, dict):
                    if "mesh" in p_plot:
                        meshes.append(p_plot["mesh"])
                    if "outline" in p_plot:
                        outlines.append(p_plot["outline"])
                else:
                    fig.add_trace(p_plot)
            if project is not None:
                p_plot = poly.plot3d(
                    name=f"{poly}",
                    color=c,
                    project=project,
                    **kwargs,
                )
                if p_plot is not None:
                    if isinstance(p_plot, dict):
                        if "mesh" in p_plot:
                            meshes.append(p_plot["mesh"])
                        if "outline" in p_plot:
                            outlines.append(p_plot["outline"])
                    else:
                        fig.add_trace(p_plot)
            if label_regions and poly.center is not None:
                fig.add_trace(
                    go.Scatter3d(
                        x=[poly.center[0]],
                        y=[poly.center[1]],
                        z=[
                            self.net(torch.tensor(poly.center, device=self.net.device, dtype=self.net.dtype).T)
                            .detach()
                            .cpu()
                            .numpy()
                            .squeeze()
                            .flatten()[:, 0]
                        ],
                        mode="text",
                        text=str(poly),
                        showlegend=False,
                    )
                )
        for outline in outlines:
            fig.add_trace(outline)
        for mesh in meshes:
            fig.add_trace(mesh)
        maxcoord = np.median([np.max(np.abs(p.interior_point)) for p in self if p.finite]) * PLOT_MARGIN_FACTOR
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=(-maxcoord, maxcoord), visible=show_axes),
                yaxis=dict(range=(-maxcoord, maxcoord), visible=show_axes),
                zaxis=dict(visible=show_axes),
            ),
        )
        return fig
