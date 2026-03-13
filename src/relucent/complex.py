import multiprocessing as mp
import os
import pickle
import random
import warnings
from collections import defaultdict
from functools import partial
from typing import Any, Generator, Iterable, Iterator, cast

import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from gurobipy import Env
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
from relucent.model import NN
from relucent.poly import Polyhedron
from relucent.ss import SSManager
from relucent.utils import BlockingQueue, NonBlockingQueue, UpdatablePriorityQueue, close_env, get_colors, get_env

# Worker process state (set by set_globals when used as pool initializer).
env: Env | None = None
net: NN | None = None
dim: int = 0
get_vol_calc: bool = True


def set_globals(get_net: NN, get_volumes: bool = True, num_threads: int | None = None) -> None:
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
    env = get_env(num_threads=num_threads)
    global net
    net = get_net
    net.save_numpy_weights()  # Refresh weight_cpu after pickle; pickled net can have stale/corrupt weight_cpu
    global dim
    dim = int(np.prod(net.input_shape))
    global get_vol_calc
    get_vol_calc = get_volumes


def poly_calculations(
    task: np.ndarray | torch.Tensor | tuple[Any, ...],
    **kwargs: Any,
) -> tuple[Any, ...]:
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
    assert net is not None, "set_globals must be used as pool initializer"
    ss = task[0] if isinstance(task, tuple) else task
    rest = task[1:] if isinstance(task, tuple) else ()
    p = Polyhedron(net, ss)

    try:
        halfspaces, W, b = p.get_hs()
        assert isinstance(halfspaces, torch.Tensor) or isinstance(halfspaces, np.ndarray)
        assert isinstance(W, torch.Tensor) or isinstance(W, np.ndarray)
        assert isinstance(b, torch.Tensor) or isinstance(b, np.ndarray)
        p._halfspaces = halfspaces
        p._W = W
        p._b = b

        p.get_center_inradius(env=env)
        p.get_interior_point(env=env)
        if p.interior_point is not None:
            p._interior_point_norm = np.linalg.norm(p.interior_point).item()
        else:
            p._interior_point_norm = float("inf")
        if isinstance(p.W, torch.Tensor):
            p._Wl2 = torch.linalg.norm(p.W).item()
        else:
            p._Wl2 = np.linalg.norm(p.W).item()

        if dim <= 6 and get_vol_calc:
            p.volume
        if p._shis is None:
            if "collect_info" in kwargs:
                shis, shi_info = p.get_shis(env=env, **kwargs)
                assert isinstance(shis, list)
                p._shis = shis
            else:
                result = p.get_shis(env=env, **kwargs)
                p._shis = result[0] if isinstance(result, tuple) else result
    except ValueError as error:
        return error, *rest
    p.clean_data()

    if isinstance(p._shis, list):
        random.shuffle(p._shis)
    return (p, *rest)


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
    try:
        ss = p.ss_np.copy()
        ss[0, shi] = -ss[0, shi]
        assert net is not None, "set_globals must be used as pool initializer"
        n = Polyhedron(net, ss)
        for max_radius in INTERIOR_POINT_RADIUS_SEQUENCE:
            try:
                n.get_interior_point(env=env, max_radius=max_radius)
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
        **kwargs: Additional arguments passed to get_shis() method, such as
            'collect_info' or 'bound'.

    Returns:
        tuple: If successful, returns (Polyhedron, *rest). If an exception occurs
            during SHI computation, returns (Polyhedron, error, *rest).
    """
    p = task[0] if isinstance(task, tuple) else task
    rest = task[1:] if isinstance(task, tuple) else ()
    if p.net is None:
        assert net is not None, "set_globals must be used as pool initializer"
        p.net = net

    if p._inradius is None:
        p.get_center_inradius(env=env)
    if p._interior_point is None:
        p.get_interior_point(env=env)

    try:
        if p._shis is None:
            if "collect_info" in kwargs:
                shis, shi_info = p.get_shis(env=env, **kwargs)
                assert isinstance(shis, list)
                p._shis = shis
            else:
                result = p.get_shis(env=env, **kwargs)
                p._shis = result[0] if isinstance(result, tuple) else result
    except Exception as error:
        return p, error, *rest
    p.clean_data()
    assert isinstance(p._shis, list), "get_shis returns a list"
    random.shuffle(p._shis)
    # p, neighbors = get_neighbors(p, (shi for shi in p._shis if shi != task[1]))
    return p, *rest


class Complex:
    """Manages the polyhedral complex of a neural network.

    This class provides methods for calculating, storing, and searching the h-representations
    (halfspace representations) of polyhedra in the complex.
    """

    def __init__(self, net: NN) -> None:
        """Initialize the complex for a given network.

        Args:
            net: The NN (neural network) instance whose polyhedral complex
                is to be built and queried.
        """
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

        self._G = None

    def __del__(self) -> None:
        close_env()

    def __getitem__(self, key: Polyhedron | np.ndarray | torch.Tensor) -> Polyhedron:
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
            raise KeyError("Complex can only be indexed by Polyhedra, arrays, or tensors")

    def str_to_poly(self, name: str, ensure_unique: bool = True) -> Polyhedron:
        """Convert a string to the first Polyhedron with that __repr__. These values may not be unique.

        Args:
            name: The string name.

        Returns:
            Polyhedron: The polyhedron associated with the given name.
        """
        match = None
        for p in self:
            if p.__repr__() == name:
                if match is not None:
                    raise ValueError(f"Multiple Polyhedra with name {name} in Complex")
                if ensure_unique:
                    match = p
                else:
                    return p
        if match is not None:
            return match
        raise KeyError(f"Polyhedron with name {name} not in Complex")

    def __contains__(self, key: Polyhedron | np.ndarray | torch.Tensor) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __iter__(self) -> Iterator[Polyhedron]:
        """Iterate over all Polyhedra in the complex.

        Yields:
            Polyhedron: Polyhedra in the order they were added to the complex.
        """
        for p in self.index2poly:
            yield p

    def __len__(self) -> int:
        return len(self.index2poly)

    def save(self, filename: str | os.PathLike[str], save_ssm: bool = True) -> None:
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
    def load(filename: str | os.PathLike[str]) -> "Complex":
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

    def __getstate__(self) -> dict[str, Any]:
        return {
            "index2poly": self.index2poly,
            "net": self.net,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
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
    def net(self) -> NN:
        """The neural network. May only be set once."""
        return self._net

    @net.setter
    def net(self, value: NN) -> None:
        if self._net is not None:
            raise ValueError("The net attribute cannot be set when it already has a non-None value")
        self._net = value

    @property
    def dim(self) -> int:
        """The input dimension of the network."""
        return int(np.prod(self.net.input_shape))

    @property
    def n(self) -> int:
        """The number of bent hyperplanes/neurons in the network."""
        return len(self.ssi2maski)

    @torch.no_grad()
    def ss_iterator(self, batch: torch.Tensor | np.ndarray) -> Generator[torch.Tensor, None, None]:
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

    def point2ss(self, batch: torch.Tensor | np.ndarray) -> np.ndarray | torch.Tensor:
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

    def add_ss(
        self,
        ss: np.ndarray | torch.Tensor,
        check_exists: bool = True,
    ) -> Polyhedron:
        """Convert a sign sequence to a Polyhedron and add it to the complex.

        Args:
            ss: A sign sequence as a torch.Tensor or np.ndarray.
            check_exists: If True, return the existing polyhedron from the complex
                if it already exists. Defaults to True.

        Returns:
            Polyhedron: The polyhedron that was added (or already existed) in the complex.
        """
        return self.add_polyhedron(Polyhedron(self.net, ss), check_exists=check_exists)

    def add_polyhedron(
        self,
        p: Polyhedron,
        overwrite: bool = False,
        check_exists: bool = True,
    ) -> Polyhedron:
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

    def add_point(
        self,
        data: torch.Tensor | np.ndarray,
        check_exists: bool = True,
    ) -> Polyhedron:
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

    def clean_data(self) -> None:
        """Clean cached data from all polyhedra in the complex.

        This method calls clean_data() on each polyhedron, which removes most of their
        computed data.
        """
        for poly in self:
            poly.clean_data()

    @torch.no_grad()
    def adjacent_polyhedra(self, poly: Polyhedron) -> set[Polyhedron]:
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

    def parallel_add(
        self,
        points: Iterable[torch.Tensor | np.ndarray],
        nworkers: int | None = None,
        bound: float = DEFAULT_PARALLEL_ADD_BOUND,
        **kwargs: Any,
    ) -> list[Polyhedron | None]:
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
        cube_radius: float | None = None,
        cube_mode: str = "unrestricted",
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

        if cube_mode not in {"unrestricted", "intersect", "clipped", "exclude"}:
            raise ValueError("cube_mode must be one of {'unrestricted', 'intersect', 'clipped', 'exclude'}")
        elif cube_mode != "unrestricted":
            assert cube_radius is not None, "cube_radius must be provided when cube_mode is not 'unrestricted'"
            if verbose:
                print(f"Applying cube filter with mode '{cube_mode}' and radius {cube_radius}")
        elif cube_radius is not None:
            warnings.warn("cube_radius is provided but cube_mode is 'unrestricted'. Ignoring cube_radius.")
            cube_radius = None

        ## TODO: Move intersection computations to the workers
        def _poly_intersects_cube(p: Polyhedron) -> bool:
            try:
                # get_bounded_halfspaces raises ValueError when the intersection is empty
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
                # Keep only polyhedra that intersect the cube, but leave them unchanged.
                return intersects
            if cube_mode == "exclude":
                # Exclude polyhedra that intersect the cube; keep only those completely outside.
                return not intersects

            # cube_mode == "clipped": intersect the polyhedron with the cube and
            # replace its halfspaces with the bounded version.
            if not intersects:
                return False
            bounded_halfspaces = p.get_bounded_halfspaces(cast(float, cube_radius))
            p._halfspaces = bounded_halfspaces  # type: ignore[assignment]
            p._halfspaces_np = bounded_halfspaces
            p._nonzero_halfspaces_np = None
            p._center = None
            p._inradius = None
            p._finite = True
            p._vertices = None
            p._volume = None
            p._hs = None
            return True

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
        result = start.get_shis(bound=bound, **kwargs)
        assert isinstance(result, list)
        start._shis = result
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
                        ## TODO: Should this be double checked?
                        node._shis.remove(shi)
                        if len(node._shis) < min(self.dim, self.n):
                            # raise ValueError(f"Polyhedron {node} has less than {min(self.dim, self.n)} SHIs")
                            warnings.warn(
                                RuntimeWarning(f"Polyhedron {node} has less than {min(self.dim, self.n)} SHIs")
                            )
                        if unprocessed == 0 or len(self) >= max_polys:
                            break
                        continue

                    if p.net is None:
                        p.net = self.net

                    if cube_mode != "unrestricted" and not _apply_cube_filter(p):
                        if unprocessed == 0 or len(self) >= max_polys:
                            break
                        continue

                    p = self.add_polyhedron(p)

                    if getattr(p, "warnings", None):
                        for warning in p.warnings:
                            try:
                                warnings.warn(warning)
                            except Exception as e:
                                print(warning, type(warning), e, end="\n\n")

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

                    assert isinstance(p._shis, list)
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

    def _greedy_path_helper(
        self, start: Polyhedron, end: Polyhedron, diffs: set[int] | None = None
    ) -> list[Polyhedron]:
        if start == end:
            # print("Start and end points are the same")
            return [start]

        if (start.ss_np == 0).any():
            raise ValueError("Start point must not be on a hyperplane")

        diffs = diffs or set(np.argwhere((start.ss_np != end.ss_np).ravel()).ravel().tolist())

        print("Diffs:", diffs)

        if not start._shis:
            start.get_shis()
        shis_set = set(start.shis)
        groupa = shis_set & diffs
        groupb = shis_set - diffs
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
            rest = self._greedy_path_helper(next_poly, end, diffs | {shi})
            if rest is not None:
                return [start] + rest
        return []

    def greedy_path(
        self,
        start: torch.Tensor | np.ndarray | Polyhedron,
        end: torch.Tensor | np.ndarray | Polyhedron,
    ) -> list[Polyhedron] | None:
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
        start_poly = self.add_polyhedron(start) if isinstance(start, Polyhedron) else self.add_point(start)
        end_poly = self.add_polyhedron(end) if isinstance(end, Polyhedron) else self.add_point(end)
        return self._greedy_path_helper(start_poly, end_poly)

    def hamming_astar(
        self,
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
        start_poly = self.add_polyhedron(start) if isinstance(start, Polyhedron) else self.add_point(start)
        end_poly = self.add_polyhedron(end) if isinstance(end, Polyhedron) else self.add_point(end)
        if start_poly == end_poly:
            print("Start and end points are in the same region")
            start_poly.get_center_inradius()
            start_poly.get_interior_point()
            start_poly.get_shis(bound=bound)
            return [start_poly]

        if (start_poly.ss_np == 0).any():
            raise ValueError("Start point must not be on a hyperplane")

        nhs = len(start_poly.halfspaces)
        nworkers = min(cast(int, nworkers if isinstance(nworkers, int) else (os.process_cpu_count() or 1)), nhs)
        print(f"Using {nworkers} workers")

        cameFrom: dict[Polyhedron, Polyhedron] = {}
        gScore = defaultdict(lambda: float("inf"))
        fScore = defaultdict(lambda: float("inf"))

        openSet = NonBlockingQueue(
            queue_class=UpdatablePriorityQueue,
            push=lambda pq, task, priority: pq.push(task, priority=priority),
            pop=lambda pq: pq.pop(),
        )
        # found_sss = SSManager()
        # nworkers = nworkers or os.process_cpu_count()

        gScore[start_poly] = 0
        fScore[start_poly] = (start_poly.ss_np != end_poly.ss_np).sum()
        # found_sss.add(start_poly.ss)
        # found = set(start_poly)

        result = start_poly.get_shis(bound=bound, **kwargs)
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
            dist = np.linalg.norm(p.interior_point - end_poly.interior_point).item()
            # bias = -1 / ((1 + dist) ** 0.1)  ## TODO: Test if this is faster
            # bias = -1 / (1 + np.log(dist + 10))
            bias = -1 / (1 + dist)
            # bias = 0
            # bias = 1 / (1 + depth) - 1
            return hamming + ASTAR_BIAS_WEIGHT * bias

        def d(p1: Polyhedron, p2: Polyhedron) -> int:
            return 1

        # with mp.Pool(nworkers, initializer=set_globals, initargs=(self.net,)) as pool:
        pool = (
            mp.Pool(nworkers, initializer=set_globals, initargs=(self.net, False, num_threads))
            if nworkers > 1
            else None
        )
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

                if nworkers == 1:
                    neighbor_iter = map(partial(get_ip, p), (i for i in p.shis if i != shi))
                else:
                    neighbor_iter = pool.imap_unordered(
                        partial(get_ip, p),
                        (i for i in p.shis if i != shi),
                        chunksize=max(nhs // nworkers, 1),
                    )

                for neighbor, neighbor_shi in neighbor_iter:
                    if not isinstance(neighbor, Polyhedron):
                        ## TODO: Should this be double checked?
                        p._shis.remove(neighbor_shi)
                    else:
                        tentative_gScore = gScore[p] + d(p, neighbor)
                        if neighbor.net is None:
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
                    if 0 < min_dist:
                        ## TODO: What if there are no/multiple differences?
                        last_shi = np.argwhere((min_p.ss_np != end_poly.ss_np).ravel()).item()
                        if last_shi in min_p.shis:
                            cameFrom[end_poly] = min_p
                            neighbor = end_poly
                            break
                    else:
                        # raise ValueError("what in tarnation???")
                        neighbor = end_poly
                        break

                if unprocessed == 0 or len(cameFrom) >= max_polys:
                    break
        except Exception:
            raise
        finally:
            # print(f"Closing out after {pbar.n} iterations")

            if pool is not None:
                pool.terminate()
                pool.join()
                pool.close()

            openSet.close()
            tqdm.get_lock().locks = []
            pbar.close()
        #     print("Closed out")
        # print("Finished A* Search")
        if neighbor == end_poly:
            path = [end_poly]
            while path[-1] != start_poly:
                assert cameFrom[path[-1]] not in path, path
                path.append(cameFrom[path[-1]])
            path.reverse()
            [(p.Wl2, p.inradius) for p in path]
            # print(f"Path found with length {len(path) - 1}:")
            # if len(path) < 100:
            #     for p1, p2 in zip(path[:-1], path[1:]):
            #         print(f"    {p1} - {np.argwhere((p1.ss != p2.ss).ravel()).item()} -> {p2}")
            return path
        else:
            # print(f"No Path Found - Final Distance: {min_dist - 1}")
            return None

    def get_poly_attrs(self, attrs: Iterable[str]) -> dict[str, list[Any]]:
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

    def get_boundary_edges(self, i):
        """Get the boundary of neuron i by returning the set of edges in the dual graph with label i."""
        return {edge for edge in self.G.edges() if self.G.edges[edge]["shi"] == i}

    def get_boundary_graph(self, i):
        """Get the induced subgraph of neuron i's BH."""
        return self.G.subgraph(
            [edge[0] for edge in self.get_boundary_edges(i)] + [edge[1] for edge in self.get_boundary_edges(i)]
        )

    def get_boundary_cells(self, i):
        """Get all (d-1)-cells in neuron i's BH."""
        faces = set()
        for edge in self.get_boundary_edges(i):
            ss = edge[0].ss_np.copy()
            ss[0, self.G.edges[edge]["shi"]] *= 0
            faces.add(self.ss2poly(ss))
        return faces

    def get_boundary_complex(self, i):
        """Get the boundary complex of neuron i."""
        cplx = Complex(self.net)
        for poly in self.get_boundary_cells(i):
            cplx.add_polyhedron(poly)
        return cplx

    @property
    def G(self) -> nx.Graph:
        if self._G is None:
            self._G = self.get_dual_graph()
        return self._G

    def get_dual_graph(
        self,
        relabel=False,
        plot=False,
        node_color=None,
        node_size=None,
        cmap="viridis",
        match_locations=False,
        show_node_labels=False,
        show_edge_labels=False,
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
            show_node_labels: If True, show node labels in the graph. Defaults to False.
            show_edge_labels: If True, show edge labels (SHI) in the graph. Defaults to False.

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

                nx.set_node_attributes(G, values=False, name="physics")  # type: ignore
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
                nx.set_node_attributes(G, values=4, name="size")  # type: ignore

            for node in G.nodes:
                G.nodes[node]["label"] = str(node) if show_node_labels else ""
                G.nodes[node]["title"] = str(node)
            for edge in G.edges:
                G.edges[edge]["label"] = str(G.edges[edge]["shi"]) if show_edge_labels else ""
                G.edges[edge]["title"] = str(G.edges[edge]["shi"])
        if plot or relabel:
            G = nx.relabel_nodes(G, {poly: i for i, poly in enumerate(self)})
        return G

    def recover_from_dual_graph(
        self,
        G: nx.Graph,
        initial_ss: np.ndarray | torch.Tensor,
        source: int,
        copy: bool = False,
    ) -> None:
        """Recover a complex from its connectivity graph.

        Reconstructs polyhedra in the complex by traversing the adjacency graph
        of top-dimensional cells, using the supporting hyperplane indices stored
        on edges to determine how to flip sign sequence elements. This is useful
        for storing large complexes efficiently, as you only need to store the
        graph structure and SHI indices on edges rather than full polyhedron data.

        Args:
            G: A networkx.Graph representing the dual graph. Edges must have
                a "shi" attribute indicating the supporting hyperplane index.
            initial_ss: The sign sequence of the starting polyhedron as
                torch.Tensor or np.ndarray.
            source: The node key in G for the polyhedron with sign sequence initial_ss.
            copy: If True, operate on a copy of G; otherwise modify G in place.
                Defaults to False.

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

    def plot_2d_complex(
        self,
        label_regions=False,
        color=None,
        highlight_regions=None,
        ss_name=False,
        bound=DEFAULT_COMPLEX_PLOT_BOUND,
        **kwargs,
    ):
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
                polyhedron's plot_2d_complex() method. Defaults to config.DEFAULT_COMPLEX_PLOT_BOUND.
            **kwargs: Additional arguments passed to each polyhedron's plot_2d_complex() method.

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
        for c, poly in tqdm(zip(colors, polys), desc="Plotting Polyhedra", total=len(polys), delay=1):
            if (highlight_regions is not None) and ((poly in highlight_regions) or (str(poly) in highlight_regions)):
                c = "red"
            if ss_name:
                name = f"{poly.ss_np.ravel().astype(int).tolist()}"
            else:
                name = f"{poly}"
            p_plot = poly.plot_2d_complex(
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
        maxcoord = (
            (np.mean(interior_points) * PLOT_MARGIN_FACTOR)
            if len(interior_points) > 0
            else min(PLOT_DEFAULT_MAXCOORD, bound if bound else PLOT_DEFAULT_MAXCOORD)
        )
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

    def plot_3d_complex(
        self,
        label_regions: bool = False,
        color: str | None = None,
        highlight_regions: Iterable[Polyhedron] | Iterable[str] | None = None,
        show_axes: bool = False,
        **kwargs: Any,
    ) -> go.Figure:
        """Plot the complex in 3D input space using plotly.

        This visualizes the polyhedral complex of a network with 3D input by
        drawing each cell as a 3D region in the input space. Axis scaling is
        automatic based on the traces returned by each polyhedron's plot function.

        Args:
            label_regions: If True, add text labels at (approximate) centers of
                polyhedra. Defaults to False.
            color: If "Wl2", color polyhedra by their Wl2 (weight norm) value.
                If None, use an equitable graph coloring. Defaults to None.
            highlight_regions: If provided, highlight these polyhedra in red and
                render them as filled meshes instead of wireframes. Can be a
                list of Polyhedron objects or their string representations.
                Defaults to None.
            show_axes: If True, display coordinate axes. Defaults to False.
            **kwargs: Additional keyword arguments forwarded to
                ``Polyhedron.plot_3d_complex``.

        Returns:
            plotly.graph_objects.Figure: A figure containing the 3D plot of the complex.
        """
        if self.dim != 3:
            raise ValueError("Complex must have 3D input to plot 3D complex")

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

        for c, poly in tqdm(zip(colors, polys), desc="Plotting 3D Polyhedra", total=len(polys), delay=1):
            is_highlighted = (highlight_regions is not None) and (
                (poly in [p for p in highlight_regions if isinstance(p, Polyhedron)])
                or (str(poly) in [p for p in highlight_regions if isinstance(p, str)])
            )
            if is_highlighted:
                c = "red"
            traces = poly.plot_3d_complex(
                showlegend=False,
                color=c,
                filled=is_highlighted,
                **kwargs,
            )
            for trace in traces:
                fig.add_trace(trace)
            if label_regions and poly.center is not None and poly.center.shape[0] >= 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=[poly.center[0]],
                        y=[poly.center[1]],
                        z=[poly.center[2]],
                        mode="text",
                        text=str(poly),
                        showlegend=False,
                    )
                )

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=show_axes),
                yaxis=dict(visible=show_axes),
                zaxis=dict(visible=show_axes),
            ),
        )
        return fig

    def plot_2d_graph(
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
            **kwargs: Additional arguments passed to each polyhedron's plot_2d_graph() method.

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
        for c, poly in tqdm(zip(colors, polys), desc="Plotting Polyhedra", total=len(polys), delay=1):
            if (highlight_regions is not None) and ((poly in highlight_regions) or (str(poly) in highlight_regions)):
                c = "red"
            p_plot = poly.plot_2d_graph(
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
                p_plot = poly.plot_2d_graph(
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
                            .ravel()[:, 0]
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
