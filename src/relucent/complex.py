from __future__ import annotations

import os
import pickle
import random
import warnings
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterable, Iterator
from itertools import combinations
from typing import Any, Literal, overload

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from gurobipy import Env
from tqdm.auto import tqdm

import relucent.config as cfg
from relucent._logging import logger
from relucent._torch_compat import TORCH_AVAILABLE, torch
from relucent.convert_model import convert
from relucent.model import Layer, LinearLayer, ReLULayer, ReLUNetwork
from relucent.poly import Polyhedron
from relucent.search import greedy_path as _greedy_path_fn
from relucent.search import hamming_astar as _hamming_astar_fn
from relucent.search import parallel_add as _parallel_add_fn
from relucent.search import parallel_compute_geometric_properties as _parallel_compute_geometric_properties_fn
from relucent.search import searcher as _searcher_fn
from relucent.ss import SSManager
from relucent.utils import (
    BlockingQueue,
    get_env,
)
from relucent.vis import get_colors, plot_complex

__all__ = ["Complex"]
RESEARCH_WARNING_DISABLE_ENV_VAR = "DISABLE_RESEARCH_WARNING"

# ``shi`` edge attribute for truncation incidences in :meth:`Complex.get_meta_graph`
# when ``truncate=True`` (not a supporting-hyperplane index of the underlying network).
TRUNCATION_META_SHI: int = -1

# Worker process state — set by set_globals() when used as a pool initializer.
env: Env | None = None
_net: ReLUNetwork | None = None
dim: int = 0
get_vol_calc: bool = True


def set_globals(get_net: ReLUNetwork, get_volumes: bool = True, num_threads: int | None = None) -> None:
    """Initialize global variables for worker processes in multiprocessing.

    This function should only be used as an initializer for multiprocessing pools,
    never called directly by the main process. It sets up the network, environment,
    and volume calculation settings that worker processes need.

    Args:
        get_net: The neural network object to be used by worker processes.
        get_volumes: Whether to compute volumes for polyhedra when input dimension <= 6.
            Defaults to True.

    Example:
        >>> import multiprocessing as mp
        >>> with mp.Pool(nworkers, initializer=set_globals, initargs=(net,)) as pool:
        ...     # Use pool for parallel processing
        ...     pass
    """
    global env
    env = get_env(num_threads=num_threads)
    global _net
    _net = get_net
    global dim
    dim = int(np.prod(_net.input_shape))
    global get_vol_calc
    get_vol_calc = get_volumes


def _delete_ss_columns(ss: np.ndarray | torch.Tensor, deleted_shis: Iterable[int]) -> np.ndarray | torch.Tensor:
    axis = int(ss.ndim - 1)
    for shi in sorted(set(int(s) for s in deleted_shis), reverse=True):
        if isinstance(ss, np.ndarray):
            ss = np.delete(ss, shi, axis=axis)
        elif TORCH_AVAILABLE and isinstance(ss, torch.Tensor):
            keep = [i for i in range(ss.shape[axis]) if i != shi]
            ss = ss.index_select(axis, torch.tensor(keep, device=ss.device))
        else:
            raise TypeError(f"Unsupported ss type: {type(ss)}")
    return ss


def _contract_dual_graph_for_shi(
    graph: nx.Graph[int],
    deleted_shi: int,
) -> tuple[nx.Graph[int], dict[int, int]]:
    """Quotient a relabeled dual graph by edges with ``shi == deleted_shi``.

    Returns the contracted graph (nodes ``0 .. n-1``) and a map from each new
    node to a representative old node id.
    """
    parent = {node: node for node in graph.nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for u, v, data in graph.edges(data=True):
        if data.get("shi") == deleted_shi:
            union(u, v)

    roots = sorted({find(node) for node in graph.nodes})
    root_to_new = {root: i for i, root in enumerate(roots)}
    old_representative = {root_to_new[root]: root for root in roots}
    node_map = {node: root_to_new[find(node)] for node in graph.nodes}

    contracted = nx.Graph()
    contracted.add_nodes_from(range(len(roots)))
    for u, v, data in graph.edges(data=True):
        shi = data.get("shi")
        if shi == deleted_shi:
            continue
        a, b = node_map[u], node_map[v]
        if a == b:
            continue
        if shi is None:
            continue
        new_shi = shi - 1 if shi > deleted_shi else shi
        if contracted.has_edge(a, b):
            assert contracted.edges[a, b]["shi"] == new_shi
        else:
            contracted.add_edge(a, b, shi=new_shi)
    return contracted, old_representative


def _net_remove_ss_layer_and_following_relu(net: ReLUNetwork, ss_layer_idx: int) -> ReLUNetwork:
    """Remove a width-1 linear layer and the ReLU immediately after it."""
    items = list(net.layers.items())
    relu_idx = ss_layer_idx + 1
    if relu_idx >= len(items) or not isinstance(items[relu_idx][1], ReLULayer):
        raise ValueError(f"Layer index {ss_layer_idx} is not immediately followed by a ReLU.")
    removed_linear = items[ss_layer_idx][1]
    if not isinstance(removed_linear, LinearLayer):
        raise ValueError(f"Layer index {ss_layer_idx} is not a LinearLayer.")
    n_removed_outputs = int(removed_linear.weight.shape[0])

    new_items: list[tuple[str, Layer]] = []
    skip_next_relu = False
    for i, (name, layer) in enumerate(items):
        if i == ss_layer_idx:
            skip_next_relu = True
            continue
        if skip_next_relu and i == relu_idx:
            skip_next_relu = False
            continue
        if skip_next_relu:
            raise RuntimeError("Expected ReLU immediately after removed linear layer.")
        if i == relu_idx + 1 and isinstance(layer, LinearLayer):
            weight = np.delete(layer.weight, np.arange(n_removed_outputs), axis=1)
            new_items.append((name, LinearLayer(weight=weight, bias=layer.bias)))
        else:
            new_items.append((name, layer))
    return ReLUNetwork(OrderedDict(new_items), input_shape=net.input_shape)


def _net_without_last_ss_layer_neuron(
    net: ReLUNetwork,
    ss_layer_idx: int,
    neuron_idx: int,
) -> ReLUNetwork:
    """Return a copy of ``net`` with one neuron removed from the given ReLU hidden layer."""
    layer = list(net.layers.values())[ss_layer_idx]
    if not isinstance(layer, LinearLayer):
        raise ValueError(f"Layer index {ss_layer_idx} is not a LinearLayer.")
    if int(layer.weight.shape[0]) == 1:
        return _net_remove_ss_layer_and_following_relu(net, ss_layer_idx)

    items = list(net.layers.items())
    new_items: list[tuple[str, Layer]] = []
    delete_column_from_next_linear = False
    for i, (name, layer) in enumerate(items):
        if i == ss_layer_idx and isinstance(layer, LinearLayer):
            weight = np.delete(layer.weight, neuron_idx, axis=0)
            bias = np.delete(layer.bias, neuron_idx, axis=1)
            new_items.append((name, LinearLayer(weight=weight, bias=bias)))
            delete_column_from_next_linear = True
        elif delete_column_from_next_linear and isinstance(layer, LinearLayer):
            weight = np.delete(layer.weight, neuron_idx, axis=1)
            new_items.append((name, LinearLayer(weight=weight, bias=layer.bias)))
            delete_column_from_next_linear = False
        else:
            new_items.append((name, layer))
    return ReLUNetwork(OrderedDict(new_items), input_shape=net.input_shape)


class Complex:
    """Manages the polyhedral complex of a neural network.

    This class provides methods for calculating, storing, and searching the h-representations
    (halfspace representations) of polyhedra in the complex.
    """

    def __init__(self, net: Any) -> None:
        """Initialize the complex for a given network.

        Args:
            net: Any model convertible to relucent's canonical ``NN``.
                is to be built and queried.
        """
        original_net = net
        if not isinstance(net, ReLUNetwork):
            net = convert(net)
        self.net = original_net
        self._net = net

        self.ssm = SSManager()
        self.index2poly: list[Polyhedron] = []

        net_layers = list(self._net.layers.values())
        self.ss_layers = [i for i, next_layer in enumerate(net_layers[1:]) if isinstance(next_layer, ReLULayer)]

        # Build mapping from global sign-sequence indices to (layer_index, neuron_index)
        self.ssi2maski = []
        for i, layer in enumerate(self._net.layers.values()):
            if i in self.ss_layers:
                assert isinstance(layer, LinearLayer), "Only linear layers should be before ReLU layers"
                for neuron_idx in range(layer.weight.shape[0]):
                    self.ssi2maski.append((i, (0, neuron_idx)))

        self._dual_graph: nx.Graph[Polyhedron] | None = None

    def __repr__(self) -> str:
        net_name = type(self._net).__name__ if getattr(self, "_net", None) is not None else "None"
        return f"Complex(dim={self.dim}, n={self.n}, n_polyhedra={len(self.index2poly)}, net={net_name}@{id(self._net):#x})"

    def __str__(self) -> str:
        return f"Complex(n_polyhedra={len(self)})"

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
        """Return the polyhedron whose ``__repr__`` equals ``name``.

        Args:
            name: The string returned by ``str(poly)`` / ``repr(poly)``.
            ensure_unique: If ``True`` (default), raise :exc:`ValueError` when more
                than one polyhedron matches. If ``False``, return the first match
                immediately without scanning for duplicates.

        Returns:
            The matching :class:`Polyhedron`.

        Raises:
            KeyError: If no polyhedron with the given name is in the complex.
            ValueError: If ``ensure_unique`` is ``True`` and multiple polyhedra match.
        """
        match = None
        for p in self:
            if p.__repr__() == name:
                if not ensure_unique:
                    return p
                if match is not None:
                    raise ValueError(f"Multiple polyhedra with name {name!r} in complex")
                match = p
        if match is not None:
            return match
        raise KeyError(f"Polyhedron with name {name!r} not in complex")

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
        yield from self.index2poly

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
    def load(filename: str | os.PathLike[str]) -> Complex:
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
            "_net": self._net,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state.get("net", state["_net"]))
        if "_net" in state:
            self._net = state["_net"]
        self.index2poly = state["index2poly"]
        if "ssm" in state:
            self.ssm = state["ssm"]
        else:
            for p in self.index2poly:
                self.ssm.add(p.ss_np)
        for p in self.index2poly:
            p._net = self._net

    @property
    def dim(self) -> int:
        """The input dimension of the network."""
        return int(np.prod(self._net.input_shape))

    @property
    def n(self) -> int:
        """The number of bent hyperplanes/neurons in the network."""
        return len(self.ssi2maski)

    def _deleted_shi_for_last_layer_neuron(self, neuron_idx: int) -> int:
        last_ss_layer = max(self.ss_layers)
        for shi, (layer_idx, (_, idx)) in enumerate(self.ssi2maski):
            if layer_idx == last_ss_layer and idx == neuron_idx:
                return shi
        raise RuntimeError(f"neuron_idx {neuron_idx} is not in the last ReLU hidden layer.")

    def without_last_layer_neuron(self, neuron_idx: int) -> Complex:
        """Return the complex obtained by deleting a neuron from the last ReLU layer.

        The last ReLU layer is the final :class:`~relucent.model.LinearLayer` that is
        immediately followed by a ReLU in the canonical network (the same layer used
        by output-neuron boundary analysis).  Top-dimensional cells that shared a
        facet on the removed neuron are merged; recovered SHIs are the union of the
        two sides' facet indices, minus the removed neuron (see :meth:`contract`).

        If that layer has only one neuron, the linear layer and its following ReLU are
        removed from the network entirely.

        Implementation: contract dual-graph edges for the removed SHI, then rebuild
        cells with :meth:`recover_from_dual_graph` on the smaller network.

        Args:
            neuron_idx: Index of the neuron within that last hidden linear layer
                (not the global supporting-hyperplane index).  Must be ``0`` when the
                layer has width ``1``.

        Returns:
            A new :class:`Complex` over the smaller network.  The dual graph is not
            copied.

        Raises:
            ValueError: If there is no ReLU hidden layer or ``neuron_idx`` is out of
                range for the last one.
        """
        if not self.ss_layers:
            raise ValueError("Network has no ReLU layers; cannot delete a neuron.")
        last_ss_layer = max(self.ss_layers)
        layer = list(self._net.layers.values())[last_ss_layer]
        if not isinstance(layer, LinearLayer):
            raise ValueError("Last sign-sequence layer is not a LinearLayer.")
        n_neurons = int(layer.weight.shape[0])
        if not (0 <= neuron_idx < n_neurons):
            raise ValueError(f"neuron_idx must be in [0, {n_neurons}), got {neuron_idx} for the last ReLU layer.")

        deleted_shi = self._deleted_shi_for_last_layer_neuron(neuron_idx)
        new_net = _net_without_last_ss_layer_neuron(self._net, last_ss_layer, neuron_idx)
        out = Complex(new_net)
        out.net = new_net if isinstance(self.net, ReLUNetwork) else self.net

        dual = self.get_dual_graph(relabel=True, verbose=False)
        if dual.number_of_nodes() == 0:
            return out

        contracted, old_rep = _contract_dual_graph_for_shi(dual, deleted_shi)
        for component in nx.connected_components(contracted):
            sub = contracted.subgraph(component).copy()
            source = min(component)
            initial_ss = _delete_ss_columns(self.index2poly[old_rep[source]].ss_np, [deleted_shi])
            out.recover_from_dual_graph(sub, initial_ss, source=source, copy=True)

        return out

    @torch.no_grad()
    def ss_iterator(self, batch: torch.Tensor | np.ndarray) -> Generator[torch.Tensor | np.ndarray, None, None]:
        """Generate sign sequences for each ReLU layer from a batch of data points.

        Args:
            batch: A batch of input data points as a torch.Tensor, np.ndarray, or array-like.
                Will be reshaped to match the network's input shape.

        Yields:
            torch.Tensor: Sign sequences for each ReLU layer in
                the network, indicating the activation pattern of that layer.
        """
        if TORCH_AVAILABLE and isinstance(batch, torch.Tensor):
            x: torch.Tensor | np.ndarray = batch.reshape((-1, *self._net.input_shape))
            use_torch = True
        else:
            x = np.asarray(batch, dtype=np.float64).reshape((-1, *self._net.input_shape))
            use_torch = False
        for i, layer in enumerate(self._net.layers.values()):
            x = self._net._apply_layer(layer, x)
            if i in self.ss_layers:
                if use_torch:
                    yield torch.sign(torch.as_tensor(x))
                else:
                    yield np.sign(np.asarray(x))
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
        ss_parts = list(self.ss_iterator(batch))
        if is_tensor and TORCH_AVAILABLE:
            return torch.hstack([torch.as_tensor(s) for s in ss_parts])
        return np.hstack([np.asarray(s) for s in ss_parts])

    def point2poly(
        self,
        point: torch.Tensor | np.ndarray,
        check_exists: bool = True,
        **kwargs: Any,
    ) -> Polyhedron:
        """Convert a data point to its corresponding Polyhedron.

        Finds the polyhedron that contains the given data point. Does not add
        the polyhedron to the complex.

        Args:
            point: A single data point as a torch.Tensor or np.ndarray.
            check_exists: If True, return the existing polyhedron from the complex
                if it already exists. Defaults to True.
            **kwargs: Additional arguments passed to the Polyhedron constructor.

        Returns:
            Polyhedron: The polyhedron containing the given point.
        """
        return self.ss2poly(self.point2ss(point), check_exists=check_exists, **kwargs)

    def ss2poly(
        self,
        ss: np.ndarray | torch.Tensor,
        check_exists: bool = True,
        **kwargs: Any,
    ) -> Polyhedron:
        """Convert a sign sequence to a Polyhedron.

        Creates a Polyhedron object from the given sign sequence. Does not add
        it to the complex.

        Args:
            ss: A sign sequence as a torch.Tensor or np.ndarray.
            check_exists: If True, return the existing polyhedron from the complex
                if it already exists. Defaults to True.
            **kwargs: Additional arguments passed to the Polyhedron constructor.

        Returns:
            Polyhedron: The polyhedron corresponding to the given sign sequence.
        """
        if check_exists and ss in self:
            return self[ss]
        else:
            return Polyhedron(self._net, ss, **kwargs)

    def add_ss(
        self,
        ss: np.ndarray | torch.Tensor,
        check_exists: bool = True,
        **kwargs: Any,
    ) -> Polyhedron:
        """Convert a sign sequence to a Polyhedron and add it to the complex.

        Args:
            ss: A sign sequence as a torch.Tensor or np.ndarray.
            check_exists: If True, return the existing polyhedron from the complex
                if it already exists. Defaults to True.
            **kwargs: Additional arguments passed to the Polyhedron constructor.

        Returns:
            Polyhedron: The polyhedron that was added (or already existed) in the complex.
        """
        return self.add_polyhedron(self.ss2poly(ss, check_exists=check_exists, **kwargs), check_exists=check_exists)

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
            self._dual_graph = None
            return p

        p_exists = p in self

        if p_exists and overwrite:
            self.index2poly[self.ssm[p.ss_np]] = p
            self._dual_graph = None
            return p
        elif p_exists:
            return self[p]
        else:
            self.index2poly.append(p)
            self.ssm.add(p.ss_np)
            self._dual_graph = None
            return p

    def add_point(
        self,
        data: torch.Tensor | np.ndarray,
        check_exists: bool = True,
        **kwargs: Any,
    ) -> Polyhedron:
        """Find the polyhedron containing a data point and add it to the complex.

        Args:
            data: A single data point as a torch.Tensor, np.ndarray, or array-like.
            check_exists: If True, check whether the polyhedron already exists in
                the complex and return it if so. Only set to false if you know it does not.
                Defaults to True.
            **kwargs: Additional arguments passed to the Polyhedron constructor.

        Returns:
            Polyhedron: The polyhedron containing the given point, now stored in the complex.
        """
        return self.add_ss(self.point2ss(data), check_exists=check_exists, **kwargs)

    def clean_data(self) -> None:
        """Clean cached data from all polyhedra in the complex.

        This method calls clean_data() on each polyhedron, which removes most of their
        computed data.
        """
        for poly in self:
            poly.clean_data()

    def parallel_add(
        self,
        points: Iterable[torch.Tensor | np.ndarray],
        nworkers: int | None = None,
        bound: float | None = None,
        verbose: int | None = None,
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
            verbose: Controls progress output. ``0`` silences all output; ``1``
                (default) shows worker count and progress bars.  When ``None``,
                falls back to :data:`relucent.config.VERBOSE`.
            **kwargs: Additional arguments passed to poly_calculations() and get_shis().

        Returns:
            list: A list of Polyhedron objects (or None for failed computations)
                corresponding to the input points.
        """
        if bound is None:
            bound = cfg.DEFAULT_PARALLEL_ADD_BOUND
        return _parallel_add_fn(
            self,
            points,
            nworkers=nworkers,
            bound=bound,
            verbose=verbose,
            **kwargs,
        )

    def searcher(
        self,
        start: torch.Tensor | np.ndarray | Polyhedron | None = None,
        max_depth: float = float("inf"),
        max_polys: float = float("inf"),
        queue: Any = None,
        bound: float | None = None,
        nworkers: int | None = None,
        verbose: int | None = None,
        cube_radius: float | None = None,
        cube_mode: str = "unrestricted",
        geometry_properties: Iterable[str] | None = None,
        keep_caches: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
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
            verbose: Controls progress output. ``0`` silences all output; ``1``
                (default) shows worker count and a progress bar.  When ``None``,
                falls back to :data:`relucent.config.VERBOSE`.
            geometry_properties: Iterable of polyhedron cache/property names to
                compute for each discovered polyhedron during search. If None,
                uses the default non-SciPy geometry set. Pass an empty iterable
                for topology-only search.
            keep_caches: If True, keep caches such as ``halfspaces``,
                ``W``, and ``b`` when polyhedra are returned from worker
                processes. Defaults to False.
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
        if bound is None:
            bound = cfg.DEFAULT_SEARCH_BOUND
        return _searcher_fn(
            self,
            start=start,
            max_depth=max_depth,
            max_polys=max_polys,
            queue=queue,
            bound=bound,
            nworkers=nworkers,
            verbose=verbose,
            cube_radius=cube_radius,
            cube_mode=cube_mode,
            geometry_properties=geometry_properties,
            keep_caches=keep_caches,
            **kwargs,
        )

    def compute_geometric_properties(
        self,
        nworkers: int | None = None,
        properties: Iterable[str] | None = None,
        keep_caches: bool = False,
        verbose: int | None = None,
    ) -> dict[str, Any]:
        """Compute selected polyhedron caches in parallel.

        This is intended to run after a topology-only search pass.

        Args:
            nworkers: Number of worker processes (defaults to CPU count).
            properties: Iterable of cache/property names to compute. If None,
                uses relucent's default non-SciPy geometric set.
            keep_caches: If True, retain heavy caches (halfspaces/W/b) after
                worker transfer; otherwise they are cleaned to reduce memory.
            verbose: Controls progress output. ``0`` silences all output; ``1``
                (default) shows worker count and a progress bar.  When ``None``,
                falls back to :data:`relucent.config.VERBOSE`.
        """
        return _parallel_compute_geometric_properties_fn(
            self,
            nworkers=nworkers,
            geometry_properties=properties,
            keep_caches=keep_caches,
            verbose=verbose,
        )

    def bfs(self, **kwargs: Any) -> dict[str, Any]:
        """Perform breadth-first search of the complex.

        Explores the complex using a breadth-first strategy, discovering all
        polyhedra at depth d before moving to depth d+1. Uses a FIFO queue.

        Args:
            **kwargs: All arguments accepted by searcher().

        Returns:
            dict: Search information dictionary (see searcher() documentation).
        """
        return self.searcher(**kwargs)

    def dfs(self, **kwargs: Any) -> dict[str, Any]:
        """Perform depth-first search of the complex.

        Explores the complex using a depth-first strategy, following paths as
        deeply as possible before backtracking. Uses a LIFO queue.

        Args:
            **kwargs: All arguments accepted by searcher().

        Returns:
            dict: Search information dictionary (see searcher() documentation).
        """
        return self.searcher(queue=BlockingQueue(pop=lambda x: x.pop()), **kwargs)

    def random_walk(self, **kwargs: Any) -> dict[str, Any]:
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
        return _greedy_path_fn(self, start, end)

    @staticmethod
    def _warn_research_use(method_name: str) -> None:
        """Emit a collaboration warning unless disabled by environment variable."""
        disable_warning = os.getenv(RESEARCH_WARNING_DISABLE_ENV_VAR, "").strip().lower()
        if disable_warning in {"1", "true", "yes", "on"}:
            return
        warnings.warn(
            (
                f"Complex.{method_name}() is actively used by the package author in ongoing research. "
                + "If you'd like to collaborate, please reach out! My email is blake@uconn.edu. "
                + f"Set {RESEARCH_WARNING_DISABLE_ENV_VAR}=1 to silence this warning."
            ),
            UserWarning,
            stacklevel=2,
        )

    def hamming_astar(
        self,
        start: torch.Tensor | np.ndarray | Polyhedron,
        end: torch.Tensor | np.ndarray | Polyhedron,
        nworkers: int | None = None,
        bound: float | None = None,
        max_polys: float = float("inf"),
        show_pbar: bool = True,
        num_threads: int = 1,
        **kwargs: Any,
    ) -> dict[str, Any]:
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
            dict[str, Any]: Dictionary containing the path (if found) and
                additional diagnostics/bounds.

        Raises:
            ValueError: If the start point lies exactly on a neuron's boundary.
        """
        self._warn_research_use("hamming_astar")
        if bound is None:
            bound = cfg.DEFAULT_SEARCH_BOUND
        return _hamming_astar_fn(
            self,
            start=start,
            end=end,
            nworkers=nworkers,
            bound=bound,
            max_polys=max_polys,
            show_pbar=show_pbar,
            num_threads=num_threads,
            **kwargs,
        )

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
        attrs = list(attrs)
        self.compute_geometric_properties(properties=attrs)
        return {attr: [getattr(poly, attr) for poly in self] for attr in attrs}

    def get_boundary_edges(self, i: int, verbose: bool = False) -> set[tuple[Polyhedron, Polyhedron]]:
        """Get the boundary of neuron i by returning the set of edges in the dual graph with label i."""
        assert 0 <= i < self.n, f"Neuron index out of range: {i} not in [0, {self.n})"
        return {
            (a, b)
            for a, b, shi in tqdm(self.G.edges(data="shi"), desc="Getting Boundary Edges", delay=1, disable=not verbose)
            if shi == i
        }

    def get_boundary_graph(self, i: int, verbose: bool = False) -> nx.Graph[Polyhedron]:
        """Get the induced subgraph of neuron i's BH."""
        return self.G.subgraph(
            [edge[0] for edge in self.get_boundary_edges(i, verbose=verbose)]
            + [edge[1] for edge in self.get_boundary_edges(i, verbose=verbose)]
        )

    def get_boundary_cells(self, i: int, verbose: bool = False) -> set[Polyhedron]:
        """Get all (d-1)-cells in neuron i's BH."""
        faces = set()
        for edge in tqdm(
            self.get_boundary_edges(i, verbose=verbose), desc="Getting Boundary Cells", delay=1, disable=not verbose
        ):
            ss = edge[0].ss_np.copy()
            ss[0, self.G.edges[edge]["shi"]] = 0
            # Do not pass ``finite`` from cofaces: ambient boundedness of neighbors does not
            # determine intrinsic boundedness of the shared (d-1)-face (e.g. a segment can be
            # bounded while both incident d-cells are unbounded in R^d). Let Chebyshev run.
            p = self.ss2poly(
                ss,
                check_exists=False,
                shis=list(set(edge[0].shis) & set(edge[1].shis) - {self.G.edges[edge]["shi"]}),
            )
            faces.add(p)
        return faces

    def get_boundary_complex(self, i: int, verbose: bool = False) -> Complex:
        """Get the boundary complex of neuron i."""
        cplx = Complex(self.net)
        for poly in tqdm(
            self.get_boundary_cells(i, verbose=verbose), desc="Getting Boundary Complex", delay=1, disable=not verbose
        ):
            cplx.add_polyhedron(poly, check_exists=False)
        return cplx

    def contract(self, verbose: bool = False) -> Complex:
        """Contract the maximal cells in the complex."""
        G = self.get_dual_graph(verbose=verbose)
        new_complex = Complex(self.net)
        for p1, p2, shi in G.edges(data="shi"):
            new_ss = p1.ss_np.copy()
            new_ss[0, shi] = 0
            new_complex.add_ss(
                new_ss,
                halfspaces=p1.halfspaces,
                shis=list(set(p1.shis) & set(p2.shis) - {shi}),
            )

        return new_complex

    def get_chain_complex(self, verbose: bool = False) -> list[Complex]:
        """Get the chain complex of the complex."""
        chain: list[Complex] = [self]
        while True:
            new_complex = chain[-1].contract(verbose=verbose)
            if len(new_complex) == 0:
                break
            chain.append(new_complex)
            if verbose:
                logger.info("Chain: %s, ...", ", ".join([f"{len(c)} {c.index2poly[0].dim}-cells" for c in chain]))
            if new_complex.index2poly[0].dim == 0:
                break
        if verbose:
            logger.info("Chain: %s", ", ".join([f"{len(c)} {c.index2poly[0].dim}-cells" for c in chain]))
        return chain

    def get_meta_graph(self, *, enrich: bool = True, verbose: bool = False, truncate: bool = False) -> nx.MultiDiGraph[Any]:
        """Return a meta-graph encoding cells across all dimensions and face relations.

        This method mirrors the face-encoding convention used by relucent's contracted
        chain complex and topology routines: a codimension-1 face of a k-cell is
        obtained by setting one supporting-hyperplane sign entry (a SHI) to 0.

        Nodes correspond to cells across dimensions k=0..d and are keyed by the
        polyhedron's stable ``tag``. Each node stores:
          - ``poly``: a representative :class:`~relucent.poly.Polyhedron`
          - ``dim``: the cell dimension k
          - ``ss``: the cell's sign-sequence array (numpy)

        Directed edges go from a k-cell to a (k-1)-cell whenever the latter is a
        codimension-1 face of the former under the SHI-zeroing rule. Edges store:
          - ``shi``: the supporting hyperplane index that was zeroed

        If ``enrich=True`` (default), the returned graph is post-processed with a
        second pass that propagates boundedness/SHI information downward, in the
        same spirit as :meth:`contract`. Concretely, for each face node ``f`` with
        cofaces ``c`` (incoming edges) and edge attributes ``shi(c→f)``, we set:

        - ``finite``: ``any(c.finite for c in cofaces)``
        - ``shis``: ``intersection((c.shis - {shi(c→f)}) for c in cofaces)``

        These are stored as node attributes on the meta-graph (they do not mutate
        the underlying :class:`~relucent.poly.Polyhedron` objects).

        If ``truncate=True``, the meta-graph is augmented to encode a combinatorial
        truncation by a large bounding halfspace that meets only unbounded cells.
        Every existing node's ``ss`` gains a trailing ``1`` (strictly inside the
        truncation halfspace). The induced subgraph on nodes with ``finite is False``
        (unbounded cells) is duplicated: each copy has trailing ``0`` on ``ss``,
        dimension decremented by one, and node keys ``("trunc", tag)`` where ``tag``
        is the original polyhedron tag. Face edges among duplicates mirror the
        induced subgraph; each original unbounded node ``n`` gains an edge
        ``n → ("trunc", n)`` with ``shi`` equal to :data:`TRUNCATION_META_SHI` (not a
        network SHI). Duplicates are not created for 0-cells (no ``-1``-dimensional
        placeholder). When ``truncate`` is set, node keys may be ``bytes`` or
        ``("trunc", bytes)`` tuples.

        Note:
            The resulting structure encodes the face relations present in the
            contracted chain returned by :meth:`get_chain_complex`. In particular,
            boundary faces that are not represented in the contracted chain will
            not appear unless they were already present in the chain complexes.

            A breadth-first search over full-dimensional regions can still leave this
            graph short of a closed cellular complex: some codimension-one faces of a
            stored cell may not appear as nodes (their sign pattern is missing from the
            explored complex or from the contracted chain), so incidence data can omit
            entries that true geometry would require. That breaks ``∂² = 0`` for the
            GF(2) boundary maps in :func:`relucent.topology.get_betti_numbers` unless
            ``verify_chain_complex`` is disabled; see that function's documentation.
        """
        if len(self) == 0:
            return nx.MultiDiGraph()

        chain = self.get_chain_complex(verbose=verbose)
        # Dimension -> complex in the chain (there is at most one per dimension).
        by_dim: dict[int, Complex] = {}
        for c in chain:
            if len(c) == 0:
                continue
            by_dim[int(c.index2poly[0].dim)] = c

        meta: nx.MultiDiGraph[Any] = nx.MultiDiGraph()

        # Add all cells as nodes, keyed by stable poly.tag (bytes).
        for k, c_k in sorted(by_dim.items(), reverse=True):
            for p in c_k:
                meta.add_node(
                    p.tag,
                    poly=p,
                    dim=int(k),
                    ss=np.asarray(p.ss_np),
                    finite=getattr(p, "finite", None),
                    shis=sorted(int(s) for s in getattr(p, "shis", [])),
                )

        # Add face edges k -> k-1 using the same SS-zeroing rule used elsewhere.
        for k, c_k in sorted(by_dim.items(), reverse=True):
            if int(k) <= 0:
                continue
            c_km1 = by_dim.get(int(k - 1))
            for p in tqdm(c_k, desc=f"Building meta-graph faces (k={k})", leave=False, disable=not verbose):
                ss = np.asarray(p.ss_np)
                for shi in p.shis:
                    face_ss = ss.copy()
                    face_ss[0, int(shi)] = 0

                    face_poly: Polyhedron | None = None
                    if c_km1 is not None and face_ss in c_km1:
                        face_poly = c_km1[face_ss]
                    elif face_ss in self:
                        face_poly = self[face_ss]

                    if face_poly is None:
                        # ``c_{k-1}`` is built by :meth:`contract` from **dual edges** only, so
                        # a geometric facet of ``p`` (zeroing ``shi``) may not appear as a stored
                        # polyhedron—e.g. missing neighbor off the explored subgraph. The
                        # ``face_ss in self`` branch rarely helps for codim-1 faces of top cells
                        # (their ``ss`` has a new zero vs. full-dimensional regions in ``self``).
                        continue

                    if face_poly.tag not in meta:
                        meta.add_node(
                            face_poly.tag,
                            poly=face_poly,
                            dim=int(face_poly.dim),
                            ss=np.asarray(face_poly.ss_np),
                            finite=getattr(face_poly, "finite", None),
                            shis=sorted(int(s) for s in getattr(face_poly, "shis", [])),
                        )
                    meta.add_edge(p.tag, face_poly.tag, shi=int(shi))

        if enrich:
            # Second pass: propagate boundedness/SHI information down the face poset.
            # Traverse high->low so coface attrs are already available.
            for k in sorted(by_dim.keys(), reverse=True):
                if int(k) <= 0:
                    continue
                face_tags = [n for n, a in meta.nodes(data=True) if int(a.get("dim", -1)) == int(k - 1)]
                for face_tag in face_tags:
                    # Consider only immediate cofaces (dim+1).
                    in_edges = [
                        (u, data)
                        for u, _, data in meta.in_edges(face_tag, data=True)
                        if int(meta.nodes[u].get("dim", -999)) == int(k)
                    ]
                    if not in_edges:
                        continue

                    # Keep cofaces aligned with `in_edges`: dropping Nones from cofaces alone would
                    # pair the wrong SHI data with the wrong poly under zip().
                    in_edges = [(u, data) for u, data in in_edges if meta.nodes[u].get("poly") is not None]
                    if not in_edges:
                        continue

                    cofaces = [meta.nodes[u].get("poly") for u, _ in in_edges]

                    finite_vals = [getattr(p, "finite", None) for p in cofaces]
                    if any(v is not None for v in finite_vals):
                        finite = any(bool(v) for v in finite_vals if v is not None)
                    else:
                        finite = None

                    # contract()-style SHI inference: intersect coface SHIs after removing crossed facet.
                    shis_sets: list[set[int]] = []
                    for (_u, data), coface in zip(in_edges, cofaces, strict=True):
                        shi = data.get("shi")
                        s = set(int(x) for x in getattr(coface, "shis", []))
                        if shi is not None:
                            s.discard(int(shi))
                        shis_sets.append(s)
                    inferred_shis = sorted(set.intersection(*shis_sets)) if shis_sets else []

                    meta.nodes[face_tag]["finite"] = finite
                    meta.nodes[face_tag]["shis"] = inferred_shis

        if truncate:
            # --- combinatorial "big ball" truncation (see docstring) ---
            # Unbounded cells are the ones that would meet the truncating boundary; grab their
            # face poset *before* we tack the extra sign bit onto everyone's `ss`.
            unbounded = {n for n, a in meta.nodes(data=True) if a.get("finite", None) is False}
            ub_faces = meta.subgraph(unbounded).copy()

            def _ss_with_extra_bit(ss: np.ndarray, bit: int) -> np.ndarray:
                a = np.asarray(ss)
                dt = np.int8 if np.issubdtype(a.dtype, np.integer) else a.dtype
                return np.hstack([a, np.full((a.shape[0], 1), bit, dtype=dt)])

            # Mark everything as strictly inside the truncation halfspace (trailing 1).
            for attrs in meta.nodes.values():
                if (ss0 := attrs.get("ss")) is not None:
                    attrs["ss"] = _ss_with_extra_bit(np.asarray(ss0), 1)

            # For each positive-dim unbounded cell: shadow copy on the cut (trailing 0, dim−1),
            # and wire it to the original with a fake SHI so it's not confused with a ReLU facet.
            dup_keys: set[Any] = set()
            for orig in unbounded:
                oa = meta.nodes[orig]
                k = int(oa.get("dim", -1))
                ss_in = oa.get("ss")
                if k <= 0 or ss_in is None:
                    continue
                dup = ("trunc", orig)
                ss_on_cut = np.asarray(ss_in).copy()
                ss_on_cut[..., -1] = 0
                dup_keys.add(dup)
                meta.add_node(
                    dup,
                    poly=oa.get("poly"),
                    dim=k - 1,
                    ss=ss_on_cut,
                    finite=True,
                    shis=list(oa.get("shis", [])),
                    truncation_duplicate=True,
                )
                meta.add_edge(orig, dup, shi=TRUNCATION_META_SHI)

            # Copy face incidences among unbounded cells onto their boundary doubles (both ends
            # need a duplicate; 0-cells deliberately have none).
            for u, v, ed in ub_faces.edges(data=True):
                tu, tv = ("trunc", u), ("trunc", v)
                if tu in dup_keys and tv in dup_keys:
                    meta.add_edge(tu, tv, **dict(ed))

        return meta

    @staticmethod
    def _gf2_rank_packed(packed: np.ndarray, ncols: int) -> int:
        """Gaussian elimination rank over GF(2) on row-major bit-packed rows (uint64 words)."""
        if packed.size == 0 or ncols == 0:
            return 0
        nrows = int(packed.shape[0])
        rank = 0
        for col in range(ncols):
            if rank >= nrows:
                break
            word = col >> 6
            sh = col & 63
            bitm = np.uint64(1) << sh
            colbits = packed[rank:, word] & bitm
            pivot_offs = np.flatnonzero(colbits)
            if pivot_offs.size == 0:
                continue
            pivot = rank + int(pivot_offs[0])
            if pivot != rank:
                packed[[rank, pivot], :] = packed[[pivot, rank], :]
            mask = (packed[:, word] & bitm) != 0
            mask[rank] = False
            inds = np.flatnonzero(mask)
            if inds.size > 0:
                packed[inds, :] ^= packed[rank, :]
            rank += 1
        return rank

    @staticmethod
    def _gf2_rank(matrix: np.ndarray) -> int:
        """Compute matrix rank over GF(2) via Gaussian elimination.

        Uses a bit-packed row representation (~8× less RAM than ``uint8``) and fills it
        in 64-column stripes so callers are not forced to materialize a huge dense matrix.
        """
        if matrix.size == 0:
            return 0
        nrows, ncols = matrix.shape
        nwords = (int(ncols) + 63) // 64
        packed = np.zeros((nrows, nwords), dtype=np.uint64)
        m = np.asarray(matrix)
        for w in range(nwords):
            c0 = w * 64
            c1 = min(c0 + 64, int(ncols))
            acc = np.zeros(nrows, dtype=np.uint64)
            for j, bc in enumerate(range(c0, c1)):
                acc |= (m[:, bc].astype(np.uint8, copy=False) & 1).astype(np.uint64) << j
            packed[:, w] = acc
        return int(Complex._gf2_rank_packed(packed, int(ncols)))

    @staticmethod
    def _simplicial_betti_gf2(
        *,
        simplices_by_dim: dict[int, list[tuple[int, ...]]],
    ) -> dict[int, int]:
        """Compute simplicial Betti numbers over GF(2).

        ``simplices_by_dim[k]`` is a list of k-simplices as tuples of vertex ids.
        Vertex ids are assumed to be 0..N-1 (not necessarily contiguous, but consistent).
        """
        if not simplices_by_dim:
            return {}

        dims = sorted(simplices_by_dim.keys())
        kmin = dims[0]
        kmax = dims[-1]

        # Ensure all lower-dimensional faces are present (closure).
        for k in range(kmax, 0, -1):
            if k not in simplices_by_dim:
                continue
            faces: set[tuple[int, ...]] = set(simplices_by_dim.get(k - 1, []))
            for s in simplices_by_dim[k]:
                for j in range(len(s)):
                    faces.add(tuple(s[:j] + s[j + 1 :]))
            simplices_by_dim[k - 1] = sorted(faces)

        ncells = {k: len(simplices_by_dim.get(k, [])) for k in range(kmin, kmax + 1)}

        boundary_rank: dict[int, int] = {}
        for k in range(kmin, kmax + 1):
            if k == kmin:
                # d_0 = 0
                boundary_rank[k] = 0
                continue
            sk = simplices_by_dim.get(k, [])
            skm1 = simplices_by_dim.get(k - 1, [])
            if not sk or not skm1:
                boundary_rank[k] = 0
                continue
            row_index = {s: i for i, s in enumerate(skm1)}
            nrows = len(skm1)
            ncols = len(sk)
            nwords = (ncols + 63) // 64
            packed = np.zeros((nrows, nwords), dtype=np.uint64)
            for col, s in enumerate(sk):
                w = col >> 6
                bit = np.uint64(1) << (col & 63)
                # Boundary is the sum of codimension-1 faces (mod 2).
                for j in range(len(s)):
                    face = tuple(s[:j] + s[j + 1 :])
                    row = row_index.get(face)
                    if row is None:
                        continue
                    packed[row, w] ^= bit
            boundary_rank[k] = int(Complex._gf2_rank_packed(packed, ncols))

        # beta_k = dim C_k - rank(d_k) - rank(d_{k+1})
        out: dict[int, int] = {}
        for k in range(kmin, kmax + 1):
            out[k] = int(ncells.get(k, 0) - boundary_rank.get(k, 0) - boundary_rank.get(k + 1, 0))
        return out

    def _get_traditional_truncation_bound(self) -> float:
        """Choose an L-infinity bounding box radius from polyhedron interior points.

        Per manuscript convention, we set the truncation box using the interior point
        (one per cell) with the largest max-norm across the complex.
        """
        if len(self) == 0:
            return 1.0
        max_norm = 0.0
        for p in self:
            ip = p.interior_point
            if ip is None:
                continue
            max_norm = max(max_norm, float(np.max(np.abs(np.asarray(ip).reshape(-1)))))
        # Avoid degenerate bound when everything is near the origin.
        return float(cfg.PLOT_MARGIN_FACTOR) * max(max_norm, 1.0)

    def get_betti_numbers(
        self,
        *,
        reduced: bool = False,
        compactify: bool = False,
        respect_finite: bool = False,
        verify_chain_complex: bool = False,
        verbose: bool = False,
    ) -> dict[int, int]:
        """Compute Betti numbers over GF(2).

        This is a thin public wrapper around :func:`relucent.topology.get_betti_numbers`.
        The ``infinity`` argument is deprecated and ignored (see that function's docstring).
        Pass ``verbose=True`` for stderr progress from topology and meta-graph construction.
        """
        self._warn_research_use("get_betti_numbers")
        from relucent.topology import get_betti_numbers as _get_betti_numbers

        return _get_betti_numbers(
            self,
            reduced=reduced,
            compactify=compactify,
            respect_finite=respect_finite,
            verify_chain_complex=verify_chain_complex,
            verbose=verbose,
        )

    def get_persistent_homology(
        self,
        filtration: object,
        *,
        compactify: bool = False,
        respect_finite: bool = False,
        lower_star: bool | None = None,
        verbose: bool = False,
    ) -> object:
        """Compute persistent homology over GF(2) for a :class:`~relucent.filtration.Filtration`.

        See :func:`relucent.persistence.compute_persistent_homology`.
        """
        self._warn_research_use("get_persistent_homology")
        from relucent.filtration import Filtration
        from relucent.persistence import compute_persistent_homology

        if not isinstance(filtration, Filtration):
            raise TypeError(f"filtration must be a Filtration instance, got {type(filtration)!r}")
        return compute_persistent_homology(
            self,
            filtration,
            compactify=compactify,
            respect_finite=respect_finite,
            lower_star=lower_star,
            verbose=verbose,
        )

    def verify_dual_graph_consistency(self) -> None:
        """Verify that dual-graph edges correspond to valid shared faces (by sign sequences).

        - For top_dim==1: each dual-graph edge must correspond to exactly one shared 0-face tag.
        - For top_dim>=2: each dual-graph edge's `shi` must induce a common codim-1 face.
        """
        if len(self) == 0:
            return
        try:
            top_dim = max(int(p.dim) for p in self)
        except ValueError:
            return
        if top_dim <= 0:
            return

        G = self.get_dual_graph(verbose=False)
        if top_dim == 1:
            endtags: dict[bytes, set[bytes]] = {}
            for p in self:
                if int(p.dim) != 1:
                    continue
                tags: set[bytes] = set()
                ss = np.asarray(p.ss_np)
                for shi in np.flatnonzero(ss[0] != 0):
                    face_ss = ss.copy()
                    face_ss[0, int(shi)] = 0
                    face = p.__class__(p._net, face_ss, bound=p.bound)
                    if face.is_face_of(p):
                        tags.add(face.tag)
                endtags[p.tag] = tags

            for u, v in G.edges():
                inter = endtags.get(u.tag, set()) & endtags.get(v.tag, set())
                if len(inter) != 1:
                    raise RuntimeError(
                        "Dual-graph adjacency is inconsistent with 0-face sign-sequence endpoints: "
                        + f"|intersection|={len(inter)} for edge ({u.tag!r}, {v.tag!r})."
                    )
            return

        for u, v, shi in G.edges(data="shi"):
            if shi is None:
                raise RuntimeError("Dual-graph edge is missing 'shi' attribute.")
            face = u.get_face(int(shi))
            if not (face.is_face_of(u) and face.is_face_of(v)):
                raise RuntimeError(
                    "Dual-graph adjacency is inconsistent with SS-implied shared face: "
                    + f"edge shi={int(shi)} did not yield a common face."
                )

    def intrinsic_vertex_coords(
        self,
        *,
        top_dim: int | None = None,
        bound: float,
        tol: float,
        verify_cube: bool = True,
    ) -> dict[bytes, np.ndarray]:
        """Identify intrinsic (non-truncation-box) vertices and optionally verify them with the dual-graph."""
        if len(self) == 0:
            return {}
        if top_dim is None:
            top_dim = max(int(p.dim) for p in self)
        if int(top_dim) < 2:
            return {}

        G = self.get_dual_graph(verbose=False)

        # vertex_tag -> (S, incident top cells)
        vtx: dict[bytes, tuple[tuple[int, ...], list[Polyhedron]]] = {}
        for p in self:
            if int(p.dim) != int(top_dim):
                continue
            shis = tuple(int(s) for s in p.shis)
            if len(shis) < int(top_dim):
                continue
            for shis_subset in combinations(shis, int(top_dim)):
                face = p.get_face_by_shis(shis_subset)
                if int(face.dim) != 0:
                    continue
                if not face.is_face_of(p):
                    continue
                tag = face.tag
                shis_sorted = tuple(sorted(int(x) for x in shis_subset))
                hit = vtx.get(tag)
                if hit is None:
                    vtx[tag] = (shis_sorted, [p])
                else:
                    S0, cells = hit
                    if shis_sorted != S0:
                        raise RuntimeError(
                            "Intrinsic vertex tag produced with inconsistent SHI sets: "
                            + f"tag={tag!r} had {S0} vs {shis_sorted}."
                        )
                    cells.append(p)

        out: dict[bytes, np.ndarray] = {}
        match_box = float(cfg.TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR) * float(tol)

        for tag, (shis_cube, cells) in vtx.items():
            if verify_cube:
                patt2cell: dict[int, Polyhedron] = {}
                for p in cells:
                    ss = np.asarray(p.ss_np)
                    patt = 0
                    for i, shi in enumerate(shis_cube):
                        sgn = int(ss[0, int(shi)])
                        if sgn == 0:
                            patt = -1
                            break
                        if sgn > 0:
                            patt |= 1 << i
                    if patt < 0:
                        continue
                    if patt in patt2cell and patt2cell[patt].tag != p.tag:
                        raise RuntimeError(
                            "Intrinsic vertex incident set contains duplicate sign patterns: "
                            + f"vertex={tag!r} pattern={patt}."
                        )
                    patt2cell[patt] = p

                patt_by_tag = {p.tag: patt for patt, p in patt2cell.items()}
                cell_tags = set(patt_by_tag.keys())
                for u, v, shi in G.edges(data="shi"):
                    if u.tag not in cell_tags or v.tag not in cell_tags:
                        continue
                    if shi is None:
                        raise RuntimeError("Dual-graph edge is missing 'shi' attribute.")
                    if int(shi) not in shis_cube:
                        raise RuntimeError(
                            "Dual-graph edge crosses a facet not in the intrinsic-vertex cube directions: "
                            + f"vertex={tag!r} edge_shi={int(shi)} S={shis_cube}."
                        )
                    pu = patt_by_tag[u.tag]
                    pv = patt_by_tag[v.tag]
                    if bin(pu ^ pv).count("1") != 1:
                        raise RuntimeError(
                            "Dual-graph edge within intrinsic-vertex incident set is not a single-bit flip: "
                            + f"vertex={tag!r} patterns=({pu},{pv}) S={shis_cube}."
                        )

            witness = cells[0]
            face = witness.get_face_by_shis(shis_cube)
            x = getattr(face, "interior_point", None)
            if x is None:
                x = getattr(face, "center", None)
            if x is None:
                continue
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            if float(np.max(np.abs(x))) <= float(bound) + match_box:
                out[tag] = x

        return out

    def truncation_vertex_id_mapper(
        self,
        *,
        top_dim: int,
        bound: float,
        tol: float,
        verify_cube: bool,
    ) -> Callable[[np.ndarray], int]:
        """Return a vertex-id mapper that canonicalizes intrinsic vertices by SS tag."""
        intrinsic_coords = self.intrinsic_vertex_coords(top_dim=top_dim, bound=bound, tol=tol, verify_cube=verify_cube)
        intrinsic_tag2vid: dict[bytes, int] = {}
        key2vid: dict[tuple[int, ...], int] = {}
        vertices: list[np.ndarray] = []

        def vid(v: np.ndarray) -> int:
            vv = np.asarray(v, dtype=np.float64).reshape(-1)
            if intrinsic_coords:
                thr = float(cfg.TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR) * float(tol)
                for t, x in intrinsic_coords.items():
                    if float(np.max(np.abs(vv - x))) <= thr:
                        hit = intrinsic_tag2vid.get(t)
                        if hit is not None:
                            return int(hit)
                        new_id = len(vertices)
                        intrinsic_tag2vid[t] = new_id
                        vertices.append(np.asarray(x, dtype=np.float64).reshape(-1))
                        return new_id
            q = tuple(np.round(vv / tol).astype(np.int64).tolist())
            hit = key2vid.get(q)
            if hit is not None:
                return int(hit)
            new_id = len(vertices)
            key2vid[q] = new_id
            vertices.append(vv)
            return new_id

        return vid

    @property
    def G(self) -> nx.Graph[Polyhedron]:
        """The adjacency graph of top-dimensional cells in the complex."""
        if self._dual_graph is None:
            self._dual_graph = self.get_dual_graph()
        return self._dual_graph

    @overload
    def get_dual_graph(
        self,
        auto_add: bool = False,
        *,
        relabel: Literal[False] = False,
        plot: bool = False,
        node_color: str | None = None,
        node_size: str | None = None,
        cmap: str = "viridis",
        match_locations: bool = False,
        show_node_labels: bool = False,
        show_edge_labels: bool = False,
        verbose: bool = False,
    ) -> nx.Graph[Polyhedron]: ...

    @overload
    def get_dual_graph(
        self,
        auto_add: bool = False,
        *,
        relabel: Literal[True],
        plot: bool = False,
        node_color: str | None = None,
        node_size: str | None = None,
        cmap: str = "viridis",
        match_locations: bool = False,
        show_node_labels: bool = False,
        show_edge_labels: bool = False,
        verbose: bool = False,
    ) -> nx.Graph[int]: ...

    def get_dual_graph(
        self,
        auto_add: bool = False,
        *,
        relabel: bool = False,
        plot: bool = False,
        node_color: str | None = None,
        node_size: str | None = None,
        cmap: str = "viridis",
        match_locations: bool = False,
        show_node_labels: bool = False,
        show_edge_labels: bool = False,
        verbose: bool = True,
    ) -> nx.Graph[Polyhedron] | nx.Graph[int]:
        """Construct the dual graph of the complex.

        The dual graph represents the connectivity structure of the complex,
        where nodes are polyhedra and edges connect adjacent polyhedra (those
        sharing a supporting hyperplane).

        Args:
            auto_add: If True, add polyhedra to the complex if they are not already present.
                Defaults to False.
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
            verbose: If True, print progress messages. Defaults to True.

        Returns:
            networkx.Graph: The dual graph of the complex. Nodes are polyhedra
                (or integers if relabel=True), edges connect adjacent polyhedra
                and have a "shi" attribute indicating which supporting hyperplane
                they cross.

        Raises:
            ValueError: If match_locations is True and the complex is not 2D.
        """
        max_dim = max(poly.dim for poly in self)
        graph = nx.Graph()
        for poly in self:
            if poly.dim == max_dim:
                graph.add_node(poly, label=str(poly))
        missing: list[tuple[np.ndarray, int, Polyhedron]] = []
        for poly in tqdm(graph.nodes(), desc="Creating Dual Graph", leave=False, disable=not verbose):
            ss = poly.ss_np.copy()
            for shi in poly.shis:
                if cfg.CAREFUL_MODE:
                    assert ss[0, shi] != 0
                ss[0, shi] *= -1
                try:
                    graph.add_edge(poly, self[ss], shi=shi)
                except KeyError:
                    missing.append((ss.copy(), shi, poly))
                ss[0, shi] *= -1

        if len(missing) > 0 and max_dim == self.dim:
            warnings.warn(
                f"Dual graph is incomplete. {len(missing)} boundary cells were not added to the complex."
                + "Set auto_add=True to add them.",
                stacklevel=2,
            )
        if auto_add:
            for ss, shi, source_poly in missing:
                neighbor = self.ss2poly(ss, check_exists=False)
                if neighbor.feasible:
                    self.add_polyhedron(neighbor, check_exists=False)
                    graph.add_edge(source_poly, neighbor, shi=shi)
        if plot:
            if match_locations:
                if self.dim != 2:
                    raise ValueError("Polyhedra must be 2D to match locations")

                nx.set_node_attributes(graph, {node: False for node in graph.nodes}, "physics")
                nx.set_node_attributes(
                    graph,
                    {poly: poly.interior_point[0].item() * 10 for poly in graph.nodes},
                    "x",
                )
                nx.set_node_attributes(
                    graph,
                    {poly: poly.interior_point[1].item() * 10 for poly in graph.nodes},
                    "y",
                )

            if node_color == "Wl2":
                colors = get_colors([poly.Wl2 for poly in graph.nodes], cmap=cmap)
                for c, poly in zip(colors, graph.nodes, strict=True):
                    graph.nodes[poly]["color"] = c
            elif node_color == "volume":
                colors = get_colors([poly.ch.volume for poly in graph.nodes], cmap=cmap)
                for c, poly in zip(colors, graph.nodes, strict=True):
                    graph.nodes[poly]["color"] = c

            if node_size == "volume":
                sizes = [poly.ch.volume for poly in graph.nodes]
                maxsize = max(sizes)
                for size, poly in zip(sizes, graph.nodes, strict=True):
                    graph.nodes[poly]["size"] = (10 + 1000 * size / maxsize) ** 1
            else:
                nx.set_node_attributes(graph, {node: 4 for node in graph.nodes}, "size")

            for node in graph.nodes:
                graph.nodes[node]["label"] = str(node) if show_node_labels else ""
                graph.nodes[node]["title"] = str(node)
            for edge in graph.edges:
                graph.edges[edge]["label"] = str(graph.edges[edge]["shi"]) if show_edge_labels else ""
                graph.edges[edge]["title"] = str(graph.edges[edge]["shi"])
        if plot or relabel:
            graph = nx.relabel_nodes(graph, {poly: i for i, poly in enumerate(self)})
        return graph

    def recover_from_dual_graph(
        self,
        graph: nx.Graph[int],
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
            graph = graph.copy()
        initial_p = self.add_ss(initial_ss)
        graph.nodes[source]["poly"] = initial_p
        # ``nx.bfs_edges`` yields only the N-1 tree edges, so the progress bar
        # total must be in terms of nodes, not the total number of dual edges.
        for edge in tqdm(
            nx.bfs_edges(graph, source=source),
            desc="Recovering Polyhedra",
            total=graph.number_of_nodes() - 1,
        ):
            poly1, shi = graph.nodes[edge[0]]["poly"], graph.edges[edge]["shi"]
            poly2_ss = poly1.ss_np.copy()
            if cfg.CAREFUL_MODE:
                assert poly2_ss[0, shi] != 0
            poly2_ss[0, shi] *= -1
            poly2 = self.add_ss(poly2_ss, check_exists=False)
            graph.nodes[edge[1]]["poly"] = poly2

        # Populate each polyhedron's ``_shis`` from the full dual graph in a
        # single pass: iterating ``graph.edges(node)`` per-node would visit each
        # edge twice and also force a redundant ``self[...]`` SSManager lookup.
        shis_per_node: dict[Any, list[int]] = {n: [] for n in graph}
        for u, v, data in graph.edges(data=True):
            shi = data["shi"]
            shis_per_node[u].append(shi)
            shis_per_node[v].append(shi)
        for node, shis in shis_per_node.items():
            graph.nodes[node]["poly"]._shis = shis

    def plot(
        self,
        *,
        plot_mode: Literal["cells", "graph", "1-skeleton"] = "cells",
        label_regions: bool = False,
        color: Any = None,
        highlight_regions: Any = None,
        ss_name: bool = False,
        bound: float | None = None,
        show_axes: bool = False,
        project: float | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Unified plotting entrypoint for all complex visualizations.

        Args:
            plot_mode: Visualization type:
                - ``"cells"``: top-dimensional cells in input space.
                - ``"graph"``: lifted 2D cells in graph/output space.
                - ``"1-skeleton"``: 1-cells from ``get_chain_complex()``.
            label_regions: If True, annotate region centers with ``str(poly)`` when
                supported by the selected mode.
            color: Coloring strategy or explicit color accepted by plotting backends.
            highlight_regions: Iterable of region identifiers (poly objects or names)
                to highlight in red where supported.
            ss_name: 2D ``"cells"`` mode only. Use sign-sequence labels for traces.
            bound: Plot bound in input coordinates.
            show_axes: If True, show axis lines/ticks.
            project: ``"graph"`` mode only; optional z-value for projected copies.
            **kwargs: Additional mode-specific Plotly kwargs forwarded to
                :func:`relucent.vis.plot_complex`.

        Returns:
            Plotly figure for the selected visualization.
        """
        if bound is None:
            bound = cfg.DEFAULT_COMPLEX_PLOT_BOUND
        plot_kwargs: dict[str, Any] = dict(
            label_regions=label_regions,
            color=color,
            highlight_regions=highlight_regions,
            bound=bound,
            show_axes=show_axes,
            **kwargs,
        )
        if plot_mode == "cells":
            plot_kwargs["ss_name"] = ss_name
            plot_kwargs["fill_mode"] = "filled"
        elif plot_mode == "graph":
            plot_kwargs["project"] = project

        return plot_complex(
            self,
            plot_mode=plot_mode,
            **plot_kwargs,
        )
