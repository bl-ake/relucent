import os
import pickle
import random
import warnings
from collections.abc import Generator, Iterable, Iterator
from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from gurobipy import Env
from tqdm.auto import tqdm

import relucent.config as cfg
from relucent.convert_model import convert
from relucent.model import NN
from relucent.poly import Polyhedron
from relucent.search import (
    greedy_path as _greedy_path_fn,
)
from relucent.search import (
    hamming_astar as _hamming_astar_fn,
)
from relucent.search import (
    parallel_add as _parallel_add_fn,
)
from relucent.search import (
    searcher as _searcher_fn,
)
from relucent.ss import SSManager
from relucent.utils import (
    BlockingQueue,
    get_env,
)
from relucent.vis import get_colors, plot_complex

__all__ = ["Complex"]

# Worker process state — set by set_globals() when used as a pool initializer.
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
        >>> import multiprocessing as mp
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


class Complex:
    """Manages the polyhedral complex of a neural network.

    This class provides methods for calculating, storing, and searching the h-representations
    (halfspace representations) of polyhedra in the complex.
    """

    def __init__(self, net: nn.Module) -> None:
        """Initialize the complex for a given network.

        Args:
            net: The nn.Module instance whose polyhedral complex
                is to be built and queried.
        """
        if not isinstance(net, NN):
            net = convert(net)
        self._net = net
        self.net.save_numpy_weights()

        self.ssm = SSManager()
        self.index2poly: list[Polyhedron] = []

        net_layers = list(net.layers.values())
        self.ss_layers = [
            i
            for i, (layer, next_layer) in enumerate(zip(net_layers[:-1], net_layers[1:], strict=True))
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

    # def __del__(self) -> None:
    #     close_env()

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

    def point2poly(self, point, check_exists=True, **kwargs: Any):
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

    def ss2poly(self, ss, check_exists=True, **kwargs: Any):
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
            return Polyhedron(self.net, ss, **kwargs)

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
        if bound is None:
            bound = cfg.DEFAULT_PARALLEL_ADD_BOUND
        return _parallel_add_fn(
            self,
            points,
            nworkers=nworkers,
            bound=bound,
            **kwargs,
        )

    def searcher(
        self,
        start=None,
        max_depth=float("inf"),
        max_polys=float("inf"),
        queue=None,
        bound=None,
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
            get_volumes=get_volumes,
            verbose=verbose,
            cube_radius=cube_radius,
            cube_mode=cube_mode,
            **kwargs,
        )

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
        return {attr: [getattr(poly, attr) for poly in self] for attr in attrs}

    def get_boundary_edges(self, i):
        """Get the boundary of neuron i by returning the set of edges in the dual graph with label i."""
        assert 0 <= i < self.n, "Neuron index out of range"
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
        """The adjacency graph of top-dimensional cells in the complex."""
        return self.get_dual_graph()

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
                for c, poly in zip(colors, G.nodes, strict=True):
                    G.nodes[poly]["color"] = c
            elif node_color == "volume":
                colors = get_colors([poly.ch.volume for poly in G.nodes], cmap=cmap)
                for c, poly in zip(colors, G.nodes, strict=True):
                    G.nodes[poly]["color"] = c

            if node_size == "volume":
                sizes = [poly.ch.volume for poly in G.nodes]
                maxsize = max(sizes)
                for size, poly in zip(sizes, G.nodes, strict=True):
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

    def plot_cells(
        self,
        label_regions=False,
        color=None,
        highlight_regions=None,
        ss_name=False,
        bound=None,
        show_axes: bool = False,
        fill_mode: str = "wireframe",
        **kwargs: Any,
    ) -> go.Figure:
        """Plot all cells in input space; 2D vs 3D is chosen from :attr:`dim`."""
        if bound is None:
            bound = cfg.DEFAULT_COMPLEX_PLOT_BOUND
        return plot_complex(
            self,
            plot_mode="cells",
            label_regions=label_regions,
            color=color,
            highlight_regions=highlight_regions,
            ss_name=ss_name,
            bound=bound,
            show_axes=show_axes,
            fill_mode=fill_mode,
            **kwargs,
        )

    def plot_graph(
        self,
        label_regions=False,
        color=None,
        highlight_regions=None,
        show_axes=False,
        project=True,
        **kwargs,
    ):
        return plot_complex(
            self,
            plot_mode="graph",
            label_regions=label_regions,
            color=color,
            highlight_regions=highlight_regions,
            show_axes=show_axes,
            project=project,
            **kwargs,
        )
