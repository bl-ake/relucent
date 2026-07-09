from __future__ import annotations

import os
import pickle
import random
import warnings
from collections.abc import Callable, Generator, Iterable, Iterator
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal, Self, overload

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from tqdm.auto import tqdm

import relucent.config as cfg
from relucent import meta_graph as mg
from relucent._logging import logger
from relucent._torch_compat import TORCH_AVAILABLE, torch
from relucent.complex_graph import (
    contract_dual_graph_for_shi,
    delete_ss_columns,
    net_without_last_ss_layer_neuron,
)
from relucent.convert_model import convert
from relucent.meta_graph import (
    INFINITY_POINT_META_NODE,
    INFINITY_POINT_META_SHI,
    TRUNCATION_META_SHI,
    NonGenericArrangementError,
)
from relucent.model import LinearLayer, ReLULayer, ReLUNetwork
from relucent.poly import Polyhedron
from relucent.search import (
    ALL_GEOMETRY_PROPERTIES,
    retain_geometry_caches,
)
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
    parallel_compute_geometric_properties as _parallel_compute_geometric_properties_fn,
)
from relucent.search import (
    searcher as _searcher_fn,
)
from relucent.ss import SSManager
from relucent.utils import (
    BlockingQueue,
    encode_ss,
    flip_ss_at_shi,
    process_aware_cpu_count,
)
from relucent.verify import (
    ComplexNotCompleteError,
    ComplexNotVerifiedError,
    DualGraphAsymmetricEdgeError,
)

if TYPE_CHECKING:
    from relucent.morse import CriticalPoint

__all__ = [
    "Complex",
    "INFINITY_POINT_META_NODE",
    "INFINITY_POINT_META_SHI",
    "IncompleteDualGraphError",
    "NonGenericArrangementError",
    "TRUNCATION_META_SHI",
    "ComplexNotCompleteError",
    "ComplexNotVerifiedError",
    "DualGraphAsymmetricEdgeError",
]


class IncompleteDualGraphError(ValueError):
    """The dual graph has missing boundary neighbors (partially explored complex).

    :meth:`Complex.contract`, :meth:`Complex.get_chain_complex`, and
    :meth:`Complex.get_meta_graph` require a complete adjacency structure among
    top-dimensional cells. Explore the complex further (e.g. BFS/DFS) before building
    the chain complex or running topology routines.
    """


RESEARCH_WARNING_DISABLE_ENV_VAR = "DISABLE_RESEARCH_WARNING"

# ``compactify=False``: combinatorial truncation (``truncate_meta_graph``).
# ``compactify=True``: Borel–Moore (``require_shared_faces``).
# ``compactify="one_point"``: one-point compactification.
# TODO: deprecate ``compactify=False`` and rename ``compactify=True`` to ``compactify="bm"``.
CompactifyMode = bool | Literal["one_point"]


class Complex:
    """Manages the polyhedral complex of a neural network.

    This class provides methods for calculating, storing, and searching the h-representations
    (halfspace representations) of polyhedra in the complex.
    """

    def __init__(self, net: Any, *, auto_tolerances: bool = True) -> None:
        """Initialize the complex for a given network.

        Args:
            net: Any model convertible to relucent's canonical ``NN``.
                is to be built and queried.
            auto_tolerances: When True (default), set :mod:`relucent.config`
                tolerance values from this network's weight scale via
                :func:`~relucent.numeric_tolerances.apply_tolerances`. This is
                the usual runtime source for values such as
                ``cfg.MIN_SEARCH_INRADIUS`` after import bootstrap.
        """
        original_net = net
        if not isinstance(net, ReLUNetwork):
            net = convert(net)
        self.net = original_net
        self._net = net

        if auto_tolerances:
            from relucent.numeric_tolerances import apply_tolerances

            apply_tolerances(net=self._net)

        self.ssm = SSManager()
        self.index2poly: list[Polyhedron] = []
        self.tag2poly: dict[bytes, Polyhedron] = {}

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
        self._betti_cache: dict[tuple[bool, CompactifyMode, bool], dict[int, int]] = {}
        self._complete: bool | None = None
        self._verified: bool | None = None

    def _invalidate_derived_caches(self) -> None:
        self._dual_graph = None
        self._betti_cache.clear()
        self._complete = None
        self._verified = None

    @property
    def complete(self) -> bool | None:
        """Whether exploration finished without an intentional cap (``None`` if unknown)."""
        return self._complete

    @property
    def verified(self) -> bool | None:
        """Whether the complex passed the last invariant verification (``None`` if unknown)."""
        return self._verified

    def set_exploration_state(self, *, complete: bool, verified: bool) -> None:
        """Record exploration / verification status after search or verify."""
        self._complete = complete
        self._verified = verified

    def assert_topology_ready(self) -> None:
        """Require a complete, verified complex before topology routines."""
        if self._complete is True and self._verified is True:
            return
        if self._complete is False:
            from relucent.verify import ComplexNotCompleteError

            raise ComplexNotCompleteError(
                "Complex is not complete or failed invariant verification. Explore further "
                + "(e.g. BFS) or pass an explicit exploration cap (max_polys) to opt into "
                + "a partial complex."
            )
        if self._complete is True and self._verified is not True:
            from relucent.verify import ComplexNotVerifiedError

            raise ComplexNotVerifiedError(
                "Complex is complete but not verified. Re-run BFS with verify=True or call " + "verify_complex."
            )
        from relucent.verify import ComplexNotCompleteError

        raise ComplexNotCompleteError(
            "Complex exploration state is unknown. Run BFS or explore_for_topology first, "
            + "or set_exploration_state(complete=True, verified=True) for trusted loads."
        )

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
            tag = key.tag
        elif isinstance(key, (np.ndarray, torch.Tensor)):
            tag = encode_ss(key)
        else:
            raise KeyError("Complex can only be indexed by Polyhedra, arrays, or tensors")
        try:
            return self.tag2poly[tag]
        except KeyError:
            raise KeyError(key) from None

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
        if isinstance(key, Polyhedron):
            return key.tag in self.tag2poly
        elif isinstance(key, (np.ndarray, torch.Tensor)):
            return encode_ss(key) in self.tag2poly
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

    @classmethod
    def load(cls, filename: str | os.PathLike[str]) -> Self:
        """Load a Complex from a pickle file.

        Intended to be called as Complex.load(filename). The file must have been
        created by save().

        Args:
            filename: Path to the pickle file.

        Returns:
            The restored complex.
        """
        with open(filename, "rb") as f:
            state = pickle.load(f)
        cplx = cls(state["net"])
        cplx.__setstate__(state)
        return cplx

    def __getstate__(self) -> dict[str, Any]:
        return {
            "index2poly": self.index2poly,
            "net": self.net,
            "_net": self._net,
            "_betti_cache": self._betti_cache,
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
            self.tag2poly[p.tag] = p
        self._betti_cache = state.get("_betti_cache", {})

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
        new_net = net_without_last_ss_layer_neuron(self._net, last_ss_layer, neuron_idx)
        out = Complex(new_net)
        out.net = new_net if isinstance(self.net, ReLUNetwork) else self.net

        dual = self.get_dual_graph(relabel=True, verbose=False)
        if dual.number_of_nodes() == 0:
            return out

        contracted, old_rep = contract_dual_graph_for_shi(dual, deleted_shi)
        for component in nx.connected_components(contracted):
            sub = contracted.subgraph(component).copy()
            source = min(component)
            initial_ss = delete_ss_columns(self.index2poly[old_rep[source]].ss_np, [deleted_shi])
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
            self.tag2poly[p.tag] = p
            self._invalidate_derived_caches()
            return p

        tag = p.tag
        p_exists = tag in self.tag2poly

        if p_exists and overwrite:
            self.index2poly[self.ssm.tag2index[tag]] = p
            self.tag2poly[tag] = p
            self._invalidate_derived_caches()
            return p
        elif p_exists:
            return self.tag2poly[tag]
        else:
            self.index2poly.append(p)
            self.ssm.add(p.ss_np)
            self.tag2poly[tag] = p
            self._invalidate_derived_caches()
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
        """Drop heavy geometry caches from all polyhedra in the complex.

        Retains lightweight search data (sign sequence, SHIs, Chebyshev
        classification, and interior points) on each polyhedron.
        """
        for poly in self:
            retain_geometry_caches(poly, ())

    def parallel_add(
        self,
        points: Iterable[torch.Tensor | np.ndarray],
        nworkers: int | None = None,
        bound: float | None = None,
        geometry_properties: Iterable[str] = ALL_GEOMETRY_PROPERTIES,
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
            geometry_properties: Iterable of cache/property names to compute and
                retain on each polyhedron. Defaults to
                :data:`~relucent.search.ALL_GEOMETRY_PROPERTIES`.
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
            geometry_properties=geometry_properties,
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
        verify: bool = True,
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
                compute and retain for each discovered polyhedron during search.
                ``None`` (default) performs topology-only search. Pass
                :data:`~relucent.search.ALL_GEOMETRY_PROPERTIES` or a subset for
                optional caches. ``finite``, ``center``, and ``inradius`` are always
                computed.
            verify: When True (default), require complete exploration and run invariant
                checks at the end. Verification is skipped when exploration hits
                ``max_polys`` before the frontier is exhausted. A finite ``max_depth``
                cap can leave ``complete=False``; with ``verify=True`` that raises
                :class:`~relucent.complex.IncompleteDualGraphError` unless the cap
                was hit. Sets ``strict=True`` on SHI LPs when verifying.
            **kwargs: Additional arguments passed to :func:`~relucent.poly.get_shis`.

        Returns:
            dict: Search information dictionary containing:
                - "Search Depth": Maximum depth reached
                - "Avg # Facets Uncorrected": Average number of facets per polyhedron
                - "Search Time": Elapsed time in seconds
                - "Bad SHI Computations": List of failed computations
                - "Complete": Whether search completed (no unprocessed items)
                - "Verified": Whether invariant checks passed (``None`` if not run)

        Raises:
            ValueError: If the start point lies on a hyperplane (has zero in SS).
            IncompleteDualGraphError: If ``verify`` is True and exploration stops early
                for reasons other than hitting ``max_polys``.
        """
        if bound is None:
            from relucent._network_scale import default_polyhedron_bound

            bound = default_polyhedron_bound(self._net)
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
            verify=verify,
            **kwargs,
        )

    def compute_geometric_properties(
        self,
        nworkers: int | None = None,
        properties: Iterable[str] = ALL_GEOMETRY_PROPERTIES,
        verbose: int | None = None,
    ) -> dict[str, Any]:
        """Compute selected polyhedron caches in parallel.

        This is intended to run after a topology-only search pass.

        Args:
            nworkers: Number of worker processes (defaults to CPU count).
            properties: Iterable of cache/property names to compute and retain.
                Defaults to :data:`~relucent.search.ALL_GEOMETRY_PROPERTIES`.
            verbose: Controls progress output. ``0`` silences all output; ``1``
                (default) shows worker count and a progress bar.  When ``None``,
                falls back to :data:`relucent.config.VERBOSE`.
        """
        return _parallel_compute_geometric_properties_fn(
            self,
            nworkers=nworkers,
            geometry_properties=properties,
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
            nworkers: Number of worker processes for parallel neighbor evaluation.
                ``None`` (default) uses ``min(CPU count, number of ReLU units)``.
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

    def slice_affine(
        self,
        x0: np.ndarray,
        V: np.ndarray,
        tol: float | None = None,
    ) -> Complex:
        """Return the non-empty intersections of each cell with an affine subspace.

        The affine subspace is given in the parametric form ``{x0 + V @ t : t in R^k}``,
        where ``x0`` is a base point in input space and the columns of ``V`` span the
        subspace direction. ``k = V.shape[1]`` is the intrinsic dimension of the subspace.

        For a cell with halfspace representation ``Ax + b <= 0``, the intersection in
        parameter space is ``{t : (A @ V) t + (A @ x0 + b) <= 0}``, a polyhedron in ``R^k``.

        Feasibility is tested via a Chebyshev-center LP using ``scipy.optimize.linprog``.
        An intersection is included when the optimal Chebyshev radius ``r >= -tol``.
        An unbounded LP (LP status 3) means ``r -> +inf``, which implies the subspace
        lies entirely inside that cell — always included.

        The returned :class:`Complex` is backed by a stub :class:`~relucent.model.ReLUNetwork`
        with ``input_shape=(k,)`` and no ReLU layers, so ``cpx.dim == k`` and
        ``cpx.plot(plot_mode="cells")`` works for ``k`` in ``{2, 3}``.  Each
        :class:`~relucent.poly.Polyhedron` in the result carries ``halfspaces = H_slice``
        (shape ``(m, k+1)``) and inherits the sign sequence of its parent cell, which
        keeps tags unique across cells and carries correct codimension information.

        Note:
            This method triggers halfspace computation for any cell that has not yet been
            computed. Pre-populate with :meth:`compute_geometric_properties` to avoid
            on-demand Gurobi calls.

        Args:
            x0: Base point of the affine subspace, shape ``(d,)``.
            V: Direction matrix, shape ``(d, k)``. Columns need not be orthonormal.
                Pass a 1-D array of shape ``(d,)`` for a line (``k=1``).
            tol: Feasibility tolerance. Defaults to ``cfg.TOL_HALFSPACE_CONTAINMENT``.

        Returns:
            A new :class:`Complex` in ``k``-dimensional parameter space, containing
            one :class:`~relucent.poly.Polyhedron` per non-empty intersection.
            The complex can be plotted directly with :meth:`plot` for ``k`` in ``{2, 3}``.
        """
        ## TODO: Switch to Gurobi / existing Polyhedron methods
        from scipy.optimize import linprog

        from relucent.model import LinearLayer, ReLUNetwork

        if tol is None:
            tol = float(cfg.TOL_HALFSPACE_CONTAINMENT)

        x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
        # Ensure V is 2-D: a 1-D vector becomes a (d, 1) column
        V_arr = np.asarray(V, dtype=np.float64).reshape(len(x0_arr), -1)
        k = V_arr.shape[1]

        # Stub network: gives out.dim = k with no ReLU layers (no halfspace LPs needed).
        stub_net = ReLUNetwork(
            {"linear": LinearLayer(np.eye(k, dtype=np.float64), np.zeros(k, dtype=np.float64))},
            input_shape=(k,),
        )
        out = Complex(stub_net)

        def _slice_poly_kwargs(parent: Polyhedron, halfspaces: np.ndarray) -> dict[str, Any]:
            kwargs: dict[str, Any] = {"halfspaces": halfspaces, "_ambient_dim": k}
            if parent._shis is not None:
                kwargs["shis"] = list(parent._shis)
            for attr in ("_codim", "_dim", "_finite"):
                val = getattr(parent, attr, None)
                if val is not None:
                    kwargs[attr.lstrip("_")] = val
            return kwargs

        for poly in self:
            H = poly.halfspaces_np  # (m, d+1)
            A = H[:, :-1]  # (m, d)
            b_col = H[:, -1]  # (m,)

            # Substitute x = x0 + V t into each constraint a_i^T x + b_i <= 0
            A_v = A @ V_arr  # (m, k)
            b_v = A @ x0_arr + b_col  # (m,)

            if k == 0:
                # Subspace is a single point; just check containment of x0
                if bool((b_v <= tol).all()):
                    out.add_polyhedron(
                        Polyhedron(
                            stub_net,
                            poly.ss_np,
                            **_slice_poly_kwargs(poly, b_v.reshape(-1, 1)),
                        ),
                        check_exists=False,
                    )
                continue

            H_slice = np.column_stack([A_v, b_v])  # (m, k+1)

            # Chebyshev-center LP: max r s.t. a_vi^T t + r ||a_vi|| + b_vi <= 0
            # Non-empty iff optimal r >= -tol; status 3 (unbounded) means r -> +inf
            norms = np.linalg.norm(A_v, axis=1)  # row norms for Chebyshev radius
            res = linprog(
                np.r_[np.zeros(k), -1.0],  # objective: maximize r
                A_ub=np.column_stack([A_v, norms]),
                b_ub=-b_v,
                bounds=[(None, None)] * (k + 1),
                method="highs",
            )
            if res.status == 3 or (res.status == 0 and float(res.x[-1]) >= -tol):
                out.add_polyhedron(
                    Polyhedron(stub_net, poly.ss_np, **_slice_poly_kwargs(poly, H_slice)),
                    check_exists=False,
                )
        if len(out) > 0:
            from relucent import meta_graph as mg

            mg.set_contracted_shis(out)
            if cfg.CAREFUL_MODE:
                mg.verify_contracted_shis(out)
        return out

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
        edges = list(self.get_boundary_edges(i, verbose=verbose))
        return self.G.subgraph([p for edge in edges for p in edge])

    def _codim_one_face_kwargs(self, p1: Polyhedron, p2: Polyhedron, shi: int) -> dict[str, Any]:
        """Shared kwargs for :meth:`contract` and :meth:`get_boundary_cells` faces.

        Candidate SHIs are all nonzero sign-sequence crossings on the face (role 2);
        :func:`mg.set_contracted_shis` keeps flip neighbors in the slice.
        Infeasible 1-cell faces are dropped at construction time via
        :meth:`~relucent.poly.Polyhedron.is_shi_face_feasible`.

        Meta-graph **face edges** always use :func:`mg.ss_nonzero_indices`; see
        ``negative-betti-meta-graph-bug.md``.
        """
        ambient = p1.ambient_dim
        codim = p1.codim + 1
        face_dim = ambient - codim
        shi_i = int(shi)
        face_ss = p1.ss_np.copy()
        face_ss[0, shi_i] = 0
        candidate_shis = list(mg.ss_nonzero_indices(face_ss))
        if face_dim == 1 and p1.halfspaces is not None:
            new_ss = p1.ss_np.copy()
            new_ss[0, shi_i] = 0
            probe = Polyhedron(
                None,
                new_ss,
                halfspaces=p1.halfspaces,
                codim=codim,
                dim=face_dim,
                _ambient_dim=ambient,
            )
            candidate_shis = [s for s in candidate_shis if probe.is_shi_face_feasible(int(s))]
        poly_kwargs: dict[str, Any] = {
            "halfspaces": p1.halfspaces,
            "shis": candidate_shis,
            "codim": codim,
            "dim": face_dim,
            "_ambient_dim": ambient,
        }
        return poly_kwargs

    def get_boundary_cells(self, i: int, verbose: bool = False, *, verify: bool = True) -> set[Polyhedron]:
        """Get all (d-1)-cells in neuron i's BH."""
        from relucent.boundary_search import _both_ambient_cofaces_feasible

        faces = set()
        edges = list(self.get_boundary_edges(i, verbose=verbose))
        for edge in tqdm(edges, desc="Getting Boundary Cells", delay=1, disable=not verbose):
            p1, p2 = edge[0], edge[1]
            shi = int(self.G.edges[edge]["shi"])
            if verify and (shi not in p1.shis or shi not in p2.shis):
                raise DualGraphAsymmetricEdgeError(
                    f"Boundary edge shi={shi} on ({p1!r}, {p2!r}) lacks bidirectional SHI support."
                )
            new_ss = p1.ss_np.copy()
            new_ss[0, shi] = 0
            p = self.ss2poly(
                new_ss,
                check_exists=False,
                **self._codim_one_face_kwargs(p1, p2, shi),
            )
            if verify and not _both_ambient_cofaces_feasible(p, i):
                raise ValueError(f"Boundary face {p!r} fails ambient coface feasibility for neuron {i}.")
            faces.add(p)
        return faces

    def get_boundary_complex(self, i: int, verbose: bool = False) -> Complex:
        """Get the boundary complex of neuron i.

        Raises:
            IncompleteDualGraphError: If top-dimensional adjacency is incomplete.
            ComplexNotCompleteError: If the input complex is not complete.
            ComplexNotVerifiedError: If the input complex is not verified.
        """
        self.assert_topology_ready()
        self._dual_graph = self.get_dual_graph(verbose=verbose, require_complete=True, verify=True, cubical=False)
        cplx = Complex(self.net)
        for poly in tqdm(
            self.get_boundary_cells(i, verbose=verbose, verify=True),
            desc="Getting Boundary Complex",
            delay=1,
            disable=not verbose,
        ):
            cplx.add_polyhedron(poly, check_exists=False)
        mg.set_contracted_shis(cplx)
        if cfg.CAREFUL_MODE:
            mg.verify_contracted_shis(cplx)
        cplx.verify_arrangement_genericity()
        from relucent.verify import verify_complex

        if len(cplx) == 0:
            cplx.set_exploration_state(complete=True, verified=True)
            return cplx
        cplx.set_exploration_state(complete=True, verified=False)
        verify_complex(cplx, level="fast", record_state=True)
        return cplx

    @overload
    def discover_boundary_complex(
        self,
        i: int,
        verbose: bool = False,
        *,
        return_stats: Literal[False] = False,
        **kwargs: Any,
    ) -> Complex: ...

    @overload
    def discover_boundary_complex(
        self,
        i: int,
        verbose: bool = False,
        *,
        return_stats: Literal[True],
        **kwargs: Any,
    ) -> tuple[Complex, Any]: ...

    def discover_boundary_complex(
        self,
        i: int,
        verbose: bool = False,
        *,
        return_stats: bool = False,
        **kwargs: Any,
    ) -> Complex | tuple[Complex, Any]:
        """Discover neuron ``i``'s boundary complex without building the full input complex.

        Uses MIP pricing to find new connected components on the slice ``ss[i]=0``,
        then slice-restricted BFS to complete each component.

        Args:
            i: Global supporting-hyperplane index (bent hyperplane).
            verbose: If True, show search progress.
            return_stats: If True, return ``(complex, stats)`` with timing metadata.
            **kwargs: Forwarded to :func:`~relucent.boundary_search.discover_boundary_complex`.

        Returns:
            A new :class:`Complex` of boundary cells, or ``(complex, stats)`` when
            ``return_stats`` is True.
        """
        from relucent.boundary_search import discover_boundary_complex as _discover

        boundary, stats = _discover(
            self._net,
            i,
            verbose=verbose,
            **kwargs,
        )
        if return_stats:
            return boundary, stats
        return boundary

    def contract(self, verbose: bool = False) -> Complex:
        """Contract the maximal cells in the complex.

        Each codimension-one face is keyed by zeroing one dual-graph SHI in the
        sign sequence.  Face ``shis`` kwargs come from
        :meth:`_codim_one_face_kwargs`, then :func:`mg.set_contracted_shis`.

        Raises:
            IncompleteDualGraphError: If top-dimensional adjacency is incomplete.
            ComplexNotCompleteError: If this complex is not complete.
            ComplexNotVerifiedError: If this complex is not verified.
        """
        self.assert_topology_ready()
        G = self.get_dual_graph(verbose=verbose, require_complete=True, verify=True, cubical=False)
        new_complex = Complex(self.net)
        for p1, p2, shi in G.edges(data="shi"):
            new_ss = p1.ss_np.copy()
            new_ss[0, shi] = 0
            new_complex.add_ss(
                new_ss,
                **self._codim_one_face_kwargs(p1, p2, int(shi)),
            )

        mg.set_contracted_shis(new_complex)
        if len(new_complex) > 0:
            from relucent.verify import verify_complex

            new_complex.set_exploration_state(complete=True, verified=False)
            verify_complex(new_complex, level="fast", record_state=True)
        return new_complex

    def get_chain_complex(self, verbose: bool = False) -> list[Complex]:
        """Get the chain complex of the complex.

        Each contraction step produces one lower-dimensional slice of the complex.
        The loop terminates when the contracted cells are 0-dimensional (vertices),
        which have no further faces to contract.

        Phantom vertices (geometrically infeasible 0-cells that would otherwise arise
        from combinatorial contraction) are prevented upstream:
        :meth:`~relucent.poly.Polyhedron.is_shi_face_feasible` at 1-cell construction
        and :func:`~relucent.meta_graph.set_contracted_shis` on each slice.

        Raises:
            IncompleteDualGraphError: If dual adjacency is incomplete at some dimension;
                see :meth:`contract`.
        """
        chain: list[Complex] = [self]
        while True:
            cur = chain[-1]
            if len(cur) == 0:
                break
            new_complex = cur.contract(verbose=verbose)
            if len(new_complex) == 0:
                break
            chain.append(new_complex)
            if verbose:
                logger.info("Chain: %s, ...", ", ".join([f"{len(c)} {c.index2poly[0].dim}-cells" for c in chain]))
            if new_complex.index2poly[0].dim == 0:
                # 0-cells are the bottom of the chain; contraction terminates here.
                break
        if verbose:
            logger.info("Chain: %s", ", ".join([f"{len(c)} {c.index2poly[0].dim}-cells" for c in chain]))
        return chain

    def partial_derivative_on_1cell(
        self,
        one_cell: Polyhedron,
        from_vertex: Polyhedron,
        *,
        value: bool = False,
    ) -> int | float:
        """Partial derivative of the network along a 1-cell from a vertex endpoint.

        See :func:`relucent.morse.partial_derivative_on_1cell`.
        """
        from relucent.morse import partial_derivative_on_1cell as _pd_1cell

        return _pd_1cell(one_cell, from_vertex, self, value=value)

    def get_critical_points(
        self,
        *,
        require_complete: bool = False,
        include_degenerate: bool = False,
        verbose: bool = False,
    ) -> list[CriticalPoint]:
        """Return PL Morse critical vertices and their indices in the discovered complex.

        Uses combinatorial edge data (Brooks & Masden, arXiv:2412.18005) and requires a
        scalar-output network.

        Args:
            require_complete: If True, require every combinatorial 1-cell incident to
                each tested vertex to appear in the complex.
            include_degenerate: If True, include flat / degenerate critical vertices
                (index ``-1``).
            verbose: Log chain-complex progress.

        Returns:
            List of :class:`~relucent.morse.CriticalPoint` records.
        """
        from relucent.morse import CriticalPoint, assert_scalar_output, is_pl_critical_vertex

        assert_scalar_output(self._net)
        chain = self.get_chain_complex(verbose=verbose)
        if not chain or chain[-1].index2poly[0].dim != 0:
            return []

        vertex_complex = chain[-1]
        one_cell_tags: set[bytes] | None = None
        if require_complete:
            meta = self.get_meta_graph(verbose=verbose)
            one_cell_tags = {tag for tag, attrs in meta.nodes(data=True) if int(attrs.get("dim", -1)) == 1}

        results: list[CriticalPoint] = []
        for vertex in vertex_complex.index2poly:
            if require_complete and one_cell_tags is not None:
                # Incident edges are inferred combinatorially; this checks they were discovered.
                from relucent.utils import encode_ss

                v_ss = vertex.ss_np.ravel()
                for shi in np.flatnonzero(v_ss == 0):
                    for sign in (-1, 1):
                        edge_ss = v_ss.copy()
                        edge_ss[int(shi)] = sign
                        tag = encode_ss(edge_ss.reshape(1, -1))
                        if tag not in one_cell_tags:
                            raise ValueError(
                                f"combinatorial 1-cell {tag!r} incident to vertex {vertex.tag!r} "
                                + "is missing from the discovered complex"
                            )

            is_critical, index = is_pl_critical_vertex(
                vertex.ss_np,
                self._net,
                ssi2maski=self.ssi2maski,
                ss_layers=self.ss_layers,
            )
            if not is_critical:
                continue
            if index is None or (index < 0 and not include_degenerate):
                continue

            point: np.ndarray | None
            try:
                # Interior point is optional; criticality is combinatorial.
                point = np.asarray(vertex.interior_point, dtype=np.float64).reshape(-1)
            except (ValueError, TypeError):
                point = None

            results.append(
                CriticalPoint(
                    polyhedron=vertex,
                    tag=vertex.tag,
                    ss=vertex.ss_np.copy(),
                    point=point,
                    index=int(index),
                )
            )
        return results

    @staticmethod
    def finite_cells_subgraph(meta: nx.MultiDiGraph[Any]) -> nx.MultiDiGraph[Any]:
        """Return the subcomplex induced by nodes with ``finite is True``."""
        return mg.finite_cells_subgraph(meta)

    @staticmethod
    def truncate_meta_graph(meta: nx.MultiDiGraph[Any]) -> None:
        """Augment ``meta`` in place with combinatorial truncation at infinity."""
        mg.truncate_meta_graph(meta)

    @staticmethod
    def one_point_compactify_meta_graph(meta: nx.MultiDiGraph[Any]) -> bool:
        """Augment ``meta`` in place with a single point-at-infinity 0-cell."""
        return mg.one_point_compactify_meta_graph(meta)

    def get_meta_graph(self, *, verify: bool = False, verbose: bool = False) -> nx.MultiDiGraph[Any]:
        """Return a meta-graph encoding cells across all dimensions and face relations.

        This method mirrors the face-encoding convention used by relucent's contracted
        chain complex and topology routines: a codimension-1 face of a k-cell is
        obtained by setting one supporting-hyperplane sign entry (a SHI) to 0.

        Nodes correspond to cells across dimensions k=0..d and are keyed by the
        polyhedron's stable ``tag``. Each node stores ``poly`` (a representative
        :class:`~relucent.poly.Polyhedron`), ``dim`` (the cell dimension k), and
        ``ss`` (the cell's sign-sequence array as numpy).

        Directed edges go from a k-cell to a (k-1)-cell whenever the latter is a
        codimension-1 face of the former under the SHI-zeroing rule. Each edge
        stores ``shi``, the supporting hyperplane index that was zeroed.

        **SHI rules (see module comment above the meta-graph helpers):**

        - Face **edges**: :func:`mg.ss_nonzero_indices` + lookup (homology-critical).
        - Node **metadata** (``shis``, ``crossings``): :func:`mg.cubical_cell_shis` on each
          dimension slice — derived at node creation, not propagated from ``contract``.
        - **Boundedness**: combinatorial classification from face edges only.

        If ``verify=True``, :func:`mg.verify_meta_graph_incidence` checks that assembled
        edges, node SHIs, and finite labels match the incidence engine.

        For combinatorial truncation (link-at-infinity homology), build the graph with
        this method and call :meth:`truncate_meta_graph` on a copy before
        :func:`relucent.topology.get_betti_numbers` or
        :meth:`get_betti_numbers_from_meta` with ``compactify=False``.

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

        Raises:
            IncompleteDualGraphError: If top-dimensional adjacency is incomplete; see
                :meth:`get_chain_complex`.
        """
        if len(self) == 0:
            logger.info("get_meta_graph: empty complex, returning empty graph")
            return nx.MultiDiGraph()

        nworkers = process_aware_cpu_count() or 1
        logger.info(
            "get_meta_graph: starting (verify=%s, verbose=%s, nworkers=%d)",
            verify,
            verbose,
            nworkers,
        )

        chain = self.get_chain_complex(verbose=verbose)
        # Dimension -> complex in the chain (there is at most one per dimension).
        by_dim: dict[int, Complex] = {}
        for c in chain:
            if len(c) == 0:
                continue
            by_dim[int(c.index2poly[0].dim)] = c

        meta: nx.MultiDiGraph[Any] = nx.MultiDiGraph()

        # Add all cells as nodes, keyed by stable poly.tag (bytes).
        # Pre-compute finite for chain-complex cells in parallel; contracted
        # lower-dim cells don't have finite cached yet (contract() only copies
        # halfspaces, not the Chebyshev solve result).
        all_chain_polys = [p for c_k in by_dim.values() for p in c_k]
        logger.info(
            "get_meta_graph: chain complex has %d dimensions, %d cells",
            len(by_dim),
            len(all_chain_polys),
        )

        lookup: dict[bytes, Polyhedron] = dict(self.tag2poly)
        for c_k in by_dim.values():
            lookup.update(c_k.tag2poly)

        top_dim = max(by_dim.keys())

        dim_neighbor_tags: dict[int, set[bytes]] = {int(k): {p.tag for p in c_k} for k, c_k in by_dim.items()}

        # Role 2: face edges from SS crossings (see module comment).
        edges_by_dim: dict[int, tuple[list[tuple[bytes, bytes, int]], list[bytes]]] = {}
        for k, c_k in sorted(by_dim.items(), reverse=True):
            if int(k) <= 0:
                continue
            valid_face_tags = set(lookup.keys())
            cells = [(p.tag, np.asarray(p.ss_np), mg.ss_nonzero_indices(np.asarray(p.ss_np))) for p in c_k]
            use_parallel = len(cells) >= mg.META_FACE_PARALLEL_MIN_CELLS and nworkers > 1
            if use_parallel:
                logger.info(
                    "get_meta_graph: k=%d face edges via multiprocessing Pool (%d workers, %d cells)",
                    int(k),
                    nworkers,
                    len(cells),
                )
                edges, extra_tags = mg.parallel_collect_meta_face_edges(
                    cells,
                    valid_face_tags,
                    nworkers=nworkers,
                )
            else:
                if len(cells) < mg.META_FACE_PARALLEL_MIN_CELLS:
                    face_mode = f"sequential (cells < {mg.META_FACE_PARALLEL_MIN_CELLS})"
                elif nworkers <= 1:
                    face_mode = "sequential (nworkers <= 1)"
                else:
                    face_mode = "sequential"
                logger.info(
                    "get_meta_graph: k=%d face edges %s (%d cells)",
                    int(k),
                    face_mode,
                    len(cells),
                )
                if verbose:
                    edges, extra_tags = mg.collect_meta_face_edges(
                        list(
                            tqdm(
                                cells,
                                desc=f"Building meta-graph faces (k={k})",
                                leave=False,
                            )
                        ),
                        valid_face_tags,
                    )
                else:
                    edges, extra_tags = mg.collect_meta_face_edges(cells, valid_face_tags)
            edges_by_dim[int(k)] = (edges, extra_tags)
            for face_tag in set(extra_tags):
                if face_tag not in lookup and face_tag in self.tag2poly:
                    lookup[face_tag] = self.tag2poly[face_tag]

        # Drop any stale finite hints on contracted cells before combinatorial
        # classification.  Boundedness is not monotone downward (a bounded coface
        # may still have unbounded faces), so hints from contraction must not
        # short-circuit the ascending sweep.
        for p in all_chain_polys:
            if 0 < p.dim < top_dim:
                p._finite_computed = False
                p._finite = None

        # Step 1: classify all 1-dim cells from 0-face incidence in meta edges.
        n_from_faces, infeasible_one_cells = mg.classify_one_cells_finite_from_face_edges(
            by_dim,
            edges_by_dim,
        )
        if infeasible_one_cells:
            logger.info(
                "get_meta_graph: %d infeasible 1-cells (no 0-faces, empty geometry)",
                len(infeasible_one_cells),
            )
        if n_from_faces:
            logger.info(
                "get_meta_graph: classified %d 1-cells from 0-face incidence (no LP)",
                n_from_faces,
            )

        # Step 2: ascending sweep — classify k-dim contracted cells (k ≥ 2) using
        # their already-classified (k-1)-dim faces.  Because every 0-dim cell is
        # trivially bounded (a point) and every 1-dim cell was classified above,
        # induction guarantees this sweep fully classifies all contracted cells:
        #   • unbounded if ANY (k-1)-face is unbounded
        #   • bounded   if ALL (k-1)-faces are bounded
        n_ascending = mg.classify_finite_ascending(by_dim, lookup, edges_by_dim)
        if n_ascending:
            logger.info(
                "get_meta_graph: ascending sweep classified %d contracted cells (no LP)",
                n_ascending,
            )

        pending_finite = sum(1 for p in all_chain_polys if not p._finite_computed)
        if pending_finite:
            msg = (
                f"get_meta_graph: {pending_finite}/{len(all_chain_polys)} chain cells "
                "still unclassified after combinatorial passes. This may indicate an "
                "incomplete BFS, missing edges, or a non-generic network."
            )
            if cfg.CAREFUL_MODE:
                raise AssertionError(msg)
            logger.warning(msg)
        else:
            logger.info(
                "get_meta_graph: all %d chain cells classified combinatorially",
                len(all_chain_polys),
            )

        excluded_tags: set[bytes] = set()
        for k, c_k in sorted(by_dim.items(), reverse=True):
            neighbor_tags = dim_neighbor_tags[int(k)]
            for p in c_k:
                if p.dim > 0 and p._finite_computed and p._finite is None:
                    excluded_tags.add(p.tag)
                    continue
                meta.add_node(p.tag, **mg.meta_node_attrs(p, neighbor_tags=neighbor_tags))

        # Add cached face edges k -> k-1.
        for k in sorted(by_dim.keys(), reverse=True):
            if int(k) <= 0:
                continue
            edges, extra_tags = edges_by_dim[int(k)]

            known_nodes = set(meta.nodes)
            new_face_tags = [tag for tag in set(extra_tags) if tag not in known_nodes and tag not in excluded_tags]
            if new_face_tags:
                mg.classify_lazy_face_polys(new_face_tags, lookup, edges_by_dim)
            face_dim = int(k) - 1
            face_neighbors = dim_neighbor_tags.get(face_dim, set())
            for face_tag_key in new_face_tags:
                meta.add_node(
                    face_tag_key,
                    **mg.meta_node_attrs(lookup[face_tag_key], neighbor_tags=face_neighbors),
                )
            known_nodes.update(new_face_tags)

            meta.add_edges_from(
                (u, v, {"shi": shi}) for u, v, shi in edges if u not in excluded_tags and v not in excluded_tags
            )

        if verify:
            logger.info("get_meta_graph: verify pass (incidence engine consistency)")
            mg.verify_meta_graph_incidence(meta, by_dim, lookup)

        logger.info(
            "get_meta_graph: done (%d nodes, %d edges, verify=%s)",
            meta.number_of_nodes(),
            meta.number_of_edges(),
            verify,
        )
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

    @classmethod
    def get_betti_numbers_from_meta(
        cls,
        meta: nx.MultiDiGraph[Any],
        *,
        reduced: bool = False,
        compactify: CompactifyMode = False,
        respect_finite: bool = False,
        verify_chain_complex: bool = False,
        verify_connected_components: bool = False,
        verbose: bool = False,
        nworkers: int | None = None,
    ) -> dict[int, int]:
        """Compute Betti numbers from an existing meta-graph.

        Args:
            meta: Output of :meth:`get_meta_graph`, optionally passed through
                :meth:`truncate_meta_graph` or :meth:`one_point_compactify_meta_graph`.
            compactify: Homology convention. ``False``: combinatorial truncation
                (caller should apply :meth:`truncate_meta_graph` first unless
                ``respect_finite``). ``True``: Borel–Moore (only faces with at least
                two cofaces). ``"one_point"``: one-point compactification at infinity
                (caller should apply :meth:`one_point_compactify_meta_graph` first).
            respect_finite: If True, restrict to the subcomplex of cells with ``finite is True``
                (no truncation). Other flags are forwarded to :func:`relucent.topology.get_betti_numbers`.
            verify_connected_components: Forwarded to :func:`relucent.topology.get_betti_numbers`.
            nworkers: Forwarded to :func:`relucent.topology.get_betti_numbers`; controls
                how many threads rank independent boundary maps concurrently.
        """
        from relucent.topology import get_betti_numbers

        if meta.number_of_nodes() == 0:
            return {}
        if respect_finite:
            meta = cls.finite_cells_subgraph(meta)
            if meta.number_of_nodes() == 0:
                return {}
        return get_betti_numbers(
            meta,
            require_shared_faces=compactify is True,
            reduced=reduced,
            verify_chain_complex=verify_chain_complex,
            verify_connected_components=verify_connected_components,
            verbose=verbose,
            nworkers=nworkers,
        )

    def get_betti_numbers(
        self,
        *,
        reduced: bool = False,
        compactify: CompactifyMode = False,
        respect_finite: bool = False,
        verify_chain_complex: bool = False,
        verify_connected_components: bool = False,
        verbose: bool = False,
        nworkers: int | None = None,
    ) -> dict[int, int]:
        """Compute Betti numbers over GF(2).

        Builds a meta-graph from this complex, applies truncation when appropriate, then
        delegates to :meth:`get_betti_numbers_from_meta`. Pass ``verbose=True`` for
        stderr progress from meta-graph construction and boundary maps.

        Results are cached per ``(reduced, compactify, respect_finite)`` and survive
        :meth:`save` / :meth:`load`. The cache is cleared when polyhedra are added or
        overwritten. Calls with ``verify_chain_complex`` or ``verify_connected_components``
        bypass the cache and always recompute.

        Args:
            compactify: ``False`` applies combinatorial truncation at infinity;
                ``True`` uses Borel–Moore face incidences; ``"one_point"`` adds an
                extra 0-cell at infinity for unbounded 1-cell ends.
            nworkers: Number of threads for ranking independent boundary maps concurrently.
                ``None`` (default): auto (one thread per map when the C backend is available).
                ``1``: always sequential.  See :func:`relucent.topology.get_betti_numbers`.
        """
        self._warn_research_use("get_betti_numbers")
        if len(self) == 0:
            return {}
        cache_key = (reduced, compactify, respect_finite)
        use_cache = not verify_chain_complex and not verify_connected_components
        if use_cache and cache_key in self._betti_cache:
            return dict(self._betti_cache[cache_key])
        meta = self.get_meta_graph(verbose=verbose)
        if compactify == "one_point":
            self.one_point_compactify_meta_graph(meta)
        elif not compactify and not respect_finite:
            self.truncate_meta_graph(meta)
        betti = self.get_betti_numbers_from_meta(
            meta,
            reduced=reduced,
            compactify=compactify,
            respect_finite=respect_finite,
            verify_chain_complex=verify_chain_complex,
            verify_connected_components=verify_connected_components,
            verbose=verbose,
            nworkers=nworkers,
        )
        if use_cache:
            self._betti_cache[cache_key] = dict(betti)
        return betti

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

    def verify_arrangement_genericity(self) -> None:
        """Raise :class:`NonGenericArrangementError` on degenerate 1-cell arrangements.

        Checks that combinatorial 0-face endpoints are geometrically distinct on each
        1-cell and that geometrically coincident endpoints share a combinatorial tag.
        """
        if len(self) == 0:
            return
        top_dim = max(int(p.dim) for p in self)
        if top_dim == 1:
            mg.verify_arrangement_genericity(self)

    def verify_dual_graph_consistency(self, graph: nx.Graph[Polyhedron] | None = None) -> None:
        """Verify cubical consistency of dual-graph edges (Layer 1)."""
        if len(self) == 0:
            return
        try:
            top_dim = max(int(p.dim) for p in self)
        except ValueError:
            return
        if top_dim <= 0:
            return

        g = graph if graph is not None else self.get_dual_graph(verbose=False)
        top_cells = [p for p in self if int(p.dim) == top_dim]
        mg.verify_dual_graph_cubical(top_cells, g, top_dim=top_dim)

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
        require_complete: bool = False,
        verify: bool = True,
        cubical: bool | None = None,
    ) -> nx.Graph[Polyhedron]: ...

    @overload
    def get_dual_graph(
        self,
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
        require_complete: bool = False,
        verify: bool = True,
        cubical: bool | None = None,
    ) -> nx.Graph[int]: ...

    def get_dual_graph(
        self,
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
        require_complete: bool = False,
        verify: bool = True,
        cubical: bool | None = None,
    ) -> nx.Graph[Polyhedron] | nx.Graph[int]:
        """Construct the dual graph of the complex.

        The dual graph represents the connectivity structure of the complex,
        where nodes are polyhedra and edges connect adjacent polyhedra (those
        sharing a supporting hyperplane).

        Edges use combinatorial cubical adjacency via :func:`~relucent.meta_graph.dual_edges_top_dim`
        (0-face sharing when ``max_dim == 1``, flip neighbors when ``max_dim >= 2``).

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
            verbose: If True, print progress messages. Defaults to True.
            require_complete: If True, raise :class:`IncompleteDualGraphError` when
                boundary neighbors are missing. Defaults to False (emit a warning).
                Used by :meth:`contract`.
            verify: If True, require bidirectional ``_shis`` on each edge and run
                cubical consistency checks. Defaults to True.

        Returns:
            networkx.Graph: The dual graph of the complex. Nodes are polyhedra
                (or integers if relabel=True), edges connect adjacent polyhedra
                and have a "shi" attribute indicating which supporting hyperplane
                they cross.

        Raises:
            ValueError: If match_locations is True and the complex is not 2D.
            IncompleteDualGraphError: If ``require_complete`` is True and boundary
                neighbors are missing.
        """
        if len(self) == 0:
            return nx.Graph()
        max_dim = max(poly.dim for poly in self)
        graph = nx.Graph()
        top_cells: list[Polyhedron] = []
        for poly in self:
            if poly.dim == max_dim:
                graph.add_node(poly, label=str(poly))
                top_cells.append(poly)

        neighbor_tags = {p.tag for p in top_cells}
        edge_top_dim = mg.dual_graph_edge_top_dim(cell_top_dim=max_dim, ambient_dim=int(self.dim))
        prefer_cell_shis = max_dim < int(self.dim)
        edges, _ = mg.dual_edges_top_dim(
            top_cells,
            neighbor_tags,
            top_dim=edge_top_dim,
            prefer_cell_shis=prefer_cell_shis,
        )
        for u, v, shi in edges:
            graph.add_edge(u, v, shi=shi)
        if top_cells:
            mg.set_shis_from_dual_graph(graph)

        if verify and top_cells:
            mg.verify_dual_graph_cubical(top_cells, graph, top_dim=edge_top_dim)
        if plot:
            from relucent.vis import get_colors

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
        if require_complete and int(max_dim) == int(self.dim) and top_cells:
            # LP completeness check for full ambient top cells only.
            from relucent.verify import verify_lp_flip_neighbors_in_complex

            verify_lp_flip_neighbors_in_complex(self)
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

        Notes:
            Sets exploration state to ``complete=True, verified=True``. Only
            recover graphs that were built from a complete, verified ambient search.
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
            if cfg.CAREFUL_MODE:
                assert poly1.ss_np.ravel()[shi] != 0
            poly2 = self.add_ss(flip_ss_at_shi(poly1.ss_np, shi), check_exists=False)
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
            # Caches are written only for complete+verified complexes; trust SHIs on reload.
            graph.nodes[node]["poly"]._shis_strict = True
        # Dual-graph recovery is a trusted reconstruct of a previously explored complex.
        self.set_exploration_state(complete=True, verified=True)

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

        from relucent.vis import plot_complex

        return plot_complex(
            self,
            plot_mode=plot_mode,
            **plot_kwargs,
        )
