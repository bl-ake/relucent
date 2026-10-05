"""Polyhedral complex container: search, graphs, topology, and certification."""

from __future__ import annotations

import os
import pickle
import random
from collections.abc import Generator, Iterable, Iterator
from typing import TYPE_CHECKING, Any, Literal, Self, cast, overload

import networkx as nx
import numpy as np
import plotly.graph_objects as go

import relucent.config as cfg
import relucent.verify.certify as certify
from relucent._internal.logging import with_verbosity
from relucent._internal.parallel import BlockingQueue
from relucent._internal.torch_compat import TORCH_AVAILABLE, torch
from relucent.core.errors import (
    ComplexNotCompleteError,
    ComplexNotVerifiedError,
    DualGraphAsymmetricEdgeError,
    IncompleteDualGraphError,
    NonGenericArrangementError,
)
from relucent.core.poly import Polyhedron
from relucent.core.ss import SSManager, encode_ss
from relucent.graph import incidence, vertex_star
from relucent.graph import meta_graph as mg
from relucent.graph.complex_graph import recover_from_dual_graph
from relucent.graph.meta_graph import (
    INFINITY_POINT_META_NODE,
    INFINITY_POINT_META_SHI,
    TRUNCATION_META_SHI,
)
from relucent.model.convert_model import convert
from relucent.model.model import LinearLayer, ReLULayer, ReLUNetwork
from relucent.search.engine import (
    ALL_GEOMETRY_PROPERTIES,
    retain_geometry_caches,
)
from relucent.search.engine import (
    greedy_path as _greedy_path_fn,
)
from relucent.search.engine import (
    hamming_astar as _hamming_astar_fn,
)
from relucent.search.engine import (
    parallel_add as _parallel_add_fn,
)
from relucent.search.engine import (
    parallel_compute_geometric_properties as _parallel_compute_geometric_properties_fn,
)
from relucent.search.engine import (
    searcher as _searcher_fn,
)
from relucent.verify.certify import CertifyLevel

if TYPE_CHECKING:
    from relucent.search.engine import CubeMode
    from relucent.search.exploration import SearchResult
    from relucent.topology.betti import Compactify
    from relucent.topology.filtration import Filtration
    from relucent.topology.morse import CriticalPoint
    from relucent.topology.persistence import PersistenceDiagram

__all__ = [
    "Complex",
    "CertifyLevel",
    "INFINITY_POINT_META_NODE",
    "INFINITY_POINT_META_SHI",
    "IncompleteDualGraphError",
    "NonGenericArrangementError",
    "TRUNCATION_META_SHI",
    "ComplexNotCompleteError",
    "ComplexNotVerifiedError",
    "DualGraphAsymmetricEdgeError",
]


class Complex:
    """Manages the polyhedral complex of a neural network.

    This class provides methods for calculating, storing, and searching the h-representations
    (halfspace representations) of polyhedra in the complex.
    """

    #: Version of the file layout :meth:`save` writes; :meth:`load` refuses newer files.
    SAVE_FORMAT_VERSION = 1

    def __init__(self, net: Any) -> None:
        """Initialize the complex for a given network.

        Args:
            net: A :class:`~relucent.model.model.ReLUNetwork`, or any model
                :func:`~relucent.model.convert_model.convert` accepts (e.g. a PyTorch
                ``nn.Sequential``).
        """
        self.source_model: Any = net
        self._net: ReLUNetwork = net if isinstance(net, ReLUNetwork) else convert(net)

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
        self._betti_cache: dict[tuple[bool, Compactify, bool], dict[int, int]] = {}
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
        """Whether the complex passed the last certification (``None`` if unknown)."""
        return self._verified

    def set_exploration_state(self, *, complete: bool, verified: bool) -> None:
        """Record exploration / certification status after search or certify."""
        self._complete = complete
        self._verified = verified

    def assert_topology_ready(self) -> None:
        """Require a complete, verified complex before topology routines."""
        if self._complete is True and self._verified is True:
            return
        if self._complete is False:
            raise ComplexNotCompleteError(
                "Complex is not complete or failed invariant verification. Explore further "
                + "(e.g. BFS) or pass an explicit exploration cap (max_polys) to opt into "
                + "a partial complex."
            )
        if self._complete is True and self._verified is not True:
            raise ComplexNotVerifiedError(
                "Complex is complete but not verified. Re-run BFS with verify=True or call " + "certify()."
            )
        raise ComplexNotCompleteError(
            "Complex exploration state is unknown. Run BFS or explore_for_topology first, "
            + "or set_exploration_state(complete=True, verified=True) for trusted loads."
        )

    def certify(
        self,
        *,
        level: CertifyLevel = CertifyLevel.COMPLETE,
        repair: bool = True,
        graph: nx.Graph[Polyhedron] | None = None,
        verbose: int | None = None,
    ) -> None:
        """Certify this complex and record the result via :meth:`set_exploration_state`.

        See :func:`relucent.verify.certify.certify_complex` for the certification levels
        and the (conservative, ``_shis``-only) repair this performs.
        """
        certify.certify_complex(self, level=level, repair=repair, graph=graph, record_state=True, verbose=verbose)

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

        The file keeps the cells, the network, cached Betti numbers and the
        :attr:`complete` / :attr:`verified` state, so a loaded complex is ready for topology
        routines without re-running :meth:`certify`.

        Args:
            filename: Path to the output file.
            save_ssm: If True, include the SSManager in the saved state so that
                sign-sequence lookups are preserved. Defaults to True.
        """
        state = self.__getstate__()
        state["format_version"] = self.SAVE_FORMAT_VERSION
        if save_ssm:
            state["ssm"] = self.ssm
        with open(filename, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filename: str | os.PathLike[str]) -> Self:
        """Load a Complex written by :meth:`save`.

        This unpickles the file, which can run arbitrary code: only load files you trust.
        Files from before relucent 1.0 carry no format version; they load without their
        :attr:`complete` / :attr:`verified` state.

        Args:
            filename: Path to the pickle file.

        Returns:
            The restored complex.

        Raises:
            ValueError: If the file was written by a newer relucent (a newer format version).
        """
        with open(filename, "rb") as f:
            state = pickle.load(f)
        version = int(state.get("format_version", 0))
        if version > cls.SAVE_FORMAT_VERSION:
            raise ValueError(
                f"{os.fspath(filename)!r} has save format {version}; this relucent reads up to "
                + f"{cls.SAVE_FORMAT_VERSION}. Upgrade relucent to load it."
            )
        cplx = cls(state["net"])
        cplx.__setstate__(state)
        return cplx

    def __getstate__(self) -> dict[str, Any]:
        return {
            "index2poly": self.index2poly,
            "net": self._net,
            "source_model": self.source_model,
            "_betti_cache": self._betti_cache,
            "_complete": self._complete,
            "_verified": self._verified,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["net"])
        self.source_model = state.get("source_model", self._net)
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
        self._complete = state.get("_complete")
        self._verified = state.get("_verified")

    @property
    def net(self) -> ReLUNetwork:
        """The network as relucent's canonical :class:`~relucent.model.model.ReLUNetwork`.

        The model passed to the constructor (e.g. a PyTorch module) is kept as
        :attr:`source_model`.
        """
        return self._net

    def _empty_like(self) -> Self:
        """A new, empty complex over the same network."""
        out = type(self)(self._net)
        out.source_model = self.source_model
        return out

    @property
    def dim(self) -> int:
        """The input dimension of the network."""
        return int(np.prod(self._net.input_shape))

    @property
    def n(self) -> int:
        """The number of bent hyperplanes/neurons in the network."""
        return len(self.ssi2maski)

    @torch.no_grad()
    def preactivation_iterator(
        self,
        batch: torch.Tensor | np.ndarray,
    ) -> Generator[torch.Tensor | np.ndarray, None, None]:
        """Yield the preactivation values used by the sign sequence."""
        if TORCH_AVAILABLE and isinstance(batch, torch.Tensor):
            x: torch.Tensor | np.ndarray = batch.reshape((-1, *self._net.input_shape))
        else:
            x = np.asarray(batch, dtype=np.float64).reshape((-1, *self._net.input_shape))
        for i, layer in enumerate(self._net.layers.values()):
            x = self._net._apply_layer(layer, x)
            if i in self.ss_layers:
                yield x
                if i == self.ss_layers[-1]:
                    break

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
        use_torch = TORCH_AVAILABLE and isinstance(batch, torch.Tensor)
        for values in self.preactivation_iterator(batch):
            if use_torch:
                yield torch.sign(torch.as_tensor(values))
            else:
                yield np.sign(np.asarray(values))

    def point2preactivations(self, batch: torch.Tensor | np.ndarray) -> np.ndarray | torch.Tensor:
        """Return stacked ReLU preactivations in sign-sequence order."""
        is_tensor = TORCH_AVAILABLE and isinstance(batch, torch.Tensor)
        values = list(self.preactivation_iterator(batch))
        if is_tensor:
            return torch.hstack([torch.as_tensor(value) for value in values])
        return np.hstack([np.asarray(value) for value in values])

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
            verbose: Output level: ``0`` quiet, ``1`` progress bars and summaries, ``2`` debug
                detail. ``None`` uses :data:`relucent.config.VERBOSE`.
            **kwargs: Additional arguments passed to :func:`~relucent.geometry.calculations.get_shis`
                and related geometry helpers.

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
        *,
        queue: Any = None,
        max_depth: float = float("inf"),
        max_polys: float = float("inf"),
        bound: float | None = None,
        nworkers: int | None = None,
        cube_radius: float | None = None,
        cube_mode: CubeMode = "unrestricted",
        geometry_properties: Iterable[str] | None = None,
        verify: bool = True,
        verbose: int | None = None,
        **kwargs: Any,
    ) -> SearchResult:
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
                Important for numerical stability. When ``None``, uses
                :func:`~relucent._internal.network_scale.default_polyhedron_bound`.
            nworkers: Number of worker processes for parallel computation. If None,
                uses the number of CPU cores. Defaults to None.
            cube_radius: Half-width of the cube ``[-r, r]^d`` that ``cube_mode`` refers to.
            cube_mode: ``"unrestricted"`` (default) ignores the cube. ``"intersect"`` only
                explores cells that meet it, ``"clipped"`` also clips their halfspaces to it,
                and ``"exclude"`` only explores cells that don't meet it.
            geometry_properties: Iterable of polyhedron cache/property names to
                compute and retain for each discovered polyhedron during search.
                ``None`` (default) performs topology-only search. Pass
                :data:`~relucent.search.ALL_GEOMETRY_PROPERTIES` or a subset for
                optional caches. ``finite``, ``center``, and ``inradius`` are always
                computed.
            verify: When True (default), require complete exploration and run
                :func:`~relucent.verify.certify.certify_complex` at the end. Certification
                is skipped when exploration hits ``max_polys`` before the frontier is
                exhausted. A finite ``max_depth`` cap can leave ``complete=False``; with
                ``verify=True`` that raises :class:`~relucent.core.complex.IncompleteDualGraphError`
                unless the cap was hit. Frontier SHIs are certified facets, so certification
                reuses them after dual-graph sync.
            verbose: Output level: ``0`` quiet, ``1`` progress bars and summaries, ``2`` debug
                detail. ``None`` uses :data:`relucent.config.VERBOSE`.
            **kwargs: Additional arguments passed to :func:`~relucent.geometry.calculations.get_shis`.

        Returns:
            A :class:`~relucent.search.exploration.SearchResult`.

        Raises:
            ValueError: If the start point lies on a hyperplane (has zero in SS).
            IncompleteDualGraphError: If ``verify`` is True and exploration stops early
                for reasons other than hitting ``max_polys``.
        """
        if bound is None:
            from relucent._internal.network_scale import default_polyhedron_bound

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
            verbose: Output level: ``0`` quiet, ``1`` progress bars and summaries, ``2`` debug
                detail. ``None`` uses :data:`relucent.config.VERBOSE`.
        """
        return _parallel_compute_geometric_properties_fn(
            self,
            nworkers=nworkers,
            geometry_properties=properties,
            verbose=verbose,
        )

    def bfs(
        self,
        start: torch.Tensor | np.ndarray | Polyhedron | None = None,
        *,
        max_depth: float = float("inf"),
        max_polys: float = float("inf"),
        bound: float | None = None,
        nworkers: int | None = None,
        cube_radius: float | None = None,
        cube_mode: CubeMode = "unrestricted",
        geometry_properties: Iterable[str] | None = None,
        verify: bool = True,
        verbose: int | None = None,
        **kwargs: Any,
    ) -> SearchResult:
        """Breadth-first search: every cell at depth ``d`` before any at ``d + 1``.

        Takes the same arguments as :meth:`searcher`, except ``queue``.
        """
        queue = None
        return self.searcher(
            start,
            queue=queue,
            max_depth=max_depth,
            max_polys=max_polys,
            bound=bound,
            nworkers=nworkers,
            cube_radius=cube_radius,
            cube_mode=cube_mode,
            geometry_properties=geometry_properties,
            verify=verify,
            verbose=verbose,
            **kwargs,
        )

    def dfs(
        self,
        start: torch.Tensor | np.ndarray | Polyhedron | None = None,
        *,
        max_depth: float = float("inf"),
        max_polys: float = float("inf"),
        bound: float | None = None,
        nworkers: int | None = None,
        cube_radius: float | None = None,
        cube_mode: CubeMode = "unrestricted",
        geometry_properties: Iterable[str] | None = None,
        verify: bool = True,
        verbose: int | None = None,
        **kwargs: Any,
    ) -> SearchResult:
        """Depth-first search: follow each path as deep as it goes before backtracking.

        Takes the same arguments as :meth:`searcher`, except ``queue``.
        """
        queue = BlockingQueue(pop=lambda x: x.pop())
        return self.searcher(
            start,
            queue=queue,
            max_depth=max_depth,
            max_polys=max_polys,
            bound=bound,
            nworkers=nworkers,
            cube_radius=cube_radius,
            cube_mode=cube_mode,
            geometry_properties=geometry_properties,
            verify=verify,
            verbose=verbose,
            **kwargs,
        )

    def random_walk(
        self,
        start: torch.Tensor | np.ndarray | Polyhedron | None = None,
        *,
        max_depth: float = float("inf"),
        max_polys: float = float("inf"),
        bound: float | None = None,
        nworkers: int | None = None,
        cube_radius: float | None = None,
        cube_mode: CubeMode = "unrestricted",
        geometry_properties: Iterable[str] | None = None,
        verify: bool = True,
        verbose: int | None = None,
        **kwargs: Any,
    ) -> SearchResult:
        """Search that expands a uniformly random frontier cell at each step.

        Takes the same arguments as :meth:`searcher`, except ``queue``.
        """
        queue = BlockingQueue(
            queue_class=list,
            pop=lambda x: x.pop(random.randrange(0, len(x))),
            push=lambda x, y: x.append(y),
        )
        return self.searcher(
            start,
            queue=queue,
            max_depth=max_depth,
            max_polys=max_polys,
            bound=bound,
            nworkers=nworkers,
            cube_radius=cube_radius,
            cube_mode=cube_mode,
            geometry_properties=geometry_properties,
            verify=verify,
            verbose=verbose,
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
        verbose: int | None = None,
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
            verbose: Output level: ``0`` quiet, ``1`` progress bars and summaries, ``2`` debug
                detail. ``None`` uses :data:`relucent.config.VERBOSE`.
            **kwargs: Additional arguments passed to :func:`~relucent.geometry.calculations.get_shis`.

        Returns:
            dict[str, Any]: Dictionary containing the path (if found) and
                additional diagnostics/bounds.

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
            verbose=verbose,
            num_threads=num_threads,
            **kwargs,
        )

    def boundary_cells(self, i: int, *, verify: bool = True, verbose: int | None = None) -> set[Polyhedron]:
        """The (d-1)-cells on neuron ``i``'s bent hyperplane. See :func:`relucent.graph.boundary.boundary_cells`."""
        from relucent.graph.boundary import boundary_cells

        return boundary_cells(self, i, verify=verify, verbose=verbose)

    def boundary_complex(self, i: int, *, verbose: int | None = None) -> Complex:
        """The certified complex on neuron ``i``'s bent hyperplane, from this explored complex.

        See :func:`relucent.graph.boundary.boundary_complex`. To find it without exploring
        the whole input space, use :meth:`discover_boundary_complex`.

        Raises:
            IncompleteDualGraphError: If top-dimensional adjacency is incomplete.
            ComplexNotCompleteError: If this complex is not complete.
            ComplexNotVerifiedError: If this complex is not verified.
        """
        from relucent.graph.boundary import boundary_complex

        return boundary_complex(self, i, verbose=verbose)

    def slice_affine(self, x0: np.ndarray, V: np.ndarray) -> Complex:
        """Intersect every cell with the affine subspace ``{x0 + V @ t}``, as a complex in ``t``.

        See :func:`relucent.geometry.slicing.slice_complex`.
        """
        from relucent.geometry.slicing import slice_complex

        return slice_complex(self, x0, V)

    @overload
    def discover_boundary_complex(
        self,
        i: int,
        verbose: int | None = None,
        *,
        return_stats: Literal[False] = False,
        **kwargs: Any,
    ) -> Complex: ...

    @overload
    def discover_boundary_complex(
        self,
        i: int,
        verbose: int | None = None,
        *,
        return_stats: Literal[True],
        **kwargs: Any,
    ) -> tuple[Complex, Any]: ...

    def discover_boundary_complex(
        self,
        i: int,
        verbose: int | None = None,
        *,
        return_stats: bool = False,
        **kwargs: Any,
    ) -> Complex | tuple[Complex, Any]:
        """Discover neuron ``i``'s boundary complex without building the full input complex.

        Uses MIP pricing to find new connected components on the slice ``ss[i]=0``,
        then slice-restricted BFS to complete each component.

        Args:
            i: Global supporting-hyperplane index (bent hyperplane).
            verbose: Output level: ``0`` quiet, ``1`` progress bars and summaries, ``2`` debug
                detail. ``None`` uses :data:`relucent.config.VERBOSE`.
            return_stats: If True, return ``(complex, stats)`` with timing metadata.
            **kwargs: Forwarded to :func:`~relucent.search.boundary_search.discover_boundary_complex`.

        Returns:
            A new :class:`Complex` of boundary cells, or ``(complex, stats)`` when
            ``return_stats`` is True.
        """
        from relucent.search.boundary_search import discover_boundary_complex as _discover

        boundary, stats = _discover(
            self._net,
            i,
            verbose=verbose,
            **kwargs,
        )
        if return_stats:
            return boundary, stats
        return boundary

    def chain_complex(self, verbose: int | None = None) -> list[Complex]:
        """Recover every cell of every dimension from verified vertices' local stars.

        Returns ``[self, (d-1)-cells, ..., 0-cells]`` as complexes. See
        :func:`relucent.graph.vertex_star.build_chain_complex`.

        Raises:
            ComplexNotCompleteError, ComplexNotVerifiedError: If this complex is not
                complete and verified.
            CubicalConsistencyError: If the labeled top-cell graph is not cubical.
        """
        return vertex_star.build_chain_complex(self, verbose=verbose)

    def contract(self, verbose: int | None = None) -> Complex:
        """The complex of codimension-one cells (dimension ``self.dim - 1``) from :meth:`chain_complex`.

        Empty if there are none.

        Raises:
            IncompleteDualGraphError: If top-dimensional adjacency is incomplete.
            ComplexNotCompleteError: If this complex is not complete.
            ComplexNotVerifiedError: If this complex is not verified.
        """
        for cells in self.chain_complex(verbose=verbose):
            if len(cells) and int(cells.index2poly[0].dim) == self.dim - 1:
                return cells
        return self._empty_like()

    def critical_points(
        self,
        *,
        require_complete: bool = False,
        include_degenerate: bool = False,
        verbose: int | None = None,
    ) -> list[CriticalPoint]:
        """Return PL Morse critical vertices and their indices (scalar-output networks only).

        See :func:`relucent.topology.morse.critical_points`.
        """
        from relucent.topology.morse import critical_points

        return critical_points(self, require_complete=require_complete, include_degenerate=include_degenerate, verbose=verbose)

    def meta_graph(self, *, verify: bool = False, verbose: int | None = None) -> nx.MultiDiGraph[Any]:
        """Return the face poset of every cell, all dimensions, as a meta-graph.

        Nodes are cells keyed by ``tag``; edges go from each k-cell to its (k-1)-faces.
        See :func:`relucent.graph.meta_graph.build_meta_graph`.

        Raises:
            IncompleteDualGraphError: If top-dimensional adjacency is incomplete; see
                :meth:`chain_complex`.
        """
        return mg.build_meta_graph(self, verify=verify, verbose=verbose)

    @with_verbosity
    def betti_numbers(
        self,
        *,
        compactify: Compactify = "truncate",
        respect_finite: bool = False,
        reduced: bool = False,
        verify_chain_complex: bool = False,
        verify_connected_components: bool = False,
        verbose: int | None = None,
        nworkers: int | None = None,
    ) -> dict[int, int]:
        """Compute Betti numbers over GF(2).

        Builds the meta-graph (:meth:`meta_graph`) and ranks it with
        :func:`relucent.topology.betti_numbers`.

        Results are cached per ``(reduced, compactify, respect_finite)`` and survive
        :meth:`save` / :meth:`load`. The cache is cleared when polyhedra are added or
        overwritten. Calls with ``verify_chain_complex`` or ``verify_connected_components``
        bypass the cache and always recompute.

        Args:
            compactify: How unbounded cells are handled: ``"truncate"`` (default) caps them
                with combinatorial truncation at infinity, ``"borel_moore"`` computes
                Borel–Moore homology, and ``"one_point"`` adds a single 0-cell at infinity.
            respect_finite: If True, use the subcomplex of bounded cells instead.
            reduced: If True, return reduced homology.
            verify_chain_complex: Check ``∂² = 0`` (see :func:`relucent.topology.betti_numbers`).
            verify_connected_components: Check β₀ against the path-component count.
            verbose: Output level: ``0`` quiet, ``1`` progress bars and summaries, ``2`` debug
                detail. ``None`` uses :data:`relucent.config.VERBOSE`.
            nworkers: Threads for ranking independent boundary maps concurrently
                (``method="dense"`` only; see :func:`relucent.topology.betti_numbers`).
        """
        from relucent.topology.betti import betti_numbers

        del verbose  # applied by @with_verbosity
        if len(self) == 0:
            return {}
        cache_key = (reduced, compactify, respect_finite)
        use_cache = not verify_chain_complex and not verify_connected_components
        if use_cache and cache_key in self._betti_cache:
            return dict(self._betti_cache[cache_key])
        betti = betti_numbers(
            self.meta_graph(),
            compactify=compactify,
            respect_finite=respect_finite,
            reduced=reduced,
            verify_chain_complex=verify_chain_complex,
            verify_connected_components=verify_connected_components,
            nworkers=nworkers,
        )
        if use_cache:
            self._betti_cache[cache_key] = dict(betti)
        return betti

    def persistent_homology(
        self,
        filtration: Filtration,
        *,
        compactify: Literal["truncate", "borel_moore"] = "truncate",
        respect_finite: bool = False,
        lower_star: bool | None = None,
        verbose: int | None = None,
    ) -> PersistenceDiagram:
        """Compute persistent homology over GF(2) for a :class:`~relucent.topology.filtration.Filtration`.

        See :func:`relucent.topology.persistence.compute_persistent_homology`.
        """
        from relucent.topology.filtration import Filtration
        from relucent.topology.persistence import compute_persistent_homology

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
            certify.verify_arrangement_genericity(self)

    @property
    def G(self) -> nx.Graph[Polyhedron]:
        """The adjacency graph of top-dimensional cells in the complex."""
        if self._dual_graph is None:
            self._dual_graph = self.dual_graph()
        return self._dual_graph

    @overload
    def dual_graph(
        self,
        *,
        relabel: Literal[False] = False,
        require_complete: bool = False,
        repair: bool = True,
    ) -> nx.Graph[Polyhedron]: ...

    @overload
    def dual_graph(
        self,
        *,
        relabel: Literal[True],
        require_complete: bool = False,
        repair: bool = True,
    ) -> nx.Graph[int]: ...

    def dual_graph(
        self,
        *,
        relabel: bool = False,
        require_complete: bool = False,
        repair: bool = True,
    ) -> nx.Graph[Polyhedron] | nx.Graph[int]:
        """Construct the dual graph of the complex.

        The dual graph represents the connectivity structure of the complex,
        where nodes are polyhedra and edges connect adjacent polyhedra (those
        sharing a supporting hyperplane). For a PyVis-ready copy, see
        :func:`relucent.vis.pyvis_dual_graph`.

        Edges use combinatorial cubical adjacency via :func:`~relucent.graph.incidence.dual_edges_top_dim`
        (0-face sharing when ``max_dim == 1``, flip neighbors when ``max_dim >= 2``).

        Args:
            relabel: If True, nodes are indexed by integers matching self.index2poly
                indices. If False, nodes are Polyhedron objects. Defaults to False.
            require_complete: If True, raise :class:`IncompleteDualGraphError` when
                boundary neighbors are missing (checked via an LP facet recompute on
                full ambient top cells). Defaults to False.
            repair: If True (default), overwrite each top cell's ``_shis`` from the
                freshly built combinatorial dual graph (see
                :func:`~relucent.graph.incidence.sync_shis_from_dual_graph`). This is the
                one repair relucent performs automatically.

        Returns:
            networkx.Graph: The dual graph of the complex. Nodes are polyhedra
                (or integers if relabel=True), edges connect adjacent polyhedra
                and have a "shi" attribute indicating which supporting hyperplane
                they cross.

        Raises:
            IncompleteDualGraphError: If ``require_complete`` is True and boundary
                neighbors are missing.
        """
        if len(self) == 0:
            return nx.Graph()
        max_dim = max(poly.dim for poly in self)
        top_cells = [poly for poly in self if poly.dim == max_dim]
        graph = incidence.build_dual_graph(top_cells, top_dim=max_dim, ambient_dim=int(self.dim), repair=repair)
        for poly in top_cells:
            graph.nodes[poly]["label"] = str(poly)

        if relabel:
            graph = nx.relabel_nodes(graph, {poly: i for i, poly in enumerate(self)})
        if require_complete and int(max_dim) == int(self.dim) and top_cells:
            # LP completeness check for full ambient top cells only.
            certify.verify_lp_flip_neighbors_in_complex(self)
        return cast(Any, graph)

    def recover_from_dual_graph(
        self,
        graph: nx.Graph[int],
        initial_ss: np.ndarray | torch.Tensor,
        source: int,
        copy: bool = False,
    ) -> None:
        """Fill this complex from a stored dual graph (top cells plus ``shi`` edge labels).

        See :func:`relucent.graph.complex_graph.recover_from_dual_graph`.
        """
        recover_from_dual_graph(self, graph, initial_ss, source, copy=copy)

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
                - ``"1-skeleton"``: 1-cells from ``chain_complex()``.
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
