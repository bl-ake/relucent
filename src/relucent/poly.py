import hashlib
import warnings
from functools import cached_property
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import torch
from scipy.spatial import ConvexHull, HalfspaceIntersection

import relucent.config as cfg
from relucent.calculations import (
    compute_properties,
    get_hs,
    get_shis,
    solve_radius,
)
from relucent.model import NN
from relucent.utils import encode_ss, get_env
from relucent.vis import bounded_plot_geometry, plot_polyhedron

__all__ = ["Polyhedron"]


class Polyhedron:
    """Represents a polyhedron (linear region) in d-dimensional space.

    Several methods use Gurobi environments for optimization. If one is not
    provided, an environment will be created automatically.

    Interior-point search radius follows :data:`relucent.config.MAX_RADIUS`
    unless overridden per call.
    """

    def __init__(
        self,
        net: NN,
        ss: np.ndarray | torch.Tensor,
        halfspaces: np.ndarray | torch.Tensor | None = None,
        W: np.ndarray | torch.Tensor | None = None,
        b: np.ndarray | torch.Tensor | None = None,
        shis: list[int] | None = None,
        bound: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a Polyhedron object.

        Args:
            net: Instance of the NN class from the "model" module.
            ss: Sign sequence defining the polyhedron (values in {-1, 0, 1}).

        The kwargs can be used to supply precomputed values for various properties.
        """
        self._net = net
        # Store the sign sequence with an integer dtype to ensure consistent
        # semantics across NumPy and PyTorch backends.
        self._ss = self._coerce_ss_to_int(ss)
        self._halfspaces: torch.Tensor | np.ndarray | None = halfspaces
        self._halfspaces_np: np.ndarray | None = None
        self._W: torch.Tensor | np.ndarray | None = W
        self._b: torch.Tensor | np.ndarray | None = b
        self._Wl2: float | None = None
        self._interior_point: np.ndarray | None = None
        self._interior_point_norm: float | None = None
        self._center: np.ndarray | None = None
        self._inradius: float | None = None
        self._num_dead_relus: int | None = None
        self.bound = bound

        self._shis: list[int] | None = shis
        self._hs: HalfspaceIntersection | None = None
        self._ch: ConvexHull | None = None
        self._finite_computed: bool = False
        self._finite: bool | None = None
        self._vertices: np.ndarray | None = None
        self._volume: float | None = None

        self._hash: int | None = None
        self._tag: bytes | None = None

        self.warnings: list[Warning] = []

        # Cached NumPy representation of the sign sequence (if/when needed).
        self._ss_np: np.ndarray | None = None

        self._attempted_compute_properties: bool = False

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _coerce_ss_to_int(self, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Return an integer-typed sign sequence (values in {-1, 0, 1})."""
        if isinstance(value, np.ndarray):
            if not np.issubdtype(value.dtype, np.integer):
                value = value.astype(np.int8, copy=False)
            return value
        if isinstance(value, torch.Tensor):
            # Preserve device but ensure integer dtype.
            if value.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                value = value.to(dtype=torch.int8)
            return value
        raise TypeError(f"Unsupported ss type: {type(value)}")

    @property
    def net(self) -> NN:
        """The neural network I belong to"""
        return self._net

    @net.setter
    def net(self, value: NN):
        if self._net is not None:
            raise ValueError("net cannot be changed after it has been set")
        self._net = value

    @property
    def ss(self) -> np.ndarray | torch.Tensor:
        """My sign sequence."""
        return self._ss

    @ss.setter
    def ss(self, value: np.ndarray | torch.Tensor) -> None:
        self._ss = self._coerce_ss_to_int(value)
        self._ss_np = None

    @property
    def ss_np(self) -> np.ndarray:
        """Cached NumPy representation of the sign sequence."""
        if self._ss_np is None:
            if isinstance(self._ss, np.ndarray):
                self._ss_np = self._ss
            elif isinstance(self._ss, torch.Tensor):
                self._ss_np = self._ss.detach().cpu().numpy().astype(np.int8, copy=False)
            else:
                raise TypeError(f"Unsupported ss type: {type(self._ss)}")
        assert self._ss_np is not None
        return self._ss_np

    @cached_property
    def zero_indices(self) -> np.ndarray:
        """Indices of sign sequence elements that are zero."""
        return np.flatnonzero(self.ss_np == 0)

    @cached_property
    def non_zero_indices(self) -> np.ndarray:
        """Indices of sign sequence elements that are not zero."""
        return np.flatnonzero(self.ss_np != 0)

    @property
    def hyperplanes(self) -> np.ndarray:
        """Hyperplanes that are safe to use for computation of polyhedron properties."""
        return self.halfspaces_np[self.zero_indices]

    @property
    def inequalities(self) -> np.ndarray:
        return self.halfspaces_np[self.non_zero_indices]

    @property
    def equalities(self) -> np.ndarray:
        """Rows of ``halfspaces_np`` corresponding to equality constraints (zeros in the sign sequence)."""
        return self.hyperplanes

    # def reparametrize_inequalities(self) -> np.ndarray:
    #     if not self.zero_indices:
    #         # No equalities: work in the original coordinates.
    #         dim = self.ambient_dim
    #         F = np.eye(dim)
    #         x0 = np.zeros((dim, 1))

    #         def remapper(x):
    #             return F @ x + x0

    #         return self.inequalities, remapper, F, x0

    #     eq_A = self.equalities[:, :-1]
    #     eq_b = self.equalities[:, -1:]
    #     # Solve eq_A @ x = -eq_b in the least-squares sense to obtain a point on
    #     # the affine subspace defined by the equalities. This works even when
    #     # eq_A is not square.
    #     x0, *_ = np.linalg.lstsq(eq_A, -eq_b, rcond=None)
    #     F = null_space(eq_A)
    #     A = self.inequalities[:, :-1] @ F
    #     b = self.inequalities[:, -1:] - self.inequalities[:, :-1] @ x0
    #     new_inequalities = np.concatenate([A, b], axis=1)

    #     def remapper(x):
    #         return F @ x + x0

    #     return new_inequalities, remapper, F, x0

    def get_interior_point(
        self,
        env: Any = None,
        max_radius: float | None = None,
    ) -> np.ndarray:
        """Get a point inside the polyhedron.

        Computes an interior point of the polyhedron.

        This method is "compute-only": it returns the interior point but does not
        mutate cached attributes like ``self._interior_point``.

        Args:
            env: Gurobi environment for optimization. If None, uses a cached
                environment. Defaults to None.
            max_radius: Maximum radius constraint for the search. If None, uses
                :data:`relucent.config.MAX_RADIUS`. Defaults to None.
            zero_indices: Indices of sign sequence elements that are zero (for
                lower-dimensional polyhedra). Defaults to None.

        Returns:
            np.ndarray: An interior point of the polyhedron.

        Raises:
            ValueError: If no interior point can be found.
        """
        max_radius = max_radius or cfg.MAX_RADIUS
        if self.finite is None:
            raise ValueError("Polyhedron is infeasible (empty).")
        if self._finite is True:
            assert self._center is not None
            interior_point = np.asarray(self._center).squeeze()
        else:
            env = env or get_env()
            interior_point = solve_radius(
                env,
                self.halfspaces_np[:],
                zero_indices=self.zero_indices,
                max_radius=max_radius,
            )[0]
            assert isinstance(interior_point, np.ndarray)
            interior_point = interior_point.squeeze()
        if interior_point is None:
            raise ValueError("Interior point not found. Check that the polyhedron is feasible and MAX_RADIUS is large enough.")
        # Match solve_radius: validate only non-degenerate halfspaces, with slack for
        # LP feasibility (__contains__ uses all rows + tighter TOL_HALFSPACE_CONTAINMENT).
        # if interior_point is not None:
        #     hs_check = _drop_degenerate_halfspaces(self.halfspaces_np[:])
        #     if hs_check.size > 0:
        #         pt = interior_point.reshape(1, -1)
        #         dists = pt @ hs_check[:, :-1].T + hs_check[:, -1]
        #         max_v = float(dists.max())
        #         if max_v > cfg.TOL_INTERIOR_VERIFY:
        #             raise ValueError(
        #                 f"Interior point invalid - {interior_point} not in {self}: "
        #                 f"max violation {max_v} (tol {cfg.TOL_INTERIOR_VERIFY})"
        #             )
        return interior_point

    def get_center_inradius(self, env: Any = None) -> tuple[np.ndarray | None, float | None]:
        """Get the Chebyshev center and inradius of the polyhedron.

        This method is "compute-only": it returns (center, inradius) but does
        not mutate cached attributes like ``self._center``, ``self._inradius``,
        or ``self._finite``.

        Args:
            env: Gurobi environment for optimization. If None, uses a cached
                environment. Defaults to None.

        Returns:
            tuple: ``(center, inradius)``. ``inradius`` is ``None`` if the halfspace
            system is infeasible (empty); ``center`` is ``None`` and ``inradius``
            is ``inf`` for nonempty unbounded polyhedra (with infinite Chebyshev formulation).
        """
        env = env or get_env()
        center, inradius = solve_radius(env, self.halfspaces_np[:], zero_indices=self.zero_indices)
        return center, inradius

    def get_bounded_halfspaces(self, bound: float, env: Any = None) -> np.ndarray:
        """Get halfspaces after adding bounding box constraints.

        Adds constraints that bound the space to a hypercube of radius ``bound``
        around the origin. Useful for plotting and visualization.

        Args:
            bound: Radius of the bounding hypercube.
            env: Gurobi environment for feasibility checking. If None, uses
                a cached environment. Defaults to None.

        Returns:
            np.ndarray: Halfspaces with bounding constraints added.

        Raises:
            ValueError: If the polyhedron does not intersect the bounded region.
        """
        bounds_lhs = np.eye(self.halfspaces_np.shape[1] - 1)
        bounds_rhs = -np.ones((self.halfspaces_np.shape[1] - 1, 1)) * bound
        halfspaces = np.vstack(
            (
                self.halfspaces_np,
                np.hstack((bounds_lhs, bounds_rhs)),
                np.hstack((-bounds_lhs, bounds_rhs)),
            )
        )
        # Drop any degenerate rows (near-zero normals). These can arise from dead/near-dead
        # constraints and are toxic for both Gurobi feasibility checks and Qhull.
        normals = halfspaces[:, :-1]
        norms = np.linalg.norm(normals, axis=1)
        deg = norms < cfg.TOL_HALFSPACE_NORMAL
        if np.any(deg):
            b = halfspaces[:, -1]
            # A degenerate row encodes 0 + b <= 0. If b>0 it's infeasible; raise rather than
            # silently widening the region by dropping it.
            if np.any(b[deg] > cfg.TOL_HALFSPACE_CONTAINMENT):
                bad = np.flatnonzero(deg & (b > cfg.TOL_HALFSPACE_CONTAINMENT)).tolist()
                raise ValueError(
                    "Degenerate halfspace(s) imply infeasibility after bounding; "
                    f"rows={bad}, tol_normal={cfg.TOL_HALFSPACE_NORMAL:g}"
                )
            halfspaces = halfspaces[~deg]
        env = env or get_env()
        feasible = solve_radius(env, halfspaces, max_radius=bound, zero_indices=self.zero_indices)[0] is not None
        if feasible:
            return halfspaces
        else:
            raise ValueError("Bounding box constraints are not feasible")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Polyhedron):
            return self.tag == other.tag  # and (self.ss == other.ss).all()
        elif isinstance(other, str):
            warnings.warn(
                "Comparing Polyhedron with string is deprecated and will be removed in a future version", stacklevel=2
            )
            return str(self) == other
        elif other is None:
            return False
        else:
            raise ValueError(f"Cannot compare Polyhedron with {type(other)}")

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.tag)
        return self._hash

    def get_neighbor(self, shi: int) -> "Polyhedron":
        """Get the neighbor polyhedron across the supporting hyperplane at index shi.

        Args:
            shi: Index of the supporting hyperplane to cross.

        Returns:
            Polyhedron: The neighbor polyhedron.
        """
        ss = self.ss_np.copy()
        if ss[0, shi] == 0:
            raise ValueError(f"SHI {shi} contains the polyhedron, cannot get neighbor")
        ss[0, shi] = -ss[0, shi]
        return Polyhedron(self.net, ss)

    def nflips(self, other: "Polyhedron") -> int:
        """Calculate the number of non-zero sign sequence elements that differ.

        Args:
            other: Another Polyhedron object to compare with.

        Returns:
            int: The number of sign sequence elements that differ.
        """
        return int((self.ss * other.ss == -1).sum().item())

    def is_face_of(self, other: "Polyhedron") -> bool:
        """Check if this polyhedron is a face of another polyhedron.

        Args:
            other: Another Polyhedron object to check against.

        Returns:
            bool: True if this polyhedron is a face of the other.
        """
        eq = (self * other).ss == other.ss
        return bool(cast(torch.Tensor, eq).all())

    def get_bounded_vertices(self, bound: float, qhull_mode: str | None = None) -> np.ndarray | None:
        """Get the vertices of the polyhedron within a bounding hypercube.

        Computes the vertices of the polyhedron after intersecting it with a
        hypercube of radius 'bound'. Primarily used for plotting and visualization.

        Args:
            bound: Radius of the bounding hypercube.

        Returns:
            np.ndarray or None: Array of vertex coordinates, or None if the
                polyhedron doesn't intersect the bounded region or computation fails.
        """

        if qhull_mode is None:
            qhull_mode = cfg.QHULL_MODE

        try:
            bounded_halfspaces = self.get_bounded_halfspaces(bound)
        except ValueError as e:
            w = RuntimeWarning(f"Error while computing bounded vertices: {e}")
            self.warnings.append(w)
            return None

        ## TODO: Move this logic into a separate method and reuse it here and in get_interior_point()
        n_rows_raw = bounded_halfspaces.shape[0]
        zero_idx = self.zero_indices[(self.zero_indices >= 0) & (self.zero_indices < n_rows_raw)]

        # Last-resort guard: Qhull requires non-degenerate halfspace normals.
        normals = bounded_halfspaces[:, :-1]
        norms = np.linalg.norm(normals, axis=1)
        deg = norms < cfg.TOL_HALFSPACE_NORMAL
        if np.any(deg):
            old_to_new = np.full(n_rows_raw, -1, dtype=np.intp)
            kept = np.flatnonzero(~deg)
            old_to_new[kept] = np.arange(kept.size, dtype=np.intp)
            bounded_halfspaces = bounded_halfspaces[~deg]
            if zero_idx.size > 0:
                zero_idx = old_to_new[zero_idx]
                zero_idx = zero_idx[zero_idx >= 0]

        # Recompute interior point
        int_point, _ = solve_radius(
            get_env(),
            bounded_halfspaces,
            max_radius=1000,
            zero_indices=zero_idx,
        )
        if int_point is None:
            raise ValueError("Interior point not found in bounded region")

        projected_halfspaces = bounded_halfspaces
        projected_int_point = np.asarray(int_point).reshape(-1)
        remap_vertices = lambda verts: verts  # noqa: E731

        # HalfspaceIntersection expects a full-dimensional interior. For k<d cells
        # (equalities induced by zero sign entries), project to nullspace coords.
        if zero_idx.size > 0:
            equalities = bounded_halfspaces[zero_idx]
            inequalities = bounded_halfspaces[~np.isin(np.arange(bounded_halfspaces.shape[0]), zero_idx)]
            eq_A = equalities[:, :-1]
            eq_b = equalities[:, -1:]
            x0, *_ = np.linalg.lstsq(eq_A, -eq_b, rcond=None)
            x0 = np.asarray(x0).reshape(-1, 1)

            _, s, vh = np.linalg.svd(eq_A, full_matrices=True)
            rank = int(np.sum(s > cfg.TOL_HALFSPACE_NORMAL))
            null_basis = vh[rank:, :].T

            if null_basis.shape[1] == 0:
                return x0.reshape(1, -1)

            A_red = inequalities[:, :-1] @ null_basis
            b_red = inequalities[:, :-1] @ x0 + inequalities[:, -1:]
            projected_halfspaces = np.hstack((A_red, b_red))

            z0, *_ = np.linalg.lstsq(null_basis, projected_int_point[:, None] - x0, rcond=None)
            projected_int_point = np.asarray(z0).reshape(-1)

            def _remap_vertices(verts: np.ndarray) -> np.ndarray:
                return (null_basis @ verts.T + x0).T

            remap_vertices = _remap_vertices

        reduced_dim = projected_halfspaces.shape[1] - 1
        if reduced_dim == 1:
            a = projected_halfspaces[:, 0]
            b = projected_halfspaces[:, 1]
            lower = -float("inf")
            upper = float("inf")
            tol_a = cfg.TOL_HALFSPACE_NORMAL
            tol_b = cfg.TOL_HALFSPACE_CONTAINMENT
            for ai, bi in zip(a, b, strict=True):
                if abs(ai) <= tol_a:
                    if bi > tol_b:
                        raise ValueError("Infeasible 1D projected halfspace system")
                    continue
                cutoff = -bi / ai
                if ai > 0:
                    upper = min(upper, cutoff)
                else:
                    lower = max(lower, cutoff)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper + tol_b:
                raise ValueError("Projected 1D intersection is empty or unbounded")
            reduced_vertices = np.array([[lower], [upper]], dtype=np.float64)
            vertices = remap_vertices(reduced_vertices)
            return np.unique(vertices, axis=0)
        try:
            # Debug aid for rare Qhull failures (e.g. QH6023: feasible point not clearly inside halfspace).
            # If any halfspace normal is ~0, Qhull can behave pathologically.
            try:
                normals = bounded_halfspaces[:, :-1]
                normal_norms = np.linalg.norm(normals, axis=1)
                tiny = np.flatnonzero(normal_norms < 1e-12)
                if tiny.size > 0:
                    print(
                        "[relucent] DEBUG: near-zero halfspace normals detected before HalfspaceIntersection\n"
                        f"  poly={self}\n"
                        f"  bound={bound}\n"
                        f"  int_point={np.asarray(int_point).ravel()}\n"
                        f"  tiny_indices={tiny.tolist()}\n"
                        f"  tiny_rows={bounded_halfspaces[tiny].tolist()}\n"
                        f"  normal_norms: min={float(normal_norms.min()):.3e}, "
                        f"median={float(np.median(normal_norms)):.3e}, max={float(normal_norms.max()):.3e}"
                    )
            except Exception:
                # Never fail geometry due to debug printing.
                pass
            with warnings.catch_warnings(record=True) as w:
                hs = HalfspaceIntersection(
                    projected_halfspaces,
                    projected_int_point,
                    qhull_options=None,
                )  # http://www.qhull.org/html/qh-optq.htm
            if w:
                msgs = "; ".join(str(wi.message) for wi in w)
                if qhull_mode == "IGNORE":
                    self.warnings.extend([RuntimeWarning(wi) for wi in w])
                if qhull_mode == "WARN_ALL":
                    warnings.warn(f"Halfspace intersection emitted warnings: {msgs}", stacklevel=2)
                elif qhull_mode == "HIGH_PRECISION":
                    raise ValueError(f"HalfspaceIntersection emitted warnings in HIGH_PRECISION mode: {msgs}")
                elif qhull_mode == "JITTERED":
                    with warnings.catch_warnings(record=True) as w2:
                        new_hs = HalfspaceIntersection(
                            projected_halfspaces,
                            projected_int_point,
                            # Triangulated output is approximately 1000 times more accurate than joggled input.
                            qhull_options="QJ",
                        )  # http://www.qhull.org/html/qh-optq.htm
                    if w2:
                        self.warnings.append(
                            RuntimeWarning(
                                "Recomputing HalfspaceIntersection with jitter option 'QJ' still had numerical problems"
                            )
                        )
                        self.warnings.extend([RuntimeWarning(wi) for wi in w2])
                    else:
                        ## Jittering solved the numerical problems
                        hs = new_hs
        except ValueError:
            raise  # Our HIGH_PRECISION raise - do not retry
        except Exception as e:
            if qhull_mode == "JITTERED":
                try:
                    hs = HalfspaceIntersection(
                        projected_halfspaces,
                        projected_int_point,
                        # Triangulated output is approximately 1000 times more accurate than joggled input.
                        qhull_options="QJ",
                    )  # http://www.qhull.org/html/qh-optq.htm
                    self.warnings.append(
                        RuntimeWarning(f"HalfspaceIntersection failed initially, succeeded with QJ retry: {e}")
                    )
                except Exception as e2:
                    raise ValueError(f"Error while computing halfspace intersection: {e}") from e2
            else:
                raise ValueError(f"Error while computing halfspace intersection: {e}") from e
        vertices = remap_vertices(hs.intersections)
        return vertices

    def _get_bounded_plot_geometry(
        self,
        bound: float,
    ) -> tuple[str, np.ndarray] | None:
        return bounded_plot_geometry(self, bound)

    def plot_cells(
        self,
        fill: str = "toself",
        showlegend: bool = False,
        bound: float | None = None,
        filled: bool = False,
        plot_halfspaces: bool = False,
        halfspace_shade: bool = True,
        **kwargs: Any,
    ) -> list[go.Scatter] | list[go.Mesh3d | go.Scatter3d]:
        """Plot this cell in input space (2D ``Scatter`` or 3D ``Mesh3d`` / ``Scatter3d`` traces).

        Chooses 2D vs 3D from :attr:`ambient_dim` (input-space dimension; typically matches
        :attr:`Complex.dim` when this polyhedron belongs to a complex).
        """
        if bound is None:
            bound = cfg.DEFAULT_PLOT_BOUND
        return plot_polyhedron(
            self,
            plot_mode="cells",
            fill=fill,
            showlegend=showlegend,
            bound=bound,
            filled=filled,
            plot_halfspaces=plot_halfspaces,
            halfspace_shade=halfspace_shade,
            **kwargs,
        )

    def plot_graph(
        self,
        fill: str = "toself",
        showlegend: bool = False,
        bound: float | None = None,
        project: float | None = None,
        **kwargs: Any,
    ) -> dict[str, go.Mesh3d | go.Scatter3d] | None:
        if bound is None:
            bound = cfg.DEFAULT_PLOT_BOUND
        return plot_polyhedron(
            self,
            plot_mode="graph",
            fill=fill,
            showlegend=showlegend,
            bound=bound,
            project=project,
            **kwargs,
        )

    def clean_data(self) -> None:
        """Clear cached data to reduce memory usage.

        Drops large tensors (halfspaces, ``W``, ``b``) and Qhull-derived geometry
        (``_hs``, vertices, hull, volume). Resets ``_attempted_compute_properties`` so
        :func:`~relucent.calculations.compute_properties` can run again. Keeps the sign
        sequence, interior-point cache, and lightweight Chebyshev classification
        (``finite``, ``center``, ``inradius``, and related flags) so search/plotting
        need not redo Gurobi Chebyshev solves after a cleanup pass.
        """
        self._halfspaces = None
        self._W = None
        self._b = None
        self._hs = None
        self._halfspaces_np = None
        self._vertices = None
        self._ch = None
        self._volume = None
        self._attempted_compute_properties = False
        # self._interior_point = None ## TODO: Does this slow down things?

    """
    All of the following properties are computed on the fly and cached
    """

    @property
    def vertices(self) -> np.ndarray | None:
        """Vertices of the polyhedron (not always reliable)."""
        if not self._attempted_compute_properties:
            compute_properties(self)
        return self._vertices

    @property
    def hs(self) -> HalfspaceIntersection:
        """Halfspace intersection object from scipy."""
        if not self._attempted_compute_properties:
            compute_properties(self)
        assert isinstance(self._hs, HalfspaceIntersection)
        return self._hs

    @property
    def ch(self) -> ConvexHull | None:
        """Convex hull of the polyhedron for finite polyhedra, or None if unbounded or computation fails."""
        if not self._attempted_compute_properties and self.finite:
            compute_properties(self)
        return self._ch

    @property
    def volume(self) -> float:
        """Volume of the polyhedron, infinity for unbounded polyhedra, or -1 if computation fails."""
        fin = self.finite
        if fin is False:
            self._volume = float("inf")
        elif fin is None:
            self._volume = -1.0
        elif not self._attempted_compute_properties:
            compute_properties(self)
        return self._volume if self._volume is not None else -1.0

    @cached_property  ## !! See if this works
    def tag(self) -> bytes:
        """Unique tag for this polyhedron, computed as a hashable representation of the sign sequence."""
        if self._tag is None:
            self._tag = encode_ss(self.ss_np)
        return self._tag

    @property
    def halfspaces(self) -> torch.Tensor | np.ndarray:
        """Halfspace representation of the polyhedron.

        Returns:
            torch.Tensor or np.ndarray: Array of shape (n_constraints, n_dim+1)
                where each row is [a1, a2, ..., ad, b] representing the
                constraint a^T x + b <= 0.
        """
        if self._halfspaces is None:
            halfspaces, W, b, num_dead_relus = get_hs(self)
            self._halfspaces = halfspaces
            self._W = W
            self._b = b
            self._halfspaces_np = None
            self._num_dead_relus = num_dead_relus
            assert isinstance(self._halfspaces, (torch.Tensor, np.ndarray))
            return self._halfspaces
        else:
            return self._halfspaces

    @property
    def halfspaces_np(self) -> np.ndarray:
        """Cached NumPy representation of halfspaces."""
        if self._halfspaces_np is None:
            hs = self.halfspaces
            if isinstance(hs, np.ndarray):
                self._halfspaces_np = hs
            elif isinstance(hs, torch.Tensor):
                self._halfspaces_np = hs.detach().cpu().numpy()
            else:
                raise TypeError(f"Unsupported halfspaces type: {type(hs)}")
            return self._halfspaces_np
        else:
            return self._halfspaces_np

    @property
    def W(self) -> torch.Tensor | np.ndarray:
        """Affine transformation matrix W such that the polyhedron maps to W*x + b.

        Returns:
            torch.Tensor or np.ndarray: Transformation matrix.
        """
        if self._W is None:
            halfspaces, W, b, num_dead_relus = get_hs(self)
            self._halfspaces = halfspaces
            self._W = W
            self._b = b
            self._halfspaces_np = None
            self._num_dead_relus = num_dead_relus
            assert isinstance(self._W, (torch.Tensor, np.ndarray))
            return self._W
        else:
            return self._W

    @property
    def b(self) -> torch.Tensor | np.ndarray:
        """Affine transformation bias vector such that the polyhedron maps to W*x + b.

        Returns:
            torch.Tensor or np.ndarray: Bias vector.
        """
        if self._b is None:
            halfspaces, W, b, num_dead_relus = get_hs(self)
            self._halfspaces = halfspaces
            self._W = W
            self._b = b
            self._halfspaces_np = None
            self._num_dead_relus = num_dead_relus
        assert isinstance(self._b, (torch.Tensor, np.ndarray))
        return self._b

    @property
    def num_dead_relus(self) -> int:
        """Number of dead ReLU neurons (neurons always outputting zero).

        Returns:
            int: Count of ReLU neurons that are always inactive for this polyhedron.
        """
        if self._num_dead_relus is None:
            force_numpy = isinstance(self._halfspaces, np.ndarray)
            halfspaces, W, b, num_dead_relus = get_hs(self, force_numpy=force_numpy)
            self._halfspaces = halfspaces
            self._W = W
            self._b = b
            self._halfspaces_np = None
            self._num_dead_relus = num_dead_relus
        assert self._num_dead_relus is not None
        return self._num_dead_relus

    @property
    def Wl2(self) -> float:
        """L2 norm of the transformation matrix W."""
        if self._Wl2 is None:
            if isinstance(self.W, torch.Tensor):
                self._Wl2 = float(torch.linalg.norm(self.W).item())
            elif isinstance(self.W, np.ndarray):
                self._Wl2 = float(np.linalg.norm(self.W))
            else:
                raise NotImplementedError
            return self._Wl2
        else:
            return self._Wl2

    @property
    def center(self) -> np.ndarray | None:
        """Chebyshev center of the polyhedron for finite polyhedra, or None for unbounded or infeasible."""
        if not self._finite_computed:
            _ = self.finite
        return self._center

    @property
    def inradius(self) -> float | None:
        """Inradius of the polyhedron, ``inf`` if unbounded feasible, ``None`` if infeasible."""
        if not self._finite_computed:
            _ = self.finite
        if self._finite is True:
            assert self._inradius is not None  # when bounded, get_center_inradius() has set it
            return self._inradius
        if self._finite is False:
            return float("inf")
        return None

    @property
    def finite(self) -> bool | None:
        """Whether the polyhedron is bounded: ``True``, unbounded nonempty ``False``, or empty ``None``."""
        if not self._finite_computed:
            center, inradius = self.get_center_inradius()
            self._center = center
            self._inradius = inradius
            if center is not None:
                self._finite = True
            elif inradius is None:
                self._finite = None
            elif inradius == float("inf"):
                self._finite = False
            else:
                raise ValueError(f"Unexpected Chebyshev result (center={center!r}, inradius={inradius!r})")
            self._finite_computed = True
            if self._finite is None:
                self._interior_point = None
        return self._finite

    @property
    def feasible(self) -> bool:
        """Whether the halfspace system is nonempty (``finite`` is not ``None``)."""
        return self.finite is not None

    @property
    def shis(self) -> list[int]:
        """Supporting halfspace indices (SHIs)."""
        if self._shis is None:
            result = get_shis(self)
            self._shis = result[0] if isinstance(result, tuple) else result
        assert isinstance(self._shis, list)
        return self._shis

    @property
    def num_shis(self) -> int:
        """Number of faces."""
        return len(self.shis)

    @property
    def num_faces(self) -> int:
        """Alias for Polyhedron.num_shis"""
        return self.num_shis

    @cached_property
    def interior_point(self) -> np.ndarray | None:
        """A point guaranteed to be inside the polyhedron."""
        if self._interior_point is None:
            self._interior_point = self.get_interior_point()
        return self._interior_point

    @property
    def interior_point_norm(self) -> float | None:
        """L2 norm of the interior point."""
        assert self.interior_point is not None
        if self._interior_point_norm is None:
            self._interior_point_norm = np.linalg.norm(self.interior_point).item()
        return self._interior_point_norm

    @property
    def codim(self) -> int:
        """Codimension of the polyhedron, equal to the number of zero sign sequence elements."""
        return np.sum(self.ss_np == 0)

    @property
    def ambient_dim(self) -> int:
        """Dimension of the ambient space."""
        return self.halfspaces.shape[1] - 1

    @property
    def dim(self) -> int:
        """Dimension of the polyhedron, equal to the dimension of the ambient space minus
        the number of zero sign sequence elements.
        """
        return self.ambient_dim - self.codim

    def __repr__(self) -> str:
        h = hashlib.blake2b(key=b"hi")
        h.update(self.tag)
        return h.hexdigest()[:8]

    def __contains__(self, point: np.ndarray | torch.Tensor) -> bool:
        """Check if a point (ndarray or Tensor) is contained in the polyhedron."""
        halfspaces = self.halfspaces_np if isinstance(point, np.ndarray) else self.halfspaces
        if isinstance(point, torch.Tensor) and isinstance(halfspaces, np.ndarray):
            point = point.detach().cpu().numpy().astype(halfspaces.dtype)
        point = point.reshape(1, -1)
        dists = point @ halfspaces[:, :-1].T + halfspaces[:, -1]
        return bool((dists <= cfg.TOL_HALFSPACE_CONTAINMENT).all().item())

    def __mul__(self, other: "Polyhedron") -> "Polyhedron":
        """Returns a new Polyhedron object based on sign sequence multiplication"""
        return Polyhedron(self.net, self.ss + other.ss * (self.ss == 0))

    """The following methods are used for pickling"""

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "_finite_computed"):
            # Legacy pickle: ``_finite`` was only True/False; None meant "not computed".
            self._finite_computed = self._finite is not None
        if self._finite is True and self._center is None:
            self._finite_computed = False

    def __getstate__(self) -> dict[str, Any]:
        return {
            "_tag": self.tag,
            "_hash": self._hash,
            "_finite_computed": self._finite_computed,
            "_finite": self._finite,
            "_center": self._center,  ## TODO: Does this slow down things? Careful when removing this!
            "_interior_point_norm": self._interior_point_norm,
            "_inradius": self._inradius,
            "_shis": self._shis,
            "_Wl2": self._Wl2,
            "_volume": self._volume,
            "_num_dead_relus": self._num_dead_relus,
            "_interior_point": self._interior_point,  ## TODO: Does this slow down things?
            "_attempted_compute_properties": self._attempted_compute_properties,
            "warnings": self.warnings,
        }

    def __reduce__(self) -> tuple[type["Polyhedron"], tuple[None, np.ndarray], dict[str, Any]]:
        return (
            Polyhedron,
            (None, self.ss_np),
            self.__getstate__(),
        )  # Control what gets saved, do not pickle the net
