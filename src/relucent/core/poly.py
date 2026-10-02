"""Polyhedron: a single linear region of a ReLU network in input space."""

import hashlib
import warnings
from collections.abc import Callable, Iterable
from functools import cached_property
from typing import Any, Literal, cast, overload

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull, HalfspaceIntersection

import relucent.config as cfg
from relucent._internal import rounding
from relucent._internal.gurobi import get_env
from relucent._internal.torch_compat import torch
from relucent.core.errors import AmbiguousGeometryError
from relucent.core.ss import encode_ss, flip_ss_at_shi
from relucent.geometry.calculations import (
    DegenerateHalfspaceInfeasibility,
    _affine_null_basis,
    _drop_degenerate_halfspaces_tracked,
    _remap_zero_indices,
    compute_properties,
    get_hs,
    get_shis,
    solve_radius,
)
from relucent.model.model import ReLUNetwork

__all__ = ["Polyhedron"]


class Polyhedron:
    """Represents a polyhedron (linear region) in d-dimensional space.

    Prefer creating instances via :meth:`~relucent.Complex.add_point`,
    :meth:`~relucent.Complex.add_ss`, or search methods — not direct construction.
    """

    def __init__(
        self,
        net: ReLUNetwork | Any,
        ss: np.ndarray | torch.Tensor,
        *,
        halfspaces: np.ndarray | torch.Tensor | None = None,
        halfspaces_err: np.ndarray | None = None,
        halfspaces_ss: np.ndarray | None = None,
        W: np.ndarray | torch.Tensor | None = None,
        b: np.ndarray | torch.Tensor | None = None,
        finite: bool | None = None,
        shis: list[int] | None = None,
        bound: float | None = None,
        ambient_dim: int | None = None,
        rows_data: bool | None = None,
        shis_strict: bool = False,
        covector_endpoint_shis: list[int] | None = None,
    ) -> None:
        """Create a Polyhedron object.

        Args:
            net: The network (a :class:`~relucent.model.model.ReLUNetwork`, or any model
                :func:`~relucent.model.convert_model.convert` accepts), or ``None`` for a
                cell given only by ``halfspaces``.
            ss: Sign sequence defining the polyhedron (values in {-1, 0, 1}).
            halfspaces: Precomputed rows ``[A | b]`` of ``Ax + b <= 0``.
            halfspaces_err: Float64 error scale of ``halfspaces`` (see
                :mod:`relucent._internal.rounding`). Rows given without it, and without
                ``halfspaces_ss``, are taken as exact data.
            halfspaces_ss: Sign sequence the ``halfspaces`` rows were composed for, when
                they come from another cell.
            W, b: Precomputed affine map of the network on this cell.
            finite: Precomputed boundedness, if known.
            shis: Precomputed supporting-hyperplane (facet) indices.
            bound: Box radius used for this cell's LPs.
            ambient_dim: Input dimension, when it can't be read from ``halfspaces`` yet.
            rows_data: Whether the rows are exact data rather than float64 compositions of
                the network's weights. Defaults to True for a net-less cell or rows given
                without provenance.
            shis_strict: Whether ``shis`` is this cell's certified facet list (rather than one
                assigned from a dual graph or a coface).
            covector_endpoint_shis: For a 1-cell, the SHIs whose faces are its verified
                endpoints.
        """
        if net is not None and not isinstance(net, ReLUNetwork):
            from relucent.model.convert_model import convert

            net = convert(net)
        self._net = net
        # Store the sign sequence with an integer dtype to ensure consistent
        # semantics across NumPy and PyTorch backends.
        self._ss = self._coerce_ss_to_int(ss)
        self._halfspaces: torch.Tensor | np.ndarray | None = halfspaces
        self._halfspaces_np: np.ndarray | None = None
        # Float64 error scale of the rows (see relucent._internal.rounding). Rows handed in
        # from elsewhere must bring theirs; rows this cell composes itself get it computed.
        self._halfspaces_err: np.ndarray | None = halfspaces_err
        self._halfspaces_own: bool = halfspaces is None
        # Rows supplied by a caller with no error scale and no source sign sequence are the
        # caller's data, taken as exact. (Every internal hand-off passes both.)
        self._halfspaces_user: bool = halfspaces is not None and halfspaces_err is None and halfspaces_ss is None
        # Whether the rows are exact data (a net-less cell, or a caller's rows) rather than rows
        # composed from a network in float64. Kept separately from ``_net``, which pickling drops.
        self._rows_data: bool = (net is None or self._halfspaces_user) if rows_data is None else rows_data
        # Sign sequence whose composed rows ``halfspaces`` are, when handed in from another cell
        # (so their error scale and exact values can be rebuilt; see relucent._internal).
        self._halfspaces_ss: np.ndarray | None = halfspaces_ss
        self._w: torch.Tensor | np.ndarray | None = W
        self._b: torch.Tensor | np.ndarray | None = b
        self._Wl2: float | None = None
        self._interior_point: np.ndarray | None = None
        self._interior_point_norm: float | None = None
        self._center: np.ndarray | None = None
        self._inradius: float | None = None
        self.bound = bound

        self._shis: list[int] | None = shis
        # Whether ``_shis`` is this cell's certified facet list (get_shis on it), as opposed to a
        # list assigned from the dual graph or a coface; certification recomputes only the latter.
        self._shis_strict: bool = shis_strict
        self._halfspace_intersection: HalfspaceIntersection | None = None
        self._convex_hull: ConvexHull | None = None
        self._finite: bool | None = finite
        self._finite_computed: bool = finite is not None
        self._vertices: np.ndarray | None = None
        self._volume: float | None = None
        self._covector_infeasible: bool = False
        self._covector_endpoint_shis: list[int] | None = covector_endpoint_shis

        self._hash: int | None = None
        self._tag: bytes | None = None

        self.warnings: list[Warning] = []

        # Cached NumPy representation of the sign sequence (if/when needed).
        self._ss_np: np.ndarray | None = None

        self._attempted_compute_properties: bool = False
        self._ambient_dim: int | None = ambient_dim

        self._apply_zero_cell_finite_hint()

    def _get_cached_halfspaces_np(self) -> np.ndarray | None:
        """Return the cached halfspace matrix as a NumPy array without triggering lazy computation.

        Unlike :attr:`halfspaces_np`, this method does not call :func:`get_hs` if the
        halfspace matrix has not yet been computed.  Returns ``None`` when halfspaces are
        unavailable, so callers can skip optional checks without paying the cost of a
        Gurobi LP.

        The result is written back to ``_halfspaces_np`` so repeated calls (e.g. once
        per SHI candidate in :meth:`is_shi_face_feasible`) pay the tensor-to-numpy
        conversion cost only once.
        """
        if self._halfspaces_np is not None:
            return self._halfspaces_np
        if self._halfspaces is not None:
            raw = self._halfspaces
            if isinstance(raw, np.ndarray):
                self._halfspaces_np = raw
                return raw
            if isinstance(raw, torch.Tensor):
                result = raw.detach().cpu().numpy()
                self._halfspaces_np = result
                return result
        return None

    @staticmethod
    def _halfspace_point(
        hs: np.ndarray,
        eq_indices: np.ndarray,
        errors: np.ndarray | None = None,
        exact_rows: Callable[[], list[list[Any]] | None] | None = None,
    ) -> np.ndarray | None:
        """The point where the ``eq_indices`` rows vanish, if it satisfies every other row.

        Solves ``hs[eq_indices, :-1] @ x = -hs[eq_indices, -1]`` and judges each other row at
        the solution against its own float64 error (``errors``, the rows' error scale from
        :mod:`relucent._internal.rounding`; exact data when omitted), widened by the solve error.

        Returns:
            The point, or ``None`` when the equality system has no solution or some other row
            is violated beyond its error.

        Raises:
            AmbiguousGeometryError: When the equality rows are dependent within their error, or
                another row vanishes at the point to within its error (the point lies on a further
                hyperplane, so whether it belongs to the cell cannot be decided).
        """
        sol = Polyhedron._halfspace_point_with_error(hs, eq_indices, errors, exact_rows)
        return None if sol is None else sol[0]

    @staticmethod
    def _halfspace_point_with_error(
        hs: np.ndarray,
        eq_indices: np.ndarray,
        errors: np.ndarray | None = None,
        exact_rows: Callable[[], list[list[Any]] | None] | None = None,
    ) -> tuple[np.ndarray, float] | None:
        """:meth:`_halfspace_point` plus a bound on the point's distance to the exact point.

        When float64 cannot decide and ``exact_rows`` can rebuild the rows exactly, the decision
        is made in exact arithmetic (:func:`relucent._internal.exact.exact_point`) instead.
        """
        try:
            return Polyhedron._halfspace_point_float(hs, eq_indices, errors)
        except AmbiguousGeometryError:
            rows = exact_rows() if exact_rows is not None else None
            if rows is None:
                raise
            from relucent._internal import exact

            return exact.exact_point(rows[: np.asarray(hs).shape[0]], np.asarray(eq_indices, dtype=np.intp))

    @staticmethod
    def _halfspace_point_float(
        hs: np.ndarray, eq_indices: np.ndarray, errors: np.ndarray | None = None
    ) -> tuple[np.ndarray, float] | None:
        """Float64 half of :meth:`_halfspace_point_with_error`; raises when it cannot decide."""
        H = np.asarray(hs, dtype=np.float64)
        E = rounding.exact_rows_error(H) if errors is None else np.asarray(errors, dtype=np.float64)
        eq_idx = np.asarray(eq_indices, dtype=np.intp)
        sol = rounding.solve_equalities(H, E, eq_idx)
        if sol is None:
            return None
        x, x_err = sol
        others = np.flatnonzero(~np.isin(np.arange(H.shape[0]), eq_idx))
        # Rows with an exactly zero normal are constants, decided when the cell was built.
        others = others[~np.all(H[others, :-1] == 0.0, axis=1)]
        if others.size == 0:
            return x, x_err
        side = rounding.classify_rows(H, E, x, x_err, rows=others)
        if np.any(side == 1):
            return None
        if np.all(side == -1):
            return x, x_err
        raise AmbiguousGeometryError(
            f"the point where rows {eq_idx.tolist()} vanish also lies on rows "
            + f"{others[side == 0].tolist()} to within float64 error"
        )

    def verify_vertex_covector(self, vertex_ss: np.ndarray) -> np.ndarray | None:
        """Recover and verify a vertex predicted from this top-dimensional coface.

        ``vertex_ss`` is this cell's sign sequence with more entries set to zero, ``ambient_dim`` in
        all. This cell's rows are those of a full-dimensional cell whose closure contains it. It is
        a vertex of this cell's closure exactly when the point where those rows vanish satisfies
        every other row of this cell strictly, and then its covector is ``vertex_ss`` (a unit's
        preactivation equals this cell's affine row on the whole closure). Both are decided in
        float64 against each row's own error by :meth:`_halfspace_point`, or exactly when that
        cannot decide and the rows can be rebuilt exactly. Rows with an exactly zero normal (dead
        units) are constants, not hyperplanes, and do not take part.

        Raises:
            AmbiguousGeometryError: When the decision falls inside float64 error.
        """
        ss = np.asarray(vertex_ss, dtype=np.int8)
        row = ss.ravel()
        zero_indices = np.flatnonzero(row == 0).astype(np.intp, copy=False)
        ambient_dim = int(self.ambient_dim)
        if zero_indices.size != ambient_dim:
            return None

        own = np.asarray(self.ss_np, dtype=np.int8).ravel()
        keep = row != 0
        if own.size != row.size or not np.array_equal(own[keep], row[keep]) or np.any(row[own == 0] != 0):
            raise ValueError(
                "verify_vertex_covector needs a covector obtained by zeroing entries of this cell's sign sequence"
            )
        hs = np.asarray(self.halfspaces_np, dtype=np.float64)[: row.size]
        err = np.asarray(self.halfspaces_err_np, dtype=np.float64)[: row.size]
        if hs.shape[0] < row.size:
            raise ValueError(f"Halfspace row count {hs.shape[0]} is smaller than sign-sequence length {row.size}.")
        if np.any(np.all(hs[zero_indices, :-1] == 0.0, axis=1)):
            return None  # a dead unit's constant row is not a hyperplane: no vertex on it
        return self._halfspace_point(hs, zero_indices, err, self._exact_rows)

    def _apply_zero_cell_finite_hint(self) -> None:
        """Mark 0-cells (vertices) as bounded without a Chebyshev LP."""
        hs = self._get_cached_halfspaces_np()
        if hs is None:
            return
        ambient = int(hs.shape[1] - 1)
        if self.codim != ambient:
            return
        self._finite = True
        self._finite_computed = True

    def _is_zero_cell(self) -> bool:
        return self.dim == 0

    def _interior_point_from_equalities(self) -> np.ndarray:
        """Recover the unique point of a 0-cell from equality rows (no Gurobi)."""
        hs = self.halfspaces_np
        zidx = self.zero_indices
        if zidx.size == 0:
            raise ValueError("0-cell has no equality (zero) constraints in its sign sequence")
        x = self._halfspace_point(hs, zidx, self.halfspaces_err_np, self._exact_rows)
        if x is None:
            raise ValueError("0-cell halfspace system is infeasible or candidate point violates active inequalities")
        return x

    def is_shi_face_feasible(self, shi: int) -> bool:
        """Check whether zeroing SHI ``shi`` produces a geometrically feasible face.

        For 1-cells (whose faces are 0-cells, i.e., points), the induced vertex is
        the unique solution of the current equality constraints extended by hyperplane
        ``shi``.  Feasibility reduces to linear-system consistency, which is checked
        cheaply via :meth:`_halfspace_point` (numpy lstsq + slack test, no LP).

        For all other dimensions this method returns ``True`` without checking.  Faces
        of k-cells with k > 1 are themselves polytopes; checking their feasibility
        requires finding an interior point via LP.  In practice the dual-graph /
        covector recovery path and construction-time
        :func:`relucent.graph.boundary._codim_one_face_kwargs` checks (boundary
        faces) prevent phantom cells at dimensions > 0, so this check is not needed there.

        **Invariant**: every 1-cell that passes through boundary-face construction
        (via :func:`relucent.graph.boundary._codim_one_face_kwargs`) is constructed
        with ``halfspaces`` set from its coface, so halfspaces are always available.
        A ``ValueError`` is raised when this invariant is violated (i.e. a 1-cell is
        encountered without cached halfspaces), which indicates the cell was
        constructed outside the normal boundary-face pipeline.

        Note:
            The ``dim != 1`` early-return is mathematically fundamental.  A 0-cell is a
            point and its feasibility is equivalent to the consistency of a linear
            system, the only dimension where a cheap (non-LP) check is exact.  All
            callers should delegate this check to this method rather than reproducing the
            dimension guard elsewhere.

        Raises:
            ValueError: If ``self.dim == 1`` and no halfspaces are cached.
        """
        if self.dim != 1:
            return True
        hs = self._get_cached_halfspaces_np()
        if hs is None:
            raise ValueError(
                f"Polyhedron {self!r} is a 1-cell but has no cached halfspaces. "
                + "Cells entering the contraction pipeline via _codim_one_face_kwargs "
                + "always receive halfspaces from their coface; missing halfspaces "
                + "indicates the cell was constructed outside the normal pipeline."
            )
        active = np.array(list(self.zero_indices) + [shi], dtype=np.intp)
        return self._halfspace_point(hs, active, self.halfspaces_err_np, self._exact_rows) is not None

    def _coerce_ss_to_int(self, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Return an integer-typed sign sequence (values in {-1, 0, 1})."""
        if isinstance(value, np.ndarray):
            # ``dtype.kind in "iu"`` is ~50x faster than ``np.issubdtype`` and
            # this path is hit once per Polyhedron construction (hot in e.g.
            # ``Complex.recover_from_dual_graph``).
            if value.dtype.kind not in "iu":
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
    def net(self) -> ReLUNetwork:
        """The :class:`~relucent.model.model.ReLUNetwork` this polyhedron belongs to.

        Raises ``ValueError`` if unset; it can be set once and not changed afterwards.
        """
        if self._net is None:
            raise ValueError("Polyhedron has no associated network.")
        return self._net

    @net.setter
    def net(self, value: ReLUNetwork):
        if self._net is not None:
            raise ValueError("net cannot be changed after it has been set")
        self._net = value

    @property
    def ss(self) -> np.ndarray | torch.Tensor:
        """My sign sequence (entries in {-1, 0, 1}, one per ReLU unit). Assigning it clears derived caches."""
        return self._ss

    @ss.setter
    def ss(self, value: np.ndarray | torch.Tensor) -> None:
        self._ss = self._coerce_ss_to_int(value)
        self._clear_ss_derived_caches()

    def _clear_ss_derived_caches(self) -> None:
        """Drop sign-sequence-derived caches after ``ss`` changes."""
        self._ss_np = None
        for name in ("codim", "dim", "zero_indices", "non_zero_indices"):
            self.__dict__.pop(name, None)

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
    def inequalities(self) -> np.ndarray:
        """Rows of ``halfspaces_np`` for the nonzero sign-sequence entries (the strict inequality constraints)."""
        return self.halfspaces_np[self.non_zero_indices]

    @property
    def equalities(self) -> np.ndarray:
        """Rows of ``halfspaces_np`` for the zero sign-sequence entries (the hyperplanes this cell lies on)."""
        return self.halfspaces_np[self.zero_indices]

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
        if self._is_zero_cell():
            return self._interior_point_from_equalities()
        if self.finite is None:
            raise ValueError("Polyhedron is infeasible (empty).")
        # ``finite=True`` may be supplied at construction (e.g. propagated from cofaces)
        # without ever running the Chebyshev solve; in that case ``_center`` is unset.
        if self._finite is True and self._center is not None:
            interior_point = np.asarray(self._center).squeeze()
        else:
            env = env or get_env()
            interior_point = solve_radius(
                env,
                self.halfspaces_np[:],
                zero_indices=self.zero_indices,
                max_radius=max_radius,
                errors=self.halfspaces_err_np,
            )[0]
            assert isinstance(interior_point, np.ndarray)
            interior_point = interior_point.squeeze()
        if interior_point is None:
            raise ValueError("Interior point not found. Check that the polyhedron is feasible and MAX_RADIUS is large enough.")
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
        if self._is_zero_cell():
            pt = self._interior_point_from_equalities()
            return pt.reshape(-1, 1), 0.0
        env = env or get_env()
        center, inradius = solve_radius(
            env,
            self.halfspaces_np[:],
            zero_indices=self.zero_indices,
            errors=self.halfspaces_err_np,
        )
        return center, inradius

    def _halfspaces_with_bounding_box(self, bound: float, env: Any = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Stack axis-aligned bounds, drop degenerates, and check feasibility.

        Returns ``(halfspaces, zero_indices)`` where ``zero_indices`` is remapped
        through degenerate-row removal. Callers must use that remapped array with
        the returned halfspaces — the raw :attr:`zero_indices` point into the
        pre-drop stack and would otherwise land on a bounding-box row.
        """
        halfspaces, zero_indices, _ = self._halfspaces_with_bounding_box_err(bound, env=env)
        return halfspaces, zero_indices

    def _halfspaces_with_bounding_box_err(
        self, bound: float, env: Any = None
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """:meth:`_halfspaces_with_bounding_box` plus the float64 error scale of the returned rows."""
        dim = self.halfspaces_np.shape[1] - 1
        bounds_lhs = np.eye(dim)
        bounds_rhs = -np.ones((dim, 1)) * bound
        halfspaces = np.vstack(
            (
                self.halfspaces_np,
                np.hstack((bounds_lhs, bounds_rhs)),
                np.hstack((-bounds_lhs, bounds_rhs)),
            )
        )
        errors = np.vstack((self.halfspaces_err_np, rounding.box_rows_error(dim, bound)))
        zero_indices: np.ndarray | None = np.asarray(self.zero_indices, dtype=np.intp)
        if zero_indices.size == 0:
            zero_indices = None
        # Drop constant rows (exactly zero normal); toxic for Gurobi / Qhull.
        try:
            kept_hs, old_to_new = _drop_degenerate_halfspaces_tracked(halfspaces, errors=errors)
        except DegenerateHalfspaceInfeasibility as error:
            raise ValueError(f"Degenerate halfspace(s) imply infeasibility after bounding: {error}") from error
        if kept_hs.shape[0] != halfspaces.shape[0]:
            zero_indices = _remap_zero_indices(zero_indices, old_to_new)
            errors = errors[old_to_new >= 0]
            halfspaces = kept_hs
        env = env or get_env()
        center, _ = solve_radius(env, halfspaces, max_radius=bound, zero_indices=zero_indices, errors=errors)
        if center is None:
            raise ValueError("Bounding box constraints are not feasible")
        return halfspaces, zero_indices, errors

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
        halfspaces, _ = self._halfspaces_with_bounding_box(bound, env=env)
        return halfspaces

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Polyhedron):
            return self.tag == other.tag
        if other is None:
            return False
        return NotImplemented

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.tag)
        return self._hash

    def _same_rows_kwargs(self) -> dict[str, Any]:
        """Constructor kwargs for a net-less cell that shares this cell's rows (and their provenance)."""
        return {
            "halfspaces": self._halfspaces,
            "halfspaces_err": self.halfspaces_err_np,
            "rows_data": self._rows_data,
            "bound": self.bound,
        }

    def get_neighbor(self, shi: int) -> "Polyhedron":
        """Get the neighbor polyhedron across the supporting hyperplane at index shi.

        Args:
            shi: Index of the supporting hyperplane to cross.

        Returns:
            Polyhedron: The neighbor polyhedron.
        """
        if self.ss_np.ravel()[shi] == 0:
            raise ValueError(f"SHI {shi} contains the polyhedron, cannot get neighbor")
        ss = flip_ss_at_shi(self.ss_np, shi)
        # If this Polyhedron was constructed directly from explicit halfspaces (no net),
        # preserve them when flipping an inequality sign. The feasible region in input
        # space is the same; only the sign sequence label changes.
        if self._net is None and self._halfspaces is not None:
            return Polyhedron(None, ss, **self._same_rows_kwargs())
        return Polyhedron(self._net, ss)

    def get_face(self, shi: int) -> "Polyhedron":
        """Get the face of the polyhedron across the supporting hyperplane at index shi.

        Args:
            shi: Index of the supporting hyperplane to cross.

        Returns:
            Polyhedron: The face polyhedron.
        """
        ss = self.ss_np.copy()
        ss[0, shi] = 0
        # Don't reuse cached geometry (halfspaces/W/b/shis) from the parent: zeroing a sign
        # changes which constraints are active, and stale caches can give an inconsistent
        # complex (and wrong Betti numbers).
        #
        # Exception: with no network and a polyhedron built from explicit halfspaces, those
        # halfspaces define the geometry. The face is the same system with one more
        # constraint treated as an equality (via the zero sign entry).
        if self._net is None and self._halfspaces is not None:
            return Polyhedron(None, ss, **self._same_rows_kwargs())
        return Polyhedron(self._net, ss, bound=self.bound)

    def get_face_by_shis(self, shis: Iterable[int]) -> "Polyhedron":
        """Get a (possibly higher-codimension) face by zeroing multiple SHIs.

        This is a purely combinatorial operation on the sign sequence: for each
        ``shi`` in ``shis``, we set ``ss[0, shi] = 0`` and construct the resulting
        Polyhedron. No cached geometry is reused (same rationale as :meth:`get_face`).
        """
        ss = self.ss_np.copy()
        for shi in shis:
            ss[0, int(shi)] = 0
        if self._net is None and self._halfspaces is not None:
            return Polyhedron(None, ss, **self._same_rows_kwargs())
        return Polyhedron(self._net, ss, bound=self.bound)

    @property
    def faces(self) -> list["Polyhedron"]:
        """All codimension-1 faces of the polyhedron."""
        return [self.get_face(shi) for shi in self.shis]

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
        if not self.feasible or not other.feasible:
            return False
        eq = (self * other).ss == other.ss
        if isinstance(eq, np.ndarray):
            return bool(eq.all())
        return bool(cast(torch.Tensor, eq).all())

    def get_bounded_vertices(self, bound: float, qhull_mode: str | None = None) -> np.ndarray | None:
        """Get the vertices of the polyhedron within a bounding hypercube.

        Computes the vertices of the polyhedron after intersecting it with a
        hypercube of radius 'bound'. Primarily used for plotting and visualization.

        Args:
            bound: Radius of the bounding hypercube.
            qhull_mode: Qhull numerical-warning handling strategy. Defaults to
                :data:`relucent.config.QHULL_MODE`.

        Returns:
            np.ndarray or None: Array of vertex coordinates, or None if the
                polyhedron doesn't intersect the bounded region or computation fails.
        """

        if qhull_mode is None:
            qhull_mode = cfg.QHULL_MODE

        try:
            bounded_halfspaces, zero_idx, bounded_err = self._halfspaces_with_bounding_box_err(bound)
        except ValueError as e:
            w = RuntimeWarning(f"Error while computing bounded vertices: {e}")
            self.warnings.append(w)
            return None

        if zero_idx is None:
            zero_idx = np.array([], dtype=np.intp)

        # Recompute interior point (equalities already remapped with the halfspaces)
        int_point, _ = solve_radius(
            get_env(),
            bounded_halfspaces,
            max_radius=1000,
            zero_indices=zero_idx if zero_idx.size > 0 else None,
            errors=bounded_err,
        )
        if int_point is None:
            raise ValueError("Interior point not found in bounded region")

        projected_halfspaces = bounded_halfspaces
        projected_int_point = np.asarray(int_point).reshape(-1)

        def remap_vertices(verts: np.ndarray) -> np.ndarray:
            return verts

        # HalfspaceIntersection expects a full-dimensional interior. For k<d cells
        # (equalities induced by zero sign entries), project to nullspace coords.
        if zero_idx.size > 0:
            x0, null_basis, ineq_mask = _affine_null_basis(bounded_halfspaces, zero_idx, errors=bounded_err)

            if null_basis.shape[1] == 0:
                return x0.reshape(1, -1)

            inequalities = bounded_halfspaces[ineq_mask]
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
            for ai, bi in zip(a, b, strict=True):
                if ai == 0.0:
                    # A row parallel to the segment; the segment was certified nonempty.
                    if bi > 0.0:
                        raise ValueError("Infeasible 1D projected halfspace system")
                    continue
                cutoff = -bi / ai
                if ai > 0:
                    upper = min(upper, cutoff)
                else:
                    lower = max(lower, cutoff)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                raise ValueError("Projected 1D intersection is empty or unbounded")
            reduced_vertices = np.array([[lower], [upper]], dtype=np.float64)
            vertices = remap_vertices(reduced_vertices)
            return np.unique(vertices, axis=0)
        try:
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
        from relucent.vis import bounded_plot_geometry

        return bounded_plot_geometry(self, bound)

    @overload
    def plot(
        self, plot_mode: Literal["cells"] = "cells", **kwargs: Any
    ) -> list[go.Scatter] | list[go.Mesh3d | go.Scatter3d]: ...

    @overload
    def plot(self, plot_mode: Literal["graph"], **kwargs: Any) -> dict[str, go.Mesh3d | go.Scatter3d] | None: ...

    def plot(
        self, plot_mode: Literal["cells", "graph"] = "cells", **kwargs: Any
    ) -> list[go.Scatter] | list[go.Mesh3d | go.Scatter3d] | dict[str, go.Mesh3d | go.Scatter3d] | None:
        """Plotly traces for this cell: ``"cells"`` in input space (2D or 3D), or ``"graph"``,
        a 2D cell lifted through the network.

        Keyword arguments go to :func:`relucent.vis.plot_polyhedron` (``bound`` defaults to
        :data:`relucent.config.DEFAULT_PLOT_BOUND`). For a 3D cell, options that only apply to
        2D traces (``fill``, ``plot_halfspaces``, ...) are ignored.
        """
        from relucent.vis import _POLY_CELLS_3D_EXCLUDE, plot_polyhedron

        if plot_mode == "cells" and self.ambient_dim == 3:
            kwargs = {k: v for k, v in kwargs.items() if k not in _POLY_CELLS_3D_EXCLUDE}
        return plot_polyhedron(self, plot_mode=plot_mode, **kwargs)

    def get_geometry(
        self,
        properties: Iterable[str],
        env: Any = None,
    ) -> None:
        """Compute selected cached properties.

        Args:
            properties: Iterable of cache/property names to ensure are computed.
                Supported names include ``"halfspaces"``, ``"W"``, ``"b"``,
                ``"finite"``, ``"center"``,
                ``"inradius"``, ``"interior_point"``, ``"interior_point_norm"``,
                ``"Wl2"``, ``"volume"``, ``"vertices"``, ``"convex_hull"``, and
                ``"halfspace_intersection"``.
            env: Optional Gurobi environment used for interior-point/feasibility
                solves when relevant.
        """
        requested = {name.strip() for name in properties}
        if not requested:
            return

        geometry_aliases = {"halfspace_intersection", "vertices", "convex_hull", "volume"}
        if "halfspaces_np" in requested:
            requested.add("halfspaces")
        if requested & geometry_aliases:
            requested.add("interior_point")

        if requested & {"halfspaces", "W", "b"}:
            _ = self.halfspaces
        if "finite" in requested or "center" in requested or "inradius" in requested:
            _ = self.finite
        if "center" in requested:
            _ = self.center
        if "inradius" in requested:
            _ = self.inradius
        if "interior_point" in requested and self._interior_point is None:
            self._interior_point = self.get_interior_point(env=env)
        if "interior_point_norm" in requested:
            if self.interior_point is not None:
                self._interior_point_norm = np.linalg.norm(self.interior_point).item()
            else:
                self._interior_point_norm = float("inf")
        if "Wl2" in requested:
            _ = self.Wl2
        if requested & geometry_aliases:
            self._compute_qhull_geometry()

    def _compute_qhull_geometry(self, qhull_mode: str | None = None) -> None:
        """Compute Qhull-derived geometry caches (halfspace intersection, vertices, convex hull, volume)."""
        compute_properties(self, qhull_mode=qhull_mode)

    def _ensure_affine_data(self, *, force_numpy: bool = False) -> None:
        """Populate halfspace and affine-map caches via :func:`~relucent.geometry.calculations.get_hs`."""
        halfspaces, w, b = get_hs(self, force_numpy=force_numpy)
        self._halfspaces = halfspaces
        self._w = w
        self._b = b
        self._halfspaces_np = None
        self._halfspaces_err = None
        self._halfspaces_own = True
        self._halfspaces_user = False
        self._rows_data = False

    @property
    def vertices(self) -> np.ndarray | None:
        """Vertices of the polyhedron (not always reliable)."""
        if not self._attempted_compute_properties:
            self._compute_qhull_geometry()
        return self._vertices

    @property
    def halfspace_intersection(self) -> HalfspaceIntersection:
        """SciPy :class:`~scipy.spatial.HalfspaceIntersection` of this cell's halfspaces."""
        if not self._attempted_compute_properties:
            self._compute_qhull_geometry()
        assert isinstance(self._halfspace_intersection, HalfspaceIntersection)
        return self._halfspace_intersection

    @property
    def convex_hull(self) -> ConvexHull | None:
        """SciPy :class:`~scipy.spatial.ConvexHull` of a bounded cell; ``None`` if unbounded or if Qhull fails."""
        if not self._attempted_compute_properties and self.finite:
            self._compute_qhull_geometry()
        return self._convex_hull

    @property
    def volume(self) -> float:
        """Volume of the polyhedron, infinity for unbounded polyhedra, or -1 if computation fails."""
        fin = self.finite
        if fin is False:
            self._volume = float("inf")
        elif fin is None:
            self._volume = -1.0
        elif not self._attempted_compute_properties:
            self._compute_qhull_geometry()
        return self._volume if self._volume is not None else -1.0

    @cached_property
    def tag(self) -> bytes:
        """Hashable bytes representation of the sign sequence; stable (possibly non-unique) identity key."""
        return encode_ss(self.ss_np)

    @property
    def halfspaces(self) -> torch.Tensor | np.ndarray:
        """Halfspace representation of the polyhedron.

        Returns:
            torch.Tensor or np.ndarray: Array of shape (n_constraints, n_dim+1)
                where each row is [a1, a2, ..., ad, b] representing the
                constraint a^T x + b <= 0.
        """
        if self._halfspaces is None:
            if self._halfspaces_np is not None:
                self._halfspaces = self._halfspaces_np
            else:
                self._ensure_affine_data()
        assert isinstance(self._halfspaces, (torch.Tensor, np.ndarray))
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

    @property
    def halfspaces_err_np(self) -> np.ndarray:
        """Float64 error scale of :attr:`halfspaces_np`, row for row (see :mod:`relucent._internal.rounding`)."""
        if self._halfspaces_err is None:
            hs = self.halfspaces_np
            if self._rows_data:
                self._halfspaces_err = rounding.exact_rows_error(hs)
            elif self._net is None:
                raise ValueError(f"Polyhedron {self!r} holds network rows without the network or their error scale")
            elif self._halfspaces_own:
                self._halfspaces_err = rounding.halfspaces_error_for_ss(self._net, self.ss_np)
            elif self._halfspaces_ss is not None:
                self._halfspaces_err = rounding.halfspaces_error_for_ss(self._net, self._halfspaces_ss)
            else:
                # Rows cached without their error scale (pickled before scales were tracked).
                # They may be a coface's, so only use this cell's own scale if they match.
                try:
                    own, *_ = get_hs(Polyhedron(self._net, self.ss_np), force_numpy=True)
                except AssertionError:
                    own = None
                own_np = None if own is None else own if isinstance(own, np.ndarray) else own.detach().cpu().numpy()
                if own_np is None or own_np.shape != hs.shape or not np.array_equal(own_np, hs):
                    raise ValueError(
                        f"Polyhedron {self!r} was given halfspaces from another sign sequence "
                        + "without their error scale; pass halfspaces_err alongside halfspaces."
                    )
                self._halfspaces_err = rounding.halfspaces_error_for_ss(self._net, self.ss_np)
            if self._halfspaces_err.shape != hs.shape:
                raise ValueError(f"halfspaces_err shape {self._halfspaces_err.shape} != halfspaces {hs.shape}")
        return self._halfspaces_err

    def _exact_rows(self) -> list[list[Any]] | None:
        """This cell's rows in exact rational arithmetic, or None when they cannot be rebuilt."""
        from fractions import Fraction

        from relucent._internal import exact

        if self._rows_data:
            # Rows given as data are exact as stored (their error scale is evaluation-only).
            return [[Fraction(float(v)) for v in r] for r in self.halfspaces_np]
        ss = self.halfspaces_rows_ss
        # No network attached (e.g. in a worker), or too wide to rebuild exactly in reasonable
        # time: callers raise instead.
        if self._net is None or ss is None or not exact.exact_rows_affordable(self._net):
            return None
        return exact.exact_rows_for_ss(self._net, ss)

    @property
    def halfspaces_rows_ss(self) -> np.ndarray | None:
        """Sign sequence that generated :attr:`halfspaces_np`, when known."""
        if self._rows_data:
            return None
        if self._halfspaces_own:
            return self.ss_np
        return self._halfspaces_ss

    @property
    def W(self) -> torch.Tensor | np.ndarray:
        """Affine transformation matrix W such that the polyhedron maps to W*x + b.

        Returns:
            torch.Tensor or np.ndarray: Transformation matrix.
        """
        if self._w is None:
            self._ensure_affine_data()
        assert isinstance(self._w, (torch.Tensor, np.ndarray))
        return self._w

    @property
    def b(self) -> torch.Tensor | np.ndarray:
        """Affine transformation bias vector such that the polyhedron maps to W*x + b.

        Returns:
            torch.Tensor or np.ndarray: Bias vector.
        """
        if self._b is None:
            self._ensure_affine_data()
        assert isinstance(self._b, (torch.Tensor, np.ndarray))
        return self._b

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

    def _ensure_chebyshev_center(self, env: Any = None) -> None:
        """Populate ``_center`` / ``_inradius`` for bounded cells (optional; not used by meta-graph)."""
        if self._is_zero_cell():
            if self._center is None:
                pt = self._interior_point_from_equalities()
                self._center = pt.reshape(-1, 1)
                self._inradius = 0.0
            return
        if self._finite is not True or (self._center is not None and self._inradius is not None):
            return
        env = env or get_env()
        center, inradius = self.get_center_inradius(env=env)
        self._center = center
        self._inradius = inradius

    @property
    def center(self) -> np.ndarray | None:
        """Chebyshev center of the polyhedron for finite polyhedra, or None for unbounded or infeasible."""
        if not self._finite_computed:
            _ = self.finite
        elif self._finite is True and self._center is None:
            self._ensure_chebyshev_center()  # TODO: Make sure this only happens once
        return self._center

    @property
    def inradius(self) -> float | None:
        """Inradius of the polyhedron, ``inf`` if unbounded feasible, ``None`` if infeasible."""
        if not self._finite_computed:
            _ = self.finite
        elif self._finite is True and self._inradius is None:
            self._ensure_chebyshev_center()
        if self._finite is True:
            if cfg.CAREFUL_MODE:
                assert self._inradius is not None  # when bounded, get_center_inradius() has set it
            return self._inradius
        if self._finite is False:
            return float("inf")
        return None

    @property
    def finite(self) -> bool | None:
        """Whether the polyhedron is bounded: ``True``, unbounded nonempty ``False``, or empty ``None``."""
        # 0-cells are vertices; bounded by definition. Never run Chebyshev on them.
        if self._is_zero_cell():
            self._finite = True
            self._finite_computed = True
            if self._center is None:
                pt = self._interior_point_from_equalities()
                self._center = pt.reshape(-1, 1)
                self._inradius = 0.0
            return True
        if self._finite_computed:
            return self._finite
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
            bound = self.bound
            if bound is None and self._net is not None:
                from relucent._internal.network_scale import default_polyhedron_bound

                bound = default_polyhedron_bound(self._net)
            elif bound is None:
                bound = cfg.DEFAULT_SEARCH_BOUND
            self._shis = get_shis(self, bound=float(bound))
            self._shis_strict = True
        assert isinstance(self._shis, list)
        return self._shis

    @property
    def num_shis(self) -> int:
        """Number of faces."""
        return len(self.shis)

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

    @cached_property
    def codim(self) -> int:
        """Codimension of the polyhedron, equal to the number of zero sign sequence elements."""
        return int(np.count_nonzero(self.ss_np == 0))

    @property
    def ambient_dim(self) -> int:
        """Dimension of the ambient space."""
        if self._ambient_dim is not None:
            return self._ambient_dim
        return self.halfspaces.shape[1] - 1

    @cached_property
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
        """Check if a point (ndarray or Tensor) is in the closed polyhedron.

        Each row is judged at the point against its own float64 error. The rows of this cell's
        zero entries (its equalities, for a lower-dimensional cell) hold when they are within that
        error of zero: no float64 point lies exactly on a face's affine hull, so a point on the
        face to within rounding is on the face.

        Raises:
            AmbiguousGeometryError: If the point lies on an inequality row to within that row's
                error and no row is clearly violated.
        """
        if not isinstance(point, np.ndarray):
            point = cast(Any, point).detach().cpu().numpy()
        x = np.asarray(point, dtype=np.float64).reshape(-1)
        side = rounding.classify_rows(self.halfspaces_np, self.halfspaces_err_np, x, 0.0)
        if np.any(side == 1):
            return False
        undecided = side == 0
        zero_idx = np.asarray(self.zero_indices, dtype=np.intp)
        undecided[zero_idx[zero_idx < undecided.size]] = False
        if not np.any(undecided):
            return True
        raise AmbiguousGeometryError(
            f"point lies on rows {np.flatnonzero(undecided).tolist()} of {self!r} to within float64 error"
        )

    def __mul__(self, other: "Polyhedron") -> "Polyhedron":
        """Returns a new Polyhedron object based on sign sequence multiplication"""
        return Polyhedron(self._net, self.ss + other.ss * (self.ss == 0))

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "_halfspaces_own" not in state and state.get("_halfspaces_np") is not None:
            # Pickled before error scales were tracked: the cached rows may be a coface's.
            self._halfspaces_own = False
        self.__dict__.setdefault("_halfspaces_ss", None)
        self.__dict__.setdefault("_halfspaces_user", False)
        self.__dict__.setdefault("_rows_data", False)
        self.__dict__.setdefault("_halfspaces_err", None)
        self.__dict__.setdefault("_halfspaces_own", self.__dict__.get("_halfspaces_np") is None)
        if self._finite is True and self._center is None:
            self._finite_computed = False

    def __getstate__(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "_tag": self.tag,
            "_hash": self._hash,
            "_finite_computed": self._finite_computed,
            "_finite": self._finite,
            "_center": self._center,
            "_interior_point_norm": self._interior_point_norm,
            "_inradius": self._inradius,
            "_shis": self._shis,
            "_shis_strict": self._shis_strict,
            "_Wl2": self._Wl2,
            "_volume": self._volume,
            "_interior_point": self._interior_point,
            "_attempted_compute_properties": self._attempted_compute_properties,
            "_covector_infeasible": self._covector_infeasible,
            "_covector_endpoint_shis": self._covector_endpoint_shis,
            "warnings": self.warnings,
            "codim": self.codim,
            "dim": self.dim,
            "_ambient_dim": self._ambient_dim,
            "_halfspaces_np": self._halfspaces_np,
            "_halfspaces_err": self._pickled_halfspaces_err(),
            "_halfspaces_own": self._halfspaces_own,
            "_halfspaces_user": self._halfspaces_user,
            "_rows_data": self._rows_data,
            "_halfspaces_ss": self._halfspaces_ss,
            "_w": self._w,
            "_b": self._b,
            "bound": self.bound,
        }
        return state

    def _pickled_halfspaces_err(self) -> np.ndarray | None:
        """Error scale to pickle with the rows: only when it cannot be rebuilt after unpickling.

        Rows that are data, the cell's own composition, or a known sign sequence's composition get
        their scale recomputed from the reattached network, so it need not double the payload.
        """
        if self._rows_data or self._halfspaces_own or self._halfspaces_ss is not None:
            return None
        if self._halfspaces_err is None and self._halfspaces_np is not None and self._net is not None:
            return self.halfspaces_err_np
        return self._halfspaces_err

    def __reduce__(self) -> tuple[type["Polyhedron"], tuple[None, np.ndarray], dict[str, Any]]:
        return (
            Polyhedron,
            (None, self.ss_np),
            self.__getstate__(),
        )  # Control what gets saved, do not pickle the net
