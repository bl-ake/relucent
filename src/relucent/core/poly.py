"""Polyhedron: a single linear region of a ReLU network in input space."""

import hashlib
import warnings
from collections.abc import Callable, Iterable
from functools import cached_property
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull, HalfspaceIntersection

import relucent.config as cfg
from relucent._internal.torch_compat import torch
from relucent.geometry.calculations import (
    _affine_null_basis,
    _remap_zero_indices,
    compute_properties,
    get_hs,
    get_shis,
    solve_radius,
)
from relucent.model.model import ReLUNetwork
from relucent.utils import encode_ss, flip_ss_at_shi, get_env

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
        halfspaces: np.ndarray | torch.Tensor | None = None,
        W: np.ndarray | torch.Tensor | None = None,
        b: np.ndarray | torch.Tensor | None = None,
        finite: bool | None = None,
        shis: list[int] | None = None,
        bound: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a Polyhedron object.

        Args:
            net: Internal canonical NN class instance used for geometry calculations.
            ss: Sign sequence defining the polyhedron (values in {-1, 0, 1}).

        The kwargs can be used to supply precomputed values for various properties.
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
        self._w: torch.Tensor | np.ndarray | None = W
        self._b: torch.Tensor | np.ndarray | None = b
        self._Wl2: float | None = None
        self._interior_point: np.ndarray | None = None
        self._interior_point_norm: float | None = None
        self._center: np.ndarray | None = None
        self._inradius: float | None = None
        self._num_dead_relus: int | None = None
        self.bound = bound

        self._shis: list[int] | None = shis
        self._shis_strict: bool = False
        self._hs: HalfspaceIntersection | None = None
        self._ch: ConvexHull | None = None
        self._finite: bool | None = finite
        self._finite_computed: bool = finite is not None
        self._vertices: np.ndarray | None = None
        self._volume: float | None = None
        self._covector_infeasible: bool = False
        self._covector_endpoint_shis: list[int] | None = None

        self._hash: int | None = None
        self._tag: bytes | None = None

        self.warnings: list[Warning] = []

        # Cached NumPy representation of the sign sequence (if/when needed).
        self._ss_np: np.ndarray | None = None

        self._attempted_compute_properties: bool = False
        self._ambient_dim: int | None = None

        for key, value in kwargs.items():
            setattr(self, key, value)

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
    def _halfspace_point(hs: np.ndarray, eq_indices: np.ndarray) -> np.ndarray | None:
        """Attempt to find the point defined by treating ``eq_indices`` rows as equalities.

        Solves the linear system ``hs[eq_indices, :-1] @ x = -hs[eq_indices, -1]`` via
        least-squares.  Two checks are applied:

        1. **Residual check** (equality rows): ``‖a_eq @ x − b_eq‖ ≤ TOL_INTERIOR_VERIFY``
           (relative to ``‖b_eq‖``).  A large residual means the equality system is
           inconsistent — no such point exists.
        2. **Slack check** (inequality rows only): each row *not* in ``eq_indices`` must
           satisfy ``a_i @ x + b_i ≤ TOL_HALFSPACE_CONTAINMENT``.  Equality rows are
           excluded because their satisfaction is already covered by the residual check
           with the intentionally looser ``TOL_INTERIOR_VERIFY`` tolerance; applying the
           stricter ``TOL_HALFSPACE_CONTAINMENT`` to them could reject valid points when
           the lstsq residual is in the ``(TOL_HALFSPACE_CONTAINMENT, TOL_INTERIOR_VERIFY]``
           range.

        Returns:
            The feasible point as a 1-D float64 array, or ``None`` if the equality
            system is inconsistent or the candidate point violates any inequality
            halfspace.
        """
        H = np.asarray(hs, dtype=np.float64)
        eq = H[eq_indices]
        a_eq = eq[:, :-1]
        b_eq = -eq[:, -1]
        x, *_ = np.linalg.lstsq(a_eq, b_eq, rcond=None)
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if a_eq.size > 0:
            res = float(np.linalg.norm(a_eq @ x - b_eq))
            scale = max(1.0, float(np.linalg.norm(b_eq)))
            if res > float(cfg.TOL_INTERIOR_VERIFY) * scale:
                return None
        n_rows = H.shape[0]
        n_eq = len(eq_indices)
        if n_rows > n_eq:
            eq_mask = np.zeros(n_rows, dtype=bool)
            eq_mask[np.asarray(eq_indices, dtype=np.intp)] = True
            ineq_slacks = H[~eq_mask, :-1] @ x + H[~eq_mask, -1]
            if np.any(ineq_slacks > float(cfg.TOL_HALFSPACE_CONTAINMENT)):
                return None
        return x

    def verify_vertex_covector(
        self,
        vertex_ss: np.ndarray,
        *,
        point2preactivations: Callable[[np.ndarray], np.ndarray],
        sign_margin: float,
    ) -> np.ndarray | None:
        """Recover and verify a vertex predicted from this top-dimensional coface.

        Only the equality solve uses floating-point arithmetic. Verification
        ignores the zero coordinates and requires every predicted nonzero
        preactivation to lie strictly beyond ``sign_margin``.

        Coordinates whose halfspace normal vanishes (dead / near-dead ReLUs,
        ``||a|| < TOL_DEAD_RELU``) are not real hyperplanes: they may still
        carry a spurious combinatorial ``±1`` in ``vertex_ss``, but they are
        skipped in the sign check so they do not reject genuine vertices.
        """
        ss = np.asarray(vertex_ss, dtype=np.int8)
        row = ss.ravel()
        zero_indices = np.flatnonzero(row == 0).astype(np.intp, copy=False)
        ambient_dim = int(self.ambient_dim)
        if zero_indices.size != ambient_dim:
            return None

        hs = np.asarray(self.halfspaces_np, dtype=np.float64)
        if hs.shape[0] < row.size:
            raise ValueError(f"Halfspace row count {hs.shape[0]} is smaller than sign-sequence length {row.size}.")
        equality_normals = hs[zero_indices, :-1]
        if equality_normals.shape != (ambient_dim, ambient_dim):
            return None
        if int(np.linalg.matrix_rank(equality_normals)) != ambient_dim:
            return None

        point = self._halfspace_point(hs[: row.size], zero_indices)
        if point is None:
            return None
        values = np.asarray(point2preactivations(point), dtype=np.float64).reshape(-1)
        if values.size != row.size:
            raise ValueError(f"Network produced {values.size} preactivations for a sign sequence of length {row.size}.")

        # Skip vanishing normals: not real cuts, only noise in the sign alphabet.
        normals = hs[: row.size, :-1]
        normal_norms = np.linalg.norm(normals, axis=1)
        active = (row != 0) & (normal_norms >= float(cfg.TOL_DEAD_RELU))
        if not np.any(active):
            return None
        signed_values = values[active] * row[active]
        if np.any(signed_values <= float(sign_margin)):
            return None
        return point

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
        x = self._halfspace_point(hs, zidx)
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
        :meth:`~relucent.core.complex.Complex._codim_one_face_kwargs` checks (boundary
        faces) prevent phantom cells at dimensions > 0, so this check is not needed there.

        **Invariant**: every 1-cell that passes through boundary-face construction
        (via :meth:`~relucent.core.complex.Complex._codim_one_face_kwargs`) is constructed
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
        return self._halfspace_point(hs, active) is not None

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
        """The neural network I belong to"""
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
        """My sign sequence."""
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
        center, inradius = solve_radius(env, self.halfspaces_np[:], zero_indices=self.zero_indices)
        return center, inradius

    def _halfspaces_with_bounding_box(self, bound: float, env: Any = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Stack axis-aligned bounds, drop degenerates, and check feasibility.

        Returns ``(halfspaces, zero_indices)`` where ``zero_indices`` is remapped
        through degenerate-row removal. Callers must use that remapped array with
        the returned halfspaces — the raw :attr:`zero_indices` point into the
        pre-drop stack and would otherwise land on a bounding-box row.
        """
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
        # Drop near-zero normals (dead constraints); toxic for Gurobi / Qhull.
        normals = halfspaces[:, :-1]
        norms = np.linalg.norm(normals, axis=1)
        deg = norms < cfg.TOL_HALFSPACE_NORMAL
        zero_indices: np.ndarray | None = np.asarray(self.zero_indices, dtype=np.intp)
        if zero_indices.size == 0:
            zero_indices = None
        if np.any(deg):
            b = halfspaces[:, -1]
            # Degenerate row is 0*x + b <= 0; b>0 is outright infeasible.
            if np.any(b[deg] > cfg.TOL_HALFSPACE_CONTAINMENT):
                bad = np.flatnonzero(deg & (b > cfg.TOL_HALFSPACE_CONTAINMENT)).tolist()
                raise ValueError(
                    "Degenerate halfspace(s) imply infeasibility after bounding; "
                    + f"rows={bad}, tol_normal={cfg.TOL_HALFSPACE_NORMAL:g}"
                )
            old_to_new = np.full(halfspaces.shape[0], -1, dtype=np.intp)
            kept = np.flatnonzero(~deg)
            old_to_new[kept] = np.arange(kept.size, dtype=np.intp)
            zero_indices = _remap_zero_indices(zero_indices, old_to_new)
            halfspaces = halfspaces[~deg]
        env = env or get_env()
        if solve_radius(env, halfspaces, max_radius=bound, zero_indices=zero_indices)[0] is None:
            raise ValueError("Bounding box constraints are not feasible")
        return halfspaces, zero_indices

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
            return Polyhedron(None, ss, halfspaces=self._halfspaces, bound=self.bound)
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
        # IMPORTANT: do not reuse cached geometry (halfspaces/W/b/shis) from the parent
        # polyhedron. Setting a sign to 0 changes which constraints are active and can
        # change the derived halfspace representation; reusing caches can yield an
        # inconsistent cell complex (and invalid Betti numbers).
        #
        # Exception: when there is no associated network and the polyhedron was created
        # from explicit halfspaces, the halfspace system itself *defines* the geometry.
        # In that case, the face is represented by the same halfspaces but with one
        # more constraint treated as an equality (via the zero sign entry).
        if self._net is None and self._halfspaces is not None:
            return Polyhedron(None, ss, halfspaces=self._halfspaces, bound=self.bound)
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
            return Polyhedron(None, ss, halfspaces=self._halfspaces, bound=self.bound)
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
            bounded_halfspaces, zero_idx = self._halfspaces_with_bounding_box(bound)
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
            x0, null_basis, ineq_mask = _affine_null_basis(bounded_halfspaces, zero_idx)

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
        plot_kwargs: dict[str, Any] = dict(
            showlegend=showlegend,
            bound=bound,
            filled=filled,
            **kwargs,
        )
        # 2D-only controls must not be forwarded to 3D plotting paths.
        if self.ambient_dim == 2:
            plot_kwargs["fill"] = fill
            plot_kwargs["plot_halfspaces"] = plot_halfspaces
            plot_kwargs["halfspace_shade"] = halfspace_shade
        from relucent.vis import plot_polyhedron

        return plot_polyhedron(self, plot_mode="cells", **plot_kwargs)

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
        from relucent.vis import plot_polyhedron

        return plot_polyhedron(
            self,
            plot_mode="graph",
            fill=fill,
            showlegend=showlegend,
            bound=bound,
            project=project,
            **kwargs,
        )

    def get_geometry(
        self,
        properties: Iterable[str],
        env: Any = None,
    ) -> None:
        """Compute selected cached properties.

        Args:
            properties: Iterable of cache/property names to ensure are computed.
                Supported names include ``"halfspaces"``, ``"W"``, ``"b"``,
                ``"num_dead_relus"``, ``"finite"``, ``"center"``,
                ``"inradius"``, ``"interior_point"``, ``"interior_point_norm"``,
                ``"Wl2"``, ``"volume"``, ``"vertices"``, ``"ch"``, and ``"hs"``.
            env: Optional Gurobi environment used for interior-point/feasibility
                solves when relevant.
        """
        requested = {name.strip() for name in properties}
        if not requested:
            return

        geometry_aliases = {"hs", "vertices", "ch", "volume"}
        if "halfspaces_np" in requested:
            requested.add("halfspaces")
        if requested & geometry_aliases:
            requested.add("interior_point")

        if requested & {"halfspaces", "W", "b", "num_dead_relus"}:
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
        """Compute Qhull-derived geometry caches (hs/vertices/ch/volume)."""
        compute_properties(self, qhull_mode=qhull_mode)

    def _ensure_affine_data(self, *, force_numpy: bool = False) -> None:
        """Populate halfspace and affine-map caches via :func:`~relucent.geometry.calculations.get_hs`."""
        halfspaces, w, b, num_dead_relus = get_hs(self, force_numpy=force_numpy)
        self._halfspaces = halfspaces
        self._w = w
        self._b = b
        self._halfspaces_np = None
        self._num_dead_relus = num_dead_relus

    @property
    def vertices(self) -> np.ndarray | None:
        """Vertices of the polyhedron (not always reliable)."""
        if not self._attempted_compute_properties:
            self._compute_qhull_geometry()
        return self._vertices

    @property
    def hs(self) -> HalfspaceIntersection:
        """Halfspace intersection object from scipy."""
        if not self._attempted_compute_properties:
            self._compute_qhull_geometry()
        assert isinstance(self._hs, HalfspaceIntersection)
        return self._hs

    @property
    def ch(self) -> ConvexHull | None:
        """Convex hull of the polyhedron for finite polyhedra, or None if unbounded or computation fails."""
        if not self._attempted_compute_properties and self.finite:
            self._compute_qhull_geometry()
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
    def num_dead_relus(self) -> int:
        """Number of dead ReLU neurons (neurons always outputting zero).

        Returns:
            int: Count of ReLU neurons that are always inactive for this polyhedron.
        """
        if self._num_dead_relus is None:
            self._ensure_affine_data(force_numpy=isinstance(self._halfspaces, np.ndarray))
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
        assert isinstance(self._shis, list)
        return self._shis

    @property
    def num_shis(self) -> int:
        """Number of faces."""
        return len(self.shis)

    @property
    def num_faces(self) -> int:
        """Number of faces; alias for :attr:`num_shis`."""
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
        """Check if a point (ndarray or Tensor) is contained in the polyhedron."""
        halfspaces = self.halfspaces_np if isinstance(point, np.ndarray) else self.halfspaces
        if isinstance(point, torch.Tensor) and isinstance(halfspaces, np.ndarray):
            point = point.detach().cpu().numpy().astype(halfspaces.dtype)
        point = point.reshape(1, -1)
        dists = point @ halfspaces[:, :-1].T + halfspaces[:, -1]
        return bool((dists <= cfg.TOL_HALFSPACE_CONTAINMENT).all().item())

    def __mul__(self, other: "Polyhedron") -> "Polyhedron":
        """Returns a new Polyhedron object based on sign sequence multiplication"""
        return Polyhedron(self._net, self.ss + other.ss * (self.ss == 0))

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
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
            "_num_dead_relus": self._num_dead_relus,
            "_interior_point": self._interior_point,
            "_attempted_compute_properties": self._attempted_compute_properties,
            "_covector_infeasible": self._covector_infeasible,
            "_covector_endpoint_shis": self._covector_endpoint_shis,
            "warnings": self.warnings,
            "codim": self.codim,
            "dim": self.dim,
            "_ambient_dim": self._ambient_dim,
            "_halfspaces_np": self._halfspaces_np,
            "_w": self._w,
            "_b": self._b,
            "bound": self.bound,
        }
        return state

    def __reduce__(self) -> tuple[type["Polyhedron"], tuple[None, np.ndarray], dict[str, Any]]:
        return (
            Polyhedron,
            (None, self.ss_np),
            self.__getstate__(),
        )  # Control what gets saved, do not pickle the net
