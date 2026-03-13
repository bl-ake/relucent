import hashlib
import warnings
from functools import cached_property
from typing import Any, Iterable, MutableSequence, cast

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from gurobipy import GRB, Model
from scipy.spatial import ConvexHull, HalfspaceIntersection
from tqdm.auto import tqdm

from relucent.config import (
    DEFAULT_PLOT_BOUND,
    GUROBI_SHI_BEST_BD_STOP,
    GUROBI_SHI_BEST_OBJ_STOP,
    MAX_RADIUS,
    QHULL_MODE,
    TOL_DEAD_RELU,
    TOL_HALFSPACE_CONTAINMENT,
    TOL_NEARLY_VERTICAL,
    TOL_SHI_HYPERPLANE,
    TOL_VERIFY_AB_ATOL,
    VERTEX_TRUST_THRESHOLD,
)
from relucent.model import NN
from relucent.utils import encode_ss, get_env


def solve_radius(
    env: Any,
    halfspaces: np.ndarray | torch.Tensor,  ## TODO: Remove redundant check for this
    max_radius: float = GRB.INFINITY,
    zero_indices: MutableSequence[int] | None = None,
    sense: int = GRB.MAXIMIZE,
) -> tuple[np.ndarray | None, float]:
    """Solve for the Chebyshev center or interior point of a polyhedron.

    Only works if all polyhedron vertices are within 2*max_radius of each other.

    Args:
        env: Gurobi environment for optimization.
        halfspaces: Halfspace representation of the polyhedron as an array with
            shape (n_constraints, n_dim+1), where the last column contains bias terms.
        max_radius: Maximum radius constraint for the polyhedron. Defaults to infinity.
        zero_indices: Indices of sign sequence elements that are zero (for
            lower-dimensional polyhedra). Defaults to None.
        sense: Optimization sense, should typically be GRB.MAXIMIZE. Defaults to GRB.MAXIMIZE.

    Returns:
        tuple: (center_point, radius) where center_point is the center/interior
            point and radius is the inradius. Returns (None, float('inf')) if the
            polyhedron is unbounded and max_radius is infinity.

    Raises:
        ValueError: If the optimization fails or produces invalid results.
    """

    if isinstance(halfspaces, torch.Tensor):
        halfspaces = halfspaces.detach().cpu().numpy()

    if zero_indices is not None and len(zero_indices) > 0:
        warnings.warn("Working with k<d polyhedron.")
        equalities = halfspaces[zero_indices]
        inequalities = halfspaces[~np.isin(np.arange(halfspaces.shape[0]), zero_indices)]
        P = (
            np.eye(equalities[:, :-1].shape[1])
            - equalities[:, :-1].T @ np.linalg.pinv(equalities[:, :-1] @ equalities[:, :-1].T) @ equalities[:, :-1]
        )
        norm_vector = np.reshape(np.linalg.norm(inequalities[:, :-1] @ P.T, axis=1), (inequalities[:, :-1].shape[0], 1))
    else:
        inequalities = halfspaces
        equalities = None
        norm_vector = np.reshape(np.linalg.norm(inequalities[:, :-1], axis=1), (inequalities[:, :-1].shape[0], 1))

    model = Model("Interior Point", env)
    x = model.addMVar((halfspaces.shape[1] - 1, 1), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    y = model.addMVar((1,), ub=max_radius, vtype=GRB.CONTINUOUS, name="y")
    try:
        model.addConstr(inequalities[:, :-1] @ x + norm_vector * y <= -inequalities[:, -1:])
        if equalities is not None:
            model.addConstr(equalities[:, :-1] @ x == -equalities[:, -1:])
        model.setObjective(y, sense)
    except Exception as e:
        raise ValueError(f"GB Error Building Model: {e}")
    model.optimize()
    status = model.status

    if status == GRB.OPTIMAL:
        objVal = model.objVal
        x, y = x.X, float(np.squeeze(y.X))
        model.close()
        if objVal <= 0 and objVal > -1e-6:
            raise ValueError(f"Inradius {objVal:.4e}")
        if objVal < -1e-6:
            raise ValueError(f"Something has gone horribly wrong: objVal={objVal:.4e}")
        assert isinstance(x, np.ndarray)
        return x, y
    elif status == GRB.INTERRUPTED:
        model.close()
        raise KeyboardInterrupt
    else:
        if max_radius == GRB.INFINITY:
            model.close()
            return None, float("inf")
        else:
            # if status == GRB.INFEASIBLE:
            #     breakpoint()
            model.close()
            raise ValueError(f"Interior Point Model Status: {status}")


class Polyhedron:
    """Represents a polyhedron (linear region) in d-dimensional space.

    Several methods use Gurobi environments for optimization. If one is not
    provided, an environment will be created automatically.
    """

    MAX_RADIUS = MAX_RADIUS  # from config; smaller is faster but may exclude polyhedra

    def __init__(
        self,
        net: NN,
        ss: np.ndarray | torch.Tensor,
        halfspaces: np.ndarray | torch.Tensor | None = None,
        W: np.ndarray | torch.Tensor | None = None,
        b: np.ndarray | torch.Tensor | None = None,
        point: np.ndarray | torch.Tensor | None = None,
        shis: list[int] | None = None,
        bound: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a Polyhedron object.

        The kwargs can be used to supply precomputed values for various properties.

        Args:
            net: Instance of the NN class from the "model" module.
            ss: Sign sequence defining the polyhedron (values in {-1, 0, 1}).
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
        if isinstance(point, torch.Tensor):
            point = point.detach().cpu().numpy()
        self._point = point
        self._interior_point: np.ndarray | None = None
        self._interior_point_norm: float | None = None
        self._center: np.ndarray | None = None
        self._inradius: float | None = None
        self._num_dead_relus: int | None = None
        self.bound = bound

        self._shis: list[int] | None = shis
        self._hs = None
        self._ch: ConvexHull | None = None
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
    def zero_indices(self) -> list[int]:
        """Indices of sign sequence elements that are zero."""
        return np.flatnonzero(self.ss_np == 0)

    @cached_property
    def non_zero_indices(self) -> list[int]:
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

    def compute_properties(self, qhull_mode: str = QHULL_MODE) -> None:
        """Compute additional geometric properties for low-dimensional polyhedra.

        Returns:
            bool: True if computation succeeded.

        Raises:
            ValueError: If input dimension > 6 or if computation fails.
        """
        if self._attempted_compute_properties:
            return
        self._attempted_compute_properties = True

        if self.net.input_shape[0] > 6:
            raise ValueError("Input shape too large to compute extra properties")
        try:
            halfspaces = self.halfspaces_np
            with warnings.catch_warnings(record=True) as w:
                hs = HalfspaceIntersection(
                    halfspaces,
                    self.interior_point,
                    qhull_options=None,
                )  # http://www.qhull.org/html/qh-optq.htm
            if w:
                msgs = "; ".join(str(wi.message) for wi in w)
                if qhull_mode == "IGNORE":
                    self.warnings.extend([RuntimeWarning(wi) for wi in w])
                if qhull_mode == "WARN_ALL":
                    warnings.warn(f"HalfspaceIntersection emitted warnings in WARN_ALL mode: {msgs}")
                elif qhull_mode == "HIGH_PRECISION":
                    raise ValueError(f"HalfspaceIntersection emitted warnings in HIGH_PRECISION mode: {msgs}")
                elif qhull_mode == "JITTERED":
                    with warnings.catch_warnings(record=True) as w2:
                        new_hs = HalfspaceIntersection(
                            halfspaces,
                            self.interior_point,
                            qhull_options="QJ",  # Triangulated output is approximately 1000 times more accurate than joggled input.
                        )  # http://www.qhull.org/html/qh-optq.htm
                    if w2:
                        self.warnings.append(
                            RuntimeWarning(
                                "Recomputing HalfspaceIntersection with jitter option 'QJ' still had numerical problems"
                            )
                        )
                        self.warnings.extend([RuntimeWarning(wi) for wi in w2])
                        msgs = "; ".join(str(wi.message) for wi in w)
                    else:
                        ## Jittering solved the numerical problems
                        hs = new_hs
        except ValueError:
            raise  # Our HIGH_PRECISION raise - do not retry
        except Exception as e:
            if qhull_mode == "JITTERED":
                try:
                    hs = HalfspaceIntersection(
                        halfspaces,
                        self.interior_point,
                        qhull_options="QJ",  # Triangulated output is approximately 1000 times more accurate than joggled input.
                    )  # http://www.qhull.org/html/qh-optq.htm
                    self.warnings.append(
                        RuntimeWarning(f"HalfspaceIntersection failed initially, succeeded with QJ retry: {e}")
                    )
                except Exception as e2:
                    raise ValueError(f"Error while computing halfspace intersection: {e}") from e2
            else:
                raise ValueError(f"Error while computing halfspace intersection: {e}")
        self._hs = hs
        # It seems like the SHIs are not always computed correctly by HalfSpaceIntersection, so we will not check them
        # try:
        #     hs_shis = np.unique([shi for shis in hs.dual_facets for shi in shis]).tolist()
        #     # hs_shis = hs.dual_vertices.ravel().tolist()
        #     if set(hs_shis) != set(self.shis):
        #         w = RuntimeWarning(
        #             f"HalfspaceIntersection SHIs on {self} do not match computed SHIs: {sorted(hs_shis)} vs {sorted(self.shis)}"
        #         )
        #         self.warnings.append(w)
        # except Exception as e:
        #     w = RuntimeWarning(f"Error while getting dual vertices: {e}")
        #     self.warnings.append(w)
        #     raise ValueError(f"Error while getting dual vertices: {e}")
        vertices = hs.intersections
        trust_vertices = ~(np.isinf(vertices).any(axis=1) | np.isnan(vertices).any(axis=1))
        trust_vertices_2 = (halfspaces[:, :-1] @ vertices[trust_vertices].T + halfspaces[:, -1, None]).sum(
            axis=0
        ) < VERTEX_TRUST_THRESHOLD
        # if not (
        #     (halfspaces[:, :-1] @ vertices[trust_vertices].T + halfspaces[:, -1, None]).sum(axis=0)
        #     < VERTEX_TRUST_THRESHOLD
        # ).all():
        #     w = RuntimeWarning(
        #         f"Vertex computation failed - Maximum Violation: {(halfspaces[:, :-1] @ vertices[trust_vertices].T + halfspaces[:, -1, None]).max()}"
        #     )
        #     # warnings.warn(w)
        #     self.warnings.append(w)
        #     self._volume = -1
        #     return
        self._vertices = vertices[trust_vertices][trust_vertices_2]
        self._vertex_set = set(tuple(x) for x in self.vertices)
        if self.finite and len(self.vertices) > self.ambient_dim:
            try:
                self._ch = ConvexHull(vertices)
                try:
                    self._volume = self._ch.volume
                except Exception as e:
                    raise ValueError(f"Error while computing convex hull volume: {e}")
            except Exception as e:
                # warnings.warn("Error while computing convex hull:", e)
                if qhull_mode == "WARN_ALL":
                    warnings.warn(f"Error while computing convex hull: {e}")
                elif qhull_mode == "HIGH_PRECISION":
                    raise ValueError(f"Error while computing convex hull: {e}")
                self._ch = None
                self._volume = -1
        else:
            self._volume = float("inf")

    def get_interior_point(
        self,
        env: Any = None,
        max_radius: float | None = None,
    ) -> np.ndarray:
        """Get a point inside the polyhedron.

        Computes an interior point of the polyhedron. If the center is already
        computed, uses that; otherwise solves for an interior point using Gurobi.

        Args:
            env: Gurobi environment for optimization. If None, uses a cached
                environment. Defaults to None.
            max_radius: Maximum radius constraint for the search. If None, uses
                self.MAX_RADIUS. Defaults to None.
            zero_indices: Indices of sign sequence elements that are zero (for
                lower-dimensional polyhedra). Defaults to None.

        Returns:
            np.ndarray: An interior point of the polyhedron.

        Raises:
            ValueError: If no interior point can be found.
        """
        max_radius = max_radius or self.MAX_RADIUS
        if isinstance(self._center, (np.ndarray, torch.Tensor)):
            self._interior_point = self._center.squeeze()
        else:
            env = env or get_env()
            self._interior_point = solve_radius(
                env,
                self.halfspaces_np[:],
                zero_indices=self.zero_indices,
                max_radius=max_radius,
            )[0]
            assert isinstance(self._interior_point, np.ndarray)
            self._interior_point = self._interior_point.squeeze()
        if self._interior_point is None:
            raise ValueError("Interior point not found")
        # if (
        #     maximum_violation := (self.inequalities[:, :-1] @ self._interior_point + self.inequalities[:-1, None]).max()
        # ) > TOL_HALFSPACE_CONTAINMENT:
        #     raise ValueError(f"Interior point invalid - Maximum Violation: {maximum_violation}")
        if self._interior_point is not None and self._interior_point not in self:
            raise ValueError(f"Interior point invalid - {self._interior_point} not in {self}")
        return self._interior_point

    def get_center_inradius(self, env: Any = None) -> tuple[np.ndarray | None, float]:
        """Get the Chebyshev center and inradius of the polyhedron.

        Also sets self._finite to indicate if the polyhedron is finite.

        Args:
            env: Gurobi environment for optimization. If None, uses a cached
                environment. Defaults to None.

        Returns:
            tuple: (center, inradius) where center is None for unbounded polyhedra.
        """
        env = env or get_env()
        self._center, self._inradius = solve_radius(
            env, self.halfspaces_np[:], zero_indices=self.zero_indices
        )  ## TODO: Change
        self._finite = self._center is not None
        return self._center, self._inradius

    def get_hs(
        self,
        data: torch.Tensor | None = None,
        get_all_Ab: bool = False,
        force_numpy: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[np.ndarray, np.ndarray, np.ndarray]
        | list[dict[str, Any]]
    ):
        """Get the halfspace representation of this polyhedron.

        Computes the halfspaces (inequality constraints) that define the polyhedron
        from all neurons in the network. The result includes constraints from
        every neuron, not just the supporting hyperplanes.

        Args:
            data: Optional input data to the network for verification. If provided,
                checks that computed outputs match network outputs. Defaults to None.
            get_all_Ab: If True, returns all intermediate affine maps (A, b) for
                each layer instead of just the final halfspaces. Defaults to False.
            force_numpy: If True, use NumPy backend even when ss is a torch.Tensor.
                Defaults to False.

        Returns:
            If get_all_Ab is False: tuple (halfspaces, W, b) where halfspaces has
            shape (n_constraints, n_dim+1), W is the affine matrix, and b is the
            affine bias. If get_all_Ab is True: list of dicts with 'A', 'b', and
            'layer' keys for each layer.
        """
        # Check underlying attribute directly to avoid property access overhead
        if isinstance(self._ss, torch.Tensor) and not force_numpy:
            return self._get_hs_torch(data, get_all_Ab)
        else:
            return self._get_hs_numpy(data, get_all_Ab)

    @torch.no_grad()
    def _get_hs_torch(
        self,
        data: torch.Tensor | None = None,
        get_all_Ab: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[dict[str, Any]]:
        """Get halfspaces when the sign sequence is a torch.Tensor.

        Computes the halfspace representation using PyTorch operations.

        Args:
            data: Optional input data to the network for verification. If provided,
                checks that computed outputs match network outputs. Defaults to None.
            get_all_Ab: If True, returns all intermediate affine maps (A, b) for
                each layer instead of just the final halfspaces. Defaults to False.

        Returns:
            If get_all_Ab is False: (halfspaces, W, b) tuple.
            If get_all_Ab is True: List of dicts with 'A', 'b', and 'layer' keys.
        """
        constr_A, constr_b = None, None
        current_A, current_b = None, None
        A, b = None, None
        if data is not None:
            outs = self.net.get_all_layer_outputs(data)
        all_Ab = []
        current_mask_index = 0
        for name, layer in self.net.layers.items():
            if isinstance(layer, nn.Linear):
                A = layer.weight
                b = layer.bias[None, :]
                if current_A is None or current_b is None:
                    constr_A = torch.empty((A.shape[1], 0), device=self.net.device, dtype=self.net.dtype)
                    constr_b = torch.empty((1, 0), device=self.net.device, dtype=self.net.dtype)
                    current_A = torch.eye(A.shape[1], device=self.net.device, dtype=self.net.dtype)
                    current_b = torch.zeros((1, A.shape[1]), device=self.net.device, dtype=self.net.dtype)

                current_A = current_A @ A.T
                current_b = current_b @ A.T + b
            elif isinstance(layer, nn.ReLU):
                assert current_A is not None
                assert current_b is not None

                mask = self.ss[0, current_mask_index : current_mask_index + current_A.shape[1]]

                ## Replce mask 0s with 1s
                nonzero_mask = torch.where(mask == 0, 1, mask)

                new_constr_A = current_A * nonzero_mask
                new_constr_b = current_b * nonzero_mask

                assert isinstance(constr_A, torch.Tensor)
                assert isinstance(constr_b, torch.Tensor)
                assert isinstance(current_A, torch.Tensor)
                assert isinstance(current_b, torch.Tensor)
                assert isinstance(mask, torch.Tensor)

                constr_A = torch.cat((constr_A, new_constr_A), dim=1)
                constr_b = torch.cat((constr_b, new_constr_b), dim=1)

                current_A = current_A * (mask == 1)
                current_b = current_b * (mask == 1)
                current_mask_index += current_A.shape[1]
            elif isinstance(layer, nn.Flatten):
                if current_A is None:
                    pass
                else:
                    raise NotImplementedError("Intermediate flatten layer not supported")
            else:
                raise ValueError(
                    f"Error while processing layer {name} - Unsupported layer type: {type(layer)} ({layer})"
                )
            if data is not None:
                assert isinstance(current_A, torch.Tensor)
                assert isinstance(current_b, torch.Tensor)
                assert torch.allclose(outs[name], (data @ current_A) + current_b, atol=TOL_VERIFY_AB_ATOL)
            if get_all_Ab:
                assert current_A is not None
                assert current_b is not None

                all_Ab.append({"A": current_A.clone(), "b": current_b.clone(), "layer": layer})

        assert constr_A is not None
        assert constr_b is not None

        self._num_dead_relus = int((torch.abs(constr_A) < TOL_DEAD_RELU).all(dim=0).sum().item())
        halfspaces = torch.hstack((-constr_A.T, -constr_b.reshape(-1, 1)))

        if get_all_Ab:
            return all_Ab

        assert isinstance(halfspaces, torch.Tensor)
        assert isinstance(current_A, torch.Tensor)
        assert isinstance(current_b, torch.Tensor)

        assert halfspaces.shape[0] == self.ss.shape[1]
        return halfspaces, current_A, current_b

    @torch.no_grad()
    def _get_hs_numpy(
        self,
        data: torch.Tensor | None = None,
        get_all_Ab: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | list[dict[str, Any]]:
        """Get halfspaces when the sign sequence is a numpy array.

        Args:
            data: Optional input data to the network for verification. If provided,
                checks that computed outputs match network outputs. Defaults to None.
            get_all_Ab: If True, returns all intermediate affine maps (A, b) for
                each layer instead of just the final halfspaces. Defaults to False.

        Returns:
            If get_all_Ab is False: (halfspaces, W, b) tuple.
            If get_all_Ab is True: List of dicts with 'A', 'b', and 'layer' keys.
        """
        constr_A, constr_b = None, None
        current_A, current_b = None, None
        A, b = None, None
        if data is not None:
            outs = self.net.get_all_layer_outputs(data)
        all_Ab = []
        current_mask_index = 0
        for name, layer in self.net.layers.items():
            if isinstance(layer, nn.Linear):
                A = layer.weight_cpu
                b = layer.bias_cpu

                assert isinstance(A, np.ndarray)
                assert isinstance(b, np.ndarray)

                if current_A is None or current_b is None:
                    constr_A = np.empty((A.shape[1], 0))
                    constr_b = np.empty((1, 0))
                    current_A = np.eye(A.shape[1])
                    current_b = np.zeros((1, A.shape[1]))

                current_A = current_A @ A.T
                current_b = current_b @ A.T + b
            elif isinstance(layer, nn.ReLU):
                if current_A is None:
                    raise ValueError("ReLU layer must follow a linear layer")
                mask = self.ss_np[0, current_mask_index : current_mask_index + current_A.shape[1]]

                nonzero_mask = np.where(mask == 0, 1, mask)

                new_constr_A = current_A * nonzero_mask
                new_constr_b = current_b * nonzero_mask

                assert constr_A is not None
                assert constr_b is not None
                assert current_A is not None
                assert current_b is not None

                constr_A = np.concatenate((constr_A, new_constr_A), axis=1)
                constr_b = np.concatenate((constr_b, new_constr_b), axis=1)

                current_A = current_A * (mask == 1)
                current_b = current_b * (mask == 1)
                current_mask_index += current_A.shape[1]
            elif isinstance(layer, nn.Flatten):
                if current_A is None:
                    pass
                else:
                    raise NotImplementedError("Intermediate flatten layer not supported")
            else:
                raise ValueError(
                    f"Error while processing layer {name} - Unsupported layer type: {type(layer)} ({layer})"
                )
            if data is not None:
                assert isinstance(current_A, np.ndarray)
                assert isinstance(current_b, np.ndarray)
                assert np.allclose(
                    outs[name].detach().cpu().numpy(), (data @ current_A) + current_b, atol=TOL_VERIFY_AB_ATOL
                )
            if get_all_Ab:
                assert current_A is not None
                assert current_b is not None

                all_Ab.append({"A": current_A.copy(), "b": current_b.copy(), "layer": layer})

        assert constr_A is not None
        assert constr_b is not None

        self._num_dead_relus = (np.abs(constr_A) < TOL_DEAD_RELU).all(axis=0).sum().item()
        halfspaces = np.hstack((-constr_A.T, -constr_b.reshape(-1, 1)))
        if get_all_Ab:
            return all_Ab
        assert isinstance(halfspaces, np.ndarray)
        assert isinstance(current_A, np.ndarray)
        assert isinstance(current_b, np.ndarray)

        assert halfspaces.shape[0] == self.ss_np.shape[1]
        return halfspaces, current_A, current_b

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
            warnings.warn("Comparing Polyhedron with string is deprecated and will be removed in a future version")
            return str(self) == other
        elif other is None:
            return False
        else:
            raise ValueError(f"Cannot compare Polyhedron with {type(other)}")

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.tag)
        return self._hash

    # def common_vertices(self, other):
    #     if not self.finite or not other.finite:
    #         raise NotImplementedError
    #     return self.vertex_set.intersection(other.vertex_set)

    def get_shis(
        self,
        collect_info: bool | str = False,
        bound: float = GRB.INFINITY,
        subset: Iterable[int] | None = None,
        tol: float = TOL_SHI_HYPERPLANE,
        new_method: bool = False,
        env: Any = None,
        shi_pbar: bool = False,
        push_size: float = 1.0,
    ) -> list[int] | tuple[list[int], list]:
        """Get supporting halfspace indices (SHIs) for this polyhedron.

        Computes the indices of non-redundant halfspaces that form the boundary
        of this polyhedron. These correspond to neurons whose boundaries (BHs)
        are actually part of the polyhedron's boundary.

        Args:
            collect_info: If True, collects additional debugging information
                about the computations. If "All", collects even more detailed info.
                Defaults to False.
            bound: Defines the hypercube bounding the space for numerical stability.
                Defaults to infinity.
            subset: Indices of neurons/halfspaces to consider. If None, considers
                all halfspaces. Defaults to None.
            tol: Inequality tolerance to improve numerical stability. Defaults to config.TOL_SHI_HYPERPLANE.
            new_method: If True, uses an extra computation that doesn't improve
                runtime. Defaults to False.
            env: Gurobi environment for optimization. If None, uses a cached
                environment. Defaults to None.
            shi_pbar: If True, shows a progress bar during computation. Defaults to False.

        Returns:
            list or tuple: If collect_info is False, returns a list of SHI indices.
                If collect_info is True, returns (shis, info) where info is a list
                of dictionaries with computation details.

        Raises:
            ValueError: If the optimization model fails.
        """
        shis = []
        A = self.halfspaces_np[:, :-1]
        b = self.halfspaces_np[:, -1:]
        env = env or get_env()
        model = Model("SHIS", env)
        x = model.addMVar((self.halfspaces.shape[1] - 1, 1), lb=-bound, ub=bound, vtype=GRB.CONTINUOUS, name="x")
        constrs = model.addConstr(A @ x == -b, name="hyperplanes")
        if model.status == GRB.INTERRUPTED:
            model.close()
            raise KeyboardInterrupt
        elif model.status == GRB.OPTIMAL:
            ## All Hyperplanes Intersect
            shis = list(range(A.shape[0]))
            if collect_info:
                return shis, []
            return shis

        constrs.setAttr("Sense", GRB.LESS_EQUAL)
        model.optimize()
        if model.status != GRB.OPTIMAL:
            raise ValueError(f"Initial Solve Failed: Model status: {model.status}")

        subset = subset or range(A.shape[0])
        subset = set(subset)

        pbar = tqdm(total=len(subset), desc="Calculating SHIs", leave=False, delay=3, disable=not shi_pbar)
        if collect_info:
            poly_info = []
        while subset:
            i = subset.pop()
            if i >= self.ss_np.shape[1] or self.ss_np[0, i] == 0:
                continue
            if (A[i] == 0).all():
                continue
            # model.update()
            pbar.set_postfix_str(f"#shis: {len(shis)}")

            ## Relax halspace i
            constrs[i].setAttr("RHS", -b[i, 0] + push_size)

            model.setObjective((A[i] @ x).item() + b[i, 0], GRB.MAXIMIZE)
            model.params.BestObjStop = GUROBI_SHI_BEST_OBJ_STOP
            model.params.BestBdStop = GUROBI_SHI_BEST_BD_STOP
            # model.update()
            model.optimize()

            if model.status == GRB.INTERRUPTED:
                model.close()
                raise KeyboardInterrupt
            if model.status == GRB.OPTIMAL or model.status == GRB.USER_OBJ_LIMIT:
                if model.objVal > 0:
                    dists = A @ x.X + b
                    if dists[(dists > 0)].sum() >= 1 + tol:
                        msg = (
                            f"Invalid Proof for SHI {i}! Violation Sizes: "
                            f"{np.argwhere(dists.ravel() > 0), dists[np.argwhere(dists.ravel() > 0)]}"
                        )
                        w = RuntimeWarning(msg)
                        self.warnings.append(w)
                        warnings.warn(w)
                    else:
                        shis.append(i)

                basis_indices = constrs.CBasis.ravel() != 0
                if new_method:
                    if basis_indices.sum() != A.shape[1]:
                        warnings.warn("Bound Constraints in Basis")
                skip_size = 0
                if new_method and basis_indices.sum() == A.shape[1]:
                    point_shis = self.halfspaces[basis_indices, :-1]  # (d(# point shis) x d)
                    others = self.halfspaces[~basis_indices, :-1]  # (num_other_hyperplanes x d)
                    try:
                        sols = torch.linalg.solve(point_shis, others.T)
                    except RuntimeError:
                        warnings.warn("Could not solve linear system")
                        sols = torch.zeros(others.T.shape, device=self.halfspaces.device)
                    all_correct = (sols > 0).all(dim=0)
                    assert all_correct.shape[0] == others.shape[0]
                    correct_indices = torch.argwhere(all_correct).reshape(-1)
                    if correct_indices.shape[0] > 0:
                        A_indices = torch.arange(A.shape[0], device=self.halfspaces.device)[~basis_indices][all_correct]

                        old_len = len(subset)
                        subset -= set(A_indices.detach().cpu().numpy().ravel().tolist())
                        new_len = len(subset)
                        skip_size = old_len - new_len
            else:
                raise ValueError(f"Model status: {model.status}")

            if collect_info:
                poly_info.append(
                    {
                        "Objective Value": model.objVal,
                        "Min Non-Basis Slack": np.min(constrs.Slack[~basis_indices]),
                        "Status": model.status,
                        "# Skipped": skip_size,
                    }
                )
                if hasattr(model, "objVal"):
                    poly_info[-1]["Objective Value"] = model.objVal
                if hasattr(model, "objBound"):
                    poly_info[-1]["Objective Bound"] = model.objBound
                if hasattr(x, "X"):
                    poly_info[-1]["x Norm"] = np.linalg.norm(x.X)
                if collect_info == "All":
                    poly_info[-1] |= {"Slacks": constrs.Slack, "-b[i]": -b[i], "Status": model.status}

                    if hasattr(x, "X"):
                        poly_info[-1]["Proof"] = x.X

            ## Restore halfspace i
            constrs[i].setAttr("RHS", -b[i, 0])

            pbar.update(A.shape[0] - len(subset) - pbar.n)
        model.close()
        if collect_info:
            return shis, poly_info
        return shis

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

    def get_bounded_vertices(self, bound: float, qhull_mode: str = QHULL_MODE) -> np.ndarray | None:
        """Get the vertices of the polyhedron within a bounding hypercube.

        Computes the vertices of the polyhedron after intersecting it with a
        hypercube of radius 'bound'. Primarily used for plotting and visualization.

        Args:
            bound: Radius of the bounding hypercube.

        Returns:
            np.ndarray or None: Array of vertex coordinates, or None if the
                polyhedron doesn't intersect the bounded region or computation fails.
        """
        if self.codimension > 0:
            raise NotImplementedError("Codimension > 0 not yet supported")

        try:
            bounded_halfspaces = self.get_bounded_halfspaces(bound)
        except ValueError as e:
            w = RuntimeWarning(f"Error while computing bounded vertices: {e}")
            self.warnings.append(w)
            return None
        # int_point, _ = solve_radius(get_env(), bounded_halfspaces, max_radius=1000)
        # if not (
        #     self.interior_point @ bounded_halfspaces[:, :-1].T + bounded_halfspaces[:, -1] <= TOL_HALFSPACE_CONTAINMENT
        # ).all():
        #     warnings.warn(f"Interior point ({self.interior_point}) out of bounds ({bound}):")
        #     return None

        # Recompute interior point
        int_point, _ = solve_radius(
            get_env(),
            bounded_halfspaces,
            max_radius=1000,
            zero_indices=self.zero_indices,
        )
        if int_point is None:
            return None
        try:
            with warnings.catch_warnings(record=True) as w:
                hs = HalfspaceIntersection(
                    bounded_halfspaces,
                    int_point.squeeze(),
                    qhull_options=None,
                )  # http://www.qhull.org/html/qh-optq.htm
            if w:
                msgs = "; ".join(str(wi.message) for wi in w)
                if qhull_mode == "IGNORE":
                    self.warnings.extend([RuntimeWarning(wi) for wi in w])
                if qhull_mode == "WARN_ALL":
                    warnings.warn(f"HalfspaceIntersection emitted warnings in WARN_ALL mode: {msgs}")
                elif qhull_mode == "HIGH_PRECISION":
                    raise ValueError(f"HalfspaceIntersection emitted warnings in HIGH_PRECISION mode: {msgs}")
                elif qhull_mode == "JITTERED":
                    with warnings.catch_warnings(record=True) as w2:
                        new_hs = HalfspaceIntersection(
                            bounded_halfspaces,
                            int_point.squeeze(),
                            qhull_options="QJ",  # Triangulated output is approximately 1000 times more accurate than joggled input.
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
                        bounded_halfspaces,
                        int_point.squeeze(),
                        qhull_options="QJ",  # Triangulated output is approximately 1000 times more accurate than joggled input.
                    )  # http://www.qhull.org/html/qh-optq.htm
                    self.warnings.append(
                        RuntimeWarning(f"HalfspaceIntersection failed initially, succeeded with QJ retry: {e}")
                    )
                except Exception as e2:
                    raise ValueError(f"Error while computing halfspace intersection: {e}") from e2
            else:
                raise ValueError(f"Error while computing halfspace intersection: {e}")
        vertices = hs.intersections
        return vertices

    def _get_bounded_plot_geometry(
        self,
        bound: float,
    ) -> tuple[str, np.ndarray] | None:
        """Classify bounded geometry for plotting as polygon, segment, or point.

        The classification follows the requested behavior:
        - If there is a single 0 in ss_np, treat the region as a line segment.
        - If there are two 0s, treat the region as a point.
        - Otherwise, use the full 2D polygon (via ConvexHull) when possible.

        Returns:
            None if vertices cannot be computed or are empty, otherwise:
            ("polygon", verts) where verts are ordered boundary vertices,
            ("segment", verts) where verts has shape (2, 2) for endpoints,
            ("point", verts) where verts has shape (1, 2) for the point.
        """
        vertices = self.get_bounded_vertices(bound)
        if vertices is None or vertices.size == 0:
            return None

        # Ensure 2D coordinates.
        if vertices.shape[1] != 2:
            return None

        # Count zeros in the sign sequence (flattened).
        num_zeros = int(np.sum(self.ss_np == 0))

        # Degenerate to a single point when there are at least two zeros.
        if num_zeros >= 2:
            point = vertices.mean(axis=0, keepdims=True)
            return "point", point

        # Line segment when there is exactly one zero.
        if num_zeros == 1:
            # With very few vertices, just fall back to unique points.
            if vertices.shape[0] == 1:
                return "point", vertices
            # Use PCA/SVD to find the main direction and endpoints along it.
            centered = vertices - vertices.mean(axis=0, keepdims=True)
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                direction = vh[0]
                t = vertices @ direction
                p_min = vertices[np.argmin(t)]
                p_max = vertices[np.argmax(t)]
                segment = np.stack([p_min, p_max], axis=0)
                return "segment", segment
            except Exception:
                # Fallback: use the first two unique vertices if SVD fails.
                uniq = np.unique(vertices, axis=0)
                if uniq.shape[0] == 1:
                    return "point", uniq
                return "segment", uniq[:2]

        # Default: use ConvexHull to get a polygonal boundary, but only when the
        # polyhedron has (at least) 2-dimensional support.
        if self.dim < 2:
            uniq = np.unique(vertices, axis=0)
            if uniq.shape[0] == 1:
                return "point", uniq
            elif uniq.shape[0] == 2:
                return "segment", uniq
            else:
                return "polygon", uniq
        try:
            hull = ConvexHull(vertices)
            boundary = vertices[hull.vertices]
            return "polygon", boundary
        except Exception:
            # If ConvexHull fails, fall back to unique vertices as a polygonal chain.
            uniq = np.unique(vertices, axis=0)
            if uniq.shape[0] == 1:
                return "point", uniq
            elif uniq.shape[0] == 2:
                return "segment", uniq
            else:
                return "polygon", uniq

    def plot_3d_complex(
        self,
        showlegend: bool = False,
        bound: float = DEFAULT_PLOT_BOUND,
        filled: bool = False,
        **kwargs: Any,
    ) -> list[go.Mesh3d | go.Scatter3d]:
        """Plot the (input-space) polyhedron as a 3D region using plotly.

        This visualizes the cell of the polyhedral complex in the 3D input space.
        Only works when the ambient input dimension is exactly 3.

        Args:
            showlegend: Whether to show the trace in the legend. Defaults to False.
            bound: Radius of the bounding cube used to clip the polyhedron for
                numerical stability. Defaults to config.DEFAULT_PLOT_BOUND.
            filled: If True, render full-dimensional 3D regions as filled meshes
                (go.Mesh3d) instead of wireframes. Defaults to False.
            **kwargs: Additional keyword arguments forwarded to ``go.Scatter3d``
                or ``go.Mesh3d``.

        Returns:
            list: A list of plotly traces. For 3D regions this will contain
            line-based ``go.Scatter3d`` traces that outline the cell (no filled
            volume), or a filled ``go.Mesh3d`` when filled=True. For degenerate
            cases (faces, edges, or points) it returns appropriate ``go.Scatter3d``
            traces.
        """
        if self.ambient_dim != 3:
            raise ValueError("Polyhedron must have ambient dimension 3 to plot 3D complex")

        # Normalize common Mesh3d-style kwargs (e.g., color) to Scatter3d-compatible
        # kwargs so existing caller code keeps working.
        base_kwargs: dict[str, Any] = dict(kwargs)
        line_color = base_kwargs.pop("color", None)

        traces: list[go.Mesh3d | go.Scatter3d] = []
        vertices = self.get_bounded_vertices(bound)
        if vertices is None or vertices.size == 0:
            return traces

        if vertices.shape[1] != 3:
            return traces

        # Handle degeneracies via effective dimensionality of the vertices.
        centered = vertices - vertices.mean(axis=0, keepdims=True)
        try:
            _, s, vh = np.linalg.svd(centered, full_matrices=False)
        except Exception:
            s = np.array([])
            vh = np.zeros((0, 3))
        if s.size == 0 or np.all(s < 1e-12):
            # Point-like.
            point_kwargs = dict(base_kwargs)
            if line_color is not None:
                marker = dict(point_kwargs.get("marker", {}))
                marker.setdefault("color", line_color)
                point_kwargs["marker"] = marker
            traces.append(
                go.Scatter3d(
                    x=[vertices[0, 0]],
                    y=[vertices[0, 1]],
                    z=[vertices[0, 2]],
                    mode="markers",
                    showlegend=showlegend,
                    **point_kwargs,
                )
            )
            return traces

        tol = 1e-6 * s[0]
        eff_dim = int(np.sum(s > tol))

        if eff_dim == 1:
            # Edge-like: show as a line segment between extremal points.
            direction = vh[0]
            t = vertices @ direction
            p_min = vertices[np.argmin(t)]
            p_max = vertices[np.argmax(t)]
            seg = np.stack([p_min, p_max], axis=0)
            line_kwargs = dict(base_kwargs)
            if line_color is not None:
                line = dict(line_kwargs.get("line", {}))
                line.setdefault("color", line_color)
                line_kwargs["line"] = line
            traces.append(
                go.Scatter3d(
                    x=seg[:, 0],
                    y=seg[:, 1],
                    z=seg[:, 2],
                    mode="lines",
                    showlegend=showlegend,
                    **line_kwargs,
                )
            )
            return traces

        if eff_dim == 2:
            # Face-like: project onto principal plane to order vertices, then
            # plot as a closed polygon in 3D.
            basis = vh[:2]  # 2 x 3
            coords_2d = vertices @ basis.T
            if self.dim < 2:
                order = np.arange(vertices.shape[0])
            else:
                try:
                    hull_2d = ConvexHull(coords_2d)
                    order = hull_2d.vertices
                except Exception:
                    order = np.arange(vertices.shape[0])
            ordered = vertices[order]
            x = ordered[:, 0].tolist() + [ordered[0, 0]]
            y = ordered[:, 1].tolist() + [ordered[0, 1]]
            z = ordered[:, 2].tolist() + [ordered[0, 2]]
            face_line_kwargs = dict(base_kwargs)
            if line_color is not None:
                line = dict(face_line_kwargs.get("line", {}))
                line.setdefault("color", line_color)
                face_line_kwargs["line"] = line
            if filled and line_color is not None:
                # Use Mesh3d for filled planar face (fan triangulation from vertex 0)
                n = len(ordered)
                i_vals = [0] * (n - 2)
                j_vals = list(range(1, n - 1))
                k_vals = list(range(2, n))
                mesh_kwargs = {k: v for k, v in base_kwargs.items() if k not in ("line", "marker")}
                traces.append(
                    go.Mesh3d(
                        x=ordered[:, 0].tolist(),
                        y=ordered[:, 1].tolist(),
                        z=ordered[:, 2].tolist(),
                        i=i_vals,
                        j=j_vals,
                        k=k_vals,
                        color=line_color,
                        opacity=0.7,
                        showlegend=showlegend,
                        **mesh_kwargs,
                    )
                )
            else:
                traces.append(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="lines",
                        showlegend=showlegend,
                        **face_line_kwargs,
                    )
                )
            return traces

        # Full-dimensional 3D region: plot only the wireframe (edges) of the
        # convex hull, not filled surfaces. Use a single trace with NaN-
        # separated segments so each polyhedron corresponds to one trace.
        if self.dim < 2:
            return traces
        try:
            hull = ConvexHull(vertices)

            # Build edge -> incident facet mapping from the triangulated hull.
            edge_to_facets: dict[tuple[int, int], list[int]] = {}
            for fi, simplex in enumerate(hull.simplices):
                a, b, c = simplex
                for u, v in ((a, b), (a, c), (b, c)):
                    if u == v:
                        continue
                    key = (u, v) if u < v else (v, u)
                    edge_to_facets.setdefault(key, []).append(fi)

            # Normalize facet planes for coplanarity tests.
            planes = hull.equations  # shape (n_facets, 4): [nx, ny, nz, d]
            normals = planes[:, :3]
            offsets = planes[:, 3]
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normals_n = normals / norms
            offsets_n = offsets / norms.ravel()

            def facets_coplanar(facet_indices: list[int], atol: float = 1e-6) -> bool:
                if not facet_indices:
                    return True
                f0 = facet_indices[0]
                n0 = normals_n[f0]
                d0 = offsets_n[f0]
                for fi in facet_indices[1:]:
                    ni = normals_n[fi]
                    di = offsets_n[fi]
                    # Same or opposite normal direction and similar offset.
                    if np.abs(np.dot(n0, ni)) < 1.0 - atol:
                        return False
                    if np.abs(d0 - di) > atol:
                        return False
                return True

            # Keep only edges that are intersections of non-coplanar facets;
            # this removes diagonals introduced by triangulation inside
            # polygonal faces.
            edges: set[tuple[int, int]] = set()
            for edge, facet_indices in edge_to_facets.items():
                if not facets_coplanar(facet_indices):
                    edges.add(edge)

            # When filled=True, add a Mesh3d trace for solid rendering instead of wireframe
            if filled and line_color is not None:
                mesh_kwargs = {k: v for k, v in base_kwargs.items() if k not in ("line", "marker")}
                traces.append(
                    go.Mesh3d(
                        x=vertices[:, 0].tolist(),
                        y=vertices[:, 1].tolist(),
                        z=vertices[:, 2].tolist(),
                        i=hull.simplices[:, 0].tolist(),
                        j=hull.simplices[:, 1].tolist(),
                        k=hull.simplices[:, 2].tolist(),
                        color=line_color,
                        opacity=0.7,
                        showlegend=showlegend,
                        **mesh_kwargs,
                    )
                )
            else:
                # Wireframe for non-filled regions
                edge_line_kwargs = dict(base_kwargs)
                if line_color is not None:
                    line = dict(edge_line_kwargs.get("line", {}))
                    line.setdefault("color", line_color)
                    edge_line_kwargs["line"] = line

                xs: list[float] = []
                ys: list[float] = []
                zs: list[float] = []
                for edge_u, edge_v in edges:
                    seg = vertices[[edge_u, edge_v]]
                    xs.extend([float(seg[0, 0]), float(seg[1, 0]), float("nan")])
                    ys.extend([float(seg[0, 1]), float(seg[1, 1]), float("nan")])
                    zs.extend([float(seg[0, 2]), float(seg[1, 2]), float("nan")])

                traces.append(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        showlegend=showlegend,
                        **edge_line_kwargs,
                    )
                )
        except Exception as e:
            warnings.warn(f"Error while computing 3D mesh for polyhedron {self}: {e}")

        return traces

    def plot_2d_complex(
        self,
        fill: str = "toself",
        showlegend: bool = False,
        bound: float = DEFAULT_PLOT_BOUND,
        plot_halfspaces: bool = False,
        halfspace_shade: bool = True,
        **kwargs: Any,
    ) -> list[go.Scatter]:
        """Plot the polyhedron in 2D using plotly.

        Args:
            fill: Fill mode passed to go.Scatter. Defaults to "toself".
            showlegend: Whether to show in legend. Defaults to False.
            bound: Radius of the bounding hypercube for vertex computation.
                Defaults to config.DEFAULT_PLOT_BOUND.
            plot_halfspaces: If True, add one Scatter trace per halfspace (inequality)
                as line or shaded region. Defaults to False.
            halfspace_shade: When plot_halfspaces is True, shade the feasible side
                of each halfspace. Defaults to True.
            **kwargs: Additional arguments passed to go.Scatter (polyhedron outline).

        Returns:
            list: A list of plotly Scatter traces: [outline_trace] when
                plot_halfspaces is False, or [outline_trace, *halfspace_traces] when
                True. If the main outline fails (e.g. vertex computation or
                ConvexHull raises), returns [] when plot_halfspaces is False, or
                only the halfspace traces when plot_halfspaces is True.

        Raises:
            ValueError: If the polyhedron is not 2D.
        """
        if self.W.shape[0] != 2:
            raise ValueError("Polyhedron must be 2D to plot")
        traces: list[go.Scatter] = []
        try:
            geom = self._get_bounded_plot_geometry(bound)
            if geom is not None:
                kind, verts = geom
                if kind == "polygon":
                    x = verts[:, 0].tolist() + [verts[0, 0]]
                    y = verts[:, 1].tolist() + [verts[0, 1]]
                    traces.append(go.Scatter(x=x, y=y, fill=fill, showlegend=showlegend, **kwargs))
                elif kind == "segment":
                    x = verts[:, 0].tolist()
                    y = verts[:, 1].tolist()
                    traces.append(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            fill=None,
                            showlegend=showlegend,
                            **kwargs,
                        )
                    )
                elif kind == "point":
                    x = verts[:, 0].tolist()
                    y = verts[:, 1].tolist()
                    traces.append(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="markers",
                            fill=None,
                            showlegend=showlegend,
                            **kwargs,
                        )
                    )
        except Exception as e:
            raise e

        if plot_halfspaces:
            W = self.halfspaces_np[:, :-1]
            b = self.halfspaces_np[:, -1]
            bounds = (-bound, bound)
            for i in range(W.shape[0]):
                w = W[i]
                if np.abs(w[1]) < TOL_NEARLY_VERTICAL:
                    # Nearly vertical line: x = -b / w[0]
                    x_line = -b[i] / w[0] if np.abs(w[0]) >= TOL_NEARLY_VERTICAL else 0.0
                    xs = [x_line, x_line]
                    ys = [bounds[0], bounds[1]]
                    halfspace_shade_this = False
                else:
                    halfspace_shade_this = halfspace_shade
                    y0 = (-b[i] - w[0] * bounds[0]) / w[1]
                    y1 = (-b[i] - w[0] * bounds[1]) / w[1]
                    if halfspace_shade_this:
                        outer = max(bounds[1], y0, y1) if w[1] < 0 else min(bounds[0], y0, y1)
                        xs = [bounds[0], bounds[0], bounds[1], bounds[1], bounds[0]]
                        ys = [outer, y0, y1, outer, outer]
                    else:
                        xs = [bounds[0], bounds[1]]
                        ys = [y0, y1]
                traces.append(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        name=f"Halfspace {i}",
                        fill="toself" if halfspace_shade_this else None,
                        visible="legendonly",
                        showlegend=True,
                    )
                )
        return traces

    def plot_2d_graph(
        self,
        fill: str = "toself",
        showlegend: bool = False,
        bound: float = DEFAULT_PLOT_BOUND,
        project: float | None = None,
        **kwargs: Any,
    ) -> dict[str, go.Mesh3d | go.Scatter3d] | None:
        """Plot the polyhedron in 3D using plotly.

        Creates a 3D mesh plot of the polyhedron. The z-coordinates are computed
        by passing the 2D vertices through the network.

        Args:
            fill: Fill mode (not used for 3D plots). Defaults to "toself".
            showlegend: Whether to show in legend. Defaults to False.
            bound: Radius of the bounding hypercube for vertex computation.
                Defaults to config.DEFAULT_PLOT_BOUND.
            project: If a number, projects the polyhedron onto this z-value
                instead of computing it from the network. Defaults to None.
            **kwargs: Additional arguments passed to go.Mesh3d.

        Returns:
            dict or None: Dictionary with 'mesh' and 'outline' keys containing
                plotly traces, or None if plotting fails.

        Raises:
            ValueError: If the polyhedron is not 2D.
        """
        if self.W.shape[0] != 2:
            raise ValueError("Polyhedron must be 2D to plot")
        geom = self._get_bounded_plot_geometry(bound)
        if geom is None:
            return None

        kind, verts = geom
        try:
            # Prepare x, y for all geometry types.
            x = verts[:, 0].tolist()
            y = verts[:, 1].tolist()
            z = (
                (
                    self.net(torch.tensor([x, y], device=self.net.device, dtype=self.net.dtype).T)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()[:, 1]
                )
                if project is None
                else [project] * len(x)
            )

            if kind == "polygon":
                # Close the loop for polygonal surfaces.
                x_closed = x + [x[0]]
                y_closed = y + [y[0]]
                z_closed = z + [z[0]]
                mesh = go.Mesh3d(x=x_closed, y=y_closed, z=z_closed, alphahull=-1, lighting=dict(ambient=1), **kwargs)
                scatter = go.Scatter3d(
                    x=x_closed,
                    y=y_closed,
                    z=z_closed,
                    mode="lines",
                    showlegend=False,
                    line=dict(width=5, color="black"),
                    visible=False,
                )
                return {"mesh": mesh, "outline": scatter}

            if kind == "segment":
                # Only an outline line in 3D.
                scatter = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    showlegend=False,
                    line=dict(width=5, color="black"),
                    **kwargs,
                )
                return {"outline": scatter}

            if kind == "point":
                # Single point plot.
                scatter = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    showlegend=False,
                    marker=dict(size=4, color="black"),
                    **kwargs,
                )
                return {"outline": scatter}
        except Exception as e:
            warnings.warn(f"Error while plotting polyhedron: {e}")
            return None

    def clean_data(self) -> None:
        """Clear cached data to reduce memory usage.

        Removes large cached properties like halfspaces, W matrix, center,
        and halfspace intersection data. Keeps small properties, the sign sequence,
        and the interior point.
        """
        self._halfspaces = None
        self._W = None
        self._b = None
        self._center = None
        self._hs = None
        # self._interior_point = None ## TODO: Does this slow down things?
        self._point = None
        self._halfspaces_np = None

    """
    All of the following properties are computed on the fly and cached
    """

    @property
    def vertex_set(self) -> set[tuple[float, ...]]:
        """Set of vertices of the polyhedron (not always reliable)."""
        if self._hs is None:
            self.compute_properties()
        return self._vertex_set

    @property
    def vertices(self) -> np.ndarray:
        """Vertices of the polyhedron (not always reliable)."""
        if self._vertices is None:
            self.compute_properties()
        return self._vertices

    @property
    def hs(self) -> HalfspaceIntersection:
        """Halfspace intersection object from scipy."""
        if self._hs is None:
            self.compute_properties()
        return self._hs

    @property
    def ch(self) -> ConvexHull | None:
        """Convex hull of the polyhedron for finite polyhedra, or None if unbounded or computation fails."""
        if self._ch is None and self.finite:
            self.compute_properties()
        return self._ch

    @property
    def volume(self) -> float:
        """Volume of the polyhedron, infinity for unbounded polyhedra, or -1 if computation fails."""
        if not self.finite:
            self._volume = float("inf")
        elif self._volume is None:
            self.compute_properties()
        return self._volume

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
            halfspaces, W, b = self.get_hs()
            assert isinstance(halfspaces, torch.Tensor) or isinstance(halfspaces, np.ndarray)
            assert isinstance(W, torch.Tensor) or isinstance(W, np.ndarray)
            assert isinstance(b, torch.Tensor) or isinstance(b, np.ndarray)
            self._halfspaces = halfspaces
            self._W = W
            self._b = b
            self._halfspaces_np = None
            assert isinstance(self._halfspaces, torch.Tensor) or isinstance(self._halfspaces, np.ndarray)
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
            halfspaces, W, b = self.get_hs()
            assert isinstance(halfspaces, torch.Tensor) or isinstance(halfspaces, np.ndarray)
            assert isinstance(W, torch.Tensor) or isinstance(W, np.ndarray)
            assert isinstance(b, torch.Tensor) or isinstance(b, np.ndarray)
            self._halfspaces = halfspaces
            self._W = W
            self._b = b
            self._halfspaces_np = None
            assert isinstance(self._W, torch.Tensor) or isinstance(self._W, np.ndarray)
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
            halfspaces, W, b = self.get_hs()
            assert isinstance(halfspaces, torch.Tensor) or isinstance(halfspaces, np.ndarray)
            assert isinstance(W, torch.Tensor) or isinstance(W, np.ndarray)
            assert isinstance(b, torch.Tensor) or isinstance(b, np.ndarray)
            self._halfspaces = halfspaces
            self._W = W
            self._b = b
            self._halfspaces_np = None
        assert isinstance(self._b, torch.Tensor) or isinstance(self._b, np.ndarray)
        return self._b

    @property
    def num_dead_relus(self) -> int:
        """Number of dead ReLU neurons (neurons always outputting zero).

        Returns:
            int: Count of ReLU neurons that are always inactive for this polyhedron.
        """
        if self._num_dead_relus is None:
            halfspaces, W, b = self.get_hs()
            assert isinstance(halfspaces, torch.Tensor) or isinstance(halfspaces, np.ndarray)
            assert isinstance(W, torch.Tensor) or isinstance(W, np.ndarray)
            assert isinstance(b, torch.Tensor) or isinstance(b, np.ndarray)
            self._halfspaces = halfspaces
            self._W = W
            self._b = b
            self._halfspaces_np = None
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
        """Chebyshev center of the polyhedron for finite polyhedra, or None for unbounded polyhedra."""
        return self._center

    @property
    def inradius(self) -> float:
        """Inradius of the polyhedron (radius of largest inscribed ball), infinity for unbounded polyhedra."""
        if self.finite:
            assert self._inradius is not None  # when finite, get_center_inradius() has set it
            return self._inradius
        return float("inf")

    @property
    def finite(self) -> bool:
        """Whether the polyhedron is bounded (finite)."""
        if self._finite is None:
            self.get_center_inradius()
        return self._finite

    @property
    def shis(self) -> list[int]:
        """Supporting halfspace indices (SHIs)."""
        if self._shis is None:
            result = self.get_shis()
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
            self.get_interior_point()
        return self._interior_point

    @property
    def point(self) -> np.ndarray | None:
        """The center if available, otherwise an interior point. May be None if set explicitly via the setter."""
        if self._point is None:
            if self._center is not None:
                self._point = self._center
            else:
                self._point = self.interior_point
        if self._point is not None:
            self._point = self._point.squeeze()
        return self._point

    @point.setter
    def point(self, value: np.ndarray | None) -> None:
        """Set the representative point manually."""
        self._point = value

    @property
    def interior_point_norm(self) -> float:
        """L2 norm of the interior point."""
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
        """Dimension of the polyhedron, equal to the dimension of the ambient space minus the number of zero sign sequence elements."""
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
        return (dists <= TOL_HALFSPACE_CONTAINMENT).all().item()

    def __mul__(self, other: "Polyhedron") -> "Polyhedron":
        """Returns a new Polyhedron object based on sign sequence multiplication"""
        return Polyhedron(self.net, self.ss + other.ss * (self.ss == 0))

    """The following methods are used for pickling"""

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "_tag": self.tag,
            "_hash": self._hash,
            "_finite": self._finite,
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
