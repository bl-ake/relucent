import hashlib
import warnings
from functools import cached_property

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from gurobipy import GRB, Model
from scipy.spatial import ConvexHull, HalfspaceIntersection
from tqdm.auto import tqdm

from relucent.utils import get_env


def solve_radius(env, halfspaces, max_radius=GRB.INFINITY, zero_indices=None, sense=GRB.MAXIMIZE):
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
            point and radius is the inradius. Returns (None, None) if the polyhedron
            is unbounded and max_radius is infinity.

    Raises:
        ValueError: If the optimization fails or produces invalid results.
    """

    if isinstance(halfspaces, torch.Tensor):
        halfspaces = halfspaces.detach().cpu().numpy()
    A = halfspaces[:, :-1]
    b = halfspaces[:, -1:]
    norm_vector = np.reshape(np.linalg.norm(A, axis=1), (A.shape[0], 1))
    if zero_indices is not None and len(zero_indices) > 0:
        warnings.warn("Working with k<d polyhedron.")
        norm_vector[zero_indices] = 0

    model = Model("Interior Point", env)
    x = model.addMVar((halfspaces.shape[1] - 1, 1), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    y = model.addMVar((1,), ub=max_radius, vtype=GRB.CONTINUOUS, name="y")
    model.addConstr(A @ x + norm_vector * y <= -b)
    model.setObjective(y, sense)
    model.optimize()
    status = model.status

    if status == GRB.OPTIMAL:
        objVal = model.objVal
        x, y = x.X, y.X.item()
        model.close()
        if objVal <= 0:
            raise ValueError(f"Something has gone horribly wrong: objVal={objVal}")
        return x, y
    else:
        if max_radius == GRB.INFINITY:
            model.close()
            return None, None
        else:
            # if status == GRB.INFEASIBLE:
            #     breakpoint()
            model.close()
            raise ValueError(f"Interior Point Model Status: {status}")


def encode_bv(bv):
    """Create a hashable representation of a sign sequence.

    Converts a sign sequence array into a bytes object that can be used as a
    dictionary key or for hashing.

    Args:
        bv: A sign sequence as np.ndarray or torch.Tensor with values in {-1, 0, 1}.

    Returns:
        bytes: A hashable bytes representation of the flattened sign sequence.
    """
    # return tuple(bv.astype(int).flatten().tolist())  ## TODO: Test these two options and compare speeds
    return bv.flatten().tobytes()


class Polyhedron:
    """Represents a polyhedron (linear region) in d-dimensional space.

    Several methods use Gurobi environments for optimization. If one is not
    provided, an environment will be created automatically.
    """

    MAX_RADIUS = 100  ## The smaller the faster, but making this value too small can exclude some polyhedrons

    def __init__(self, net, bv, halfspaces=None, W=None, b=None, point=None, shis=None, bound=None, **kwargs):
        """Create a Polyhedron object.

        The kwargs can be used to supply precomputed values for various properties.

        Args:
            net: Instance of the NN class from the "model" module.
            bv: Sign sequence defining the polyhedron (values in {-1, 0, 1}).
        """
        self.net = net
        self.bv = bv
        if not isinstance(bv, torch.Tensor):
            bv = torch.from_numpy(bv)
            if net is not None:
                bv = bv.to(net.device, net.dtype)
        self._halfspaces = halfspaces
        self._W = W
        self._b = b
        self._Wl2 = None
        if isinstance(point, torch.Tensor):
            point = point.detach().cpu().numpy()
        self._point = point
        self._interior_point = None
        self._interior_point_norm = None
        self._center = None
        self._inradius = None
        self._num_dead_relus = None
        self.bound = bound

        self._shis = shis
        self._hs = None
        self._ch = None
        self._finite = None
        self._vertices = None
        self._volume = None

        self._hash = None
        self._tag = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def compute_properties(self):
        """Compute additional geometric properties for low-dimensional polyhedra.

        Returns:
            bool: True if computation succeeded.

        Raises:
            ValueError: If input dimension > 6 or if computation fails.
        """
        if self.net.input_shape[0] > 6:
            raise ValueError("Input shape too large to compute extra properties")
        try:
            # warnings.warn("Computing Additional Properties")
            halfspaces = (
                self.halfspaces.detach().cpu().numpy() if isinstance(self.halfspaces, torch.Tensor) else self.halfspaces
            )
            hs = HalfspaceIntersection(
                halfspaces,
                self.interior_point,
                # qhull_options="Qx",
            )  ## http://www.qhull.org/html/qh-optq.htm
        except Exception:
            raise ValueError("Error while computing halfspace intersection")
        self._hs = hs
        self._shis = hs.dual_vertices.flatten().tolist()
        vertices = hs.intersections
        trust_verticies = (halfspaces[self.shis, :-1] @ vertices.T + halfspaces[self.shis, -1, None]).sum(axis=0) < 0.01
        self._vertices = vertices[trust_verticies]
        self._vertex_set = set(tuple(x) for x in self.vertices)
        if self.finite:
            try:
                self._ch = ConvexHull(vertices)
            except Exception:
                # warnings.warn("Error while computing convex hull:", e)
                self._ch = None
        return True

    def get_interior_point(self, env=None, max_radius=None, zero_indices=None):
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
        if self._center is not None:
            self._interior_point = self._center.squeeze()
        else:
            env = env or get_env()
            self._interior_point = solve_radius(
                env, self.halfspaces[:], max_radius=max_radius, zero_indices=zero_indices
            )[0].squeeze()
        if self._interior_point is None:
            raise ValueError("Interior point not found")
        return self._interior_point

    def get_center_inradius(self, env=None):
        """Get the Chebyshev center and inradius of the polyhedron.

        Also sets self._finite to indicate if the polyhedron is finite.

        Args:
            env: Gurobi environment for optimization. If None, uses a cached
                environment. Defaults to None.

        Returns:
            tuple: (center, inradius) where center is None for unbounded polyhedra.
        """
        env = env or get_env()
        self._center, self._inradius = solve_radius(env, self.halfspaces[:])
        self._finite = self._center is not None
        return self._center, self._inradius

    def get_hs(self, data=None, get_all_Ab=False):
        """Get the halfspace representation of this polyhedron.

        Computes the halfspaces (inequality constraints) that define the polyhedron
        from all neurons in the network. The result includes constraints from
        every neuron, not just the supporting hyperplanes.

        Args:
            data: Optional input data to the network for verification. If provided,
                checks that computed outputs match network outputs. Defaults to None.
            get_all_Ab: If True, returns all intermediate affine maps (A, b) for
                each layer instead of just the final halfspaces. Defaults to False.

        Returns:
            tuple: (halfspaces, W, b) where:
                - halfspaces: Array of shape (n_constraints, n_dim+1) with bias terms
                - W: Affine transformation matrix
                - b: Affine transformation bias vector
        """
        if isinstance(self.bv, torch.Tensor):
            return self._get_hs_torch(data, get_all_Ab)
        elif isinstance(self.bv, np.ndarray):
            return self._get_hs_numpy(data, get_all_Ab)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def _get_hs_torch(self, data=None, get_all_Ab=False):
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
                if current_A is None:
                    constr_A = torch.empty((A.shape[1], 0), device=self.net.device, dtype=self.net.dtype)
                    constr_b = torch.empty((1, 0), device=self.net.device, dtype=self.net.dtype)
                    current_A = torch.eye(A.shape[1], device=self.net.device, dtype=self.net.dtype)
                    current_b = torch.zeros((1, A.shape[1]), device=self.net.device, dtype=self.net.dtype)

                current_A = current_A @ A.T
                current_b = current_b @ A.T + b
            elif isinstance(layer, nn.ReLU):
                mask = self.bv[0, current_mask_index : current_mask_index + current_A.shape[1]]

                new_constr_A = current_A * mask
                new_constr_b = current_b * mask

                constr_A = torch.concatenate(
                    (constr_A, new_constr_A[:, mask != 0], current_A[:, mask == 0], -current_A[:, mask == 0]), axis=1
                )
                constr_b = torch.concatenate(
                    (constr_b, new_constr_b[:, mask != 0], current_b[:, mask == 0], -current_b[:, mask == 0]), axis=1
                )

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
                assert torch.allclose(outs[name], (data @ current_A) + current_b, atol=1e-5)
            if get_all_Ab:
                all_Ab.append({"A": current_A.clone(), "b": current_b.clone(), "layer": layer})
        self._num_dead_relus = (torch.abs(constr_A) < 1e-8).all(dim=0).sum().item()
        halfspaces = torch.hstack((-constr_A.T, -constr_b.reshape(-1, 1)))
        if get_all_Ab:
            return all_Ab
        return halfspaces, current_A, current_b

    @torch.no_grad()
    def _get_hs_numpy(self, data=None, get_all_Ab=False):
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
                if hasattr(layer, "weight_cpu"):
                    A = layer.weight_cpu
                    b = layer.bias_cpu
                else:
                    A = layer.weight.detach().cpu().numpy()
                    b = layer.bias[None, :].detach().cpu().numpy()
                    layer.weight_cpu = A
                    layer.bias_cpu = b
                if current_A is None:
                    constr_A = np.empty((A.shape[1], 0))
                    constr_b = np.empty((1, 0))
                    current_A = np.eye(A.shape[1])
                    current_b = np.zeros((1, A.shape[1]))

                current_A = current_A @ A.T
                current_b = current_b @ A.T + b
            elif isinstance(layer, nn.ReLU):
                if current_A is None:
                    raise ValueError("ReLU layer must follow a linear layer")
                mask = self.bv[0, current_mask_index : current_mask_index + current_A.shape[1]]

                new_constr_A = current_A * mask
                new_constr_b = current_b * mask

                constr_A = np.concatenate(
                    (constr_A, new_constr_A[:, mask != 0], current_A[:, mask == 0], -current_A[:, mask == 0]), axis=1
                )
                constr_b = np.concatenate(
                    (constr_b, new_constr_b[:, mask != 0], current_b[:, mask == 0], -current_b[:, mask == 0]), axis=1
                )

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
                assert np.allclose(outs[name].detach().cpu().numpy(), (data @ current_A) + current_b, atol=1e-5)
            if get_all_Ab:
                all_Ab.append({"A": current_A.copy(), "b": current_b.copy(), "layer": layer})
        self._num_dead_relus = (np.abs(constr_A) < 1e-8).all(axis=0).sum().item()
        halfspaces = np.hstack((-constr_A.T, -constr_b.reshape(-1, 1)))
        if get_all_Ab:
            return all_Ab
        return halfspaces, current_A, current_b

    def get_bounded_halfspaces(self, bound, env=None):
        """Get halfspaces after adding bounding box constraints.

        Adds constraints that bound the space to a hypercube of radius 'bound'
        around the origin. Useful for plotting and visualization. Returns None
        if the polyhedron doesn't intersect the bounded region.

        Args:
            bound: Radius of the bounding hypercube.
            env: Gurobi environment for feasibility checking. If None, uses
                a cached environment. Defaults to None.

        Returns:
            np.ndarray or None: Halfspaces with bounding constraints added, or
                None if the polyhedron doesn't intersect the bounded region.
        """
        bounds_lhs = np.eye(self.halfspaces.shape[1] - 1)
        bounds_rhs = -np.ones((self.halfspaces.shape[1] - 1, 1)) * bound
        halfspaces = np.vstack(
            (
                self.halfspaces if isinstance(self.halfspaces, np.ndarray) else self.halfspaces.detach().cpu().numpy(),
                np.hstack((bounds_lhs, bounds_rhs)),
                np.hstack((-bounds_lhs, bounds_rhs)),
            )
        )
        env = env or get_env()
        feasible = solve_radius(env, halfspaces, max_radius=bound)[0] is not None
        if feasible:
            return halfspaces
        else:
            return None

    def __eq__(self, other):
        if isinstance(other, Polyhedron):
            return self.tag == other.tag  # and (self.bv == other.bv).all()

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
        collect_info=False,
        bound=GRB.INFINITY,
        subset=None,
        tol=1e-6,
        new_method=False,
        env=None,
        shi_pbar=False,
    ):
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
            tol: Inequality tolerance to improve numerical stability. Defaults to 1e-6.
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
        A = (self.halfspaces.detach().cpu().numpy() if isinstance(self.halfspaces, torch.Tensor) else self.halfspaces)[
            :, :-1
        ]
        b = (self.halfspaces.detach().cpu().numpy() if isinstance(self.halfspaces, torch.Tensor) else self.halfspaces)[
            :, -1:
        ]
        env = env or get_env()
        model = Model("SHIS", env)
        x = model.addMVar((self.halfspaces.shape[1] - 1, 1), lb=-bound, ub=bound, vtype=GRB.CONTINUOUS, name="x")
        constrs = model.addConstr(A @ x == -b - tol, name="hyperplanes")
        model.optimize()
        if model.status == GRB.OPTIMAL:
            # print("All Hyperplanes Intersect")
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
            if (A[i] == 0).all():
                continue
            model.update()
            pbar.set_postfix_str(f"#shis: {len(shis)}")
            constrs[i].setAttr("RHS", constrs[i].getAttr("RHS") + 1)
            # breakpoint()
            model.setObjective((A[i] @ x).item() + b[i, 0], GRB.MAXIMIZE)
            # model.setObjective(gp.quicksum([(A[i] @ x).item(), b[i]]), GRB.MAXIMIZE)
            model.params.BestObjStop = 1e-5
            model.params.BestBdStop = -1e-5
            model.update()
            model.optimize()

            if model.status == GRB.OPTIMAL or model.status == GRB.USER_OBJ_LIMIT:
                if model.objVal >= 0:
                    dists = A @ x.X + b
                    if (dists > 0).sum() != 1:
                        warnings.warn(
                            f"Invalid Proof for SHI {i}! Violation Sizes: {np.argwhere(dists.flatten() > 0), dists[np.argwhere(dists.flatten() > 0)]}"
                        )
                    else:
                        shis.append(i)

                basis_indices = constrs.CBasis.flatten() != 0
                if new_method:
                    if basis_indices.sum() != A.shape[1]:
                        warnings.warn("Bound Constraints in Basis")
                skip_size = 0
                if new_method and basis_indices.sum() == A.shape[1]:
                    point_shis = self.halfspaces[basis_indices, :-1]  # (d(# point shis) x d)
                    others = self.halfspaces[~basis_indices, :-1]  # (num_other_hyperplanes x d)
                    try:
                        sols = torch.linalg.solve(point_shis, others.T)
                    except torch._C._LinAlgError:
                        warnings.warn("Could not solve linear system")
                        sols = torch.zeros_like(others.T, device=self.halfspaces.device)
                    all_correct = (sols > 0).all(axis=0)
                    assert all_correct.shape[0] == others.shape[0]
                    correct_indices = torch.argwhere(all_correct).reshape(-1)
                    if correct_indices.shape[0] > 0:
                        A_indices = torch.arange(A.shape[0], device=self.halfspaces.device)[~basis_indices][all_correct]

                        old_len = len(subset)
                        subset -= set(A_indices.detach().cpu().numpy().flatten().tolist())
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
                    poly_info[-1]["x Norm"] = (np.linalg.norm(x.X),)
                if collect_info == "All":
                    poly_info[-1] |= {"Slacks": constrs.Slack, "-b[i]": -b[i], "Status": model.status}

                    if hasattr(x, "X"):
                        poly_info[-1]["Proof"] = x.X

            constrs[i].setAttr("RHS", -b[i] - tol)
            pbar.update(A.shape[0] - len(subset) - pbar.n)
        model.close()
        if collect_info:
            return shis, poly_info
        return shis

    def nflips(self, other):
        """Calculate the number of non-zero sign sequence elements that differ.

        Args:
            other: Another Polyhedron object to compare with.

        Returns:
            int: The number of sign sequence elements that differ.
        """
        return (self.bv * other.bv == -1).sum().item()

    def is_face_of(self, other):
        """Check if this polyhedron is a face of another polyhedron.

        Args:
            other: Another Polyhedron object to check against.

        Returns:
            bool: True if this polyhedron is a face of the other.
        """
        return ((self * other).bv == other.bv).all()

    def get_bounded_vertices(self, bound):
        """Get the vertices of the polyhedron within a bounding hypercube.

        Computes the vertices of the polyhedron after intersecting it with a
        hypercube of radius 'bound'. Primarily used for plotting and visualization.

        Args:
            bound: Radius of the bounding hypercube.

        Returns:
            np.ndarray or None: Array of vertex coordinates, or None if the
                polyhedron doesn't intersect the bounded region or computation fails.
        """
        try:
            bounded_halfspaces = self.get_bounded_halfspaces(bound)
        except ValueError as e:
            print("Could not get bounded halfspaces")
            print(e)
            return None
        # int_point, _ = solve_radius(get_env(), bounded_halfspaces, max_radius=1000)
        if not (self.interior_point @ bounded_halfspaces[:, :-1].T + bounded_halfspaces[:, -1] <= 1e-8).all():
            warnings.warn(f"Interior point ({self.interior_point}) out of bounds ({bound}):")
            return None
        hs = HalfspaceIntersection(
            bounded_halfspaces,
            self.interior_point,
            # qhull_options="QbB",
        )  ## http://www.qhull.org/html/qh-optq.htm
        vertices = hs.intersections
        return vertices

    def plot2d(self, fill="toself", showlegend=False, bound=1000, **kwargs):
        """Plot the polyhedron in 2D using plotly.

        Args:
            fill: Fill mode passed to go.Scatter. Defaults to "toself".
            showlegend: Whether to show in legend. Defaults to False.
            bound: Radius of the bounding hypercube for vertex computation.
                Defaults to 1000.
            **kwargs: Additional arguments passed to go.Scatter.

        Returns:
            plotly.graph_objects.Scatter or None: A plotly Scatter trace, or None
                if plotting fails.

        Raises:
            ValueError: If the polyhedron is not 2D.
        """
        if self.W.shape[0] != 2:
            raise ValueError("Polyhedron must be 2D to plot")
        vertices = self.get_bounded_vertices(bound)
        if vertices is not None:
            try:
                hull = ConvexHull(vertices)
                x = vertices[hull.vertices, 0].tolist() + [vertices[hull.vertices, 0][0]]
                y = vertices[hull.vertices, 1].tolist() + [vertices[hull.vertices, 1][0]]
                return go.Scatter(x=x, y=y, fill=fill, showlegend=showlegend, **kwargs)
            except Exception as e:
                print(self, e)
                return None
        else:
            return None

    def plot3d(self, fill="toself", showlegend=False, bound=1000, project=None, **kwargs):
        """Plot the polyhedron in 3D using plotly.

        Creates a 3D mesh plot of the polyhedron. The z-coordinates are computed
        by passing the 2D vertices through the network.

        Args:
            fill: Fill mode (not used for 3D plots). Defaults to "toself".
            showlegend: Whether to show in legend. Defaults to False.
            bound: Radius of the bounding hypercube for vertex computation.
                Defaults to 1000.
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
        vertices = self.get_bounded_vertices(bound)
        if vertices is not None:
            try:
                hull = ConvexHull(vertices)
                x = vertices[hull.vertices, 0].tolist() + [vertices[hull.vertices, 0][0]]
                y = vertices[hull.vertices, 1].tolist() + [vertices[hull.vertices, 1][0]]
                z = (
                    (
                        self.net(torch.tensor([x, y], device=self.net.device, dtype=self.net.dtype).T)
                        .detach()
                        .cpu()
                        .numpy()
                        .squeeze()[:, 1]
                    )
                    if project is None
                    else [project] * (len(x))
                )
                mesh = go.Mesh3d(x=x, y=y, z=z, alphahull=-1, lighting=dict(ambient=1), **kwargs)

                scatter = go.Scatter3d(
                    x=x, y=y, z=z, mode="lines", showlegend=False, line=dict(width=5, color="black"), visible=False
                )
            except Exception as e:
                warnings.warn(f"Error while plotting polyhedron: {e}")
                return None

            return {"mesh": mesh, "outline": scatter}

        else:
            return None

    def clean_data(self):
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

    """
    All of the following properties are computed on the fly and cached
    """

    @property
    def vertex_set(self):
        """Set of vertices of the polyhedron (not always reliable)."""
        if self._hs is None:
            self.compute_properties()
        return self._vertex_set

    @property
    def vertices(self):
        """Vertices of the polyhedron (not always reliable).

        Returns:
            np.ndarray: Array of vertex coordinates.
        """
        if self._vertices is None:
            self.compute_properties()
        return self._vertices

    @property
    def hs(self):
        """Halfspace intersection object from scipy."""
        if self._hs is None:
            self.compute_properties()
        return self._hs

    @property
    def ch(self):
        """Convex hull of the polyhedron for finite polyhedra."""
        if self._ch is None and self.finite:
            self.compute_properties()
        return self._ch

    @property
    def volume(self):
        """Volume of the polyhedron, infinity for unbounded polyhedra,
        or -1 if computation fails.
        """
        if not self.finite:
            self._volume = float("inf")
        elif self._volume is None:
            try:
                if self.ch is None:
                    self._volume = -1
                else:
                    self._volume = self.ch.volume
            except Exception:
                self._volume = -1
        return self._volume

    @cached_property  ## !! See if this works
    def tag(self):
        """Unique tag for this polyhedron, computed as a hashable
        representation of the sign sequence.
        """
        if self._tag is None:
            self._tag = encode_bv(
                self.bv.detach().cpu().numpy().squeeze() if isinstance(self.bv, torch.Tensor) else self.bv
            )
        return self._tag

    @property
    def halfspaces(self):
        """Halfspace representation of the polyhedron.

        Returns:
            torch.Tensor or np.ndarray: Array of shape (n_constraints, n_dim+1)
                where each row is [a1, a2, ..., ad, b] representing the
                constraint a^T x + b <= 0.
        """
        if self._halfspaces is None:
            self._halfspaces, self._W, self._b = self.get_hs()
        return self._halfspaces

    @property
    def W(self):
        """Affine transformation matrix W such that the polyhedron maps to W*x + b.

        Returns:
            torch.Tensor or np.ndarray: Transformation matrix.
        """
        if self._W is None:
            self._halfspaces, self._W, self._b = self.get_hs()
        return self._W

    @property
    def b(self):
        """Affine transformation bias vector such that the polyhedron maps to W*x + b.

        Returns:
            torch.Tensor or np.ndarray: Bias vector.
        """
        if self._b is None:
            self._halfspaces, self._W, self._b = self.get_hs()
        return self._b

    @property
    def num_dead_relus(self):
        """Number of dead ReLU neurons (neurons always outputting zero).

        Returns:
            int: Count of ReLU neurons that are always inactive for this polyhedron.
        """
        if self._num_dead_relus is None:
            self._halfspaces, self._W, self._b = self.get_hs()
        return self._num_dead_relus

    @property
    def Wl2(self):
        """L2 norm of the transformation matrix W."""
        if self._Wl2 is None:
            if isinstance(self.W, torch.Tensor):
                self._Wl2 = torch.linalg.norm(self.W).item()
            elif isinstance(self.W, np.ndarray):
                self._Wl2 = np.linalg.norm(self.W)
            else:
                raise NotImplementedError
        return self._Wl2

    @property
    def center(self):
        """Chebyshev center of the polyhedron for finite polyhedra, or None for unbounded polyhedra."""
        if self.finite:
            return self._center

    @property
    def inradius(self):
        """Inradius of the polyhedron (radius of largest inscribed ball), infinity for unbounded polyhedra."""
        if self.finite:
            return self._inradius
        else:
            return float("inf")

    @property
    def finite(self):
        """Whether the polyhedron is bounded (finite)."""
        if self._finite is None:
            self.get_center_inradius()
        return self._finite

    @property
    def shis(self):
        """Supporting halfspace indices (SHIs)."""
        if self._shis is None:
            self._shis = self.get_shis()
        return self._shis

    @property
    def num_shis(self):
        """Number of faces."""
        return len(self.shis)

    @property
    def num_faces(self):
        """Alias for Polyhedron.num_shis"""
        return self.num_shis

    @property
    def interior_point(self):
        """np.ndarray: A point guaranteed to be inside the polyhedron."""
        # if (self.bv == 0).any():
        #     raise NotImplementedError("Interior point for non-maximal cells is not implemented")
        if self._interior_point is None:
            self.get_interior_point(zero_indices=np.argwhere((self.bv == 0).any(axis=1)).flatten())
        return self._interior_point

    @property
    def point(self):
        """The center if available, otherwise an interior point."""
        if self._point is None:
            if self._center is not None:
                self._point = self._center
            else:
                self._point = self.interior_point
        if self._point is not None:
            self._point = self._point.squeeze()
        return self._point

    @point.setter
    def point(self, value):
        """Set the representative point manually."""
        self._point = value

    @property
    def interior_point_norm(self):
        """L2 norm of the interior point."""
        if self._interior_point_norm is None:
            self._interior_point_norm = np.linalg.norm(self.interior_point).item()
        return self._interior_point_norm

    def __getitem__(self, key):
        return self.bv[key]

    def __repr__(self):
        h = hashlib.blake2b(key=b"hi")
        h.update(self.tag)
        return h.hexdigest()[:8]

    def __contains__(self, point):
        """Check if a point (ndarray or Tensor) is contained in the polyhedron."""
        if not isinstance(point, torch.Tensor):
            point = torch.Tensor(point).to(self.net.device, self.net.dtype)
        point = point.reshape(1, -1)
        return (point @ self.halfspaces[:, :-1].T + self.halfspaces[:, -1] <= 1e-8).all()

    def __mul__(self, other):
        """Returns a new Polyhedron object based on sign sequence multiplication"""
        return Polyhedron(self.net, self.bv + other.bv * (self.bv == 0))

    """The following methods are used for pickling"""

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
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
        }

    def __reduce__(self):
        return (
            Polyhedron,
            (None, self.bv.detach().cpu().numpy() if isinstance(self.bv, torch.Tensor) else self.bv),
            self.__getstate__(),
        )  # Control what gets saved, do not pickle the net
