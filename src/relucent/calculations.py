"""Polyhedral computation helpers (halfspaces, SHIs, qhull interior / SHI models).

Functions here take a :class:`~relucent.poly.Polyhedron` instance; the class lives in
``poly.py`` to avoid import cycles.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, overload

if TYPE_CHECKING:
    from relucent.poly import Polyhedron

import numpy as np
import torch
import torch.nn as nn
from gurobipy import GRB, Model
from scipy.spatial import ConvexHull, HalfspaceIntersection
from tqdm.auto import tqdm

from relucent.config import (
    GUROBI_SHI_BEST_BD_STOP,
    GUROBI_SHI_BEST_OBJ_STOP,
    QHULL_MODE,
    TOL_DEAD_RELU,
    TOL_SHI_HYPERPLANE,
    TOL_VERIFY_AB_ATOL,
    VERTEX_TRUST_THRESHOLD,
)
from relucent.utils import get_env

def solve_radius(
    env: Any,
    halfspaces: np.ndarray | torch.Tensor,  ## TODO: Remove redundant check for this
    max_radius: float = GRB.INFINITY,
    zero_indices: np.ndarray | None = None,
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
        raise NotImplementedError("Working with k<d polyhedron is not supported yet.")
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


@torch.no_grad()
def adjacent_polyhedra(
    poly: Polyhedron,
    ss2poly: Callable[..., Polyhedron],
) -> set[Polyhedron]:
    """Polyhedra adjacent to ``poly`` across one bounding hyperplane (one SHI flip).

    Also works on lower-dimensional polyhedra. ``ss2poly`` maps a sign sequence
    array to the corresponding :class:`Polyhedron` (e.g. ``Complex.ss2poly``).
    """
    ps: set[Polyhedron] = set()
    for shi in poly.shis:
        if poly.ss_np[0, shi] == 0:
            continue
        ss = poly.ss_np.copy()
        ss[0, shi] = -ss[0, shi]
        ps.add(ss2poly(ss))
    return ps


@overload
def get_hs(
    poly: Polyhedron,
    data: torch.Tensor | None = None,
    get_all_Ab: Literal[False] = False,
    force_numpy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | tuple[np.ndarray, np.ndarray, np.ndarray, int]: ...


@overload
def get_hs(
    poly: Polyhedron,
    data: torch.Tensor | None = None,
    get_all_Ab: Literal[True] = True,
    force_numpy: bool = False,
) -> list[dict[str, Any]]: ...


def get_hs(
    poly: Polyhedron,
    data: torch.Tensor | None = None,
    get_all_Ab: bool = False,
    force_numpy: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    | tuple[np.ndarray, np.ndarray, np.ndarray, int]
    | list[dict[str, Any]]
):
    """Halfspace representation of ``poly`` from all neurons in the network.

    Includes constraints from every neuron, not only supporting hyperplanes.

    Args:
        poly: Polyhedron whose sign sequence defines the region.
        data: Optional network input for verifying intermediate affine maps.
        get_all_Ab: If True, return per-layer ``A``, ``b`` instead of final halfspaces.
        force_numpy: If True, use the NumPy path even when ``ss`` is a tensor.

    Returns:
        If ``get_all_Ab`` is False: ``(halfspaces, W, b, num_dead_relus)``.
        If True: list of dicts with ``A``, ``b``, and ``layer`` keys.
    """
    if isinstance(poly._ss, torch.Tensor) and not force_numpy:
        return _get_hs_torch(poly, data, get_all_Ab)
    return _get_hs_numpy(poly, data, get_all_Ab)


@overload
def _get_hs_torch(
    poly: Polyhedron,
    data: torch.Tensor | None = None,
    get_all_Ab: Literal[False] = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]: ...


@overload
def _get_hs_torch(
    poly: Polyhedron,
    data: torch.Tensor | None = None,
    get_all_Ab: Literal[True] = True,
) -> list[dict[str, Any]]: ...


@torch.no_grad()
def _get_hs_torch(
    poly: Polyhedron,
    data: torch.Tensor | None = None,
    get_all_Ab: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | list[dict[str, Any]]:
    assert isinstance(poly._ss, torch.Tensor)
    constr_A, constr_b = None, None
    current_A, current_b = None, None
    A, b = None, None
    if data is not None:
        outs = poly.net.get_all_layer_outputs(data)
    all_Ab = []
    current_mask_index = 0
    for name, layer in poly.net.layers.items():
        if isinstance(layer, nn.Linear):
            A = layer.weight
            b = layer.bias[None, :]
            if current_A is None or current_b is None:
                constr_A = torch.empty((A.shape[1], 0), device=poly.net.device, dtype=poly.net.dtype)
                constr_b = torch.empty((1, 0), device=poly.net.device, dtype=poly.net.dtype)
                current_A = torch.eye(A.shape[1], device=poly.net.device, dtype=poly.net.dtype)
                current_b = torch.zeros((1, A.shape[1]), device=poly.net.device, dtype=poly.net.dtype)

            current_A = current_A @ A.T
            current_b = current_b @ A.T + b
        elif isinstance(layer, nn.ReLU):
            assert current_A is not None
            assert current_b is not None

            mask = poly._ss[0, current_mask_index : current_mask_index + current_A.shape[1]]

            ## Replace mask 0s with 1s
            nonzero_mask = torch.where(mask == 0, torch.ones_like(mask), mask)

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
            raise ValueError(f"Error while processing layer {name} - Unsupported layer type: {type(layer)} ({layer})")
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

    num_dead_relus = int((torch.abs(constr_A) < TOL_DEAD_RELU).all(dim=0).sum().item())
    halfspaces = torch.hstack((-constr_A.T, -constr_b.reshape(-1, 1)))

    if get_all_Ab:
        return all_Ab

    assert isinstance(halfspaces, torch.Tensor)
    assert isinstance(current_A, torch.Tensor)
    assert isinstance(current_b, torch.Tensor)

    assert halfspaces.shape[0] == poly._ss.shape[1]
    return halfspaces, current_A, current_b, num_dead_relus


@overload
def _get_hs_numpy(
    poly: Polyhedron,
    data: torch.Tensor | None = None,
    get_all_Ab: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]: ...


@overload
def _get_hs_numpy(
    poly: Polyhedron,
    data: torch.Tensor | None = None,
    get_all_Ab: Literal[True] = True,
) -> list[dict[str, Any]]: ...


@torch.no_grad()
def _get_hs_numpy(
    poly: Polyhedron,
    data: torch.Tensor | None = None,
    get_all_Ab: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | list[dict[str, Any]]:
    constr_A, constr_b = None, None
    current_A, current_b = None, None
    A, b = None, None
    if data is not None:
        outs = poly.net.get_all_layer_outputs(data)
    all_Ab = []
    current_mask_index = 0
    for name, layer in poly.net.layers.items():
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
            mask = poly.ss_np[0, current_mask_index : current_mask_index + current_A.shape[1]]

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
            raise ValueError(f"Error while processing layer {name} - Unsupported layer type: {type(layer)} ({layer})")
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

    num_dead_relus = (np.abs(constr_A) < TOL_DEAD_RELU).all(axis=0).sum().item()
    halfspaces = np.hstack((-constr_A.T, -constr_b.reshape(-1, 1)))
    if get_all_Ab:
        return all_Ab
    assert isinstance(halfspaces, np.ndarray)
    assert isinstance(current_A, np.ndarray)
    assert isinstance(current_b, np.ndarray)

    assert halfspaces.shape[0] == poly.ss_np.shape[1]
    return halfspaces, current_A, current_b, num_dead_relus


def get_shis(
    poly: Polyhedron,
    collect_info: bool | str = False,
    bound: float = GRB.INFINITY,
    subset: Iterable[int] | None = None,
    tol: float = TOL_SHI_HYPERPLANE,
    new_method: bool = False,
    env: Any = None,
    shi_pbar: bool = False,
    push_size: float = 1.0,
) -> list[int] | tuple[list[int], list]:
    """Supporting halfspace indices (SHIs) for ``poly``.

    Indices of non-redundant halfspaces on the boundary (neurons whose BHs are
    actually faces of the polyhedron).

    Args:
        poly: Polyhedron to analyze.
        collect_info: If true, also return debug info; ``"All"`` adds more detail.
        bound: Hypercube bound for the Gurobi variable box.
        subset: Halfspace indices to consider; default is all.
        tol: Inequality tolerance for proofs.
        new_method: Extra basis-based skipping (does not improve runtime).
        env: Gurobi environment; default uses :func:`~relucent.utils.get_env`.
        shi_pbar: Show a progress bar.
        push_size: RHS relaxation size when testing a candidate SHI.

    Returns:
        List of SHI indices, or ``(shis, info)`` if ``collect_info`` is truthy.

    Raises:
        ValueError: If the initial Gurobi solve fails.
    """
    shis = []
    A = poly.halfspaces_np[:, :-1]
    b = poly.halfspaces_np[:, -1:]
    env = env or get_env()
    model = Model("SHIS", env)
    x = model.addMVar((poly.halfspaces_np.shape[1] - 1, 1), lb=-bound, ub=bound, vtype=GRB.CONTINUOUS, name="x")
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
        if i >= poly.ss_np.shape[1] or poly.ss_np[0, i] == 0:
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
                    poly.warnings.append(w)
                    warnings.warn(w)
                else:
                    shis.append(i)

            basis_indices = constrs.CBasis.ravel() != 0
            if new_method:
                if basis_indices.sum() != A.shape[1]:
                    warnings.warn("Bound Constraints in Basis")
            skip_size = 0
            if new_method and basis_indices.sum() == A.shape[1]:
                point_shis = poly.halfspaces[basis_indices, :-1]  # (d(# point shis) x d)
                others = poly.halfspaces[~basis_indices, :-1]  # (num_other_hyperplanes x d)
                try:
                    sols = torch.linalg.solve(point_shis, others.T)
                except RuntimeError:
                    warnings.warn("Could not solve linear system")
                    sols = torch.zeros(others.T.shape, device=poly.halfspaces.device)
                all_correct = (sols > 0).all(dim=0)
                assert all_correct.shape[0] == others.shape[0]
                correct_indices = torch.argwhere(all_correct).reshape(-1)
                if correct_indices.shape[0] > 0:
                    A_indices = torch.arange(A.shape[0], device=poly.halfspaces.device)[~basis_indices][all_correct]

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


def compute_properties(poly: Polyhedron, qhull_mode: str = QHULL_MODE) -> None:
    """Compute additional geometric properties for low-dimensional polyhedra (vertices, hull, volume).

    Mutates ``poly`` cache fields (``_hs``, ``_vertices``, ``_ch``, ``_volume``,
    ``_attempted_compute_properties``). No-op if already attempted.

    Raises:
        ValueError: If input dimension > 6, interior point is missing, or qhull fails
            (depending on ``qhull_mode``).
    """
    if poly._attempted_compute_properties:
        return
    poly._attempted_compute_properties = True

    if poly.net.input_shape[0] > 6:
        raise ValueError("Input shape too large to compute extra properties")
    try:
        halfspaces = poly.halfspaces_np
        if poly.interior_point is None:
            raise ValueError("Interior point not found")
        with warnings.catch_warnings(record=True) as w:
            hs = HalfspaceIntersection(
                halfspaces,
                poly.interior_point,
                qhull_options=None,
            )  # http://www.qhull.org/html/qh-optq.htm
        if w:
            msgs = "; ".join(str(wi.message) for wi in w)
            if qhull_mode == "IGNORE":
                poly.warnings.extend([RuntimeWarning(wi) for wi in w])
            if qhull_mode == "WARN_ALL":
                warnings.warn(f"HalfspaceIntersection emitted warnings in WARN_ALL mode: {msgs}")
            elif qhull_mode == "HIGH_PRECISION":
                raise ValueError(f"HalfspaceIntersection emitted warnings in HIGH_PRECISION mode: {msgs}")
            elif qhull_mode == "JITTERED":
                with warnings.catch_warnings(record=True) as w2:
                    new_hs = HalfspaceIntersection(
                        halfspaces,
                        poly.interior_point,
                        qhull_options="QJ",  # Triangulated output is approximately 1000 times more accurate than joggled input.
                    )  # http://www.qhull.org/html/qh-optq.htm
                if w2:
                    poly.warnings.append(
                        RuntimeWarning(
                            "Recomputing HalfspaceIntersection with jitter option 'QJ' still had numerical problems"
                        )
                    )
                    poly.warnings.extend([RuntimeWarning(wi) for wi in w2])
                    msgs = "; ".join(str(wi.message) for wi in w)
                else:
                    ## Jittering solved the numerical problems
                    hs = new_hs
    except ValueError:
        raise  # Our HIGH_PRECISION raise - do not retry
    except Exception as e:
        if qhull_mode == "JITTERED":
            try:
                if poly.interior_point is None:
                    raise ValueError("Interior point not found")
                hs = HalfspaceIntersection(
                    halfspaces,
                    poly.interior_point,
                    qhull_options="QJ",  # Triangulated output is approximately 1000 times more accurate than joggled input.
                )  # http://www.qhull.org/html/qh-optq.htm
                poly.warnings.append(
                    RuntimeWarning(f"HalfspaceIntersection failed initially, succeeded with QJ retry: {e}")
                )
            except Exception as e2:
                raise ValueError(f"Error while computing halfspace intersection: {e}") from e2
        else:
            raise ValueError(f"Error while computing halfspace intersection: {e}")

    poly._hs = hs
    # It seems like the SHIs are not always computed correctly by HalfSpaceIntersection, so we will not check them
    # try:
    #     hs_shis = np.unique([shi for shis in hs.dual_facets for shi in shis]).tolist()
    #     # hs_shis = hs.dual_vertices.ravel().tolist()
    #     if set(hs_shis) != set(poly.shis):
    #         w = RuntimeWarning(
    #             f"HalfspaceIntersection SHIs on {poly} do not match computed SHIs: {sorted(hs_shis)} vs {sorted(poly.shis)}"
    #         )
    #         poly.warnings.append(w)
    # except Exception as e:
    #     w = RuntimeWarning(f"Error while getting dual vertices: {e}")
    #     poly.warnings.append(w)
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
    #     poly.warnings.append(w)
    #     poly._volume = -1
    #     return
    poly._vertices = vertices[trust_vertices][trust_vertices_2]
    if poly.finite and len(poly._vertices) > poly.ambient_dim:
        try:
            poly._ch = ConvexHull(vertices)
            try:
                poly._volume = poly._ch.volume
            except Exception as e:
                raise ValueError(f"Error while computing convex hull volume: {e}")
        except Exception as e:
            # warnings.warn("Error while computing convex hull:", e)
            if qhull_mode == "WARN_ALL":
                warnings.warn(f"Error while computing convex hull: {e}")
            elif qhull_mode == "HIGH_PRECISION":
                raise ValueError(f"Error while computing convex hull: {e}")
            poly._ch = None
            poly._volume = -1
    else:
        poly._volume = float("inf")

