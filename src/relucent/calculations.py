"""Polyhedral computation helpers (halfspaces, SHIs, qhull interior / SHI models).

Functions here take a :class:`~relucent.poly.Polyhedron` instance; the class lives in
``poly.py`` to avoid import cycles.
"""

import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import torch
import torch.nn as nn
from gurobipy import GRB, Env, Model
from scipy.spatial import ConvexHull, HalfspaceIntersection
from tqdm.auto import tqdm

import relucent.config as cfg
from relucent.utils import get_env

if TYPE_CHECKING:
    from relucent.poly import Polyhedron

__all__ = [
    "adjacent_polyhedra",
    "compute_properties",
    "get_hs",
    "get_shis",
    "solve_radius",
]


class DegenerateHalfspaceInfeasibility(ValueError):
    """Near-zero normal with positive bias: :math:`a^\\top x + b \\le 0` is empty when ``||a||≈0`` and ``b>0``."""


def _drop_degenerate_halfspaces_tracked(
    halfspaces: np.ndarray,
    *,
    tol_normal: float | None = None,
    tol_bias: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`_drop_degenerate_halfspaces` but also returns an old-row → new-row map.

    ``old_to_new[i]`` is the row index in the returned array corresponding to
    original row ``i``, or ``-1`` if that row was dropped.
    """
    n_old = halfspaces.shape[0]
    if tol_normal is None:
        tol_normal = cfg.TOL_HALFSPACE_NORMAL
    if tol_bias is None:
        tol_bias = cfg.TOL_HALFSPACE_CONTAINMENT
    if halfspaces.size == 0:
        return halfspaces, np.arange(0, dtype=np.intp)

    a = halfspaces[:, :-1]
    b = halfspaces[:, -1]
    norms = np.linalg.norm(a, axis=1)
    deg = norms < tol_normal
    if not np.any(deg):
        return halfspaces, np.arange(n_old, dtype=np.intp)

    if np.any(b[deg] > tol_bias):
        bad = np.flatnonzero(deg & (b > tol_bias)).tolist()
        raise DegenerateHalfspaceInfeasibility(
            f"Degenerate halfspace(s) imply infeasibility (||a||<{tol_normal:g} with b>{tol_bias:g}) at rows {bad}"
        )

    kept = np.flatnonzero(~deg)
    old_to_new = np.full(n_old, -1, dtype=np.intp)
    old_to_new[kept] = np.arange(kept.size, dtype=np.intp)
    return halfspaces[~deg], old_to_new


def _drop_degenerate_halfspaces(
    halfspaces: np.ndarray,
    *,
    tol_normal: float | None = None,
    tol_bias: float | None = None,
) -> np.ndarray:
    """Remove degenerate halfspaces with near-zero normals.

    A halfspace row represents a^T x + b <= 0. If ||a|| ~ 0, then the constraint
    is either always satisfied (b <= 0) or infeasible (b > 0). Keeping such rows
    can trigger Qhull errors (e.g. QH6023) and destabilize interior-point solves.
    """
    dropped, _ = _drop_degenerate_halfspaces_tracked(halfspaces, tol_normal=tol_normal, tol_bias=tol_bias)
    return dropped


def _remap_zero_indices(zero_indices: np.ndarray | None, old_to_new: np.ndarray) -> np.ndarray | None:
    """Map sign-sequence indices through a degenerate-row removal map."""
    if zero_indices is None or len(zero_indices) == 0:
        return None
    zm = old_to_new[np.asarray(zero_indices, dtype=np.intp)]
    zm = zm[zm >= 0]
    if zm.size == 0:
        return None
    return zm


def _halfspaces_feasible(
    env: Env,
    halfspaces: np.ndarray,
    zero_indices: np.ndarray | None,
) -> bool:
    """Return True iff the linear system has at least one feasible point."""
    if zero_indices is not None and len(zero_indices) > 0:
        equalities = halfspaces[zero_indices]
        inequalities = halfspaces[~np.isin(np.arange(halfspaces.shape[0]), zero_indices)]
    else:
        inequalities = halfspaces
        equalities = None

    model = Model("Feasibility", env)
    x = model.addMVar((halfspaces.shape[1] - 1, 1), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    model.addConstr(inequalities[:, :-1] @ x <= -inequalities[:, -1:])
    if equalities is not None:
        model.addConstr(equalities[:, :-1] @ x == -equalities[:, -1:])
    model.setObjective(0.0, GRB.MINIMIZE)
    model.optimize()
    status = model.status
    model.close()
    # Minimize 0 over Ax<=b, Ex=f: INFEASIBLE iff the linear system is empty.
    if status == GRB.INFEASIBLE:
        return False
    if status == GRB.INF_OR_UNBD:
        # Rare for this model; treat as undetermined and assume nonempty if not proven empty.
        return True
    if status == GRB.NUMERIC:
        raise ValueError("Feasibility model ended with NUMERIC status (ill-conditioned?)")
    return True


def solve_radius(
    env: Env,
    halfspaces: np.ndarray | torch.Tensor,
    max_radius: float = GRB.INFINITY,
    zero_indices: np.ndarray | None = None,
    sense: int = GRB.MAXIMIZE,
) -> tuple[np.ndarray | None, float | None]:
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
        tuple: ``(center_point, radius)``. Radius is the largest ball in the **affine
        hull** of the feasible region (relative Chebyshev inradius), except that
        ``max_radius`` caps the LP when finite. For ``max_radius`` infinite:
        ``(None, None)`` if infeasible; ``(None, inf)`` if the region is all of
        :math:`\\mathbb{R}^d` with no constraints; otherwise ``(x, r)`` with
        ``r = inf`` when the hull has positive dimension and no inequalities cut it,
        ``r = 0`` when the hull is a **single point**, and ``r > 0`` when bounded
        with nonempty relative interior in the usual LP.

    Raises:
        ValueError: If the optimization fails or produces invalid results.
    """

    if isinstance(halfspaces, torch.Tensor):
        halfspaces = halfspaces.detach().cpu().numpy()

    if not np.isfinite(halfspaces).all():
        raise ValueError("Halfspaces contain NaN or Inf coefficients")

    # Remove degenerate constraints (near-zero normals) before building the model.
    # This prevents pathologies like 0*x + 0*y <= -b and makes results more stable across platforms.
    try:
        halfspaces, old_to_new = _drop_degenerate_halfspaces_tracked(halfspaces)
    except DegenerateHalfspaceInfeasibility:
        # Same conclusion as an infeasible Chebyshev LP: empty intersection.
        return None, None
    zero_indices_eff = _remap_zero_indices(zero_indices, old_to_new)

    if zero_indices_eff is not None and len(zero_indices_eff) > 0:
        # warnings.warn("Working with k<d polyhedron.", stacklevel=2)
        equalities = halfspaces[zero_indices_eff]
        inequalities = halfspaces[~np.isin(np.arange(halfspaces.shape[0]), zero_indices_eff)]
        P = (
            np.eye(equalities[:, :-1].shape[1])
            - equalities[:, :-1].T @ np.linalg.pinv(equalities[:, :-1] @ equalities[:, :-1].T) @ equalities[:, :-1]
        )
        norm_vector = np.reshape(np.linalg.norm(inequalities[:, :-1] @ P.T, axis=1), (inequalities[:, :-1].shape[0], 1))
    else:
        inequalities = halfspaces
        equalities = None
        norm_vector = np.reshape(np.linalg.norm(inequalities[:, :-1], axis=1), (inequalities[:, :-1].shape[0], 1))

    if not np.isfinite(norm_vector).all():
        raise ValueError("Norm vector contains NaN or Inf coefficients")

    # No inequality rows: the Chebyshev LP has no constraints linking x and y.
    # Building ``norm_vector * y`` with shape (0, 1) hits a gurobipy matrix-API
    # AssertionError on some platforms (e.g. certain Python/gurobipy combos).
    if inequalities.shape[0] == 0:
        dim = halfspaces.shape[1] - 1
        if equalities is not None and equalities.shape[0] > 0:
            A_eq = equalities[:, :-1]
            b_vec = -equalities[:, -1:].ravel()
            x_feas, *_ = np.linalg.lstsq(A_eq, b_vec, rcond=None)
            x_feas = np.asarray(x_feas, dtype=np.float64).reshape(dim, 1)
            if A_eq.shape[0] > 0:
                rnorm = np.linalg.norm(A_eq @ x_feas.ravel() - b_vec)
                if rnorm > 1e-5 * max(1.0, float(np.linalg.norm(b_vec))):
                    return None, None
            # No strict inequalities: feasible set is an affine subspace {x : A_eq x = b}.
            # Relative inradius in that hull is infinite unless the hull is 0-dimensional.
            rnk = int(np.linalg.matrix_rank(A_eq))
            aff_dim = dim - rnk
            if aff_dim <= 0:  ## TODO: Either this or avoid calling solve_radius by checking the poly's codimension
                # Unique feasible point (affine hull is a single point).
                return x_feas, 0.0
            if max_radius == GRB.INFINITY:
                return x_feas, float("inf")
            return x_feas, float(max_radius)
        if max_radius == GRB.INFINITY:
            return None, float("inf")
        return np.zeros((dim, 1), dtype=np.float64), float(max_radius)

    model = Model("Interior Point", env)
    x = model.addMVar((halfspaces.shape[1] - 1, 1), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    y = model.addMVar((1,), ub=max_radius, vtype=GRB.CONTINUOUS, name="y")
    try:
        model.addConstr(inequalities[:, :-1] @ x + norm_vector * y <= -inequalities[:, -1:])
        if equalities is not None:
            model.addConstr(equalities[:, :-1] @ x == -equalities[:, -1:])
        model.setObjective(y, sense)
    except Exception as e:
        raise ValueError(f"GB Error Building Model: {e}") from e
    model.optimize()
    status = model.status

    if status == GRB.INF_OR_UNBD:
        model.setParam(GRB.Param.DualReductions, 0)
        model.reset()
        model.optimize()
        status = model.status

    # Rarely, the Chebyshev LP can end with NUMERIC due to ill-conditioning.
    # A second attempt with NumericFocus can recover; avoid doing this for INFEASIBLE
    # since infeasibility is common during local search and the retry is expensive.
    if status == GRB.NUMERIC:
        model.setParam("NumericFocus", 1)
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
        model.close()
        if max_radius == GRB.INFINITY:
            # Chebyshev LP can be INFEASIBLE while the linear system is still feasible
            # (e.g. intrinsic inradius 0).
            if status == GRB.INFEASIBLE:
                # if _halfspaces_feasible(env, halfspaces, zero_indices_eff):
                #     warnings.warn(
                #         "Chebyshev LP is INFEASIBLE but halfspace system is feasible.",
                #         stacklevel=2,
                #     )
                # return None, float("inf")
                return None, None
            elif status == GRB.INF_OR_UNBD:
                raise ValueError("Unable to disambiguate INF_OR_UNBD status.")
            elif status == GRB.UNBOUNDED:
                return None, float("inf")
            raise ValueError(f"Interior Point Model Status: {status}")
        else:
            # Finite max_radius: same Chebyshev outcomes as above (intrinsic inradius 0, etc.).
            if status == GRB.INFEASIBLE:
                return None, None
            if status == GRB.NUMERIC:
                return None, None
            if status == GRB.INF_OR_UNBD:
                raise ValueError("Unable to disambiguate INF_OR_UNBD status.")
            if status == GRB.UNBOUNDED:
                return None, None
            raise ValueError(f"Interior Point Model Status: {status}")


@torch.no_grad()
def adjacent_polyhedra(
    poly: "Polyhedron",
    ss2poly: Callable[..., "Polyhedron"],
) -> "set[Polyhedron]":
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
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: Literal[False] = False,
    force_numpy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | tuple[np.ndarray, np.ndarray, np.ndarray, int]: ...


@overload
def get_hs(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: Literal[True],
    force_numpy: bool = False,
) -> list[dict[str, object]]: ...


def get_hs(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: bool = False,
    force_numpy: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    | tuple[np.ndarray, np.ndarray, np.ndarray, int]
    | list[dict[str, object]]
):
    """Halfspace representation of ``poly`` from all neurons in the network.

    Includes constraints from every neuron, not only supporting hyperplanes.

    Args:
        poly: "Polyhedron" whose sign sequence defines the region.
        data: Optional network input for verifying intermediate affine maps.
        get_all_Ab: If True, return per-layer ``A``, ``b`` instead of final halfspaces.
        force_numpy: If True, use the NumPy path even when ``ss`` is a tensor.

    Returns:
        If ``get_all_Ab`` is False: ``(halfspaces, W, b, num_dead_relus)``.
        If True: list of dicts with ``A``, ``b``, and ``layer`` keys.
    """
    if isinstance(poly._ss, torch.Tensor) and not force_numpy:
        return _get_hs_torch(poly, data, get_all_Ab=get_all_Ab)
    return _get_hs_numpy(poly, data, get_all_Ab=get_all_Ab)


@overload
def _get_hs_torch(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: Literal[False] = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]: ...


@overload
def _get_hs_torch(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: Literal[True],
) -> list[dict[str, object]]: ...


@torch.no_grad()
def _get_hs_torch(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | list[dict[str, object]]:
    assert isinstance(poly._ss, torch.Tensor)
    constr_A, constr_b = None, None
    current_A, current_b = None, None
    layer_W, layer_b = None, None
    if data is not None:
        outs: dict[str, torch.Tensor] | None = poly.net.get_all_layer_outputs(data)
    else:
        outs = None
    all_Ab = []
    current_mask_index = 0
    for name, layer in poly.net.layers.items():
        if isinstance(layer, nn.Linear):
            layer_W = layer.weight
            layer_b = layer.bias[None, :]
            if current_A is None or current_b is None:
                constr_A = torch.empty((layer_W.shape[1], 0), device=poly.net.device, dtype=poly.net.dtype)
                constr_b = torch.empty((1, 0), device=poly.net.device, dtype=poly.net.dtype)
                current_A = torch.eye(layer_W.shape[1], device=poly.net.device, dtype=poly.net.dtype)
                current_b = torch.zeros((1, layer_W.shape[1]), device=poly.net.device, dtype=poly.net.dtype)

            current_A = current_A @ layer_W.T
            current_b = current_b @ layer_W.T + layer_b
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
            assert outs is not None
            assert torch.allclose(outs[name], (data @ current_A) + current_b, atol=cfg.TOL_VERIFY_AB_ATOL)
        if get_all_Ab:
            assert current_A is not None
            assert current_b is not None

            all_Ab.append({"A": current_A.clone(), "b": current_b.clone(), "layer": layer})

    assert constr_A is not None
    assert constr_b is not None

    num_dead_relus = int((torch.abs(constr_A) < cfg.TOL_DEAD_RELU).all(dim=0).sum().item())
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
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]: ...


@overload
def _get_hs_numpy(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: Literal[True],
) -> list[dict[str, object]]: ...


@torch.no_grad()
def _get_hs_numpy(
    poly: "Polyhedron",
    data: torch.Tensor | None = None,
    *,
    get_all_Ab: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | list[dict[str, object]]:
    constr_A, constr_b = None, None
    current_A, current_b = None, None
    layer_W, layer_b = None, None
    if data is not None:
        outs: dict[str, torch.Tensor] | None = poly.net.get_all_layer_outputs(data)
    else:
        outs = None
    all_Ab = []
    current_mask_index = 0
    for name, layer in poly.net.layers.items():
        if isinstance(layer, nn.Linear):
            layer_W = layer.weight_cpu
            layer_b = layer.bias_cpu

            assert isinstance(layer_W, np.ndarray)
            assert isinstance(layer_b, np.ndarray)

            if current_A is None or current_b is None:
                constr_A = np.empty((layer_W.shape[1], 0))
                constr_b = np.empty((1, 0))
                current_A = np.eye(layer_W.shape[1])
                current_b = np.zeros((1, layer_W.shape[1]))

            current_A = current_A @ layer_W.T
            current_b = current_b @ layer_W.T + layer_b
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
            assert outs is not None
            assert np.allclose(
                outs[name].detach().cpu().numpy(),
                (data @ current_A) + current_b,
                atol=cfg.TOL_VERIFY_AB_ATOL,
            )
        if get_all_Ab:
            assert current_A is not None
            assert current_b is not None

            all_Ab.append({"A": current_A.copy(), "b": current_b.copy(), "layer": layer})

    assert constr_A is not None
    assert constr_b is not None

    num_dead_relus = (np.abs(constr_A) < cfg.TOL_DEAD_RELU).all(axis=0).sum().item()
    halfspaces = np.hstack((-constr_A.T, -constr_b.reshape(-1, 1)))
    if get_all_Ab:
        return all_Ab
    assert isinstance(halfspaces, np.ndarray)
    assert isinstance(current_A, np.ndarray)
    assert isinstance(current_b, np.ndarray)

    assert halfspaces.shape[0] == poly.ss_np.shape[1]
    return halfspaces, current_A, current_b, num_dead_relus


@overload
def get_shis(
    poly: "Polyhedron",
    collect_info: Literal[False] = False,
    bound: float = GRB.INFINITY,
    subset: Iterable[int] | None = None,
    tol: float | None = None,
    new_method: bool = False,
    env: Env | None = None,
    shi_pbar: bool = False,
    push_size: float = 1.0,
) -> list[int]: ...


@overload
def get_shis(
    poly: "Polyhedron",
    collect_info: Literal[True] | Literal["All"],
    bound: float = GRB.INFINITY,
    subset: Iterable[int] | None = None,
    tol: float | None = None,
    new_method: bool = False,
    env: Env | None = None,
    shi_pbar: bool = False,
    push_size: float = 1.0,
) -> tuple[list[int], list[dict[str, object]]]: ...


def get_shis(
    poly: "Polyhedron",
    collect_info: bool | str = False,
    bound: float = GRB.INFINITY,
    subset: Iterable[int] | None = None,
    tol: float | None = None,
    new_method: bool = False,
    env: Env | None = None,
    shi_pbar: bool = False,
    push_size: float = 1.0,
) -> list[int] | tuple[list[int], list[dict[str, object]]]:
    """Supporting halfspace indices (SHIs) for ``poly``.

    Indices of non-redundant halfspaces on the boundary (neurons whose BHs are
    actually faces of the polyhedron).

    Args:
        poly: "Polyhedron" to analyze.
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
    if tol is None:
        tol = cfg.TOL_SHI_HYPERPLANE
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
    poly_info: list[dict[str, object]] | None = [] if collect_info else None
    while subset:
        i = subset.pop()
        if i >= poly.ss_np.shape[1] or poly.ss_np[0, i] == 0:
            continue
        if (A[i, :-1] == 0).all():
            continue
        # model.update()
        pbar.set_postfix_str(f"#shis: {len(shis)}")

        ## Relax halspace i
        constrs[i].setAttr("RHS", -b[i, 0] + push_size)

        model.setObjective((A[i] @ x).item() + b[i, 0], GRB.MAXIMIZE)
        model.params.BestObjStop = cfg.GUROBI_SHI_BEST_OBJ_STOP
        model.params.BestBdStop = cfg.GUROBI_SHI_BEST_BD_STOP
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
                    warnings.warn(w, stacklevel=2)
                else:
                    shis.append(i)

            basis_indices = constrs.CBasis.ravel() != 0
            if new_method and basis_indices.sum() != A.shape[1]:
                warnings.warn(
                    "SHI computation: bound constraints detected in LP basis; basis-based shortcut skipped.",
                    stacklevel=2,
                )
            skip_size = 0
            if new_method and basis_indices.sum() == A.shape[1]:
                point_shis = poly.halfspaces[basis_indices, :-1]  # (d(# point shis) x d)
                others = poly.halfspaces[~basis_indices, :-1]  # (num_other_hyperplanes x d)
                try:
                    sols = torch.linalg.solve(point_shis, others.T)
                except RuntimeError:
                    warnings.warn(
                        "SHI computation: failed to solve linear system for basis shortcut; falling back.",
                        stacklevel=2,
                    )
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
            assert poly_info is not None
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
        assert poly_info is not None
        return shis, poly_info
    return shis


def compute_properties(poly: "Polyhedron", qhull_mode: str | None = None) -> None:
    """Compute additional geometric properties for low-dimensional polyhedra (vertices, hull, volume).

    Mutates ``poly`` cache fields (``_hs``, ``_vertices``, ``_ch``, ``_volume``,
    ``_attempted_compute_properties``). No-op if already attempted.

    Raises:
        ValueError: If input dimension > 6, interior point is missing, or qhull fails
            (depending on ``qhull_mode``).
    """
    if qhull_mode is None:
        qhull_mode = cfg.QHULL_MODE
    if poly._attempted_compute_properties:
        return
    poly._attempted_compute_properties = True

    if poly.net.input_shape[0] > 6:
        raise ValueError("Input shape too large to compute extra properties")
    # Filter degenerate constraints before calling Qhull (also used by retry paths below).
    halfspaces = _drop_degenerate_halfspaces(poly.halfspaces_np)
    try:
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
                warnings.warn(f"Halfspace intersection emitted warnings: {msgs}", stacklevel=2)
            elif qhull_mode == "HIGH_PRECISION":
                raise ValueError(f"HalfspaceIntersection emitted warnings in HIGH_PRECISION mode: {msgs}")
            elif qhull_mode == "JITTERED":
                with warnings.catch_warnings(record=True) as w2:
                    new_hs = HalfspaceIntersection(
                        halfspaces,
                        poly.interior_point,
                        # Triangulated output is approximately 1000 times more accurate than joggled input.
                        qhull_options="QJ",
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
                    # Triangulated output is approximately 1000 times more accurate than joggled input.
                    qhull_options="QJ",
                )  # http://www.qhull.org/html/qh-optq.htm
                poly.warnings.append(RuntimeWarning(f"HalfspaceIntersection failed initially, succeeded with QJ retry: {e}"))
            except Exception as e2:
                raise ValueError(f"Error while computing halfspace intersection: {e}") from e2
        else:
            raise ValueError(f"Error while computing halfspace intersection: {e}") from e

    poly._hs = hs
    # It seems like the SHIs are not always computed correctly by HalfSpaceIntersection, so we will not check them
    # try:
    #     hs_shis = np.unique([shi for shis in hs.dual_facets for shi in shis]).tolist()
    #     # hs_shis = hs.dual_vertices.ravel().tolist()
    #     if set(hs_shis) != set(poly.shis):
    #         w = RuntimeWarning(
    #             f"HalfspaceIntersection SHIs on {poly} != computed SHIs: {sorted(hs_shis)} vs {sorted(poly.shis)}"
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
    ) < cfg.VERTEX_TRUST_THRESHOLD
    poly._vertices = vertices[trust_vertices][trust_vertices_2]
    if poly.finite and len(poly._vertices) > poly.ambient_dim:
        try:
            poly._ch = ConvexHull(vertices)
            try:
                poly._volume = poly._ch.volume
            except Exception as e:
                raise ValueError(f"Error while computing convex hull volume: {e}") from e
        except Exception as e:
            # warnings.warn("Error while computing convex hull:", e)
            if qhull_mode == "WARN_ALL":
                warnings.warn(f"Error while computing convex hull: {e}", stacklevel=2)
            elif qhull_mode == "HIGH_PRECISION":
                raise ValueError(f"Error while computing convex hull: {e}") from e
            poly._ch = None
            poly._volume = -1
    else:
        poly._volume = float("inf")
