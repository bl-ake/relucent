"""Compute and apply float64-safe tolerance defaults for :mod:`relucent.config`."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

import relucent.config as cfg
from relucent._network_scale import estimate_input_bound
from relucent.config import update_settings
from relucent.model import FlattenLayer, LinearLayer, ReLULayer, ReLUNetwork

__all__ = ["apply_tolerances", "compute_tolerances"]

_EPS = float(np.finfo(np.float64).eps)
_GUROBI_FEAS_TOL = 1e-6
_MIN_SHI_OBJECTIVE = 1e-18
_MIN_DEAD_RELU = 1e-8
_MIN_BOUNDARY_MIP_EPS = 1e-4
_SHI_PROOF_MAX_VIOLATIONS = 100


def _gamma(n: int) -> float:
    n = max(int(n), 1)
    denom = 1.0 - n * _EPS
    return float("inf") if denom <= 0.0 else n * _EPS / denom


def _safe(value: float, *, safety_factor: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return value
    return float(np.nextafter(value * safety_factor, np.inf))


def _scan_halfspace_magnitudes(net: ReLUNetwork) -> dict[str, float | int]:
    layers = list(net.layers.values())
    n_relus = sum(
        int(layer.weight.shape[0])
        for i, layer in enumerate(layers)
        if isinstance(layer, LinearLayer) and i + 1 < len(layers) and isinstance(layers[i + 1], ReLULayer)
    )
    if n_relus == 0:
        return {
            "max_halfspace_norm": 1.0,
            "max_abs_bias": 1.0,
            "max_constraints": 0,
            "max_column_inf": 0.0,
            "relu_depth": 0,
            "max_preactivation": 1.0,
        }

    mask = np.ones(n_relus, dtype=np.int8)
    constr_a: np.ndarray | None = None
    constr_b: np.ndarray | None = None
    current_a: np.ndarray | None = None
    current_b: np.ndarray | None = None
    mask_index = 0
    relu_depth = 0
    max_preactivation = 1.0

    for layer in net.layers.values():
        if isinstance(layer, LinearLayer):
            layer_w = np.asarray(layer.weight, dtype=np.float64)
            layer_b = np.asarray(layer.bias, dtype=np.float64)
            if current_a is None or current_b is None:
                constr_a = np.empty((layer_w.shape[1], 0), dtype=np.float64)
                constr_b = np.empty((1, 0), dtype=np.float64)
                current_a = np.eye(layer_w.shape[1], dtype=np.float64)
                current_b = np.zeros((1, layer_w.shape[1]), dtype=np.float64)
            current_a = current_a @ layer_w.T
            current_b = current_b @ layer_w.T + layer_b
            row_bounds = np.sum(np.abs(layer_w), axis=1) * max_preactivation + np.abs(layer_b)
            max_preactivation = float(np.max(row_bounds))
        elif isinstance(layer, ReLULayer):
            assert current_a is not None and current_b is not None and constr_a is not None and constr_b is not None
            relu_depth += 1
            layer_mask = mask[mask_index : mask_index + current_a.shape[1]]
            nonzero_mask = np.where(layer_mask == 0, 1, layer_mask)
            constr_a = np.concatenate((constr_a, current_a * nonzero_mask), axis=1)
            constr_b = np.concatenate((constr_b, current_b * nonzero_mask), axis=1)
            active_a = current_a * (layer_mask == 1)
            current_a = active_a
            current_b = current_b * (layer_mask == 1)
            mask_index += int(active_a.shape[1])
        elif isinstance(layer, FlattenLayer) and current_a is not None:
            raise NotImplementedError("Intermediate flatten layer not supported for magnitude scan")

    assert constr_a is not None and constr_b is not None
    halfspaces = np.hstack((-constr_a.T, -constr_b.reshape(-1, 1)))
    row_norms = np.linalg.norm(halfspaces[:, :-1], axis=1)
    return {
        "max_halfspace_norm": max(float(np.max(row_norms)) if row_norms.size else 1.0, 1.0),
        "max_abs_bias": max(float(np.max(np.abs(halfspaces[:, -1]))) if halfspaces.size else 1.0, 1.0),
        "max_constraints": n_relus,
        "max_column_inf": float(np.max(np.abs(constr_a))) if constr_a.size else 0.0,
        "relu_depth": relu_depth,
        "max_preactivation": max(max_preactivation, 1.0),
    }


def compute_tolerances(
    *,
    net: ReLUNetwork | None = None,
    ambient_dim: int = 2,
    max_coord: float | None = None,
    safety_factor: float = 2.0,
) -> dict[str, float]:
    """Return recommended values for derived tolerance settings in :mod:`relucent.config`."""
    gurobi_tol = _GUROBI_FEAS_TOL
    push_size = 1.0
    max_halfspace_norm = 1.0
    max_abs_bias = 1.0
    max_constraints = 10_000
    max_column_inf = 0.0
    relu_depth = 0
    max_preactivation = 1.0

    if net is not None:
        scan = _scan_halfspace_magnitudes(net)
        ambient_dim = int(np.prod(net.input_shape))
        max_coord = estimate_input_bound(net, margin=float(cfg.BOUNDARY_MIP_BOUND_MARGIN))
        max_halfspace_norm = float(scan["max_halfspace_norm"])
        max_abs_bias = float(scan["max_abs_bias"])
        max_constraints = int(scan["max_constraints"])
        max_column_inf = float(scan["max_column_inf"])
        relu_depth = int(scan["relu_depth"])
        max_preactivation = float(scan["max_preactivation"])
    elif max_coord is None:
        max_coord = float(cfg.DEFAULT_SEARCH_BOUND)

    assert max_coord is not None
    g = _gamma(ambient_dim)
    dot_floor = g * max_halfspace_norm * max_coord
    norm_floor = g * max_halfspace_norm

    tol_halfspace_normal = _safe(norm_floor, safety_factor=safety_factor)
    tol_halfspace_containment = _safe(dot_floor + _EPS * max_abs_bias, safety_factor=safety_factor)

    lstsq_floor = _gamma(ambient_dim) * max(max_halfspace_norm, max_abs_bias, 1.0)
    interior_cap = max(tol_halfspace_containment, gurobi_tol, lstsq_floor, 10.0 * tol_halfspace_containment)
    tol_interior_verify = max(_safe(interior_cap, safety_factor=safety_factor), tol_halfspace_containment)

    tol_vertex_trust = _safe(float(max_constraints) * tol_halfspace_containment, safety_factor=safety_factor)
    dead_floor = max(
        _gamma(max(relu_depth, 1)) * max_column_inf if max_column_inf > 0.0 else _MIN_DEAD_RELU,
        _MIN_DEAD_RELU,
    )
    tol_dead_relu = _safe(dead_floor, safety_factor=safety_factor)
    tol_verify_ab = _safe(dot_floor, safety_factor=safety_factor)

    shi_obj_floor = max(dot_floor * push_size, _MIN_SHI_OBJECTIVE)
    tol_shi_objective = _safe(shi_obj_floor, safety_factor=safety_factor)
    # BestObjStop should not be below solver feasibility tolerance, but the
    # acceptance threshold (TOL_SHI_OBJECTIVE) can be smaller.
    tol_gurobi_obj_stop = _safe(max(tol_shi_objective, gurobi_tol), safety_factor=safety_factor)

    k_viol = min(max_constraints, _SHI_PROOF_MAX_VIOLATIONS)
    tol_shi_hyperplane = _safe(float(k_viol) * dot_floor, safety_factor=safety_factor)
    tol_nearly_vertical = _safe(math.sqrt(_EPS) * max_halfspace_norm, safety_factor=safety_factor)
    boundary_floor = max(_MIN_BOUNDARY_MIP_EPS, gurobi_tol * max_preactivation)

    return {
        "TOL_HALFSPACE_NORMAL": tol_halfspace_normal,
        "TOL_HALFSPACE_CONTAINMENT": tol_halfspace_containment,
        "TOL_INTERIOR_VERIFY": tol_interior_verify,
        "VERTEX_TRUST_THRESHOLD": tol_vertex_trust,
        "TOL_DEAD_RELU": tol_dead_relu,
        "TOL_VERIFY_AB_ATOL": tol_verify_ab,
        "TOL_SHI_OBJECTIVE": tol_shi_objective,
        "GUROBI_SHI_BEST_OBJ_STOP": tol_gurobi_obj_stop,
        "GUROBI_SHI_BEST_BD_STOP": -tol_gurobi_obj_stop,
        # Search uses the same SHI-scale threshold, with a half-step margin, so
        # thin cells are rejected before relaxed-face LPs drift into tolerance noise.
        "MIN_SEARCH_INRADIUS": tol_shi_objective / 2.0,
        "TOL_SHI_HYPERPLANE": tol_shi_hyperplane,
        "TOL_NEARLY_VERTICAL": tol_nearly_vertical,
        "BOUNDARY_MIP_EPS": _safe(boundary_floor, safety_factor=safety_factor),
    }


def apply_tolerances(
    *,
    net: ReLUNetwork | None = None,
    ambient_dim: int = 2,
    max_coord: float | None = None,
    safety_factor: float = 2.0,
    respect_env: bool = True,
    **overrides: Any,
) -> None:
    """Push recommended tolerances into :mod:`relucent.config` via :func:`update_settings`."""
    values = compute_tolerances(
        net=net,
        ambient_dim=ambient_dim,
        max_coord=max_coord,
        safety_factor=safety_factor,
    )
    if respect_env:
        values = {name: value for name, value in values.items() if os.getenv(f"RELUCENT_{name}") is None}
    values.update(overrides)
    update_settings(**values)
