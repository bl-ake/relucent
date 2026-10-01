"""Compute and apply float64-safe tolerance defaults."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from relucent.config import update_settings
from relucent.model.model import FlattenLayer, LinearLayer, ReLULayer, ReLUNetwork

__all__ = ["apply_tolerances", "compute_tolerances"]

_EPS = float(np.finfo(np.float64).eps)
_GUROBI_FEAS_TOL = 1e-6
_MIN_BOUNDARY_MIP_EPS = 1e-4


def _gamma(n: int) -> float:
    n = max(int(n), 1)
    denom = 1.0 - n * _EPS
    return float("inf") if denom <= 0.0 else n * _EPS / denom


def _safe(value: float, *, safety_factor: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return value
    return float(np.nextafter(value * safety_factor, np.inf))


def _max_preactivation(net: ReLUNetwork) -> float:
    """Layerwise ``|W|_1`` bound on any preactivation magnitude (at least 1)."""
    layers = list(net.layers.values())
    max_preactivation = 1.0
    seen_relu = False
    for layer in layers:
        if isinstance(layer, LinearLayer):
            layer_w = np.asarray(layer.weight, dtype=np.float64)
            layer_b = np.asarray(layer.bias, dtype=np.float64)
            row_bounds = np.sum(np.abs(layer_w), axis=1) * max_preactivation + np.abs(layer_b)
            max_preactivation = float(np.max(row_bounds))
        elif isinstance(layer, ReLULayer):
            seen_relu = True
        elif isinstance(layer, FlattenLayer) and seen_relu:
            raise NotImplementedError("Intermediate flatten layer not supported for magnitude scan")
    return max(max_preactivation, 1.0) if seen_relu else 1.0


def compute_tolerances(
    *,
    net: ReLUNetwork | None = None,
    ambient_dim: int = 2,
    max_coord: float | None = None,
    safety_factor: float = 2.0,
) -> dict[str, float]:
    """Return static values for the tolerance settings in :mod:`relucent.config`.

    The topology path no longer relies on these. Each geometric decision (facets, emptiness,
    vertices, genericity, membership, Morse signs) is checked against the float64 error of
    the rows involved (:mod:`relucent._internal.rounding`), and raises
    :class:`~relucent.core.errors.AmbiguousGeometryError` if that error could flip it.
    The settings here only matter for plotting and the ``boundary_bfs`` MIP
    (``BOUNDARY_MIP_EPS``).

    They used to be scaled per network, but one number can't fit rows whose errors differ by
    five or more orders of magnitude: it rejected every facet on deep nets and was too tight
    elsewhere. So ``max_coord`` is ignored, and ``net`` only sets the ambient dimension and
    the ``boundary_bfs`` margin.
    """
    del max_coord
    max_preactivation = 1.0
    if net is not None:
        ambient_dim = int(np.prod(net.input_shape))
        max_preactivation = _max_preactivation(net)

    g = _gamma(ambient_dim)
    boundary_floor = max(_MIN_BOUNDARY_MIP_EPS, _GUROBI_FEAS_TOL * max_preactivation)

    return {
        "TOL_HALFSPACE_NORMAL": _safe(g, safety_factor=safety_factor),
        "TOL_VERIFY_AB_ATOL": _safe(g, safety_factor=safety_factor),
        "TOL_NEARLY_VERTICAL": _safe(math.sqrt(_EPS), safety_factor=safety_factor),
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
