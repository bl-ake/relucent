"""Shared network magnitude estimates for tolerances and boundary MIP."""

from __future__ import annotations

import numpy as np

from relucent.model.model import LinearLayer, ReLULayer, ReLUNetwork

__all__ = [
    "boundary_mip_eps",
    "count_relu_units",
    "default_polyhedron_bound",
    "estimate_input_bound",
    "relu_linear_blocks",
]

# Gurobi's default FeasibilityTol, and the smallest boundary MIP margin used.
_GUROBI_FEAS_TOL = 1e-6
_MIN_BOUNDARY_MIP_EPS = 1e-4


def relu_linear_blocks(net: ReLUNetwork) -> list[LinearLayer]:
    """Linear layers immediately followed by ReLU (sign-sequence layers)."""
    layers = list(net.layers.values())
    blocks: list[LinearLayer] = []
    for i, layer in enumerate(layers):
        if isinstance(layer, LinearLayer) and i + 1 < len(layers) and isinstance(layers[i + 1], ReLULayer):
            blocks.append(layer)
    return blocks


def count_relu_units(net: ReLUNetwork) -> int:
    """Total number of ReLU units (rows of linear layers that feed a ReLU)."""
    return sum(int(layer.weight.shape[0]) for layer in relu_linear_blocks(net))


def estimate_input_bound(net: ReLUNetwork, *, margin: float) -> float:
    """Conservative input box radius from layerwise ``|W|_1`` propagation."""
    radius = 1.0
    for block in relu_linear_blocks(net):
        w = np.asarray(block.weight, dtype=np.float64)
        b = np.asarray(block.bias, dtype=np.float64).reshape(-1)
        radius = float(np.max(np.sum(np.abs(w), axis=1) + np.abs(b)) * radius)
    return max(radius * margin, 1.0)


def default_polyhedron_bound(net: ReLUNetwork) -> float:
    """Network-scaled box radius for SHI LPs (matches boundary MIP pricing)."""
    import relucent.config as cfg

    return estimate_input_bound(net, margin=float(cfg.BOUNDARY_MIP_BOUND_MARGIN))


def max_preactivation(net: ReLUNetwork) -> float:
    """Layerwise ``|W|_1`` bound on any preactivation magnitude over the unit box (at least 1)."""
    bound = 1.0
    for block in relu_linear_blocks(net):
        w = np.asarray(block.weight, dtype=np.float64)
        b = np.asarray(block.bias, dtype=np.float64).reshape(-1)
        bound = float(np.max(np.sum(np.abs(w), axis=1) * bound + np.abs(b)))
    return max(bound, 1.0)


def boundary_mip_eps(net: ReLUNetwork) -> float:
    """Default boundary MIP margin: twice the larger of 1e-4 and Gurobi's feasibility tolerance at this scale.

    Used when :data:`relucent.config.BOUNDARY_MIP_EPS` is ``None``.
    """
    return 2.0 * max(_MIN_BOUNDARY_MIP_EPS, _GUROBI_FEAS_TOL * max_preactivation(net))
