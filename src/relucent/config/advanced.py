"""Solver-tuning settings. Unstable: these may change or disappear in any release.

They tune performance (heuristic weights, queue timeouts) and solver details rather
than what relucent computes. Change them like the settings in :mod:`relucent.config`:
assign ``relucent.config.advanced.NAME``, pass them to
:func:`~relucent.config.update_settings`, or set ``RELUCENT_<NAME>`` before import.
"""

from __future__ import annotations

from relucent.config._env import env_float, env_float_list, env_int

# -----------------------------------------------------------------------------
# SHI linear programs
# -----------------------------------------------------------------------------

# Gurobi OptimalityTol for the SHI LP (default 1e-6, minimum 1e-9). At 1e-6 the basis
# can have a slightly negative exact multiplier, which the exact facet check rejects.
GUROBI_SHI_OPTIMALITY_TOL: float = env_float("GUROBI_SHI_OPTIMALITY_TOL", 1e-9)

# Gurobi ScaleFlag for the SHI LP (-1 auto, 0 off, 1-3 scaling methods). Rows can span
# orders of magnitude and be nearly parallel, and auto scaling has called feasible LPs
# infeasible. On five checkpoints that failed this way, geometric-mean scaling (2)
# finished four; auto finished none.
GUROBI_SHI_SCALE_FLAG: int = env_int("GUROBI_SHI_SCALE_FLAG", 2)

# max_radius values tried in turn when finding a neighbor's interior point (get_ip).
INTERIOR_POINT_RADIUS_SEQUENCE: list[float] = env_float_list("INTERIOR_POINT_RADIUS_SEQUENCE", [0.01, 0.1, 1, 10, 100])

# -----------------------------------------------------------------------------
# Search heuristics & concurrency
# -----------------------------------------------------------------------------

# A* heuristic: f = hamming + ASTAR_BIAS_WEIGHT * bias. The bias is negative and
# favors cells closer to the goal in input space.
ASTAR_BIAS_WEIGHT: float = env_float("ASTAR_BIAS_WEIGHT", 0.9)

# BlockingQueue: seconds to wait on the lock before rechecking.
BLOCKING_QUEUE_WAIT_TIMEOUT: float = env_float("BLOCKING_QUEUE_WAIT_TIMEOUT", 0.5)

# String settings and their allowed values, checked by update_settings.
CHOICES: dict[str, tuple[str, ...]] = {}

__all__ = [
    "ASTAR_BIAS_WEIGHT",
    "BLOCKING_QUEUE_WAIT_TIMEOUT",
    "GUROBI_SHI_OPTIMALITY_TOL",
    "GUROBI_SHI_SCALE_FLAG",
    "INTERIOR_POINT_RADIUS_SEQUENCE",
]
