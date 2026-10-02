"""Solver-tuning settings. Unstable: these may change or disappear in any release.

They tune performance (cut ordering, batching, worker counts) and solver details rather
than what relucent computes. Change them like the settings in :mod:`relucent.config`:
assign ``relucent.config.advanced.NAME``, pass them to
:func:`~relucent.config.update_settings`, or set ``RELUCENT_<NAME>`` before import.
"""

from __future__ import annotations

from typing import Literal

from relucent.config._env import env_bool, env_choice, env_float, env_float_list, env_int

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
# Boundary pricing MIP
# -----------------------------------------------------------------------------

# For total ReLU width ``n`` up to this, also try a brute-force sign-pattern scan.
BOUNDARY_PRICING_BRUTE_FORCE_MAX_N: int = env_int("BOUNDARY_PRICING_BRUTE_FORCE_MAX_N", 18)

# Compile ``exclude_tags`` into trie no-goods at this many tags (``0`` = always, huge = never).
BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS: int = env_int("BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS", 1000)

# At this many excluded tags, add per-tag no-goods up front, before optimize.
BOUNDARY_MIP_STATIC_EXCLUSION_MIN_TAGS: int = env_int("BOUNDARY_MIP_STATIC_EXCLUSION_MIN_TAGS", 1000)

# Add all leaf nogoods up front if the trie compression ratio is below this.
BOUNDARY_MIP_STATIC_EXCLUSION_MIN_RATIO: float = env_float("BOUNDARY_MIP_STATIC_EXCLUSION_MIN_RATIO", 2.0)

# Batch size for Gurobi ``addConstrs`` when emitting nogoods.
BOUNDARY_MIP_EXCLUSION_BATCH_SIZE: int = env_int("BOUNDARY_MIP_EXCLUSION_BATCH_SIZE", 4096)

# Workers for compiling nogood specs; ``0`` = auto, ``1`` = serial.
BOUNDARY_MIP_EXCLUSION_WORKERS: int = env_int("BOUNDARY_MIP_EXCLUSION_WORKERS", 0)

CutOrder = Literal["as_is", "tag_lex", "layer_major", "hamming_median", "literal_count_asc", "trie_depth_desc", "random"]
CUT_ORDERS: tuple[CutOrder, ...] = (
    "as_is",
    "tag_lex",
    "layer_major",
    "hamming_median",
    "literal_count_asc",
    "trie_depth_desc",
    "random",
)

# Cut ordering for static/lazy nogoods (one of CUT_ORDERS).
BOUNDARY_MIP_CUT_ORDER: CutOrder = env_choice("BOUNDARY_MIP_CUT_ORDER", "tag_lex", CUT_ORDERS)  # pyright: ignore[reportAssignmentType]

BulkNogoodEmit = Literal["auto", "on", "off"]
BULK_NOGOOD_EMIT_MODES: tuple[BulkNogoodEmit, ...] = ("auto", "on", "off")

# Bulk matrix emit for static nogoods: auto, on, off (auto = on at >= 500 specs).
BOUNDARY_MIP_BULK_NOGOOD_EMIT: BulkNogoodEmit = env_choice(  # pyright: ignore[reportAssignmentType]
    "BOUNDARY_MIP_BULK_NOGOOD_EMIT", "auto", BULK_NOGOOD_EMIT_MODES
)

# Static nogood wave size; ``0`` = add everything before one optimize.
BOUNDARY_MIP_STATIC_WAVE_SIZE: int = env_int("BOUNDARY_MIP_STATIC_WAVE_SIZE", 0)

# Give trie-compressed / deeper-path cuts higher Gurobi priority.
BOUNDARY_MIP_CUT_PRIORITY_ENABLED: bool = env_bool("BOUNDARY_MIP_CUT_PRIORITY_ENABLED", False)

# At this many ``exclude_tags``, skip precompilation and use lazy MIPSOL cuts instead.
BOUNDARY_MIP_LAZY_ONLY_MIN_TAGS: int = env_int("BOUNDARY_MIP_LAZY_ONLY_MIN_TAGS", 50_000)

# -----------------------------------------------------------------------------
# Search heuristics & concurrency
# -----------------------------------------------------------------------------

# A* heuristic: f = hamming + ASTAR_BIAS_WEIGHT * bias. The bias is negative and
# favors cells closer to the goal in input space.
ASTAR_BIAS_WEIGHT: float = env_float("ASTAR_BIAS_WEIGHT", 0.9)

# BlockingQueue: seconds to wait on the lock before rechecking.
BLOCKING_QUEUE_WAIT_TIMEOUT: float = env_float("BLOCKING_QUEUE_WAIT_TIMEOUT", 0.5)

# Match a geometric vertex to an intrinsic one if
# ||x - x_intrinsic||_inf <= TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR * tol.
TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR: float = env_float("TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR", 2.0)

# String settings and their allowed values, checked by update_settings.
CHOICES: dict[str, tuple[str, ...]] = {
    "BOUNDARY_MIP_CUT_ORDER": CUT_ORDERS,
    "BOUNDARY_MIP_BULK_NOGOOD_EMIT": BULK_NOGOOD_EMIT_MODES,
}

__all__ = [
    "ASTAR_BIAS_WEIGHT",
    "BLOCKING_QUEUE_WAIT_TIMEOUT",
    "BOUNDARY_MIP_BULK_NOGOOD_EMIT",
    "BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS",
    "BOUNDARY_MIP_CUT_ORDER",
    "BOUNDARY_MIP_CUT_PRIORITY_ENABLED",
    "BOUNDARY_MIP_EXCLUSION_BATCH_SIZE",
    "BOUNDARY_MIP_EXCLUSION_WORKERS",
    "BOUNDARY_MIP_LAZY_ONLY_MIN_TAGS",
    "BOUNDARY_MIP_STATIC_EXCLUSION_MIN_RATIO",
    "BOUNDARY_MIP_STATIC_EXCLUSION_MIN_TAGS",
    "BOUNDARY_MIP_STATIC_WAVE_SIZE",
    "BOUNDARY_PRICING_BRUTE_FORCE_MAX_N",
    "GUROBI_SHI_OPTIMALITY_TOL",
    "GUROBI_SHI_SCALE_FLAG",
    "INTERIOR_POINT_RADIUS_SEQUENCE",
    "TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR",
]
