"""Tunable constants and defaults for relucent.

Code reads these at use time (``cfg.NAME``), so you can change them before or
between calls.

Ways to change a setting:

* Assign directly::

    import relucent
    relucent.config.TOL_VERIFY_AB_ATOL = 1e-7

* Set several at once::

    from relucent.config import update_settings
    update_settings(TOL_VERIFY_AB_ATOL=1e-7, MAX_RADIUS=200)

* Use an environment variable before import, e.g.
  ``RELUCENT_CAREFUL_MODE=1``. Any public setting works as
  ``RELUCENT_<SETTING_NAME>``.

See :doc:`configuration` for the full reference.
"""

from __future__ import annotations

import os
from typing import Any


def _env_name(setting: str) -> str:
    return f"RELUCENT_{setting}"


def _env_str(setting: str, default: str) -> str:
    return os.getenv(_env_name(setting), default)


def _env_float(setting: str, default: float) -> float:
    raw = os.getenv(_env_name(setting))
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float value for {_env_name(setting)!r}: {raw!r}") from exc


def _env_int(setting: str, default: int) -> int:
    raw = os.getenv(_env_name(setting))
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid int value for {_env_name(setting)!r}: {raw!r}") from exc


def _env_bool(setting: str, default: bool) -> bool:
    raw = os.getenv(_env_name(setting))
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid bool value for {_env_name(setting)!r}: {raw!r}")


def _env_float_list(setting: str, default: list[float]) -> list[float]:
    raw = os.getenv(_env_name(setting))
    if raw is None:
        return default

    # Accepts "0.1,1,10" or "[0.1, 1, 10]".
    cleaned = raw.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid float list value for {_env_name(setting)!r}: {raw!r}")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid float list value for {_env_name(setting)!r}: {raw!r}") from exc


# -----------------------------------------------------------------------------
# Polyhedron & halfspace geometry
# -----------------------------------------------------------------------------

# Run extra consistency checks (assertions, forward pass vs. affine
# recomputation, conversion spot-checks). Handy for tests and debugging, but slower.
CAREFUL_MODE: bool = _env_bool("CAREFUL_MODE", False)

# What to do when Qhull warns during HalfspaceIntersection:
#   - "IGNORE" (default): carry on.
#   - "WARN_ALL": pass every warning along.
#   - "HIGH_PRECISION": treat a warning as an error.
#   - "JITTERED": retry with Qhull's "QJ" (joggle) option.
QHULL_MODE: str = _env_str("QHULL_MODE", "IGNORE")

# Max radius when solving for a Chebyshev center / interior point. Smaller is
# faster, but drops cells with no point within MAX_RADIUS of every face.
MAX_RADIUS: float = _env_float("MAX_RADIUS", 100)

# Normals with ||a|| below this are degenerate: a^T x + b <= 0 is redundant if
# b <= 0 and infeasible otherwise. They can make Qhull fail, so they're handled separately.
TOL_HALFSPACE_NORMAL: float = _env_float("TOL_HALFSPACE_NORMAL", 1e-12)

# Gurobi OptimalityTol for the SHI LP (default 1e-6, minimum 1e-9). At 1e-6 the basis
# can have a slightly negative exact multiplier, which the exact facet check rejects.
GUROBI_SHI_OPTIMALITY_TOL: float = _env_float("GUROBI_SHI_OPTIMALITY_TOL", 1e-9)

# Gurobi ScaleFlag for the SHI LP (-1 auto, 0 off, 1-3 scaling methods). Rows can span
# orders of magnitude and be nearly parallel, and auto scaling has called feasible LPs
# infeasible. On five checkpoints that failed this way, geometric-mean scaling (2)
# finished four; auto finished none.
GUROBI_SHI_SCALE_FLAG: int = _env_int("GUROBI_SHI_SCALE_FLAG", 2)

# In 2D plotting, normal components below this count as zero
# (nearly vertical line: |w[1]| < TOL_NEARLY_VERTICAL).
TOL_NEARLY_VERTICAL: float = _env_float("TOL_NEARLY_VERTICAL", 1e-10)

# Default hypercube half-width for plotting and get_bounded_vertices.
DEFAULT_PLOT_BOUND: float = _env_float("DEFAULT_PLOT_BOUND", 10)

# allclose atol for checking computed (A, b) against network outputs.
TOL_VERIFY_AB_ATOL: float = _env_float("TOL_VERIFY_AB_ATOL", 1e-6)

# -----------------------------------------------------------------------------
# Complex search & parallel add
# -----------------------------------------------------------------------------

# max_radius values tried in turn when finding a neighbor's interior point (get_ip).
INTERIOR_POINT_RADIUS_SEQUENCE: list[float] = _env_float_list("INTERIOR_POINT_RADIUS_SEQUENCE", [0.01, 0.1, 1, 10, 100])

# Default bound for halfspace computation in parallel_add.
DEFAULT_PARALLEL_ADD_BOUND: float = _env_float("DEFAULT_PARALLEL_ADD_BOUND", 1e8)

# Default bound for halfspace computation in the searcher and hamming_astar.
# Too large can upset the solver.
DEFAULT_SEARCH_BOUND: float = _env_float("DEFAULT_SEARCH_BOUND", 1e8)

# Strict margin for boundary MIP witness pricing (``z_j = 0`` at ``boundary_shi``,
# ``|z_j| >= eps`` elsewhere).
BOUNDARY_MIP_EPS: float = _env_float("BOUNDARY_MIP_EPS", 1e-4)

# For total ReLU width ``n`` up to this, also try a brute-force sign-pattern scan.
BOUNDARY_PRICING_BRUTE_FORCE_MAX_N: int = _env_int("BOUNDARY_PRICING_BRUTE_FORCE_MAX_N", 18)

# Gurobi time limit (s) per boundary pricing MIP; ``0`` = none.
BOUNDARY_MIP_TIME_LIMIT: float = _env_float("BOUNDARY_MIP_TIME_LIMIT", 0.0)

# Show full Gurobi logs during boundary pricing MIPs (separate from :data:`VERBOSE`).
BOUNDARY_MIP_GUROBI_LOG: bool = _env_bool("BOUNDARY_MIP_GUROBI_LOG", False)

# Multiplier on the propagated preactivation bound used as the MIP big-M.
BOUNDARY_MIP_BOUND_MARGIN: float = _env_float("BOUNDARY_MIP_BOUND_MARGIN", 5.0)

# Compile ``exclude_tags`` into trie no-goods at this many tags (``0`` = always, huge = never).
BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS: int = _env_int(
    "BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS",
    1000,
)

# At this many excluded tags, add per-tag no-goods up front, before optimize.
BOUNDARY_MIP_STATIC_EXCLUSION_MIN_TAGS: int = _env_int(
    "BOUNDARY_MIP_STATIC_EXCLUSION_MIN_TAGS",
    1000,
)

# Add all leaf nogoods up front if the trie compression ratio is below this.
BOUNDARY_MIP_STATIC_EXCLUSION_MIN_RATIO: float = _env_float(
    "BOUNDARY_MIP_STATIC_EXCLUSION_MIN_RATIO",
    2.0,
)

# Batch size for Gurobi ``addConstrs`` when emitting nogoods.
BOUNDARY_MIP_EXCLUSION_BATCH_SIZE: int = _env_int(
    "BOUNDARY_MIP_EXCLUSION_BATCH_SIZE",
    4096,
)

# Workers for compiling nogood specs; ``0`` = auto, ``1`` = serial.
BOUNDARY_MIP_EXCLUSION_WORKERS: int = _env_int(
    "BOUNDARY_MIP_EXCLUSION_WORKERS",
    0,
)

# Cut ordering for static/lazy nogoods: as_is, tag_lex, layer_major, hamming_median,
# literal_count_asc, trie_depth_desc, random.
BOUNDARY_MIP_CUT_ORDER: str = os.environ.get("BOUNDARY_MIP_CUT_ORDER", "tag_lex")

# Bulk matrix emit for static nogoods: auto, on, off (auto = on at >= 500 specs).
BOUNDARY_MIP_BULK_NOGOOD_EMIT: str = os.environ.get("BOUNDARY_MIP_BULK_NOGOOD_EMIT", "auto")

# Static nogood wave size; ``0`` = add everything before one optimize.
BOUNDARY_MIP_STATIC_WAVE_SIZE: int = _env_int("BOUNDARY_MIP_STATIC_WAVE_SIZE", 0)

# Give trie-compressed / deeper-path cuts higher Gurobi priority.
BOUNDARY_MIP_CUT_PRIORITY_ENABLED: bool = _env_bool("BOUNDARY_MIP_CUT_PRIORITY_ENABLED", False)

# At this many ``exclude_tags``, skip precompilation and use lazy MIPSOL cuts instead.
BOUNDARY_MIP_LAZY_ONLY_MIN_TAGS: int = _env_int("BOUNDARY_MIP_LAZY_ONLY_MIN_TAGS", 50_000)

# A* heuristic: f = hamming + ASTAR_BIAS_WEIGHT * bias. The bias is negative and
# favors cells closer to the goal in input space.
ASTAR_BIAS_WEIGHT: float = _env_float("ASTAR_BIAS_WEIGHT", 0.9)

# Plot axis range = max interior coordinate * this factor.
PLOT_MARGIN_FACTOR: float = _env_float("PLOT_MARGIN_FACTOR", 1.1)

# -----------------------------------------------------------------------------
# Topology / Betti computation
# -----------------------------------------------------------------------------

# Match a geometric vertex to an intrinsic one if
# ||x - x_intrinsic||_inf <= TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR * tol.
TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR: float = _env_float("TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR", 2.0)

# -----------------------------------------------------------------------------
# Logging / verbosity
# -----------------------------------------------------------------------------

# How much relucent prints:
#   0 → WARNING (quiet)
#   1 → INFO (default; progress, worker counts, ...)
# 2+ is reserved for DEBUG.
VERBOSE: int = _env_int("VERBOSE", 1)

# Max coordinate for 2D plots with no interior points.
PLOT_DEFAULT_MAXCOORD: float = _env_float("PLOT_DEFAULT_MAXCOORD", 10)

# Default bound for Complex.plot (2D).
DEFAULT_COMPLEX_PLOT_BOUND: float = _env_float("DEFAULT_COMPLEX_PLOT_BOUND", 10000)

# -----------------------------------------------------------------------------
# Utilities & visualization
# -----------------------------------------------------------------------------

# BlockingQueue: seconds to wait on the lock before rechecking.
BLOCKING_QUEUE_WAIT_TIMEOUT: float = _env_float("BLOCKING_QUEUE_WAIT_TIMEOUT", 0.5)

__all__ = [
    "ASTAR_BIAS_WEIGHT",
    "BLOCKING_QUEUE_WAIT_TIMEOUT",
    "BOUNDARY_MIP_EPS",
    "BOUNDARY_MIP_TIME_LIMIT",
    "BOUNDARY_MIP_BOUND_MARGIN",
    "BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS",
    "BOUNDARY_MIP_STATIC_EXCLUSION_MIN_TAGS",
    "BOUNDARY_MIP_STATIC_EXCLUSION_MIN_RATIO",
    "BOUNDARY_MIP_EXCLUSION_BATCH_SIZE",
    "BOUNDARY_MIP_EXCLUSION_WORKERS",
    "BOUNDARY_MIP_GUROBI_LOG",
    "BOUNDARY_MIP_CUT_ORDER",
    "BOUNDARY_MIP_BULK_NOGOOD_EMIT",
    "BOUNDARY_MIP_STATIC_WAVE_SIZE",
    "BOUNDARY_MIP_CUT_PRIORITY_ENABLED",
    "BOUNDARY_MIP_LAZY_ONLY_MIN_TAGS",
    "BOUNDARY_PRICING_BRUTE_FORCE_MAX_N",
    "CAREFUL_MODE",
    "DEFAULT_COMPLEX_PLOT_BOUND",
    "DEFAULT_PARALLEL_ADD_BOUND",
    "DEFAULT_PLOT_BOUND",
    "DEFAULT_SEARCH_BOUND",
    "GUROBI_SHI_OPTIMALITY_TOL",
    "GUROBI_SHI_SCALE_FLAG",
    "INTERIOR_POINT_RADIUS_SEQUENCE",
    "MAX_RADIUS",
    "PLOT_DEFAULT_MAXCOORD",
    "PLOT_MARGIN_FACTOR",
    "QHULL_MODE",
    "TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR",
    "TOL_HALFSPACE_NORMAL",
    "TOL_NEARLY_VERTICAL",
    "TOL_VERIFY_AB_ATOL",
    "VERBOSE",
    "update_settings",
]


def update_settings(**kwargs: Any) -> None:
    """Set one or more config attributes.

    Changing :data:`VERBOSE` also updates the ``"relucent"`` logger level.

    Args:
        **kwargs: ``NAME=value`` pairs for public settings.

    Raises:
        TypeError: If any key is not a known setting name.
    """
    allowed = set(__all__) - {"update_settings"}
    unknown = set(kwargs) - allowed
    if unknown:
        raise TypeError(f"Unknown config keys: {sorted(unknown)}")
    mod = globals()
    for k, v in kwargs.items():
        mod[k] = v
    if "VERBOSE" in kwargs:
        from relucent._internal.logging import _apply_verbose

        _apply_verbose(int(kwargs["VERBOSE"]))
