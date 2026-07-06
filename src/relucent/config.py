"""Central configuration for relucent package constants.

All tunable numerical constants, tolerances, and defaults used across the
package are defined here with brief explanations. Code reads values from this
module at **use time** (via ``import relucent.config as cfg`` and ``cfg.NAME``)
so you can adjust behavior by assigning to attributes on :mod:`relucent.config`
before or between calls.

**How to change settings**

* Assign directly::

    import relucent
    relucent.config.TOL_HALFSPACE_CONTAINMENT = 1e-7

* Or use :func:`update_settings` to set several keys at once::

    from relucent.config import update_settings
    update_settings(TOL_HALFSPACE_CONTAINMENT=1e-7, MAX_RADIUS=200)

* Or set environment variables before import (for example
  ``RELUCENT_TOL_HALFSPACE_CONTAINMENT=1e-7`` or ``RELUCENT_CAREFUL_MODE=1``).
  Every public setting can be overridden with ``RELUCENT_<SETTING_NAME>``.

See the :doc:`configuration` chapter in the HTML documentation for a full
settings reference.
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

    # Accept either comma-separated values ("0.1,1,10") or a bracketed form
    # ("[0.1, 1, 10]") for convenience in shell environments.
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

# When True, run extra consistency checks (assertions, forward-pass vs. affine
# recomputation, conversion spot-checks, etc.). Intended for tests and
# debugging; default is False for lower overhead in production runs.
# Set ``RELUCENT_CAREFUL_MODE=1`` before import, or assign after import.
CAREFUL_MODE: bool = _env_bool("CAREFUL_MODE", False)

# QHULL_MODE controls how Qhull numerical warnings from HalfspaceIntersection
# are handled:
#   - "IGNORE" (default): proceed with the original call even if Qhull warns.
#   - "WARN_ALL": warn about all warnings.
#   - "HIGH_PRECISION": treat any warning as a hard error.
#   - "JITTERED": retry with Qhull's "QJ" (joggled) option on warning.
QHULL_MODE: str = _env_str("QHULL_MODE", "IGNORE")

# Maximum radius when solving for Chebyshev center / interior point. Smaller
# values speed up Gurobi but exclude any polyhedra for which no point is
# within MAX_RADIUS of every face.
MAX_RADIUS: float = _env_float("MAX_RADIUS", 100)

# Threshold for trusting vertex positions from HalfspaceIntersection.
# Vertices are trusted only if the sum of (halfspace coords @ vertex + bias)
# over supporting halfspaces is below this value. Used in vertex validation.
VERTEX_TRUST_THRESHOLD: float = _env_float("VERTEX_TRUST_THRESHOLD", 1e-6)

# Tolerance for "point in halfspace": point satisfies a^T x + b <= 0 if
# a^T x + b <= TOL_HALFSPACE_CONTAINMENT. Used in interior-point checks,
# __contains__, and get_bounded_vertices.
TOL_HALFSPACE_CONTAINMENT: float = _env_float("TOL_HALFSPACE_CONTAINMENT", 1e-6)

# After Chebyshev / interior LP (Gurobi), lstsq residuals in ``solve_radius`` and
# equality checks in ``Polyhedron._halfspace_point`` use this tolerance (relative to
# ``||b||`` where applicable). Slightly looser than :data:`TOL_HALFSPACE_CONTAINMENT`
# to absorb LP FeasibilityTol and floating-point noise (often platform-dependent).
TOL_INTERIOR_VERIFY: float = _env_float("TOL_INTERIOR_VERIFY", 1e-5)

# Threshold below which a ReLU is considered dead (always off).
# Dead ReLUs have |constr_A| < TOL_DEAD_RELU along their column.
TOL_DEAD_RELU: float = _env_float("TOL_DEAD_RELU", 1e-8)

# Tolerance for equality in get_shis: hyperplanes are considered to intersect
# when A @ x == -b - TOL_SHI_HYPERPLANE (relaxed for numerical stability).
TOL_SHI_HYPERPLANE: float = _env_float("TOL_SHI_HYPERPLANE", 1e-6)

# Threshold for treating a halfspace normal as "degenerate".
# If ||a|| < TOL_HALFSPACE_NORMAL, the constraint a^T x + b <= 0 is either
# redundant (if b <= 0) or infeasible (if b > 0). Degenerate constraints can
# cause Qhull (HalfspaceIntersection) failures and numerical pathologies.
TOL_HALFSPACE_NORMAL: float = _env_float("TOL_HALFSPACE_NORMAL", 1e-12)

# Gurobi MIP tolerances when computing supporting halfspace indices (SHIs).
# BestObjStop: stop when objective >= this (for maximization).
# BestBdStop: stop when best bound <= this.
GUROBI_SHI_BEST_OBJ_STOP: float = _env_float("GUROBI_SHI_BEST_OBJ_STOP", 1e-6)
GUROBI_SHI_BEST_BD_STOP: float = _env_float("GUROBI_SHI_BEST_BD_STOP", -1e-6)

# Minimum SHI LP objective (margin along hyperplane i after relaxing that face) to
# certify a supporting hyperplane. The objective is in the same units as halfspace
# slacks a^T x + b. Values near zero (e.g. 1e-18) are floating-point noise and
# correspond to sign patterns with no open cell at this tolerance; default matches
# :data:`GUROBI_SHI_BEST_OBJ_STOP` and :data:`TOL_SHI_HYPERPLANE`.
TOL_SHI_OBJECTIVE: float = _env_float("TOL_SHI_OBJECTIVE", 1e-8)

# During complex search (BFS/A*), raise if a discovered cell's Chebyshev inradius is
# below this floor.  Defaults to half of :data:`TOL_SHI_OBJECTIVE` so that when a thin
# cell is sandwiched between similarly thin neighbors, SHI LPs are not operating in a
# regime where opposing halfspaces can sit within tolerance of the relaxed boundary.
MIN_SEARCH_INRADIUS: float = _env_float("MIN_SEARCH_INRADIUS", TOL_SHI_OBJECTIVE / 2)

# In 2D cell plotting (``Polyhedron.plot_cells`` when ``ambient_dim == 2``), halfspace normal
# components below this are treated as zero
# (e.g. nearly vertical line: |w[1]| < TOL_NEARLY_VERTICAL).
TOL_NEARLY_VERTICAL: float = _env_float("TOL_NEARLY_VERTICAL", 1e-10)

# Default bound (hypercube half-width) for polyhedron plotting and
# get_bounded_vertices when not specified by the caller.
DEFAULT_PLOT_BOUND: float = _env_float("DEFAULT_PLOT_BOUND", 10)

# Tolerance for asserting that computed (A, b) match network outputs
# when verifying halfspace construction (torch/np allclose atol).
TOL_VERIFY_AB_ATOL: float = _env_float("TOL_VERIFY_AB_ATOL", 1e-6)

# -----------------------------------------------------------------------------
# Complex search & parallel add
# -----------------------------------------------------------------------------

# Sequence of max_radius values tried when finding an interior point for
# a neighbor polyhedron in relucent.search.get_ip (increasing on failure).
INTERIOR_POINT_RADIUS_SEQUENCE: list[float] = _env_float_list("INTERIOR_POINT_RADIUS_SEQUENCE", [0.01, 0.1, 1, 10, 100])

# Default bound for numerical stability when computing halfspaces in
# relucent.search.parallel_add (larger than searcher to allow broader exploration).
DEFAULT_PARALLEL_ADD_BOUND: float = _env_float("DEFAULT_PARALLEL_ADD_BOUND", 1e8)

# Default bound for halfspace computation in searcher and hamming_astar.
# Important for numerical stability; too large can cause solver issues.
DEFAULT_SEARCH_BOUND: float = _env_float("DEFAULT_SEARCH_BOUND", 1e8)

# Strict-margin tolerance for boundary MIP witness pricing (``z_j = 0`` at ``boundary_shi``,
# ``|z_j| >= eps`` elsewhere on top-dimensional boundary cells).
BOUNDARY_MIP_EPS: float = _env_float("BOUNDARY_MIP_EPS", 1e-4)

# When total ReLU width ``n`` is at most this value, try brute-force sign-pattern scan
# before or after MIP pricing (robust on tiny networks).
BOUNDARY_PRICING_BRUTE_FORCE_MAX_N: int = _env_int("BOUNDARY_PRICING_BRUTE_FORCE_MAX_N", 18)

# Optional Gurobi time limit (seconds) for a single boundary pricing MIP; ``0`` means no limit.
BOUNDARY_MIP_TIME_LIMIT: float = _env_float("BOUNDARY_MIP_TIME_LIMIT", 0.0)

# When True, emit full Gurobi solver logs during boundary pricing MIPs. Independent of
# :data:`VERBOSE`, which still controls Relucent progress messages and tqdm bars.
BOUNDARY_MIP_GUROBI_LOG: bool = _env_bool("BOUNDARY_MIP_GUROBI_LOG", False)

# Multiplier applied to the layerwise propagated preactivation bound for pricing MIP ``big-M``.
BOUNDARY_MIP_BOUND_MARGIN: float = _env_float("BOUNDARY_MIP_BOUND_MARGIN", 10.0)

# Compile ``exclude_tags`` into compressed trie no-goods when at least this many tags
# are present; set to ``0`` to always compile, or a large value to disable.
BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS: int = _env_int(
    "BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS",
    1000,
)

# When at least this many tags are excluded, bulk-add per-tag no-goods statically before optimize.
BOUNDARY_MIP_STATIC_EXCLUSION_MIN_TAGS: int = _env_int(
    "BOUNDARY_MIP_STATIC_EXCLUSION_MIN_TAGS",
    1000,
)

# Static-add all leaf nogoods when trie compression ratio falls below this threshold.
BOUNDARY_MIP_STATIC_EXCLUSION_MIN_RATIO: float = _env_float(
    "BOUNDARY_MIP_STATIC_EXCLUSION_MIN_RATIO",
    2.0,
)

# Chunk size for batched Gurobi ``addConstrs`` when emitting exclusion nogoods.
BOUNDARY_MIP_EXCLUSION_BATCH_SIZE: int = _env_int(
    "BOUNDARY_MIP_EXCLUSION_BATCH_SIZE",
    4096,
)

# Workers for parallel nogood spec compilation; ``0`` = auto, ``1`` = serial.
BOUNDARY_MIP_EXCLUSION_WORKERS: int = _env_int(
    "BOUNDARY_MIP_EXCLUSION_WORKERS",
    0,
)

# Cut ordering for static/lazy nogoods: as_is, tag_lex, layer_major, hamming_median,
# literal_count_asc, trie_depth_desc, random.
BOUNDARY_MIP_CUT_ORDER: str = os.environ.get("BOUNDARY_MIP_CUT_ORDER", "tag_lex")

# Bulk matrix emit for static nogoods: auto, on, off (auto uses bulk when n_specs >= 500).
BOUNDARY_MIP_BULK_NOGOOD_EMIT: str = os.environ.get("BOUNDARY_MIP_BULK_NOGOOD_EMIT", "auto")

# Static nogood wave size; ``0`` = add all constraints before a single optimize.
BOUNDARY_MIP_STATIC_WAVE_SIZE: int = _env_int("BOUNDARY_MIP_STATIC_WAVE_SIZE", 0)

# When True, assign higher Gurobi priority to trie-compressed / deeper-path cuts.
BOUNDARY_MIP_CUT_PRIORITY_ENABLED: bool = _env_bool("BOUNDARY_MIP_CUT_PRIORITY_ENABLED", False)

# When ``len(exclude_tags)`` reaches this threshold, skip trie/static precompilation
# and rely on lazy MIPSOL cuts against the visited-tag set instead.
BOUNDARY_MIP_LAZY_ONLY_MIN_TAGS: int = _env_int("BOUNDARY_MIP_LAZY_ONLY_MIN_TAGS", 50_000)

# Weight for Euclidean-distance bias in A* heuristic: f = hamming + ASTAR_BIAS_WEIGHT * bias.
# bias is negative and tends to favor polyhedra closer to the goal in input space.
ASTAR_BIAS_WEIGHT: float = _env_float("ASTAR_BIAS_WEIGHT", 0.9)

# When computing plot axis range from interior points: multiply max coordinate
# by this factor to add margin (e.g. maxcoord * PLOT_MARGIN_FACTOR).
PLOT_MARGIN_FACTOR: float = _env_float("PLOT_MARGIN_FACTOR", 1.1)

# -----------------------------------------------------------------------------
# Topology / Betti computation
# -----------------------------------------------------------------------------

# When True, :meth:`~relucent.complex.Complex.get_dual_graph` builds adjacency by
# grouping top-dimensional cells on shared codimension-one face tags (cubical rule).
# When False, use the legacy flip-on-``_shis`` walk (search / contraction SHI lists).
CUBICAL_DUAL_GRAPH: bool = _env_bool("CUBICAL_DUAL_GRAPH", True)

# When True, run geometric genericity / transversality checks (e.g. degenerate
# 1-cell endpoints). Expensive on large complexes; enabled automatically in
# :meth:`~relucent.complex.Complex.get_boundary_complex`.
VERIFY_GENERICITY: bool = _env_bool("VERIFY_GENERICITY", False)

# When merging geometrically-computed vertices with combinatorially-identified intrinsic vertices,
# accept a match if ||x - x_intrinsic||_inf <= TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR * tol.
TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR: float = _env_float("TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR", 2.0)

# -----------------------------------------------------------------------------
# Logging / verbosity
# -----------------------------------------------------------------------------

# Controls how much relucent prints during search and parallel operations.
# The integer maps to Python logging levels:
#   0  → WARNING  (silent — only errors/warnings are shown)
#   1  → INFO     (default — normal progress: worker counts, cube-filter info, …)
# Values >= 2 are reserved for future DEBUG-level output.
# Can be overridden at runtime via update_settings(VERBOSE=0) or by setting
# the environment variable RELUCENT_VERBOSE before importing relucent.
VERBOSE: int = _env_int("VERBOSE", 1)

# Fallback max coordinate for 2D plot when there are no interior points.
PLOT_DEFAULT_MAXCOORD: float = _env_float("PLOT_DEFAULT_MAXCOORD", 10)

# Default bound passed to polyhedron plotting from Complex.plot (2D).
DEFAULT_COMPLEX_PLOT_BOUND: float = _env_float("DEFAULT_COMPLEX_PLOT_BOUND", 10000)

# -----------------------------------------------------------------------------
# Utilities & visualization
# -----------------------------------------------------------------------------

# BlockingQueue: how long (seconds) to wait on the lock before rechecking.
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
    "GUROBI_SHI_BEST_BD_STOP",
    "GUROBI_SHI_BEST_OBJ_STOP",
    "INTERIOR_POINT_RADIUS_SEQUENCE",
    "MIN_SEARCH_INRADIUS",
    "MAX_RADIUS",
    "PLOT_DEFAULT_MAXCOORD",
    "PLOT_MARGIN_FACTOR",
    "QHULL_MODE",
    "TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR",
    "TOL_DEAD_RELU",
    "TOL_HALFSPACE_CONTAINMENT",
    "TOL_INTERIOR_VERIFY",
    "TOL_HALFSPACE_NORMAL",
    "TOL_NEARLY_VERTICAL",
    "TOL_SHI_HYPERPLANE",
    "TOL_SHI_OBJECTIVE",
    "TOL_VERIFY_AB_ATOL",
    "VERBOSE",
    "VERTEX_TRUST_THRESHOLD",
    "update_settings",
]


def update_settings(**kwargs: Any) -> None:
    """Set one or more :mod:`relucent.config` attributes.

    Keys must be names listed in :data:`relucent.config.__all__` (excluding
    ``update_settings`` itself). Values replace the current module-level
    constants.

    Changing :data:`VERBOSE` also immediately updates the level of the
    ``"relucent"`` package logger (see :mod:`relucent._logging`).

    Args:
        **kwargs: ``NAME=value`` pairs matching public config attributes.

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
        from relucent._logging import _apply_verbose

        _apply_verbose(int(kwargs["VERBOSE"]))
