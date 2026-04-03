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

See the :doc:`configuration` chapter in the HTML documentation for a full
settings reference.
"""

from __future__ import annotations

from typing import Any

# -----------------------------------------------------------------------------
# Polyhedron & halfspace geometry
# -----------------------------------------------------------------------------

# QHULL_MODE controls how Qhull numerical warnings from HalfspaceIntersection
# are handled:
#   - "IGNORE" (default): proceed with the original call even if Qhull warns.
#   - "WARN_ALL": warn about all warnings.
#   - "HIGH_PRECISION": treat any warning as a hard error.
#   - "JITTERED": retry with Qhull's "QJ" (joggled) option on warning.
QHULL_MODE: str = "IGNORE"

# Maximum radius when solving for Chebyshev center / interior point. Smaller
# values speed up Gurobi but exclude any polyhedra for which no point is
# within MAX_RADIUS of every face.
MAX_RADIUS: float = 100

# Threshold for trusting vertex positions from HalfspaceIntersection.
# Vertices are trusted only if the sum of (halfspace coords @ vertex + bias)
# over supporting halfspaces is below this value. Used in vertex validation.
VERTEX_TRUST_THRESHOLD: float = 1e-6

# Tolerance for "point in halfspace": point satisfies a^T x + b <= 0 if
# a^T x + b <= TOL_HALFSPACE_CONTAINMENT. Used in interior-point checks,
# __contains__, and get_bounded_vertices.
TOL_HALFSPACE_CONTAINMENT: float = 1e-6

# After Chebyshev / interior LP (Gurobi), max halfspace violation allowed vs.
# :func:`_drop_degenerate_halfspaces` rows — same set enforced in ``solve_radius``.
# Slightly looser than :data:`TOL_HALFSPACE_CONTAINMENT` to absorb LP
# FeasibilityTol and floating-point noise (often platform-dependent).
TOL_INTERIOR_VERIFY: float = 1e-5

# Threshold below which a ReLU is considered dead (always off).
# Dead ReLUs have |constr_A| < TOL_DEAD_RELU along their column.
TOL_DEAD_RELU: float = 1e-8

# Tolerance for equality in get_shis: hyperplanes are considered to intersect
# when A @ x == -b - TOL_SHI_HYPERPLANE (relaxed for numerical stability).
TOL_SHI_HYPERPLANE: float = 1e-6

# Threshold for treating a halfspace normal as "degenerate".
# If ||a|| < TOL_HALFSPACE_NORMAL, the constraint a^T x + b <= 0 is either
# redundant (if b <= 0) or infeasible (if b > 0). Degenerate constraints can
# cause Qhull (HalfspaceIntersection) failures and numerical pathologies.
TOL_HALFSPACE_NORMAL: float = 1e-12

# Gurobi MIP tolerances when computing supporting halfspace indices (SHIs).
# BestObjStop: stop when objective >= this (for maximization).
# BestBdStop: stop when best bound <= this.
GUROBI_SHI_BEST_OBJ_STOP: float = 1e-6
GUROBI_SHI_BEST_BD_STOP: float = -1e-6

# In 2D cell plotting (``Polyhedron.plot_cells`` when ``ambient_dim == 2``), halfspace normal
# components below this are treated as zero
# (e.g. nearly vertical line: |w[1]| < TOL_NEARLY_VERTICAL).
TOL_NEARLY_VERTICAL: float = 1e-10

# Default bound (hypercube half-width) for polyhedron plotting and
# get_bounded_vertices when not specified by the caller.
DEFAULT_PLOT_BOUND: float = 10

# Tolerance for asserting that computed (A, b) match network outputs
# when verifying halfspace construction (torch/np allclose atol).
TOL_VERIFY_AB_ATOL: float = 1e-6

# -----------------------------------------------------------------------------
# Complex search & parallel add
# -----------------------------------------------------------------------------

# Sequence of max_radius values tried when finding an interior point for
# a neighbor polyhedron in relucent.search.get_ip (increasing on failure).
INTERIOR_POINT_RADIUS_SEQUENCE: list[float] = [0.01, 0.1, 1, 10, 100]

# Default bound for numerical stability when computing halfspaces in
# relucent.search.parallel_add (larger than searcher to allow broader exploration).
DEFAULT_PARALLEL_ADD_BOUND: float = 1e8

# Default bound for halfspace computation in searcher and hamming_astar.
# Important for numerical stability; too large can cause solver issues.
DEFAULT_SEARCH_BOUND: float = 1e8

# Weight for Euclidean-distance bias in A* heuristic: f = hamming + ASTAR_BIAS_WEIGHT * bias.
# bias is negative and tends to favor polyhedra closer to the goal in input space.
ASTAR_BIAS_WEIGHT: float = 0.9

# When computing plot axis range from interior points: multiply max coordinate
# by this factor to add margin (e.g. maxcoord * PLOT_MARGIN_FACTOR).
PLOT_MARGIN_FACTOR: float = 1.1

# Fallback max coordinate for 2D plot when there are no interior points.
PLOT_DEFAULT_MAXCOORD: float = 10

# Default bound passed to polyhedron plotting from Complex.plot (2D).
DEFAULT_COMPLEX_PLOT_BOUND: float = 10000

# -----------------------------------------------------------------------------
# Utilities & visualization
# -----------------------------------------------------------------------------

# BlockingQueue: how long (seconds) to wait on the lock before rechecking.
BLOCKING_QUEUE_WAIT_TIMEOUT: float = 0.5

# Pie chart label distance in pyvis graph (matplotlib pie).
PIE_LABEL_DISTANCE: float = 0.6

# create_pyvis_graph: max number of node images to generate.
MAX_IMAGES_PYVIS: int = 3000

# create_pyvis_graph: max number of data examples to show per node.
MAX_NUM_EXAMPLES_PYVIS: int = 3

# create_pyvis_graph: default path for saving the HTML graph.
DEFAULT_PYVIS_SAVE_FILE: str = "./graph.html"

# -----------------------------------------------------------------------------
# Model / grid
# -----------------------------------------------------------------------------

# get_grid / output_grid: default half-width of the grid (span is [-bounds, bounds]).
DEFAULT_GRID_BOUNDS: float = 2

# get_grid / output_grid: default resolution (points per dimension).
DEFAULT_GRID_RES: int = 100


__all__ = [
    "ASTAR_BIAS_WEIGHT",
    "BLOCKING_QUEUE_WAIT_TIMEOUT",
    "DEFAULT_COMPLEX_PLOT_BOUND",
    "DEFAULT_GRID_BOUNDS",
    "DEFAULT_GRID_RES",
    "DEFAULT_PARALLEL_ADD_BOUND",
    "DEFAULT_PLOT_BOUND",
    "DEFAULT_PYVIS_SAVE_FILE",
    "DEFAULT_SEARCH_BOUND",
    "GUROBI_SHI_BEST_BD_STOP",
    "GUROBI_SHI_BEST_OBJ_STOP",
    "INTERIOR_POINT_RADIUS_SEQUENCE",
    "MAX_IMAGES_PYVIS",
    "MAX_RADIUS",
    "MAX_NUM_EXAMPLES_PYVIS",
    "PIE_LABEL_DISTANCE",
    "PLOT_DEFAULT_MAXCOORD",
    "PLOT_MARGIN_FACTOR",
    "QHULL_MODE",
    "TOL_DEAD_RELU",
    "TOL_HALFSPACE_CONTAINMENT",
    "TOL_INTERIOR_VERIFY",
    "TOL_HALFSPACE_NORMAL",
    "TOL_NEARLY_VERTICAL",
    "TOL_SHI_HYPERPLANE",
    "TOL_VERIFY_AB_ATOL",
    "VERTEX_TRUST_THRESHOLD",
    "update_settings",
]


def update_settings(**kwargs: Any) -> None:
    """Set one or more :mod:`relucent.config` attributes.

    Keys must be names listed in :data:`relucent.config.__all__` (excluding
    ``update_settings`` itself). Values replace the current module-level
    constants.

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
