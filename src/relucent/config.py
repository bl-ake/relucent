"""Central configuration for relucent package constants.

All tunable numerical constants, tolerances, and defaults used across the
package are defined here with brief explanations. Code should import from
this module rather than hard-coding values.

You can override values at runtime by mutating attributes on this module,
e.g. ``relucent.config.TOL_HALFSPACE_CONTAINMENT = 1e-7``.
"""

# -----------------------------------------------------------------------------
# Polyhedron & halfspace geometry
# -----------------------------------------------------------------------------

# Maximum radius when solving for Chebyshev center / interior point.
# Smaller values speed up Gurobi but exclude any polyhedra for which no point
# is within MAX_RADIUS of every face.
MAX_RADIUS = 100

# Threshold for trusting vertex positions from HalfspaceIntersection.
# Vertices are trusted only if the sum of (halfspace coords @ vertex + bias)
# over supporting halfspaces is below this value. Used in vertex validation.
VERTEX_TRUST_THRESHOLD = 1e-6

# Tolerance for "point in halfspace": point satisfies a^T x + b <= 0 if
# a^T x + b <= TOL_HALFSPACE_CONTAINMENT. Used in interior-point checks,
# __contains__, and get_bounded_vertices.
TOL_HALFSPACE_CONTAINMENT = 1e-6

# Threshold below which a ReLU is considered dead (always off).
# Dead ReLUs have |constr_A| < TOL_DEAD_RELU along their column.
TOL_DEAD_RELU = 1e-8

# Tolerance for equality in get_shis: hyperplanes are considered to intersect
# when A @ x == -b - TOL_SHI_HYPERPLANE (relaxed for numerical stability).
TOL_SHI_HYPERPLANE = 1e-6

# Gurobi MIP tolerances when computing supporting halfspace indices (SHIs).
# BestObjStop: stop when objective >= this (for maximization).
# BestBdStop: stop when best bound <= this.
GUROBI_SHI_BEST_OBJ_STOP = 1e-6
GUROBI_SHI_BEST_BD_STOP = -1e-6

# In plot2d, halfspace normal components below this are treated as zero
# (e.g. nearly vertical line: |w[1]| < TOL_NEARLY_VERTICAL).
TOL_NEARLY_VERTICAL = 1e-10

# Default bound (hypercube half-width) for polyhedron plotting and
# get_bounded_vertices when not specified by the caller.
DEFAULT_PLOT_BOUND = 10000

# Tolerance for asserting that computed (A, b) match network outputs
# when verifying halfspace construction (torch/np allclose atol).
TOL_VERIFY_AB_ATOL = 1e-6

# -----------------------------------------------------------------------------
# Complex search & parallel add
# -----------------------------------------------------------------------------

# Sequence of max_radius values tried when finding an interior point for
# a neighbor polyhedron in get_ip (increasing on failure).
INTERIOR_POINT_RADIUS_SEQUENCE = [0.01, 0.1, 1, 10, 100]

# Default bound for numerical stability when computing halfspaces in
# parallel_add (larger than searcher to allow broader exploration).
DEFAULT_PARALLEL_ADD_BOUND = 1e8

# Default bound for halfspace computation in searcher and hamming_astar.
# Important for numerical stability; too large can cause solver issues.
DEFAULT_SEARCH_BOUND = 1e8

# Weight for Euclidean-distance bias in A* heuristic: f = hamming + ASTAR_BIAS_WEIGHT * bias.
# bias is negative and tends to favor polyhedra closer to the goal in input space.
ASTAR_BIAS_WEIGHT = 0.9

# When computing plot axis range from interior points: multiply max coordinate
# by this factor to add margin (e.g. maxcoord * PLOT_MARGIN_FACTOR).
PLOT_MARGIN_FACTOR = 1.1

# Fallback max coordinate for 2D plot when there are no interior points.
PLOT_DEFAULT_MAXCOORD = 10

# Default bound passed to polyhedron plotting from Complex.plot (2D).
DEFAULT_COMPLEX_PLOT_BOUND = 10000

# -----------------------------------------------------------------------------
# Utilities & visualization
# -----------------------------------------------------------------------------

# BlockingQueue: how long (seconds) to wait on the lock before rechecking.
BLOCKING_QUEUE_WAIT_TIMEOUT = 0.5

# Pie chart label distance in pyvis graph (matplotlib pie).
PIE_LABEL_DISTANCE = 0.6

# create_pyvis_graph: max number of node images to generate.
MAX_IMAGES_PYVIS = 3000

# create_pyvis_graph: max number of data examples to show per node.
MAX_NUM_EXAMPLES_PYVIS = 3

# create_pyvis_graph: default path for saving the HTML graph.
DEFAULT_PYVIS_SAVE_FILE = "./graph.html"

# -----------------------------------------------------------------------------
# Model / grid
# -----------------------------------------------------------------------------

# get_grid / output_grid: default half-width of the grid (span is [-bounds, bounds]).
DEFAULT_GRID_BOUNDS = 2

# get_grid / output_grid: default resolution (points per dimension).
DEFAULT_GRID_RES = 100
