"""Tunable constants and defaults for relucent.

Code reads these at use time (``cfg.NAME``), so you can change them before or
between calls.

Ways to change a setting:

* Assign directly::

    import relucent
    relucent.config.MAX_RADIUS = 500

* Set several at once::

    from relucent.config import update_settings
    update_settings(MAX_RADIUS=500, VERBOSE=0)

* Use an environment variable before import, e.g.
  ``RELUCENT_CAREFUL_MODE=1``. Any setting works as ``RELUCENT_<SETTING_NAME>``.

Solver-tuning knobs live in :mod:`relucent.config.advanced`; they are unstable and may
change in any release. See :doc:`configuration` for the full reference.
"""

from __future__ import annotations

from typing import Any, Literal

from relucent.config import advanced
from relucent.config._env import check_choice, env_bool, env_choice, env_float, env_int, env_optional_float

# -----------------------------------------------------------------------------
# Polyhedron geometry
# -----------------------------------------------------------------------------

# Run extra consistency checks (assertions, forward pass vs. affine
# recomputation, conversion spot-checks). Handy for tests and debugging, but slower.
CAREFUL_MODE: bool = env_bool("CAREFUL_MODE", False)

QhullMode = Literal["IGNORE", "WARN_ALL", "HIGH_PRECISION", "JITTERED"]
QHULL_MODES: tuple[QhullMode, ...] = ("IGNORE", "WARN_ALL", "HIGH_PRECISION", "JITTERED")

# What to do when Qhull warns during HalfspaceIntersection:
#   - "IGNORE" (default): carry on.
#   - "WARN_ALL": pass every warning along.
#   - "HIGH_PRECISION": treat a warning as an error.
#   - "JITTERED": retry with Qhull's "QJ" (joggle) option.
QHULL_MODE: QhullMode = env_choice("QHULL_MODE", "IGNORE", QHULL_MODES)  # pyright: ignore[reportAssignmentType]

# Max radius when solving for a Chebyshev center / interior point. Smaller is
# faster, but drops cells with no point within MAX_RADIUS of every face.
MAX_RADIUS: float = env_float("MAX_RADIUS", 100)

# -----------------------------------------------------------------------------
# Complex search & parallel add
# -----------------------------------------------------------------------------

# Default bound for halfspace computation in parallel_add.
DEFAULT_PARALLEL_ADD_BOUND: float = env_float("DEFAULT_PARALLEL_ADD_BOUND", 1e8)

# Default bound for halfspace computation in the searcher and hamming_astar.
# Too large can upset the solver.
DEFAULT_SEARCH_BOUND: float = env_float("DEFAULT_SEARCH_BOUND", 1e8)

# Multiplier on the layerwise preactivation bound used as the input box radius for SHI
# LPs (when no ``bound`` is given) and as the boundary MIP big-M.
BOUNDARY_MIP_BOUND_MARGIN: float = env_float("BOUNDARY_MIP_BOUND_MARGIN", 5.0)

# Strict margin for boundary MIP witness pricing (``z_j = 0`` at ``boundary_shi``,
# ``|z_j| >= eps`` elsewhere). ``None`` (default): computed per network as
# ``2 * max(1e-4, 1e-6 * max preactivation)``, above Gurobi's feasibility tolerance.
BOUNDARY_MIP_EPS: float | None = env_optional_float("BOUNDARY_MIP_EPS")

# Gurobi time limit (s) per boundary pricing MIP; ``0`` = none.
BOUNDARY_MIP_TIME_LIMIT: float = env_float("BOUNDARY_MIP_TIME_LIMIT", 0.0)

# Show full Gurobi logs during boundary pricing MIPs (separate from :data:`VERBOSE`).
BOUNDARY_MIP_GUROBI_LOG: bool = env_bool("BOUNDARY_MIP_GUROBI_LOG", False)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

# Default hypercube half-width for plotting and Polyhedron.bounded_vertices.
DEFAULT_PLOT_BOUND: float = env_float("DEFAULT_PLOT_BOUND", 10)

# Default bound for Complex.plot (2D).
DEFAULT_COMPLEX_PLOT_BOUND: float = env_float("DEFAULT_COMPLEX_PLOT_BOUND", 10000)

# Plot axis range = max interior coordinate * this factor.
PLOT_MARGIN_FACTOR: float = env_float("PLOT_MARGIN_FACTOR", 1.1)

# Max coordinate for 2D plots with no interior points.
PLOT_DEFAULT_MAXCOORD: float = env_float("PLOT_DEFAULT_MAXCOORD", 10)

# -----------------------------------------------------------------------------
# Logging / verbosity
# -----------------------------------------------------------------------------

# Default for every ``verbose=None`` argument, and the "relucent" logger level:
#   0 → warnings only
#   1 → progress bars and one-line summaries (default)
#   2 → per-stage DEBUG detail
VERBOSE: int = env_int("VERBOSE", 1)

# String settings and their allowed values, checked by update_settings.
_CHOICES: dict[str, tuple[str, ...]] = {"QHULL_MODE": QHULL_MODES, **advanced.CHOICES}

__all__ = [
    "BOUNDARY_MIP_BOUND_MARGIN",
    "BOUNDARY_MIP_EPS",
    "BOUNDARY_MIP_GUROBI_LOG",
    "BOUNDARY_MIP_TIME_LIMIT",
    "CAREFUL_MODE",
    "DEFAULT_COMPLEX_PLOT_BOUND",
    "DEFAULT_PARALLEL_ADD_BOUND",
    "DEFAULT_PLOT_BOUND",
    "DEFAULT_SEARCH_BOUND",
    "MAX_RADIUS",
    "PLOT_DEFAULT_MAXCOORD",
    "PLOT_MARGIN_FACTOR",
    "QHULL_MODE",
    "VERBOSE",
    "advanced",
    "update_settings",
]


def update_settings(**kwargs: Any) -> None:
    """Set one or more settings from :mod:`relucent.config` or :mod:`relucent.config.advanced`.

    Changing :data:`VERBOSE` also updates the ``"relucent"`` logger level.

    Args:
        **kwargs: ``NAME=value`` pairs.

    Raises:
        TypeError: If any key is not a known setting name.
        ValueError: If a string setting gets a value outside its allowed set.
    """
    public = set(__all__) - {"advanced", "update_settings"}
    unknown = set(kwargs) - public - set(advanced.__all__)
    if unknown:
        raise TypeError(f"Unknown config keys: {sorted(unknown)}")
    for k, v in kwargs.items():
        if k in _CHOICES:
            check_choice(k, v, _CHOICES[k])
    mod = globals()
    for k, v in kwargs.items():
        if k in public:
            mod[k] = v
        else:
            setattr(advanced, k, v)
    if "VERBOSE" in kwargs:
        from relucent._internal.logging import _apply_verbose

        _apply_verbose(int(kwargs["VERBOSE"]))
