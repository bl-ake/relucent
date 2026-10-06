Configuration
=============

.. py:module:: relucent.config

Relucent exposes numeric defaults as **module-level attributes** on
:mod:`relucent.config`. Library code reads these values when routines run, so you
can tune behavior for your model or hardware without editing source files.

Nothing in relucent writes to these settings: importing the package and creating a
:class:`~relucent.core.complex.Complex` leave them as you set them. Settings that depend
on the network (the boundary MIP margin ``BOUNDARY_MIP_EPS``, the default SHI LP box) are
``None`` or omitted by default and computed per network where they are used.

The topology path does not rely on tolerance settings for its decisions. Facets,
emptiness, vertices, genericity, membership, and Morse signs are each checked against a
float64 error bound computed from the rows involved, decided exactly when that bound
can't settle them, and otherwise raise :class:`~relucent.core.errors.AmbiguousGeometryError`.
None of this has a setting to tune.

Changing settings
-----------------

Import the module (or the package namespace) and assign new values::

   import relucent
   relucent.config.VERBOSE = 0
   relucent.config.MAX_RADIUS = 500

To set several attributes at once, use :func:`relucent.config.update_settings`::

   from relucent.config import update_settings
   update_settings(
       VERBOSE=0,
       DEFAULT_SEARCH_BOUND=1e7,
   )

``update_settings`` accepts the names in ``relucent.config.__all__`` and
``relucent.config.advanced.__all__``. Unknown keys raise ``TypeError``, and a string
setting given a value outside its allowed set raises ``ValueError``.

Environment variables
---------------------

Every setting, including the advanced ones, can also be set with an environment variable
named ``RELUCENT_<SETTING_NAME>``:

.. code-block:: bash

   export RELUCENT_MAX_RADIUS=500
   export RELUCENT_VERBOSE=0

Values are read when :mod:`relucent.config` is imported, so set environment variables
before importing ``relucent``.

``INTERIOR_POINT_RADIUS_SEQUENCE`` accepts either comma-separated values or a
bracketed comma-separated form:

.. code-block:: bash

   export RELUCENT_INTERIOR_POINT_RADIUS_SEQUENCE=0.01,0.1,1,10,100
   # or:
   export RELUCENT_INTERIOR_POINT_RADIUS_SEQUENCE='[0.01, 0.1, 1, 10, 100]'

**Defaults in function signatures:** Parameters documented as “defaults to
``relucent.config.X``” resolve that value **when the function runs**, not when
Python first imports the package. If you pass an argument explicitly, it always
overrides the module setting for that call.

Settings reference
------------------

Geometry
~~~~~~~~

.. py:data:: CAREFUL_MODE
   :type: bool
   :value: False

   When True, run extra consistency checks (Chebyshev vs. propagated boundedness, forward-pass vs. affine reconstruction, conversion spot-checks, graph invariants). The test suite and CI enable this (see ``tests/conftest.py`` and ``RELUCENT_CAREFUL_MODE``).

.. py:data:: QHULL_MODE
   :type: str
   :value: "IGNORE"

   How Qhull warnings from ``HalfspaceIntersection`` are handled: ``"IGNORE"``, ``"WARN_ALL"``, ``"HIGH_PRECISION"``, or ``"JITTERED"`` (retry with QJ).

.. py:data:: MAX_RADIUS
   :type: float
   :value: 100

   Maximum Chebyshev / interior-point search radius when solving with Gurobi.

Search and boundary discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:data:: DEFAULT_PARALLEL_ADD_BOUND
   :type: float
   :value: 1e8

   Default halfspace bound for :meth:`~relucent.core.complex.Complex.parallel_add`.

.. py:data:: DEFAULT_SEARCH_BOUND
   :type: float
   :value: 1e8

   Fallback halfspace bound for :meth:`~relucent.core.complex.Complex.hamming_astar` when ``bound`` is omitted. BFS / :meth:`~relucent.core.complex.Complex.searcher` instead use a network-scaled bound (layerwise ``|W|_1`` propagation × ``BOUNDARY_MIP_BOUND_MARGIN``) when ``bound`` is ``None``.

.. py:data:: BOUNDARY_MIP_BOUND_MARGIN
   :type: float
   :value: 5.0

   Multiplier on the layerwise preactivation bound, used as the default SHI LP box radius and as the boundary MIP big-M.

.. py:data:: BOUNDARY_MIP_EPS
   :type: float | None
   :value: None

   Strict margin for boundary MIP witness pricing (``z_j = 0`` at the boundary SHI; ``|z_j| >= eps`` elsewhere). ``None`` computes ``2 · max(1e-4, 1e-6 · max preactivation)`` for each network, which stays above Gurobi's feasibility tolerance at that network's scale.

.. py:data:: BOUNDARY_MIP_TIME_LIMIT
   :type: float
   :value: 0.0

   Optional Gurobi time limit (seconds) per pricing MIP; ``0`` means no limit.

.. py:data:: BOUNDARY_MIP_GUROBI_LOG
   :type: bool
   :value: False

   Emit full Gurobi solver logs during boundary pricing (independent of ``VERBOSE``).

Plotting
~~~~~~~~

.. py:data:: DEFAULT_PLOT_BOUND
   :type: float
   :value: 10

   Default half-width of the bounding box for polyhedron plotting and bounded vertices when not passed explicitly.

.. py:data:: DEFAULT_COMPLEX_PLOT_BOUND
   :type: float
   :value: 10000

   Default bound for :meth:`~relucent.core.complex.Complex.plot` of a 2D complex when ``bound`` is omitted.

.. py:data:: PLOT_MARGIN_FACTOR
   :type: float
   :value: 1.1

   Axis margin multiplier when deriving plot extent from interior points.

.. py:data:: PLOT_DEFAULT_MAXCOORD
   :type: float
   :value: 10

   Fallback axis half-extent when no interior points exist (2D complex plots).

Logging
~~~~~~~

.. py:data:: VERBOSE
   :type: int
   :value: 1

   Default for every ``verbose=None`` argument: ``0`` → warnings only; ``1`` → progress bars and one-line summaries; ``2`` → per-stage debug detail. Each call sets the ``"relucent"`` logger to the matching level (``WARNING``, ``INFO``, ``DEBUG``) while it runs; a relucent call made inside another inherits the outer call's level. Progress bars appear only after a one-second delay, so quick calls print nothing.

Advanced settings (unstable)
----------------------------

.. py:module:: relucent.config.advanced

:mod:`relucent.config.advanced` holds solver-tuning knobs. They change performance or
solver details rather than what relucent computes, and **may change or disappear in any
release**. Set them the same ways as the settings above, e.g.
``relucent.config.advanced.BOUNDARY_MIP_EXCLUSION_WORKERS = 4``.

.. py:data:: GUROBI_SHI_OPTIMALITY_TOL
   :type: float
   :value: 1e-9

   Gurobi ``OptimalityTol`` for the SHI LP (Gurobi's minimum is ``1e-9``). At the Gurobi default of ``1e-6`` the final basis can have a slightly negative exact multiplier, which the exact facet check rejects.

.. py:data:: GUROBI_SHI_SCALE_FLAG
   :type: int
   :value: 2

   Gurobi ``ScaleFlag`` for the SHI LP (``-1`` auto, ``0`` off, ``1``-``3`` scaling methods). Rows can span orders of magnitude and be nearly parallel; geometric-mean scaling (``2``) avoids spurious infeasibility from automatic scaling. If an SHI LP still fails, it is re-solved from scratch with no scaling (``0``), and if that fails too, decided in exact arithmetic; see :doc:`search_shi_and_graphs`.

.. py:data:: INTERIOR_POINT_RADIUS_SEQUENCE
   :type: list[float]
   :value: [0.01, 0.1, 1, 10, 100]

   Radii tried in order when locating an interior point for a neighbor during search.

.. py:data:: ASTAR_BIAS_WEIGHT
   :type: float
   :value: 0.9

   Weight on Euclidean-distance bias in the A* heuristic.

.. py:data:: BLOCKING_QUEUE_WAIT_TIMEOUT
   :type: float
   :value: 0.5

   Seconds to wait on ``Condition.wait`` when polling the work queue shared by search workers.

.. py:data:: BOUNDARY_PRICING_BRUTE_FORCE_MAX_N
   :type: int
   :value: 18

   When total ReLU width is at most this value, try brute-force sign-pattern scan before/after MIP pricing.

.. py:data:: BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS
   :type: int
   :value: 1000

   Compile ``exclude_tags`` into a compressed trie when at least this many tags are present (``0`` = always).

.. py:data:: BOUNDARY_MIP_STATIC_EXCLUSION_MIN_TAGS
   :type: int
   :value: 1000

   Bulk-add per-tag no-goods statically before optimize when at least this many tags are excluded.

.. py:data:: BOUNDARY_MIP_STATIC_EXCLUSION_MIN_RATIO
   :type: float
   :value: 2.0

   Static-add all leaf nogoods when trie compression ratio falls below this threshold.

.. py:data:: BOUNDARY_MIP_EXCLUSION_BATCH_SIZE
   :type: int
   :value: 4096

   Chunk size for batched Gurobi ``addConstrs`` when emitting exclusion nogoods.

.. py:data:: BOUNDARY_MIP_EXCLUSION_WORKERS
   :type: int
   :value: 0

   Workers for parallel nogood compilation; ``0`` = auto, ``1`` = serial.

.. py:data:: BOUNDARY_MIP_CUT_ORDER
   :type: str
   :value: "tag_lex"

   Cut ordering for static/lazy nogoods: ``as_is``, ``tag_lex``, ``layer_major``, ``hamming_median``, ``literal_count_asc``, ``trie_depth_desc``, or ``random``.

.. py:data:: BOUNDARY_MIP_BULK_NOGOOD_EMIT
   :type: str
   :value: "auto"

   Bulk matrix emit for static nogoods: ``auto``, ``on``, or ``off``.

.. py:data:: BOUNDARY_MIP_STATIC_WAVE_SIZE
   :type: int
   :value: 0

   Static nogood wave size; ``0`` = add all constraints before a single optimize.

.. py:data:: BOUNDARY_MIP_CUT_PRIORITY_ENABLED
   :type: bool
   :value: False

   Assign higher Gurobi priority to trie-compressed / deeper-path cuts.

.. py:data:: BOUNDARY_MIP_LAZY_ONLY_MIN_TAGS
   :type: int
   :value: 50000

   Skip trie/static precompilation and rely on lazy MIPSOL cuts when this many tags are excluded.

API
---

.. py:currentmodule:: relucent.config

.. autofunction:: relucent.config.update_settings
