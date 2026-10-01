Configuration
=============

Relucent exposes numeric defaults and tolerances as **module-level attributes** on
:mod:`relucent.config`. Library code reads these values when routines run, so you
can tune behavior for your model or hardware without editing source files.

Automatic tolerance defaults
----------------------------

``import relucent`` computes float64-safe defaults for the tolerance settings
(``TOL_*``, ``BOUNDARY_MIP_EPS``) with
:func:`~relucent.config.numeric_tolerances.apply_tolerances` and writes them to
:mod:`relucent.config`. Per-key ``RELUCENT_<SETTING_NAME>`` environment variables are
not overwritten. Set ``RELUCENT_SKIP_NUMERIC_BOOTSTRAP=1`` before import to keep the
literals in :mod:`relucent.config`.

:class:`~relucent.core.complex.Complex` calls ``apply_tolerances`` again for its
network by default (``auto_tolerances=True``). Pass ``auto_tolerances=False`` to skip
that step.

These values are static: they depend on the ambient dimension (and, for the
``boundary_bfs`` MIP margin ``BOUNDARY_MIP_EPS``, the network's layerwise scale), not on a
per-network scan of the halfspace magnitudes. The topology path does not use them for
its decisions. Facets, emptiness, vertices, genericity, membership, and Morse signs are
each checked against a float64 error bound computed from the rows involved, decided
exactly when that bound can't settle them, and otherwise raise
:class:`~relucent.core.errors.AmbiguousGeometryError`. See
:mod:`relucent._internal.rounding` and :mod:`relucent._internal.exact`.

Changing settings
-----------------

Import the module (or the package namespace) and assign new values::

   import relucent
   relucent.config.TOL_VERIFY_AB_ATOL = 1e-7
   relucent.config.MAX_RADIUS = 500

To set several attributes at once, use :func:`relucent.config.update_settings`::

   from relucent.config import update_settings
   update_settings(
       TOL_VERIFY_AB_ATOL=1e-7,
       DEFAULT_SEARCH_BOUND=1e7,
   )

``update_settings`` only accepts names listed under :data:`relucent.config.__all__`
(excluding the helper itself). Unknown keys raise ``TypeError``.

Environment variables
---------------------

All public config settings can also be controlled with environment variables.
The naming convention is:

``RELUCENT_<SETTING_NAME>``

For example, to override ``MAX_RADIUS`` and ``TOL_VERIFY_AB_ATOL``:

.. code-block:: bash

   export RELUCENT_MAX_RADIUS=500
   export RELUCENT_TOL_VERIFY_AB_ATOL=1e-7

Read values are parsed at import time of :mod:`relucent.config`, so set
environment variables before importing :mod:`relucent`.

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

The following tables summarize each public setting. Types and defaults match the
shipped :mod:`relucent.config` module.

Polyhedron and halfspace geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 28 12 50

   * - Name
     - Type
     - Default
     - Role
   * - ``QHULL_MODE``
     - ``str``
     - ``"IGNORE"``
     - How Qhull warnings from ``HalfspaceIntersection`` are handled: ``"IGNORE"``, ``"WARN_ALL"``, ``"HIGH_PRECISION"``, or ``"JITTERED"`` (retry with QJ).
   * - ``CAREFUL_MODE``
     - ``bool``
     - ``False``
     - When True, run extra consistency checks (Chebyshev vs. propagated boundedness, forward-pass vs. affine reconstruction, conversion spot-checks, graph invariants). The test suite and CI enable this (see ``tests/conftest.py`` and ``RELUCENT_CAREFUL_MODE``).
   * - ``MAX_RADIUS``
     - ``float``
     - ``100``
     - Maximum Chebyshev / interior-point search radius when solving with Gurobi.
   * - ``TOL_HALFSPACE_NORMAL``
     - ``float``
     - ``1e-12``
     - Norms below this are treated as degenerate halfspace normals (used when plotting).
   * - ``GUROBI_SHI_OPTIMALITY_TOL``
     - ``float``
     - ``1e-9``
     - Gurobi ``OptimalityTol`` for the SHI LP (Gurobi's minimum is ``1e-9``). At the Gurobi default of ``1e-6`` the final basis can have a slightly negative exact multiplier, which the exact facet check rejects.
   * - ``GUROBI_SHI_SCALE_FLAG``
     - ``int``
     - ``2``
     - Gurobi ``ScaleFlag`` for the SHI LP (``-1`` auto, ``0`` off, ``1``-``3`` scaling methods). Rows can span orders of magnitude and be nearly parallel; geometric-mean scaling (``2``) avoids spurious infeasibility from automatic scaling.
   * - ``TOL_NEARLY_VERTICAL``
     - ``float``
     - ``1e-10``
     - 2D plotting: halfspace normals with small ``w[1]`` are treated as vertical.
   * - ``DEFAULT_PLOT_BOUND``
     - ``float``
     - ``10``
     - Default half-width of the bounding box for polyhedron plotting and bounded vertices when not passed explicitly.
   * - ``TOL_VERIFY_AB_ATOL``
     - ``float``
     - ``1e-6``
     - ``allclose`` atol when verifying halfspace ``(A, b)`` against network outputs (used in :mod:`relucent.vis` checks).

Complex search and parallel add
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 12 50

   * - Name
     - Type
     - Default
     - Role
   * - ``INTERIOR_POINT_RADIUS_SEQUENCE``
     - ``list[float]``
     - ``[0.01, 0.1, 1, 10, 100]``
     - Radii tried in order when locating an interior point for a neighbor in :func:`~relucent.search.get_ip`.
   * - ``DEFAULT_PARALLEL_ADD_BOUND``
     - ``float``
     - ``1e8``
     - Default halfspace bound for :func:`~relucent.search.parallel_add` and :meth:`~relucent.core.complex.Complex.parallel_add`.
   * - ``DEFAULT_SEARCH_BOUND``
     - ``float``
     - ``1e8``
     - Fallback halfspace bound for :func:`~relucent.search.hamming_astar` when ``bound`` is omitted. Ambient BFS / :func:`~relucent.search.searcher` instead use a network-scaled bound from :func:`~relucent._internal.network_scale.default_polyhedron_bound` (``estimate_input_bound`` × ``BOUNDARY_MIP_BOUND_MARGIN``) when ``bound`` is ``None``.
   * - ``ASTAR_BIAS_WEIGHT``
     - ``float``
     - ``0.9``
     - Weight on Euclidean-distance bias in the A* heuristic.
   * - ``PLOT_MARGIN_FACTOR``
     - ``float``
     - ``1.1``
     - Axis margin multiplier when deriving plot extent from interior points.
   * - ``PLOT_DEFAULT_MAXCOORD``
     - ``float``
     - ``10``
     - Fallback axis half-extent when no interior points exist (2D complex plots).
   * - ``DEFAULT_COMPLEX_PLOT_BOUND``
     - ``float``
     - ``10000``
     - Default bound for :meth:`~relucent.core.complex.Complex.plot` of a 2D complex when ``bound`` is omitted.
   * - ``BOUNDARY_MIP_BOUND_MARGIN``
     - ``float``
     - ``5.0``
     - Multiplier applied to layerwise bound propagation when estimating network scale for SHI LPs, boundary MIPs, and automatic tolerance selection.

Boundary discovery (MIP pricing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 28 12 50

   * - Name
     - Type
     - Default
     - Role
   * - ``BOUNDARY_MIP_EPS``
     - ``float``
     - ``1e-4``
     - Strict-margin tolerance for boundary MIP witness pricing (``z_j = 0`` at the boundary SHI; ``|z_j| >= eps`` elsewhere).
   * - ``BOUNDARY_PRICING_BRUTE_FORCE_MAX_N``
     - ``int``
     - ``18``
     - When total ReLU width is at most this value, try brute-force sign-pattern scan before/after MIP pricing.
   * - ``BOUNDARY_MIP_TIME_LIMIT``
     - ``float``
     - ``0.0``
     - Optional Gurobi time limit (seconds) per pricing MIP; ``0`` means no limit.
   * - ``BOUNDARY_MIP_GUROBI_LOG``
     - ``bool``
     - ``False``
     - Emit full Gurobi solver logs during boundary pricing (independent of ``VERBOSE``).
   * - ``BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS``
     - ``int``
     - ``1000``
     - Compile ``exclude_tags`` into a compressed trie when at least this many tags are present (``0`` = always).
   * - ``BOUNDARY_MIP_STATIC_EXCLUSION_MIN_TAGS``
     - ``int``
     - ``1000``
     - Bulk-add per-tag no-goods statically before optimize when at least this many tags are excluded.
   * - ``BOUNDARY_MIP_STATIC_EXCLUSION_MIN_RATIO``
     - ``float``
     - ``2.0``
     - Static-add all leaf nogoods when trie compression ratio falls below this threshold.
   * - ``BOUNDARY_MIP_EXCLUSION_BATCH_SIZE``
     - ``int``
     - ``4096``
     - Chunk size for batched Gurobi ``addConstrs`` when emitting exclusion nogoods.
   * - ``BOUNDARY_MIP_EXCLUSION_WORKERS``
     - ``int``
     - ``0``
     - Workers for parallel nogood compilation; ``0`` = auto, ``1`` = serial.
   * - ``BOUNDARY_MIP_CUT_ORDER``
     - ``str``
     - ``"tag_lex"``
     - Cut ordering for static/lazy nogoods (``as_is``, ``tag_lex``, ``layer_major``, …).
   * - ``BOUNDARY_MIP_BULK_NOGOOD_EMIT``
     - ``str``
     - ``"auto"``
     - Bulk matrix emit for static nogoods: ``auto``, ``on``, or ``off``.
   * - ``BOUNDARY_MIP_STATIC_WAVE_SIZE``
     - ``int``
     - ``0``
     - Static nogood wave size; ``0`` = add all constraints before a single optimize.
   * - ``BOUNDARY_MIP_CUT_PRIORITY_ENABLED``
     - ``bool``
     - ``False``
     - Assign higher Gurobi priority to trie-compressed / deeper-path cuts.
   * - ``BOUNDARY_MIP_LAZY_ONLY_MIN_TAGS``
     - ``int``
     - ``50000``
     - Skip trie/static precompilation and rely on lazy MIPSOL cuts when this many tags are excluded.

Topology and logging
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 28 12 50

   * - Name
     - Type
     - Default
     - Role
   * - ``TOPOLOGY_INTRINSIC_VERTEX_MATCH_TOL_FACTOR``
     - ``float``
     - ``2.0``
     - When merging geometric vertices with combinatorial intrinsic vertices, accept a match if :math:`\|x - x_\mathrm{intrinsic}\|_\infty \le` this factor times the containment tolerance.
   * - ``VERBOSE``
     - ``int``
     - ``1``
     - Package logging level during search and parallel work: ``0`` → WARNING only; ``1`` → INFO (worker counts, meta-graph progress, …). Also adjustable at runtime via :func:`~relucent.config.update_settings`.

Utilities and visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 28 12 50

   * - Name
     - Type
     - Default
     - Role
   * - ``BLOCKING_QUEUE_WAIT_TIMEOUT``
     - ``float``
     - ``0.5``
     - Seconds to wait on ``Condition.wait`` when polling a :class:`~relucent.utils.BlockingQueue`.

API
---

.. autofunction:: relucent.config.update_settings

Valid keys for :func:`~relucent.config.update_settings` are the names listed in
``relucent.config.__all__``, except ``update_settings`` itself.
