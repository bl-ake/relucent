Configuration
=============

Relucent exposes numeric defaults and tolerances as **module-level attributes** on
:mod:`relucent.config`. Library code reads these values when routines run, so you
can tune behavior for your model or hardware without editing source files.

Changing settings
-----------------

Import the module (or the package namespace) and assign new values::

   import relucent
   relucent.config.TOL_HALFSPACE_CONTAINMENT = 1e-7
   relucent.config.MAX_RADIUS = 500

To set several attributes at once, use :func:`relucent.config.update_settings`::

   from relucent.config import update_settings
   update_settings(
       TOL_HALFSPACE_CONTAINMENT=1e-7,
       DEFAULT_SEARCH_BOUND=1e7,
   )

``update_settings`` only accepts names listed under :data:`relucent.config.__all__`
(excluding the helper itself). Unknown keys raise ``TypeError``.

Environment variables
---------------------

All public config settings can also be controlled with environment variables.
The naming convention is:

``RELUCENT_<SETTING_NAME>``

For example, to override ``MAX_RADIUS`` and ``TOL_HALFSPACE_CONTAINMENT``:

.. code-block:: bash

   export RELUCENT_MAX_RADIUS=500
   export RELUCENT_TOL_HALFSPACE_CONTAINMENT=1e-7

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
   * - ``MAX_RADIUS``
     - ``float``
     - ``100``
     - Maximum Chebyshev / interior-point search radius when solving with Gurobi.
   * - ``VERTEX_TRUST_THRESHOLD``
     - ``float``
     - ``1e-6``
     - Threshold for trusting vertex positions from HalfspaceIntersection.
   * - ``TOL_HALFSPACE_CONTAINMENT``
     - ``float``
     - ``1e-6``
     - Feasibility tolerance for halfspace containment (:math:`a^\top x + b \le 0`) in checks and related geometry.
   * - ``TOL_DEAD_RELU``
     - ``float``
     - ``1e-8``
     - Column norm below which a ReLU is treated as dead.
   * - ``TOL_SHI_HYPERPLANE``
     - ``float``
     - ``1e-6``
     - Hyperplane equality tolerance in SHI computation.
   * - ``TOL_HALFSPACE_NORMAL``
     - ``float``
     - ``1e-12``
     - Norms below this are treated as degenerate halfspace normals.
   * - ``GUROBI_SHI_BEST_OBJ_STOP`` / ``GUROBI_SHI_BEST_BD_STOP``
     - ``float``
     - ``1e-6`` / ``-1e-6``
     - Gurobi early-stop tolerances for the SHI MIP models.
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
     - ``allclose`` atol when verifying halfspace ``(A, b)`` against network outputs.

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
     - Default halfspace bound for :func:`~relucent.search.parallel_add` and :meth:`~relucent.complex.Complex.parallel_add`.
   * - ``DEFAULT_SEARCH_BOUND``
     - ``float``
     - ``1e8``
     - Default halfspace bound for :func:`~relucent.search.searcher`, :func:`~relucent.search.hamming_astar`, and matching :class:`~relucent.complex.Complex` methods.
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
     - Default bound for :meth:`~relucent.complex.Complex.plot_cells` when ``bound`` is omitted.

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

Search-time cache controls
--------------------------

Search and post-processing APIs now accept explicit cache/property lists, so you
can control exactly what gets computed:

- :meth:`relucent.complex.Complex.searcher` takes ``geometry_properties`` and
  ``keep_caches``.
- :meth:`relucent.complex.Complex.compute_geometric_properties` takes
  ``properties`` and ``keep_caches``.
- :meth:`relucent.complex.Complex.get_poly_attrs` precomputes only the
  attributes requested in ``attrs``.

Common property names include ``"halfspaces"``, ``"W"``, ``"b"``,
``"num_dead_relus"``, ``"finite"``, ``"center"``, ``"inradius"``,
``"interior_point"``, ``"interior_point_norm"``, ``"Wl2"``, ``"vertices"``,
``"hs"``, ``"ch"``, and ``"volume"``.

Examples::

   # Topology-only search (default): SHIs + feasibility only
   cx.searcher(max_polys=1000)

   # Compute only finite/interior-point info during search
   cx.searcher(
       max_polys=1000,
       geometry_properties=["finite", "interior_point", "interior_point_norm"],
   )

   # Keep heavy caches (halfspaces/W/b) when workers return polyhedra
   cx.searcher(
       max_polys=1000,
       geometry_properties=["halfspaces", "W", "b", "finite"],
       keep_caches=True,
   )
