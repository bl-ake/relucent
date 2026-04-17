Search Geometry and Cache Strategy
==================================

Relucent separates **topology discovery** from **geometric cache computation** so
you can control runtime and memory for your workload.

Why these options exist
-----------------------

During search, relucent must discover adjacency (via SHIs) and feasibility.
Many geometric quantities (for example ``interior_point``, ``Wl2``, and
``volume``) are optional and can be expensive. In large searches, computing every
property for every region can dominate runtime and memory.

The ``geometry_properties`` and ``keep_caches`` options let you choose between:

1. **Faster/lighter topology-first search** (minimal work per discovered region).
2. **Richer per-region data during search** (more work per region, but ready-to-use).
3. **Two-phase pipelines** (topology first, then targeted geometry pass).

Important tradeoff
------------------

A common assumption is that topology-first search is always fastest overall. That
is not always true:

- If you will later need geometry for *most* regions, doing a second pass can be
  slower end-to-end because each region may incur additional recomputation (e.g. getting its h-representation).
- If you only need topology (or geometry for a small subset), delaying geometry
  usually saves substantial time and memory.

In practice:

- Use **topology-first** for exploration and large frontier growth.
- Use **in-search geometry** when downstream steps immediately need those values.
- Keep ``keep_caches=False`` unless repeated heavy cache reuse clearly outweighs
  memory/serialization cost.

Common workflows
----------------

Topology-only search
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # SHIs + feasibility only (disable default geometry computation)
   cplx.bfs(max_polys=1000, geometry_properties=[])

Compute selected geometry during search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   cplx.bfs(
       max_polys=1000,
       geometry_properties=["finite", "interior_point", "interior_point_norm"],
   )

Keep heavy caches after worker transfer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   cplx.bfs(
       max_polys=1000,
       geometry_properties=["halfspaces", "W", "b", "finite"],
       keep_caches=True,
   )

Two-phase approach: topology first, geometry later
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   cplx.bfs(max_polys=1000)  # fast discovery
   cplx.compute_geometric_properties(
       properties=["finite", "center", "inradius", "Wl2"],
       keep_caches=False,
   )

Useful property names
---------------------

Examples of valid names for ``geometry_properties`` / ``properties``:

- ``"halfspaces"``, ``"halfspaces_np"``, ``"W"``, ``"b"``, ``"num_dead_relus"``
- ``"finite"``, ``"center"``, ``"inradius"``
- ``"interior_point"``, ``"interior_point_norm"``
- ``"Wl2"``, ``"vertices"``, ``"hs"``, ``"ch"``, ``"volume"``
