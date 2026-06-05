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

1. **Default search** (standard geometry set computed per discovered region).
2. **Topology-only search** (fastest option, SHIs and feasibility only; pass ``geometry_properties=[]``).
3. **Custom in-search geometry** (a chosen subset of caches).
4. **Two-phase pipelines** (topology first, then a targeted geometry pass).

Default behavior
----------------

When ``geometry_properties`` is ``None``, search uses
:data:`relucent.search.DEFAULT_GEOMETRY_PROPERTIES`:

``halfspaces``, ``W``, ``b``, ``num_dead_relus``, ``finite``, ``center``,
``inradius``, ``interior_point``, ``interior_point_norm``, and ``Wl2``.

Workers always compute SHIs (supporting hyperplane indices) and feasibility;
those are required for adjacency discovery regardless of ``geometry_properties``.

Important tradeoff
------------------

A common assumption is that topology-only search is always fastest overall. That
is not always true:

- If you will later need geometry for *most* regions, doing a second pass can be
  slower end-to-end because each region may incur additional recomputation (e.g. getting its h-representation).
- If you only need topology (or geometry for a small subset), pass
  ``geometry_properties=[]`` to save substantial time and memory.

In practice:

- Use **topology-only** (``geometry_properties=[]``) for large frontier growth
  when you only need adjacency or meta-graph structure.
- Use the **default** or a **custom geometry set** when downstream steps
  immediately need those values (for example, filtrations that read interior
  points or affine maps).
- Keep ``keep_caches=False`` unless repeated heavy cache reuse clearly outweighs
  memory/serialization cost.

Common workflows
----------------

The examples below assume a :class:`~relucent.complex.Complex` named ``cplx``:

.. code-block:: python

   import relucent
   cplx = relucent.Complex(relucent.mlp(widths=[2, 8, 1]))

Topology-only search
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # SHIs + feasibility only (skip DEFAULT_GEOMETRY_PROPERTIES)
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

   cplx.bfs(max_polys=1000, geometry_properties=[])  # fast discovery
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
