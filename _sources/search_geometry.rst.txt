Search Geometry and Memory Management
==================================

Relucent separates **topology discovery** from **geometric cache computation** so
you can control runtime and memory for your workload.

Why these options exist
-----------------------

During search, relucent must discover adjacency (via SHIs) and feasibility, which involves computing the h-representation / affine map of all the activation regions. 
In large searches, this data can accumulate in memory and slow the search, so by default it is dropped after each region is processed. However,
some geometric properties, such as ``volume``, rely on this data, so you can choose to compute and them for each region as the search progresses.

The ``geometry_properties`` option lets you choose between:

1. **Default search** (topology-only: SHIs, ``finite``, ``center``, and ``inradius`` only).
2. **Full geometry search** (pass :data:`~relucent.search.ALL_GEOMETRY_PROPERTIES`).
3. **Custom in-search geometry** (a chosen subset of optional geometric properties).
4. **Two-phase pipelines** (topology first, then a targeted geometry pass).

Default behavior
----------------

When ``geometry_properties`` is ``None`` (the default for
:meth:`~relucent.core.complex.Complex.searcher`, :meth:`~relucent.core.complex.Complex.bfs`,
and related search methods), search is **topology-only**: workers compute the Minimum
required geometry for adjacency and feasibility, namely ``finite``, ``center``, ``inradius``, and SHIs.
If you then run a command like ``cplx.compute_geometric_properties(properties=["volume"])``, relucent will 
have to recompute the h-representation / affine map for each region, which can be slow.

Pass :data:`~relucent.search.ALL_GEOMETRY_PROPERTIES` to compute every property supported by
:meth:`~relucent.core.poly.Polyhedron.get_geometry` (including Qhull-derived
``vertices``, ``volume``).

Any property listed in ``geometry_properties`` is retained on each polyhedron after the search is complete.
Properties that are not listed may have their heavy data dropped to
save memory (for example, requesting only ``interior_point_norm`` does not retain
``interior_point`` or the ``halfspaces`` used to compute it).

- If you will later need geometry for *most* regions, doing a second pass can be
  slower end-to-end because each region may incur additional recomputation (e.g. getting its h-representation).
- If you only need topology (or geometry for a small subset), the default search
  saves substantial time and memory.

In practice:

- Use the **default** (topology-only) for large frontier growth when you only need
  adjacency or meta-graph structure.
- Pass :data:`~relucent.search.ALL_GEOMETRY_PROPERTIES` or a **custom geometry set** when downstream steps immediately
  need those values (for example, filtrations that read interior points or affine
  maps).

Certification and caps
----------------------

:meth:`~relucent.core.complex.Complex.bfs` accepts ``verify=True`` by default (see
:doc:`exploration_verification`). Examples below use ``max_polys`` as a safety cap;
if search finishes under the cap, certification still runs. If the cap is hit first,
``complete`` and ``verified`` are ``False``. Pass ``verify=False`` for intentional
partial exploration.

Common workflows
----------------

The examples below assume a :class:`~relucent.core.complex.Complex` named ``cplx``:

.. code-block:: python

   import relucent
   cplx = relucent.Complex(relucent.mlp(widths=[2, 8, 1]))

Topology-focused search (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   cplx.bfs(max_polys=1000)

Full geometry during search
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from relucent.search import ALL_GEOMETRY_PROPERTIES

   cplx.bfs(max_polys=1000, geometry_properties=ALL_GEOMETRY_PROPERTIES)

Compute selected geometry during search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   cplx.bfs(
       max_polys=1000,
       geometry_properties=["halfspaces", "volume", "Wl2"],
   )

Two-phase approach: topology first, geometry later
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   cplx.bfs(max_polys=1000)  # fast discovery (default)
   cplx.compute_geometric_properties(properties=["Wl2"])

Useful property names
---------------------

Examples of valid names for ``geometry_properties`` / ``properties``:

- :data:`~relucent.search.ALL_GEOMETRY_PROPERTIES` — every supported property
- ``"halfspaces"``, ``"halfspaces_np"``, ``"W"``, ``"b"``, ``"num_dead_relus"``
- ``"interior_point"``, ``"interior_point_norm"``
- ``"Wl2"``, ``"vertices"``, ``"hs"`` (SciPy's HalfspaceIntersection), ``"ch"`` (SciPy's ConvexHull), ``"volume"``
- Always computed during search: ``"finite"``, ``"center"``, ``"inradius"``
