Search, SHI Assignment, Verification, and Graph Structures
=============================================================

This page explains how Relucent discovers activation regions, how **Supporting
Hyperplane Indices (SHIs)** are assigned to polyhedra, how those assignments are
verified, and how the **dual graph** and **meta-graph** relate to search.

For memory/geometry trade-offs during search, see :doc:`search_geometry`. For
exploration flags and default ``verify`` behavior, see
:doc:`exploration_verification`.

Background: sign sequences and SHIs
-----------------------------------

Each ReLU activation region is a convex polyhedron in input space. Relucent
identifies regions by their **sign sequence** (``ss``): an array of values in
``{-1, 0, +1}`` recording the sign of each ReLU pre-activation.

* A **nonzero** entry means the neuron is active on one side of its hyperplane.
* A **zero** entry means the input lies on that hyperplane (a face of the region).

A **Supporting Hyperplane Index (SHI)** is an index into the stacked halfspace
rows ``Ax + b ≤ 0`` (one row per ReLU unit) that is a **non-redundant facet** of
the polyhedron — a hyperplane the region actually touches.

Neighbors across a codimension-one face differ by flipping one nonzero ``ss``
entry to ``0`` (crossing that hyperplane). The SHI labels which hyperplane was
crossed.

+----------------------------+--------------------------------------------------+
| Concept                    | Role                                             |
+============================+==================================================+
| ``ss``                     | Stable cell identity (injective for nonempty     |
|                            | regions)                                         |
+----------------------------+--------------------------------------------------+
| ``Polyhedron._shis``       | Which halfspace rows are true boundary facets    |
+----------------------------+--------------------------------------------------+
| Dual-graph edge ``shi``    | Hyperplane shared by two adjacent top cells      |
+----------------------------+--------------------------------------------------+

SHIs are computed by Gurobi LPs in :func:`~relucent.calculations.get_shis` and
cached on ``Polyhedron._shis``. The public :attr:`~relucent.poly.Polyhedron.shis`
property lazily calls ``get_shis()`` when the cache is empty.

How search works
----------------

Entry points
~~~~~~~~~~~~

+----------------------------+-------------+-----------------------------------+
| Method                     | Queue       | Use                               |
+============================+=============+===================================+
| :meth:`~relucent.complex.Complex.bfs` | FIFO        | Default local exploration          |
+----------------------------+-------------+-----------------------------------+
| :meth:`~relucent.complex.Complex.dfs` | LIFO        | Depth-first variant                |
+----------------------------+-------------+-----------------------------------+
| :meth:`~relucent.complex.Complex.random_walk` | Random pop  | Stochastic exploration             |
+----------------------------+-------------+-----------------------------------+
| :meth:`~relucent.complex.Complex.searcher` | Configurable | Generic traversal                 |
+----------------------------+-------------+-----------------------------------+

All delegate to :func:`~relucent.search.searcher`, which runs a parallel frontier
expansion over flip-neighbors.

Boundary complexes use a separate path:
:func:`~relucent.boundary_search.discover_boundary_complex` (MIP pricing +
slice-restricted BFS). See `Boundary discovery`_ below.

Search loop
~~~~~~~~~~~

::

   add_point(start) → get_shis(start) → seed frontier
        ↓
   worker pool: search_calculations per (ss, crossed_shi, depth, parent)
        ↓
   add_polyhedron on success → enqueue flip-neighbors via p.shis
        ↓
   repeat until frontier empty or cap hit
        ↓
   finalize_ambient_search (dual graph + optional verify)

1. **Initialize** — Add the start polyhedron. Compute its SHIs and enqueue one
   task per SHI: ``(neighbor_ss, crossed_shi, depth, parent_index)``.
2. **Workers** — Each task builds a :class:`~relucent.poly.Polyhedron`, runs
   Chebyshev geometry (``finite``, ``center``, ``inradius``), then
   ``get_shis()``. See :func:`~relucent.search.search_calculations` and
   ``_worker_prepare_poly`` in :mod:`relucent.search`.
3. **Main process** — On success, add the polyhedron and enqueue new neighbors
   for every SHI except the one just crossed. Failed flips are recorded so the
   same ``(poly, shi)`` pair is not retried.
4. **Termination** — Search is **complete** when the frontier is empty, no
   ``max_polys`` cap was hit, and no ``max_depth`` neighbor was left unqueued.
5. **Finalize** — :func:`~relucent.exploration.finalize_ambient_search`
   rebuilds the dual graph, syncs top-cell ``_shis`` from edges, and optionally
   runs verification.

Workers always compute ``finite``, ``center``, and ``inradius`` even in
topology-only mode (``geometry_properties=None``). SHI reliability checks and
thin-cell guards depend on Chebyshev data. Optional geometry (halfspaces,
vertices, volume, …) is controlled by ``geometry_properties``; see
:doc:`search_geometry`.

Multiprocessing
~~~~~~~~~~~~~~~

Search uses a process pool with initializer
:func:`~relucent.worker_context.set_worker_context`. Workers read
``(net, env, dim)`` from module-level state — do not call ``set_worker_context``
from the main process.

When ``verify=True`` (the default for ``bfs``), frontier search keeps SHI LPs
non-strict (heuristic neighbor discovery). After a complete search,
:func:`~relucent.exploration.finalize_ambient_search` syncs top-cell SHIs from the
dual graph and runs :func:`~relucent.certify.certify_complex`, which applies strict
facet checks.

SHI assignment: three roles
---------------------------

Relucent uses SHI information in three related but distinct ways. The helpers in
:mod:`relucent.incidence` document these as **roles 1–3**. Using the wrong rule
in the wrong place breaks adjacency or homology (``∂² = 0``).

Role 1 — Contraction and contracted slices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When building lower-dimensional cells via :meth:`~relucent.complex.Complex.contract`,
each codimension-one face seeds SHI candidates from its sign sequence::

   SHI_candidates(face) = { i : ss_i ≠ 0 on the face sign sequence }

The crossing hyperplane is already zeroed, so it is not included.
:meth:`~relucent.complex.Complex._codim_one_face_kwargs` applies this at face
creation (via :func:`~relucent.incidence.ss_nonzero_indices`). Infeasible
1-cells are dropped with
:meth:`~relucent.poly.Polyhedron.is_shi_face_feasible`. After the full slice is
known, :func:`~relucent.incidence.set_contracted_shis` sets authoritative
``_shis`` to :func:`~relucent.incidence.cubical_cell_shis` — the same
flip-neighbor rule as role 3, restricted to cells in the slice.

**Critical:** meta-graph **face edges** still use role 2 (all SS crossings), not
propagated ``_shis``. Using only ``_shis`` for face discovery omits valid faces
and breaks ``∂² = 0``.

Role 2 — Meta-graph face edges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~relucent.incidence.collect_meta_face_edges` builds directed
``k-cell → (k−1)-face`` incidences by zeroing **every** index where
``ss_i ≠ 0`` (via :func:`~relucent.incidence.ss_nonzero_indices`), then keeping
an edge only if the resulting face tag exists in the complex.

This rule is **complete** for cubical incidence: propagated ``_shis`` can be a
strict subset of SS crossings, so using only ``_shis`` here would omit valid
faces.

Role 3 — Meta-graph node metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~relucent.incidence.meta_node_attrs` derives ``shis`` (flip-neighbor
crossings) and ``crossings`` (``ss_nonzero_indices``) from each cell's sign
sequence and same-dimension slice.
1-cell boundedness uses 0-face incidence in meta face edges
(:func:`~relucent.incidence.classify_one_cells_finite_from_face_edges`), not
``len(shis)``.

SHI assignment over the pipeline
--------------------------------

During search (frontier heuristic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each discovered cell, workers call ``get_shis()`` and store the result in
``poly._shis``. This list drives which neighbors to enqueue.

The LP algorithm (:func:`~relucent.calculations.get_shis`):

1. Build halfspaces from the sign sequence; drop degenerate rows.
2. Work in intrinsic coordinates (null-space of zero-sign equalities).
3. For each candidate index ``i``: relax halfspace ``i``, maximize along its normal.
4. If the objective exceeds ``TOL_SHI_OBJECTIVE``, ``i`` is a facet SHI.
5. With ``strict=True`` (opt-in via ``get_shis`` kwargs), invalid proofs raise
   :class:`~relucent.errors.ShiProofError`; during default ambient search they
   emit warnings instead.

This is a **heuristic for exploration**: it must be good enough to find neighbors,
but it is not the final authority on top-cell SHIs after a complete search.

After complete ambient search (authoritative top cells)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~relucent.exploration.finalize_ambient_search` calls
:meth:`~relucent.complex.Complex.get_dual_graph`, which:

1. Builds combinatorial edges with
   :func:`~relucent.incidence.dual_edges_top_dim` (flip neighbors for
   ``max_dim ≥ 2``; shared 0-face tags for 1D ambient complexes).
2. Syncs ``poly._shis`` from incident edge labels via
   :func:`~relucent.incidence.sync_shis_from_dual_graph` when ``repair=True``
   (the default).

Top-cell ``_shis`` are therefore **re-derived from dual-graph edges**, not left
as raw LP output. This is the combinatorial cubical model used by contraction
and topology.

On contracted slices (boundary, chain complex)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After :meth:`~relucent.complex.Complex.contract` creates faces, SHIs are seeded
from SS crossings (role 1) and finalized by
:func:`~relucent.incidence.set_contracted_shis` to
:func:`~relucent.incidence.cubical_cell_shis`. Contracted 1-skeleton dual
graphs walk each cell's finalized ``poly.shis`` (flip neighbors in the slice).

Boundary discovery seeds slice ``_shis`` from SS crossings
(``_apply_ambient_boundary_shis`` in :mod:`relucent.boundary_search`), then
runs the same ``set_contracted_shis`` pass during finalize.

Certification
-------------

Certification is orchestrated by :func:`~relucent.certify.certify_complex`.
:meth:`~relucent.complex.Complex.bfs` with ``verify=True`` runs it automatically
after a **complete** search via ``finalize_ambient_search`` at
:class:`~relucent.certify.CertifyLevel.COMPLETE`. Call
:meth:`~relucent.complex.Complex.certify` to re-run certification manually.

:class:`~relucent.certify.CertifyLevel` is cumulative:

* ``COMBINATORIAL`` — flip-SHI symmetry
  (:func:`~relucent.incidence.verify_flip_shi_symmetry`), dual-graph certification
  (:func:`~relucent.incidence.certify_dual_graph`), and contracted-slice SHI checks
  (:func:`~relucent.incidence.verify_contracted_shis`) when applicable.
* ``COMPLETE`` — additionally requires every LP facet on a top cell to flip to a
  same-dimension neighbor in the complex
  (:func:`~relucent.certify.verify_lp_flip_neighbors_in_complex`).
* ``GEOMETRIC`` — additionally recompute SHIs with ``strict=True`` on every cached
  cell and require an exact match
  (:func:`~relucent.certify.verify_shi_geometry`).

When certification runs
~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------------+------------------------------------------------+
| Situation                     | Behavior                                       |
+===============================+================================================+
| Complete search,              | Dual-graph sync +                              |
| ``verify=True``               | ``certify_complex(level=COMPLETE)``            |
+-------------------------------+------------------------------------------------+
| ``max_polys`` cap hit         | Certification skipped                          |
|                               | (``complete=False``, ``verified=False``)       |
+-------------------------------+------------------------------------------------+
| ``max_depth`` cap with        | ``complete=False``; with ``verify=True``       |
| unqueued neighbors            | raises :class:`~relucent.complex.IncompleteDualGraphError` |
+-------------------------------+------------------------------------------------+
| Intentional partial           | Pass ``verify=False``                          |
| exploration                   |                                                |
+-------------------------------+------------------------------------------------+

Boundary-specific checks
~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~relucent.exploration.finalize_boundary_complex` also:

* Verifies ambient coface feasibility per cell
  (:func:`~relucent.certify.verify_boundary_cell`)
* Builds the dual graph with ``require_complete=verify``
* Runs :func:`~relucent.certify.verify_arrangement_genericity`
* Calls :func:`~relucent.incidence.set_contracted_shis` before ``certify_complex``

Exploration flags
~~~~~~~~~~~~~~~~~

After search or certification, :class:`~relucent.complex.Complex` records:

* ``complete`` — frontier exhausted without caps/depth limits
* ``verified`` — last invariant check passed

Topology routines that need a full arrangement call
:meth:`~relucent.complex.Complex.assert_topology_ready` (requires both flags).
Tests often use :func:`~relucent.exploration.explore_for_topology`.

Dual graph
----------

**Purpose:** adjacency among **top-dimensional** cells only.

**Entry point:** :meth:`~relucent.complex.Complex.get_dual_graph`

+------------------+----------------------------------------------------------+
| Aspect           | Detail                                                   |
+==================+==========================================================+
| Type             | ``networkx.Graph``                                       |
+------------------+----------------------------------------------------------+
| Nodes            | Top-dimensional ``Polyhedron`` objects (or relabeled ints)|
+------------------+----------------------------------------------------------+
| Edges            | Adjacent cells sharing a codimension-one face            |
+------------------+----------------------------------------------------------+
| Edge attribute   | ``shi`` — hyperplane crossed between endpoints           |
+------------------+----------------------------------------------------------+
| Cache            | Stored on ``Complex._dual_graph``; property ``G``        |
+------------------+----------------------------------------------------------+

Edge construction rules
~~~~~~~~~~~~~~~~~~~~~~~

The rule depends on the complex dimension:

* **``max_dim ≥ 2`` or ambient top level** — Combinatorial **flip neighbors**:
  two cells are adjacent if one sign sequence is obtained from the other by
  flipping a single nonzero entry.
* **1D ambient complex** — Cells sharing a combinatorial **0-face** (vertex tag).
* **Contracted 1-skeleton** (``max_dim == 1`` but ambient ``dim > 1``) — Walk
  each cell's finalized ``poly.shis`` (from :func:`~relucent.incidence.cubical_cell_shis`)
  and add edges only when the flip neighbor exists in the complex.

After edge construction, :func:`~relucent.incidence.sync_shis_from_dual_graph`
sets each node's ``_shis`` from incident edge labels when ``repair=True``.

Uses
~~~~

* Finalize after ambient BFS (``finalize_ambient_search``)
* :meth:`~relucent.complex.Complex.contract` and
  :meth:`~relucent.complex.Complex.get_chain_complex`
* :meth:`~relucent.complex.Complex.recover_from_dual_graph` — reconstruct from
  stored graph + SHI edge labels
* Visualization (``plot=True`` prepares a PyVis layout)

Meta-graph
----------

**Purpose:** the full **face poset** across all dimensions — the combinatorial
input to Betti numbers and persistent homology.

**Entry point:** :meth:`~relucent.complex.Complex.get_meta_graph`

+------------------+----------------------------------------------------------+
| Aspect           | Detail                                                   |
+==================+==========================================================+
| Type             | ``networkx.MultiDiGraph``                                |
+------------------+----------------------------------------------------------+
| Nodes            | All cells ``k = 0 … d`` from the chain complex           |
+------------------+----------------------------------------------------------+
| Node attrs       | ``poly``, ``dim``, ``ss``, ``finite``, ``crossings``, ``shis`` |
+------------------+----------------------------------------------------------+
| Edges            | Directed ``k-cell → (k−1)-face``                         |
+------------------+----------------------------------------------------------+
| Edge attr        | ``shi`` — hyperplane zeroed to reach the face            |
+------------------+----------------------------------------------------------+

Construction pipeline
~~~~~~~~~~~~~~~~~~~~~

1. :meth:`~relucent.complex.Complex.get_chain_complex` — Repeated contraction
   from the ambient complex down to 0-cells.
2. **Face edges** — :func:`~relucent.incidence.collect_meta_face_edges` per
   dimension (``ss_nonzero_indices`` + lookup); parallelized when cell count ≥
   ``META_FACE_PARALLEL_MIN_CELLS``.
3. **Boundedness** — 0-face incidence on 1-cells and ascending sweep
   (:func:`~relucent.incidence.classify_one_cells_finite_from_face_edges`,
   :func:`~relucent.incidence.classify_finite_ascending`).
4. **Node assembly** — :func:`~relucent.incidence.meta_node_attrs` derives
   ``crossings`` and flip-neighbor ``shis`` per dimension slice.
5. **Optional truncation** — :func:`~relucent.meta_graph.truncate_meta_graph` or
   :func:`~relucent.meta_graph.one_point_compactify_meta_graph` for homology at
   infinity.

Pass ``verify=True`` to ``get_meta_graph`` only for debugging:
:func:`~relucent.meta_graph.verify_meta_graph_incidence` checks edges, SHIs, and
finite labels match the incidence engine.

Dual graph vs meta-graph
~~~~~~~~~~~~~~~~~~~~~~~~

::

   BFS / finalize  →  dual graph (top-cell adjacency, edge shi)
                         ↓
                    contract()
                         ↓
                    meta-graph (all dims, face incidences)
                         ↓
                    get_betti_numbers()

+----------------------+---------------------------+---------------------------+
|                      | Dual graph                | Meta-graph                |
+======================+===========================+===========================+
| Dimension scope      | Top cells only            | All dims in chain complex |
+----------------------+---------------------------+---------------------------+
| Edge meaning         | Same-dim adjacency        | Codim-one face incidence  |
+----------------------+---------------------------+---------------------------+
| SHI on edges         | Crossing hyperplane       | Hyperplane zeroed to face |
+----------------------+---------------------------+---------------------------+
| SHI on nodes         | Synced from edges (top)   | ``cubical_cell_shis``     |
|                      | or ``cubical_cell_shis``  | (role 3)                  |
|                      | on contracted slices      |                           |
+----------------------+---------------------------+---------------------------+
| Face discovery       | Flip neighbors /          | All ``ss_i ≠ 0`` crossings|
|                      | 0-face sharing            | (role 2)                  |
+----------------------+---------------------------+---------------------------+
| Primary consumers    | Search finalize,          | Topology, persistence,    |
|                      | contraction, dual-graph   | Morse                     |
|                      | finalize                  |                           |
+----------------------+---------------------------+---------------------------+

**Critical invariant:** meta-graph **face edges** use
:func:`~relucent.incidence.ss_nonzero_indices` (role 2), not propagated
``_shis`` (role 3). Node ``_shis`` can be a strict subset of SS crossings;
conflating the two breaks ``∂² = 0`` for GF(2) boundary maps.

Boundary discovery
------------------

:meth:`~relucent.complex.Complex.discover_boundary_complex` builds the
decision-boundary complex for neuron ``i`` without a full ambient BFS:

1. **MIP pricing** — :func:`~relucent.boundary_mip.price_boundary_witness` finds
   witnesses for new connected components on the slice ``ss[i] = 0``.
2. **Slice BFS** — :func:`~relucent.boundary_search.boundary_searcher` explores
   each component with ``ss[boundary_shi] = 0`` fixed and SHI subsets that
   exclude the boundary hyperplane.
3. **Finalize** — :func:`~relucent.exploration.finalize_boundary_complex`:
   slice SHI assignment (``ss_nonzero_indices`` + ``set_contracted_shis``),
   dual graph, genericity check, ``certify_complex``.

Alternatively, explore the full ambient complex first, then
:meth:`~relucent.complex.Complex.get_boundary_complex` via contraction (requires
``assert_topology_ready``).

Related reading
---------------

* :doc:`exploration_verification` — flags, caps, dual-graph SHI model
* :doc:`search_geometry` — ``geometry_properties`` and memory
* ``docs/betti_computation.md`` — chain complex → meta-graph → Betti pipeline
* :doc:`topology` — Betti-number prerequisites and caveats
