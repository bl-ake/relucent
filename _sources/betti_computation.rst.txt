How Betti numbers are computed
==============================

This page walks through what happens when you call ``cplx.betti_numbers()`` with no extra
arguments, the default path most users hit.

**What you get back:** a dictionary ``{k: β_k}`` of Betti numbers computed over GF(2). For
example ``{0: 1, 1: 2}`` means one connected component and two independent 1-cycles.

**What you need first:** a complete, verified complex discovered by search (BFS, etc.;
:meth:`~relucent.core.complex.Complex.chain_complex` calls
:meth:`~relucent.core.complex.Complex.assert_topology_ready`). Faces are recovered
algebraically from verified vertices' local stars (Masden 2022, Theorem 20; see
:mod:`relucent.graph.vertex_star`), so no coverage heuristic is needed: every cell whose
generating vertex is verified is guaranteed present. A partial BFS would leave the complex
short of the full arrangement (fewer top cells means fewer vertices to seed from), so
``cplx.complete`` and ``cplx.verified`` must both be ``True``; see
`Exploration and certification`_ below.

The default pipeline
--------------------

When you call ``betti_numbers()`` with defaults (``compactify="truncate"``,
``respect_finite=False``, no verification flags), the library runs these steps in order:

1. BFS discovers the top-dimensional cells.
2. :meth:`~relucent.core.complex.Complex.chain_complex` seeds and verifies vertices and
   expands their local stars.
3. :meth:`~relucent.core.complex.Complex.meta_graph` records face edges and labels each cell
   bounded or unbounded.
4. :func:`~relucent.graph.meta_graph.truncate_meta_graph` caps unbounded cells at infinity.
5. :func:`relucent.topology.betti_numbers` ranks the boundary matrices.

Each step is described below.

Step 1: Build the chain complex
-------------------------------

**Entry point:** :meth:`~relucent.core.complex.Complex.chain_complex`

The chain complex is a list of :class:`~relucent.core.complex.Complex` objects, one per
dimension, from the top-dimensional cells down to 0-cells (vertices). Lower-dimensional
cells are recovered directly from verified vertices' local stars by
:func:`~relucent.graph.vertex_star.recover_cells_from_vertices`, not by iterative dual-edge
contraction, and not by requiring the complete ``2^c`` cube of top-cell cofaces to already
exist.

:meth:`~relucent.core.complex.Complex.contract` returns the complex of codimension-one cells
(dimension ``dim - 1``) from ``chain_complex()``.

How vertex-star recovery works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Masden, *Algorithmic Determination of the Combinatorial Structure of the Linear Regions of
ReLU Neural Networks* (2022), Theorem 20: for a generic, supertransversal network with at
least ``ambient_dim`` units in its first layer, the sign-sequence complex is a **pure,
ambient-dimensional cubical complex**. Every vertex has exactly ``ambient_dim`` zero sign
entries (Lemma 16), and once such a vertex is verified, *every* sign assignment on those
zero entries, holding all other entries fixed, is a real, present cell (Lemma 18's
sign-product semigroup). Result numbers are those of arXiv:2207.07696v1.

1. Build the **dual graph**, the combinatorial adjacency among top-dimensional cells
   (:meth:`~relucent.core.complex.Complex.dual_graph` →
   :func:`~relucent.graph.incidence.dual_edges_top_dim`), then
   :func:`~relucent.graph.incidence.sync_shis_from_dual_graph` so each cell's ``_shis`` list
   matches incident edge labels. Certify the labeled graph with
   :func:`~relucent.graph.incidence.certify_dual_graph`.
2. **Seed and verify vertices** (:func:`~relucent.graph.vertex_star.find_vertices`): for each
   top cell, choose ``top_dim`` of its dual-graph-incident SHIs (a top cell already has
   ``ambient_dim - top_dim`` zero entries by Lemma 16) and zero them to get a candidate vertex
   sign sequence. Verify it with one float64 equality solve plus a check of every other row of
   the witness cell, made exactly when float64 cannot decide
   (:meth:`~relucent.core.poly.Polyhedron.verify_vertex_covector`). No facet or boundedness LP
   is used.
3. **Expand each verified vertex's local star**
   (:func:`~relucent.graph.vertex_star.expand_vertex_star`): vary the ``top_dim`` coordinates
   that distinguish the vertex from its witness top cell independently over ``{-1, 0, 1}``,
   producing every cell of every dimension from 0 (the vertex itself) up to ``top_dim`` in
   that star. Every generated cell of dimension ``k ≥ 1`` has, by construction, at least one
   verified vertex among its own faces (its generating vertex). A cell with every endpoint
   unverifiable can never be produced, so no separate "cascade drop" pass is needed.
4. Materialize each recovered cell with ``add_ss``, then run
   :func:`~relucent.graph.incidence.set_contracted_shis` on each lower-dimensional slice so
   authoritative ``_shis`` match :func:`~relucent.graph.incidence.cubical_cell_shis`. In
   ``CAREFUL_MODE``, :func:`~relucent.graph.incidence.verify_contracted_shis` asserts
   flip-neighbor symmetry. Top-dimensional ambient cells do not use this step; their
   ``_shis`` come from the dual graph when the search finishes.

Dual-graph rules
~~~~~~~~~~~~~~~~

:func:`~relucent.graph.incidence.build_dual_graph` always routes top cells through
flip-neighbor adjacency (:func:`~relucent.graph.incidence.dual_graph_edge_top_dim` maps
``max_dim == 1`` complexes to the ``top_dim >= 2`` flip path). For each candidate crossing,
it adds an edge when the same-dimension flip neighbor exists in the slice
(``_dual_edges_flip_neighbors()``).

* **Ambient top cells**: candidates from
  :func:`~relucent.graph.incidence.ss_nonzero_indices`.
* **Contracted 1-skeleton** (``max_dim == 1``, ambient ``self.dim > 1``): candidates from each
  cell's finalized ``poly._shis`` after :func:`~relucent.graph.incidence.set_contracted_shis`.

Then :func:`~relucent.graph.incidence.sync_shis_from_dual_graph` overwrites ``poly._shis``
from edge labels. The legacy 0-face pairing in ``_dual_edges_one_dim()`` remains for direct
``dual_edges_top_dim(..., top_dim=1)`` callers only, not for ``Complex.dual_graph()``.

**Related (not the ambient chain complex):**
:meth:`~relucent.core.complex.Complex.boundary_cells` and
:meth:`~relucent.core.complex.Complex.boundary_complex` still create faces from dual edges
(``relucent.graph.boundary._codim_one_face_kwargs()``).

Step 2: Build the meta-graph
----------------------------

**Entry point:** :meth:`~relucent.core.complex.Complex.meta_graph`

The meta-graph is a directed graph whose nodes are cells (at every dimension found in the
chain complex) and whose edges record codimension-one face incidences. This is the
combinatorial input to the rank computation.

Each node stores ``poly``, ``dim``, ``ss``, ``finite`` (bounded or not), and ``shis``.

Face edges: what defines the Betti numbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For every cell of dimension ``k > 0``:

1. Look at every index ``i`` where ``ss[i] ≠ 0`` (via
   :func:`~relucent.graph.incidence.ss_nonzero_indices`).
2. Zero that entry to get a candidate face tag (:func:`~relucent.graph.incidence.face_tag`).
3. If a cell with that tag exists in the complex, add a directed edge
   ``k-cell → (k−1)-face`` with attribute ``shi = i``.

This is implemented by :func:`~relucent.graph.incidence.collect_meta_face_edges` (per
dimension, in parallel when there are enough cells).

**0-cells:** the face-edge loop skips ``k ≤ 0``, so 0-cells never have outgoing face edges.
They can still appear as targets of edges from 1-cells.

**1-cells:** use the same face-edge rule as higher-dimensional cells: zero each nonzero
sign-sequence entry and connect to the resulting 0-face if it exists.

Bounded and unbounded labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These labels decide which cells get truncated at infinity (Step 3). They do not change how
face edges are built.

Classification is **combinatorial** on the default path, with no linear programs.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Dimension
     - Rule
   * - 0-cells
     - Always bounded.
   * - 1-cells
     - Bounded if at least two distinct combinatorial 0-faces appear in meta face edges
       (:func:`~relucent.graph.incidence.classify_one_cells_finite_from_face_edges`). One
       0-face means a ray (unbounded). With no 0-faces a geometric check decides: empty
       phantoms (``finite is None``) are excluded, and feasible full lines are unbounded.
   * - k ≥ 2
     - Unbounded if **any** ``(k−1)``-face is unbounded; bounded if **all** ``(k−1)``-faces
       are bounded (:func:`~relucent.graph.incidence.classify_finite_ascending`).

Node ``shis`` and face edges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Face edges always scan the full sign sequence
(:func:`~relucent.graph.incidence.ss_nonzero_indices`). Meta-graph node ``shis`` are
flip-neighbor crossings derived by :func:`~relucent.graph.incidence.cubical_cell_shis` at
node creation. They are metadata only, not used for boundedness or incidence.

Step 3: Truncate at infinity
----------------------------

**Entry point:** :func:`~relucent.graph.meta_graph.truncate_meta_graph`

This runs automatically when ``compactify="truncate"`` (the default). Truncation is a
single incidence pipeline: extend sign sequences, materialize cap cells, then rebuild all
face edges with the same rule as :meth:`~relucent.core.complex.Complex.meta_graph`.

**Phase A, extend sign sequences:** every node's sign sequence gains **two** trailing
truncation bits. Bounded cells and rays use ``[..., 1, 0]``; cells with two open ends use
``[..., 1, 1]``. Node keys are relabeled to ``encode_ss(extended_ss)``.

**Phase B, openness:** bottom-up from codimension-one face structure:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Situation
     - Caps on this cell
   * - ``k ≥ 2``, has a bounded ``(k−1)``-face and **any** unbounded facet
     - 1 cap (single sphere-cut patch)
   * - ``k ≥ 2``, has a bounded ``(k−1)``-face, no unbounded facets
     - None
   * - ``k = 1``, 2 bounded 0-endpoints
     - None (bounded segment)
   * - ``k = 1``, 1 bounded 0-endpoint
     - 1 cap (ray)
   * - ``k = 1``, 0 bounded 0-endpoints
     - 2 caps (line)
   * - ``k ≥ 2``, all ``(k−1)``-faces unbounded
     - Inherit the maximum openness of its *sidedness* facets

"Any unbounded facet" for the anchored 0-vs-1 rule uses the raw facet list (before
bi-infinite sidedness filtering). Sidedness filtering drops bi-infinite line facets when
``poly`` is set. That filter drives *inheritance only*, so it cannot silently force
``n_caps=0`` on an anchored cell that still has unbounded facets.

**Phase C, cap cells:** for each needed cap,
``cap_tag = face_tag(parent_ss, truncation_bit_index)``. If absent, add an ordinary
byte-tagged node at dimension ``k−1`` with the appropriate zeroed truncation bit.

**Phase D, incidence:** clear face edges and rebuild via
:func:`~relucent.graph.incidence.collect_meta_face_edges` and
:func:`~relucent.graph.incidence.assemble_face_edges_by_dim`, reclassify ``finite``, refresh
``crossings`` and ``shis``, and run
:func:`~relucent.graph.meta_graph.verify_meta_graph_one_cells`.

Rebuilding with the cubical ``face_tag`` recovers network faces only when parent and face
share truncation bits. Faces between cells with disagreeing openness (e.g. a unilateral
coface ``(1,0)`` and a bi-infinite line ``(1,1)``) are omitted: restoring them without a
matching sphere-cut breaks ``∂²=0``. Those cofaces keep their truncation cap instead.

**0-cells are never duplicated**, even if marked unbounded.

The caps model the region at infinity, so the homology of the truncated complex reflects
the topology of unbounded regions. Each unbounded ``k``-cell gets its own sphere-cut
``(k−1)``-cell. If it has a bounded facet, the cut is a single connected patch (one cap),
built from the caps of its truncation-compatible unbounded facets.

Step 4: Rank boundary matrices
------------------------------

**Entry point:** :func:`relucent.topology.betti_numbers`

From the (possibly truncated) meta-graph:

1. Group nodes by dimension.
2. For each ``k``, build a GF(2) boundary matrix ``∂_k`` from directed edges
   (``k-cell → (k−1)-face``). Columns index ``k``-cells; rows index ``(k−1)``-cells.
3. Compute ranks. The default ``method="sparse"`` builds each map as sparse row sets
   (``_sparse_boundary_maps()``) and ranks it with
   :func:`~relucent.topology.gf2_rank_sparse_rowsets`: low-fill-pivot elimination whose cost
   follows the number of incidences, switching to bit-packed ranking only for a remainder
   that has become dense. ``method="dense"`` builds ``_packed_boundary_matrix()`` and ranks
   it with :func:`~relucent.topology.gf2_rank_boundary` (C extension when available, Python
   fallback otherwise). It needs ``rows × columns / 8`` bytes per map and is kept for
   cross-checking; :class:`~relucent.core.complex.Complex` does not expose it.
4. Apply the cellular formula
   ``β_k = (number of k-cells) − rank(∂_k) − rank(∂_{k+1})``.

When there are no 0-cells (e.g. a boundary complex with only 1- and 2-cells), the lowest key
in the returned dictionary is ``1``, not ``0``.

Zero entries are dropped from the result. With ``verify_connected_components=True`` (the
default for :func:`relucent.topology.betti_numbers`, ``False`` via ``Complex``), β₀ is
checked against the number of connected components.

Exploration and certification
-----------------------------

After a **complete** ambient BFS (``verify=True`` by default), relucent:

1. Rebuilds combinatorial dual-graph edges and syncs top-cell ``_shis``
   (:func:`~relucent.search.exploration.finalize_ambient_search` via
   :meth:`~relucent.core.complex.Complex.dual_graph`).
2. Runs certification (:func:`~relucent.verify.certify.certify_complex` at
   ``CertifyLevel.COMPLETE``), including an LP facet completeness test when
   ``cplx.complete is True``.

Certification is skipped if ``max_polys`` is hit before the frontier empties. Check
``cplx.complete`` and ``cplx.verified`` before calling ``contract()``, ``chain_complex()``,
or ``boundary_complex()``. Re-run certification manually with
:meth:`~relucent.core.complex.Complex.certify`.

See :doc:`exploration_verification` for user-facing detail.

Other options
-------------

These change behavior when you pass extra flags to ``betti_numbers()`` or ``meta_graph()``:

* ``compactify="borel_moore"``: Borel–Moore homology. No truncation; only faces with at
  least two cofaces contribute to boundary maps.
* ``compactify="one_point"``: adds a single 0-cell at infinity for unbounded 1-cell ends
  (:func:`~relucent.graph.meta_graph.one_point_compactify_meta_graph`).
* ``respect_finite=True``: restrict to cells with ``finite is True`` before ranking
  (:func:`~relucent.graph.meta_graph.finite_cells_subgraph`); no truncation.
* ``verify_chain_complex=True``: require ``∂² = 0``; raises
  :class:`~relucent.topology.ChainComplexInconsistent` if not. Bypasses the Betti cache.
* ``verify_connected_components=True``: check the rank-formula β₀ against the
  connected-component count; raises :class:`~relucent.topology.ConnectedComponentsMismatch`
  on mismatch.
* ``method="dense"`` (:func:`relucent.topology.betti_numbers` only): bit-packed ranking
  instead of sparse elimination; ``nworkers`` applies only here.
* ``meta_graph(verify=True)``: runs
  :func:`~relucent.graph.meta_graph.verify_meta_graph_incidence` to assert that assembled
  edges, node SHIs, and finite labels match the incidence engine (debugging).
* :meth:`~relucent.core.complex.Complex.verify_arrangement_genericity`: geometric
  transversality check on 1-cells, run unconditionally in ``boundary_complex``.

Function map
------------

By pipeline step
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Step
     - Main functions
   * - Search
     - ``Complex.bfs``, ``exploration.finalize_ambient_search``, ``Complex.dual_graph``,
       ``incidence.*``
   * - Chain complex
     - ``chain_complex``, ``vertex_star.find_vertices``, ``vertex_star.expand_vertex_star``,
       ``dual_graph``, ``dual_edges_top_dim``, ``set_contracted_shis``
   * - Meta-graph
     - ``meta_graph``, ``meta_graph.truncate_meta_graph``, ``incidence.cubical_cell_shis``,
       ``incidence.ss_nonzero_indices``, ``incidence.face_tag``,
       ``incidence.collect_meta_face_edges``, ``incidence.classify_finite_ascending``,
       ``incidence.meta_node_attrs``, ``meta_graph.verify_meta_graph_incidence``
   * - Certification
     - ``certify.certify_complex``, ``Complex.certify``, ``Complex.complete``,
       ``Complex.verified``
   * - Truncation
     - ``truncate_meta_graph``
   * - Ranks
     - ``Complex.betti_numbers``, ``topology.betti_numbers``, ``_sparse_boundary_maps``,
       ``gf2_rank_sparse_rowsets`` (default); ``_packed_boundary_matrix``,
       ``gf2_rank_boundary`` (``method="dense"``)

0-cells and 1-cells at a glance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * -
     - 0-cells
     - 1-cells
   * - Face edges
     - None outgoing
     - Zero each ``ss[i] ≠ 0`` and connect to 0-faces
   * - Boundedness
     - Always bounded
     - Two or more distinct 0-faces in face edges: bounded segment; one: ray
   * - Dual graph (top dim 1)
     - Not applicable
     - Pair by shared 0-face tag
   * - Truncation
     - Not duplicated
     - Caps at infinity on unbounded cells (byte tags)
   * - One-point compactification
     - Synthetic ``("infty",)`` node added
     - Open ends get an edge to ∞

See also :doc:`topology` for the API overview and examples.
