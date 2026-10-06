Topology and Persistent Homology
================================

Relucent can compute **Betti numbers** and **persistent homology** over
**GF(2)** on the face poset of a discovered ReLU polyhedral (sub)complex. These
routines build on the same meta-graph convention as dual-graph and face-poset
analysis: a codimension-one face of a cell is obtained by zeroing one supporting
hyperplane index (SHI) in the cell's sign sequence.

Prerequisites
-------------

**Complete exploration.** Local search methods like BFS are sufficient for computing
topology *only if they run to completion*. Missing neighbors can leave the meta-graph
short of a closed cellular complex, which breaks ``∂² = 0`` for the GF(2) boundary maps.
:meth:`~relucent.core.complex.Complex.chain_complex` recovers faces algebraically
from verified vertices' local stars (Masden 2022, Theorem 20; see
:mod:`relucent.graph.vertex_star`), so every cell whose generating vertex is verified
is guaranteed present — no coverage heuristic is required. A partial BFS still means
fewer top cells to seed vertices from, so completeness of the underlying exploration
still matters for how much of the true arrangement is recovered.

:meth:`~relucent.core.complex.Complex.chain_complex`,
:meth:`~relucent.core.complex.Complex.meta_graph`, and therefore
:meth:`~relucent.core.complex.Complex.betti_numbers` and
:meth:`~relucent.core.complex.Complex.persistent_homology` all call
:meth:`~relucent.core.complex.Complex.assert_topology_ready`, so they raise
:class:`~relucent.core.errors.ComplexNotCompleteError` or
:class:`~relucent.core.errors.ComplexNotVerifiedError` unless the complex is complete and
verified.

After BFS, check :attr:`~relucent.core.complex.Complex.complete` and
:attr:`~relucent.core.complex.Complex.verified` (see :doc:`exploration_verification`).
:meth:`~relucent.core.complex.Complex.contract` and
:meth:`~relucent.core.complex.Complex.boundary_complex` require a complete, verified
ambient complex via :meth:`~relucent.core.complex.Complex.assert_topology_ready`.
:meth:`~relucent.core.complex.Complex.discover_boundary_complex` finds a boundary without
exploring the whole input space, but it is experimental and can miss components (see
:doc:`boundary_search`).

**Geometry for filtrations.** Built-in filtrations such as
:class:`~relucent.topology.filtration.AffineOutputFiltration` and
:class:`~relucent.topology.filtration.TrainingDistanceFiltration` need interior points on
cells. Either compute geometry during search (done by default) or run
:meth:`~relucent.core.complex.Complex.compute_geometric_properties` afterward with
``properties=["interior_point", "finite"]`` (and ``"W"``, ``"b"`` when affine
outputs are required).

Graph representations
---------------------

The :class:`~relucent.core.complex.Complex` class exposes three related graph views:

* **Dual graph** (:meth:`~relucent.core.complex.Complex.dual_graph`): adjacency of
  top-dimensional cells only.
* **Chain complex** (:meth:`~relucent.core.complex.Complex.chain_complex`): lower-
  dimensional faces recovered by seeding and verifying vertices, then expanding each
  verified vertex's local cubical star, via :mod:`relucent.graph.vertex_star`
  (``Complex.contract()`` returns the codimension-one slice).
* **Meta-graph** (:meth:`~relucent.core.complex.Complex.meta_graph`): face poset
  over all cell dimensions, used by Betti and persistence code.

Betti numbers
-------------

:meth:`~relucent.core.complex.Complex.betti_numbers` builds a meta-graph, applies
the chosen homology convention, and returns ``{dimension: β_k}``.

**``compactify``** selects how unbounded cells are handled:

* ``"truncate"`` (default): **combinatorial truncation** at infinity via
  :func:`~relucent.graph.meta_graph.truncate_meta_graph`.
* ``"borel_moore"``: **Borel–Moore** homology (only faces with at least two
  cofaces contribute to incidence).
* ``"one_point"``: **one-point compactification** via
  :func:`~relucent.graph.meta_graph.one_point_compactify_meta_graph`.

To rank a meta-graph you built or edited yourself, call
:func:`relucent.topology.betti_numbers` on it with the same ``compactify`` values
(``None``, its default, ranks the graph exactly as given).

**``respect_finite``**: restrict to the subcomplex of cells with ``finite is True``
(no truncation).

**``verify_chain_complex``**: when ``True``, require ``∂² = 0`` on the assembled
boundary maps; raises :class:`~relucent.topology.ChainComplexInconsistent` if they are
inconsistent. Passing it bypasses the per-complex Betti cache.

**``verify_connected_components``**: when ``True``, check that β₀ from the rank formula
agrees with the number of connected components, raising
:class:`~relucent.topology.ConnectedComponentsMismatch` otherwise (default ``False`` on
:class:`~relucent.core.complex.Complex`; ``True`` when calling
:func:`relucent.topology.betti_numbers` directly).

**``reduced``**: return reduced homology (β̃₀ = β₀ − 1).

**``nworkers``**: thread count for ranking boundary maps concurrently. It only applies to
the ``method="dense"`` ranking in :func:`relucent.topology.betti_numbers` (see
*Performance*), so it has no effect on the default path.

Example:

.. code-block:: python

   import relucent
   from relucent.topology.filtration import ConstantFiltration
   from relucent.topology.persistence import betti_at_filtration_end, compute_persistent_homology

   model = relucent.mlp(widths=[2, 8, 4, 1])
   # We append a final ReLU to the model so that the constructed complex includes it's decision boundary.
   # See the Network Definitions section for more details.
   cplx = relucent.Complex(relucent.add_output_relu(model))
   # Run to completion; verification is skipped only if max_polys is hit before the frontier empties.
   cplx.bfs(max_polys=500)

   # Decision boundary of the last ReLU neuron (requires complete ambient complex)
   db = cplx.boundary_complex(cplx.n - 1)

   betti = db.betti_numbers()
   print(betti)  # e.g. {0: 1}

   # Or discover the boundary directly without a full ambient BFS:
   # db = cplx.discover_boundary_complex(cplx.n - 1, verbose=0)

   # Cross-check via persistent homology with a constant filtration
   diagram = compute_persistent_homology(
       db,
       ConstantFiltration(0.0),
       lower_star=False,
   )
   ph_betti = betti_at_filtration_end(diagram)
   for k in set(betti) | set(ph_betti):
       assert betti.get(k, 0) == ph_betti.get(k, 0)

Persistent homology workflow
----------------------------

:meth:`~relucent.core.complex.Complex.persistent_homology` accepts any
:class:`~relucent.topology.filtration.Filtration` and returns a
:class:`~relucent.topology.persistence.PersistenceDiagram`.

Built-in filtrations:

* :class:`~relucent.topology.filtration.ConstantFiltration` — all cells enter at one value.
  Use ``lower_star=False`` to match static Betti numbers on the same complex.
* :class:`~relucent.topology.filtration.LogitSublevelFiltration` — sublevel sets of a scalar
  logit (last output or class difference).
* :class:`~relucent.topology.filtration.AffineOutputFiltration` — general affine output
  functional on each cell.
* :class:`~relucent.topology.filtration.NeuronActivationFiltration` — combinatorial
  filtration by ReLU sign on a chosen SHI.
* :class:`~relucent.topology.filtration.TrainingDistanceFiltration` — distance from a cell
  representative point to training data.

Lower-star extension (:func:`~relucent.topology.filtration.lower_star_extension`) promotes
vertex values to higher cells by ``f(σ) = max_{τ face of σ} f(τ)`` when
``lower_star=True`` (the default for most filtrations).

Example:

.. code-block:: python

   import numpy as np
   import torch.nn as nn
   import relucent
   from relucent.topology.filtration import LogitSublevelFiltration

   model = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 1), nn.ReLU())
   cplx = relucent.Complex(model)
   cplx.bfs(start=np.array([[0.0]]), max_polys=5000)
   cplx.dual_graph(require_complete=True)
   cplx.compute_geometric_properties(properties=["interior_point", "finite", "W", "b"])

   diagram = cplx.persistent_homology(LogitSublevelFiltration())
   fig = diagram.plot()
   fig.show()

Use :func:`~relucent.topology.persistence.betti_curve` to track β_k across filtration
thresholds, and :func:`~relucent.topology.persistence.betti_at_filtration_end` to read
Betti numbers after all cells have entered.

Performance
-----------

:func:`relucent.topology.betti_numbers` ranks each GF(2) boundary map with
``method="sparse"`` by default: Gaussian elimination on the incidence sets with low-fill
pivots (:func:`relucent.topology.gf2_rank_sparse_rowsets`), so cost and memory follow the
number of incidences rather than ``rows × columns``. If fill-in makes the remainder dense,
that remainder is ranked bit-packed.

``method="dense"`` ranks the full bit-packed matrices and needs ``rows × columns / 8``
bytes per map (about 59 GB for a 688k × 688k ∂₂), so it is only kept for cross-checking.
It is not exposed through :class:`~relucent.core.complex.Complex`; call
:func:`relucent.topology.betti_numbers` on a meta-graph to use it.

Bit-packed ranking uses an optional **C extension** (``relucent.topology._gf2``),
JIT-compiled from ``_gf2_rank.c`` with ``gcc`` on first import. The public flag
:data:`relucent.topology.C_BACKEND_AVAILABLE` reports whether the fast path is loaded;
otherwise relucent logs a warning and falls back to pure Python.

The compiled library is cached in ``$RELUCENT_CACHE_DIR`` if set, else the user cache
directory (``~/.cache/relucent`` on Linux, ``~/Library/Caches/relucent`` on macOS), else
the system temp directory. It is built with ``-march=native``, and the cached file is keyed
by the CPU's feature flags, so an install shared between machines with different CPUs keeps
one build per CPU type.

Topology, persistence, and search calls take ``verbose``: ``0`` is quiet, ``1`` shows
progress bars (after a one-second delay) and one-line summaries, and ``2`` adds per-stage
detail such as boundary-map shapes and ranks. ``None`` (the default) uses
:data:`relucent.config.VERBOSE`.
