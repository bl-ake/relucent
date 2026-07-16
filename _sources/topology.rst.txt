Topology and Persistent Homology
================================

Relucent can compute **Betti numbers** and **persistent homology** over
**GF(2)** on the face poset of a discovered ReLU polyhedral (sub)complex. These
routines build on the same meta-graph convention as dual-graph and face-poset
analysis: a codimension-one face of a cell is obtained by zeroing one supporting
hyperplane index (SHI) in the cell's sign sequence.

This is an experimental, research-oriented API. :meth:`~relucent.core.complex.Complex.get_betti_numbers`
and :meth:`~relucent.core.complex.Complex.get_persistent_homology` emit a collaboration
``UserWarning`` unless you set ``DISABLE_RESEARCH_WARNING=1``.

Prerequisites
-------------

**Complete exploration.** Local search methods like BFS are sufficient for computing
topology *only if they run to completion*. Missing neighbors can leave the meta-graph
short of a closed cellular complex, which breaks ``∂² = 0`` for the GF(2) boundary maps.

After BFS, check :attr:`~relucent.core.complex.Complex.complete` and
:attr:`~relucent.core.complex.Complex.verified` (see :doc:`exploration_verification`).
:meth:`~relucent.core.complex.Complex.contract` and
:meth:`~relucent.core.complex.Complex.get_boundary_complex` require a complete, verified
ambient complex via :meth:`~relucent.core.complex.Complex.assert_topology_ready`.
For boundary components not covered by ambient exploration, use
:meth:`~relucent.core.complex.Complex.discover_boundary_complex`.

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

* **Dual graph** (:meth:`~relucent.core.complex.Complex.get_dual_graph`): adjacency of
  top-dimensional cells only.
* **Chain complex** (:meth:`~relucent.core.complex.Complex.get_chain_complex`): lower-
  dimensional faces recovered from cubical stars in the dual graph via
  :mod:`relucent.graph.covectors` (``Complex.contract()`` returns the codimension-one
  slice).
* **Meta-graph** (:meth:`~relucent.core.complex.Complex.get_meta_graph`): face poset
  over all cell dimensions, used by Betti and persistence code.

Betti numbers
-------------

:meth:`~relucent.core.complex.Complex.get_betti_numbers` builds a meta-graph, applies
the chosen homology convention, and returns ``{dimension: β_k}``.

**``compactify``** selects how unbounded cells are handled:

* ``False`` (default): **combinatorial truncation** at infinity via
  :meth:`~relucent.core.complex.Complex.truncate_meta_graph`.
* ``True``: **Borel–Moore** style boundaries (only faces with at least two
  cofaces contribute to incidence).
* ``"one_point"``: **one-point compactification** via
  :meth:`~relucent.core.complex.Complex.one_point_compactify_meta_graph`.

**``respect_finite``**: restrict to the subcomplex of cells with ``finite is True``
(no truncation).

**``verify_chain_complex``**: when ``True``, require ``∂² = 0`` on the assembled
boundary maps; raises :class:`~relucent.topology.ChainComplexInconsistent` if the
explored complex is incomplete.

**``nworkers``**: thread count for ranking independent boundary maps (``None``
auto-selects when the optional C GF(2) backend is available).

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
   db = cplx.get_boundary_complex(cplx.n - 1)

   betti = db.get_betti_numbers()
   print(betti)  # e.g. {0: 1}

   # Or discover the boundary directly without a full ambient BFS:
   # db = cplx.discover_boundary_complex(cplx.n - 1, verbose=False)

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

:meth:`~relucent.core.complex.Complex.get_persistent_homology` accepts any
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
   cplx.get_dual_graph(require_complete=True)
   cplx.compute_geometric_properties(properties=["interior_point", "finite", "W", "b"])

   diagram = cplx.get_persistent_homology(LogitSublevelFiltration())
   fig = diagram.plot()
   fig.show()

Use :func:`~relucent.topology.persistence.betti_curve` to track β_k across filtration
thresholds, and :func:`~relucent.topology.persistence.betti_at_filtration_end` to read
Betti numbers after all cells have entered.

Performance
-----------

Boundary-matrix rank computation can use an optional **C extension**
(``relucent.topology._gf2``), JIT-compiled from ``_gf2_rank.c`` when a C compiler is
available. The public flag :data:`relucent.topology.C_BACKEND_AVAILABLE` reports
whether the fast path is loaded; otherwise relucent falls back to pure Python.

Set ``verbose=True`` on topology and persistence calls for progress on stderr.
Package-wide search logging is controlled by :data:`relucent.config.VERBOSE`.
