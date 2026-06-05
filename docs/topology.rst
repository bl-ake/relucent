Topology and Persistent Homology
================================

Relucent can compute **Betti numbers** and **persistent homology** over
**GF(2)** on the face poset of a discovered ReLU polyhedral complex. These
routines build on the same meta-graph convention as contraction and dual-graph
analysis: a codimension-one face of a cell is obtained by zeroing one supporting
hyperplane index (SHI) in the cell's sign sequence.

This is an experimental, research-oriented API. :meth:`~relucent.complex.Complex.get_betti_numbers`
and :meth:`~relucent.complex.Complex.get_persistent_homology` emit a collaboration
``UserWarning`` unless you set ``DISABLE_RESEARCH_WARNING=1``.

Prerequisites
-------------

**Complete exploration.** Local search (BFS, DFS, random walk) usually discovers
only a *partial* complex. Missing neighbors can leave the meta-graph short of a
closed cellular complex, which breaks ``∂² = 0`` for the GF(2) boundary maps.
Before topology work, explore until adjacency is complete (for example, run BFS
with ``require_complete=True`` on :meth:`~relucent.complex.Complex.get_dual_graph`,
or add explicit seed points until no new regions appear).

**Geometry for filtrations.** Built-in filtrations such as
:class:`~relucent.filtration.AffineOutputFiltration` and
:class:`~relucent.filtration.TrainingDistanceFiltration` need interior points on
cells. Either compute geometry during search or run
:meth:`~relucent.complex.Complex.compute_geometric_properties` afterward with
``properties=["interior_point", "finite"]`` (and ``"W"``, ``"b"`` when affine
outputs are required).

Graph layers
------------

Relucent uses three related graph views:

* **Dual graph** (:meth:`~relucent.complex.Complex.get_dual_graph`): adjacency of
  top-dimensional cells only.
* **Chain complex** (:meth:`~relucent.complex.Complex.get_chain_complex`): iterated
  contraction to lower-dimensional boundary complexes.
* **Meta-graph** (:meth:`~relucent.complex.Complex.get_meta_graph`): face poset
  over all cell dimensions, used by Betti and persistence code.

Betti numbers
-------------

:meth:`~relucent.complex.Complex.get_betti_numbers` builds a meta-graph, applies
the chosen homology convention, and returns ``{dimension: β_k}``.

**``compactify``** selects how unbounded cells are handled:

* ``False`` (default): **combinatorial truncation** at infinity via
  :meth:`~relucent.complex.Complex.truncate_meta_graph`.
* ``True``: **Borel–Moore** style boundaries (only faces with at least two
  cofaces contribute to incidence).
* ``"one_point"``: **one-point compactification** via
  :meth:`~relucent.complex.Complex.one_point_compactify_meta_graph`.

 Note: ``compactify=True`` is deprecated and will be replaced with ``compactify="bm"`` in a future release.

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
   from relucent.filtration import ConstantFiltration
   from relucent.persistence import betti_at_filtration_end, compute_persistent_homology

   cplx = relucent.Complex(relucent.mlp(widths=[2, 8, 4, 1]))
   cplx.bfs(max_polys=500)

   betti = cplx.get_betti_numbers()
   print(betti)  # e.g. {0: 1, 1: 0, 2: 0}

   # Cross-check via persistent homology with a constant filtration
   diagram = compute_persistent_homology(
       cplx,
       ConstantFiltration(0.0),
       lower_star=False,
   )
   ph_betti = betti_at_filtration_end(diagram)
   for k in set(betti) | set(ph_betti):
       assert betti.get(k, 0) == ph_betti.get(k, 0)

For an existing meta-graph (after manual truncation or compactification), use
:meth:`~relucent.complex.Complex.get_betti_numbers_from_meta`.

Persistent homology workflow
----------------------------

:meth:`~relucent.complex.Complex.get_persistent_homology` accepts any
:class:`~relucent.filtration.Filtration` and returns a
:class:`~relucent.persistence.PersistenceDiagram`.

Built-in filtrations:

* :class:`~relucent.filtration.ConstantFiltration` — all cells enter at one value.
  Use ``lower_star=False`` to match static Betti numbers on the same complex.
* :class:`~relucent.filtration.LogitSublevelFiltration` — sublevel sets of a scalar
  logit (last output or class difference).
* :class:`~relucent.filtration.AffineOutputFiltration` — general affine output
  functional on each cell.
* :class:`~relucent.filtration.NeuronActivationFiltration` — combinatorial
  filtration by ReLU sign on a chosen SHI.
* :class:`~relucent.filtration.TrainingDistanceFiltration` — distance from a cell
  representative point to training data.

Lower-star extension (:func:`~relucent.filtration.lower_star_extension`) promotes
vertex values to higher cells by ``f(σ) = max_{τ face of σ} f(τ)`` when
``lower_star=True`` (the default for most filtrations).

Example:

.. code-block:: python

   import numpy as np
   import torch.nn as nn
   import relucent
   from relucent.filtration import LogitSublevelFiltration

   relucent.set_seeds(0)
   model = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 1), nn.ReLU())
   cplx = relucent.Complex(model)
   cplx.bfs(start=np.array([[0.0]]), max_polys=5000)
   cplx.get_dual_graph(require_complete=True)
   cplx.compute_geometric_properties(properties=["interior_point", "finite", "W", "b"])

   diagram = cplx.get_persistent_homology(LogitSublevelFiltration())
   fig = diagram.plot()
   fig.show()

Use :func:`~relucent.persistence.betti_curve` to track β_k across filtration
thresholds, and :func:`~relucent.persistence.betti_at_filtration_end` to read
Betti numbers after all cells have entered.

Performance
-----------

Boundary-matrix rank computation can use an optional **C extension**
(``relucent._gf2``), JIT-compiled from ``_gf2_rank.c`` when a C compiler is
available. The public flag :data:`relucent.topology.C_BACKEND_AVAILABLE` reports
whether the fast path is loaded; otherwise relucent falls back to pure Python.

Set ``verbose=True`` on topology and persistence calls for progress on stderr.
Package-wide search logging is controlled by :data:`relucent.config.VERBOSE`.
