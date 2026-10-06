:tocdepth: 0

Topology (GF(2))
================

.. automodule:: relucent.topology
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: AffineOutputFiltration, ConstantFiltration, Filtration, LogitSublevelFiltration,
      NeuronActivationFiltration, TrainingDistanceFiltration, PersistenceDiagram, PersistencePair,
      betti_at_filtration_end, betti_curve, compute_persistent_homology

.. py:data:: C_BACKEND_AVAILABLE
   :type: bool

   ``True`` when the C GF(2) rank backend (``_gf2_rank.c``) compiled and loaded. Without it,
   ranks are computed by the pure-Python backend, which gives the same results more slowly.

.. py:data:: Compactify
   :value: Literal["truncate", "borel_moore", "one_point"]

   How unbounded cells are handled when computing homology: ``"truncate"`` caps them with
   new faces (combinatorial truncation at infinity), ``"borel_moore"`` computes Borel–Moore
   homology, and ``"one_point"`` adds a single 0-cell at infinity.

The filtration and persistence names exported here are documented in :doc:`filtration` and
:doc:`persistence`.
