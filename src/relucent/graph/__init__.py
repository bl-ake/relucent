"""Dual graph, meta-graph, and cubical incidence.

Submodules:

- :mod:`~relucent.graph.incidence` — sign-sequence face rules, dual-graph assembly, and
  meta-graph edge collection. Called from :meth:`~relucent.core.complex.Complex.get_dual_graph`,
  :meth:`~relucent.core.complex.Complex.get_meta_graph`, and :func:`~relucent.verify.certify.certify_complex`.
- :mod:`~relucent.graph.vertex_star` — recovers the full face lattice by seeding and
  verifying vertices, then expanding each verified vertex's local cubical star
  (Masden 2022, Theorem 20).
- :mod:`~relucent.graph.meta_graph` — truncation, compactification, and post-assembly audits.
  Called from :meth:`~relucent.core.complex.Complex.get_betti_numbers` and
  :mod:`relucent.topology.persistence`.
- :mod:`~relucent.graph.complex_graph` — quotient the dual graph and shrink the network when
  deleting a neuron. Called from :meth:`~relucent.core.complex.Complex.without_last_layer_neuron`.
"""
