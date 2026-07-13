Exploration and Verification
============================

After BFS (or related search), relucent records whether exploration finished and
whether invariant checks passed. This page describes those flags, the default
``verify`` behavior, and how top-cell SHI lists relate to the dual graph.

Exploration flags
-----------------

:class:`~relucent.core.complex.Complex` exposes two read-only properties:

* ``complete`` — ``True`` when search exhausted the frontier without hitting an
  intentional cap (``max_polys``) or a depth limit (``max_depth``). ``False`` for
  partial exploration. ``None`` if no search has recorded state yet.
* ``verified`` — ``True`` when the complex passed the last invariant verification.
  ``False`` when verification was skipped or failed. ``None`` if unknown.

Use :meth:`~relucent.core.complex.Complex.set_exploration_state` only from library
internals; user code should rely on search and certification routines to set these flags.

Default verify behavior
-----------------------

:meth:`~relucent.core.complex.Complex.bfs` and :meth:`~relucent.core.complex.Complex.searcher`
accept ``verify=True`` by default. After a **complete** search, relucent:

1. Rebuilds combinatorial dual-graph edges and syncs top-cell ``_shis`` from them
   (:func:`~relucent.search.exploration.finalize_ambient_search` via
   :meth:`~relucent.core.complex.Complex.get_dual_graph`).
2. Runs certification (:func:`~relucent.verify.certify.certify_complex` at
   :class:`~relucent.verify.certify.CertifyLevel.COMPLETE`).

Verification is **skipped** when exploration hits ``max_polys`` before the
frontier empties — the complex may still have undiscovered neighbors and an LP
completeness check would false-fail.

A finite ``max_depth`` cap can leave ``complete=False`` even when the queue is
empty (neighbors beyond the depth limit were not queued). With ``verify=True``,
that raises :class:`~relucent.core.complex.IncompleteDualGraphError` unless
``max_polys`` was hit.

For intentional partial exploration, pass ``verify=False`` or accept
``complete=False``.

Examples
~~~~~~~~

Complete ambient search (default):

.. code-block:: python

   import relucent
   import numpy as np

   cplx = relucent.Complex(relucent.mlp(widths=[2, 4, 1]))
   cplx.bfs(start=np.zeros((1, 2)))
   assert cplx.complete is True
   assert cplx.verified is True

Capped search (verification skipped if cap is hit):

.. code-block:: python

   info = cplx.bfs(max_polys=100)
   # complete is False if the cap was hit; verified is False when verify was skipped

Partial exploration by choice:

.. code-block:: python

   cplx.bfs(max_polys=50, verify=False)
   # complete may be False; no IncompleteDualGraphError

Tests often use :func:`~relucent.search.exploration.explore_for_topology`, which runs BFS
and requires ``complete`` and ``verified`` to both be ``True``.

Topology prerequisites
----------------------

:meth:`~relucent.core.complex.Complex.contract` and
:meth:`~relucent.core.complex.Complex.get_boundary_complex` call
:meth:`~relucent.core.complex.Complex.assert_topology_ready`, which requires
``complete=True`` and ``verified=True``. Run BFS or
:func:`~relucent.search.exploration.explore_for_topology` first. For trusted loads
(deserialized complexes), call
:meth:`~relucent.core.complex.Complex.set_exploration_state` explicitly.

:meth:`~relucent.core.complex.Complex.get_betti_numbers` does **not** require
``assert_topology_ready`` — it can run on partial complexes, but results may be
wrong if neighbors are missing.

Boundary discovery
------------------

Two paths build a decision-boundary complex:

* **Full ambient complex first** — explore the input space (BFS or
  :func:`~relucent.search.exploration.explore_for_topology`), then
  :meth:`~relucent.core.complex.Complex.get_boundary_complex(i)` contracts faces on
  neuron ``i``. Requires ``assert_topology_ready`` (complete and verified).
* **Direct boundary discovery** — :meth:`~relucent.core.complex.Complex.discover_boundary_complex(i)`
  uses MIP pricing plus slice-restricted BFS per connected component, then
  :func:`~relucent.search.exploration.finalize_boundary_complex` for slice SHI assignment,
  dual graph, and verification. Does not require a full ambient BFS first.

Dual-graph SHI model
--------------------

During search, ``Polyhedron._shis`` is a frontier heuristic from LP facet solves.
At finalize on a **complete** ambient search, top-cell ``_shis`` are **re-derived**
from combinatorial dual-graph edges.

On **contracted** slices (boundary complexes, chain-complex steps),
:func:`~relucent.graph.incidence.set_contracted_shis` sets ``_shis`` to
:func:`~relucent.graph.incidence.cubical_cell_shis` (flip neighbors in the slice).
Contracted 1-skeleton dual graphs walk each cell's finalized ``poly.shis`` lists.

You can re-run certification manually with :meth:`~relucent.core.complex.Complex.certify`.

See also :doc:`topology` for Betti-number prerequisites. For the full search →
SHI → dual/meta-graph pipeline, see :doc:`search_shi_and_graphs`. The markdown
file ``docs/betti_computation.md`` walks through the homology pipeline in more
detail.
