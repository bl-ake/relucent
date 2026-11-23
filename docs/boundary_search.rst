:tocdepth: 0

Boundary Components
===================

:meth:`Complex.boundary_component(i, start) <relucent.core.complex.Complex.boundary_component>`
explores one connected component of neuron ``i``'s bent hyperplane: a slice-restricted BFS over
cells with ``ss[i] = 0``, starting from a top cell you supply, followed by the same certification
as any other complex. Its cost grows with the component, not with the input space.

It finds only the component that contains ``start`` and says nothing about whether other
components exist. Finding a starting cell is left to the caller. For the whole hyperplane, explore
the input space with :meth:`~relucent.core.complex.Complex.bfs` and take
:meth:`~relucent.core.complex.Complex.boundary_complex`.

.. automodule:: relucent.search.boundary_search
   :members:
   :show-inheritance:
