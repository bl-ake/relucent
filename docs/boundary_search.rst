:tocdepth: 0

Slices and Boundary Components
==============================

A search with ``fixed`` indices stays on the slice where those neurons are zero: one neuron's bent
hyperplane, or the set where several bent hyperplanes cross.
:meth:`Complex.bfs(start, fixed=[i, j, ...]) <relucent.core.complex.Complex.bfs>` (and
:meth:`~relucent.core.complex.Complex.searcher`, ``dfs``, ``random_walk``) never crosses the fixed
hyperplanes, keeps only real cells of the slice (every sign choice on the fixed entries gives a
nonempty cell), and certifies the result like any other complex. ``start`` is a Polyhedron or sign
sequence that is zero exactly at the fixed indices. Its cost grows with the component, not with
the input space.

:meth:`Complex.boundary_component(i, start) <relucent.core.complex.Complex.boundary_component>`
is the one-neuron case: a new complex holding the component of neuron ``i``'s bent hyperplane that
contains ``start``.

A slice search finds only the connected component that contains ``start`` and says nothing about
whether other components exist. Finding a starting cell is left to the caller. For a whole bent
hyperplane, explore the input space with :meth:`~relucent.core.complex.Complex.bfs` and take
:meth:`~relucent.core.complex.Complex.boundary_complex`. Fixing a nonzero sign (staying on one side
of a hyperplane) is not supported.

.. automodule:: relucent.search.boundary_search
   :members:
   :show-inheritance:
