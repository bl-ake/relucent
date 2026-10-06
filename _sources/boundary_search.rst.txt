:tocdepth: 0

Boundary Discovery
==================

.. warning::

   Boundary discovery is **experimental** and outside relucent's stability guarantees.
   Certification covers only the components it finds. MIP pricing can miss a component
   that lies outside its input box (``estimate_input_bound``, which shrinks as the weights
   shrink while the boundary moves outward) or on which some other unit stays within
   ``BOUNDARY_MIP_EPS`` of zero (for example a dead unit under weight decay), and the
   result is still reported as complete. For a verified boundary, explore the input space
   with :meth:`~relucent.core.complex.Complex.bfs` and take
   :meth:`~relucent.core.complex.Complex.boundary_complex`.

.. automodule:: relucent.search.boundary_search
   :members:
   :show-inheritance:

.. automodule:: relucent.search.boundary_mip
   :members:
   :show-inheritance:

.. automodule:: relucent.search.boundary_exclusion_trie
   :members:
   :show-inheritance:
