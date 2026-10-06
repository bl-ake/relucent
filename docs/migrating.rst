Stability and migrating to 1.0
==============================

API stability
-------------

From 1.0, relucent follows `semantic versioning <https://semver.org>`_: a release that
breaks the public API bumps the major version. The public API is

* every name in ``relucent.__all__``, and
* every name in the ``__all__`` of a module documented in the API reference,

with their documented arguments and behavior. A subpackage's ``__init__`` (such as
``relucent.search`` or ``relucent.core``) re-exports names for convenience. A re-exported name
is public only if the module that defines it is documented. These are excluded and may change
in any release:

* ``relucent._internal`` and any name that starts with an underscore;
* the solver-tuning settings in :mod:`relucent.config.advanced`;
* boundary discovery (:meth:`~relucent.core.complex.Complex.discover_boundary_complex`,
  :mod:`relucent.search.boundary_search`, :mod:`relucent.search.boundary_mip`), which is
  experimental (see :doc:`boundary_search`);
* the layout of files written by :meth:`~relucent.core.complex.Complex.save`, beyond the
  promise that :meth:`~relucent.core.complex.Complex.load` reads files from the same or an
  older 1.x release. Saved files are pickles that name relucent's classes by module path.

Renamed in 1.0
--------------

Methods and functions that return a computed result are named for the result; actions keep
verb names (``bfs``, ``certify``, ``plot``, ``slice_affine``,
``compute_geometric_properties``).

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 0.9
     - 1.0
   * - ``Complex.get_dual_graph()``
     - ``Complex.dual_graph()``
   * - ``Complex.get_meta_graph()``
     - ``Complex.meta_graph()``
   * - ``Complex.get_chain_complex()``
     - ``Complex.chain_complex()``
   * - ``Complex.get_betti_numbers()``
     - ``Complex.betti_numbers()``
   * - ``Complex.get_persistent_homology()``
     - ``Complex.persistent_homology()``
   * - ``Complex.get_critical_points()``
     - ``Complex.critical_points()``
   * - ``Complex.get_boundary_cells(i)``
     - ``Complex.boundary_cells(i)``
   * - ``Complex.get_boundary_complex(i)``
     - ``Complex.boundary_complex(i)``
   * - ``Complex.get_betti_numbers_from_meta(meta)``
     - ``relucent.topology.betti_numbers(meta, compactify=...)``
   * - ``relucent.topology.get_betti_numbers``
     - ``relucent.topology.betti_numbers``
   * - ``Polyhedron.get_neighbor(shi)``
     - ``Polyhedron.neighbor(shi)``
   * - ``Polyhedron.get_face(shi)``, ``Polyhedron.get_face_by_shis(shis)``
     - ``Polyhedron.face(shis)`` (one index or several)
   * - ``Polyhedron.get_bounded_halfspaces(bound)``
     - ``Polyhedron.bounded_halfspaces(bound)``
   * - ``Polyhedron.get_bounded_vertices(bound)``
     - ``Polyhedron.bounded_vertices(bound)``
   * - ``Polyhedron.get_interior_point()``
     - ``Polyhedron.find_interior_point()`` (or the cached ``interior_point`` property)
   * - ``Polyhedron.get_center_inradius()``
     - the ``center`` and ``inradius`` properties
   * - ``Polyhedron.get_geometry(props)``
     - ``Polyhedron.compute_geometric_properties(props)``
   * - ``Polyhedron.hs``, ``Polyhedron.ch``
     - ``Polyhedron.halfspace_intersection``, ``Polyhedron.convex_hull`` (also as property
       names passed to ``compute_geometric_properties`` and the search functions)
   * - ``Polyhedron.plot_cells()``, ``Polyhedron.plot_graph()``
     - ``Polyhedron.plot(plot_mode="cells" | "graph")``
   * - ``relucent.geometry.get_hs(poly, get_all_Ab=...)``
     - ``relucent.geometry.halfspaces(poly, per_layer=...)``
   * - ``relucent.geometry.get_shis(poly)``
     - ``relucent.geometry.shis(poly)``
   * - ``ReLUNetwork.get_all_layer_outputs(data)``
     - ``ReLUNetwork.all_layer_outputs(data)``
   * - ``relucent.topology.morse.get_layer_jacobians(net, ss)``
     - ``relucent.topology.morse.layer_jacobians(net, ss)``
   * - ``relucent.get_colors(data)``
     - ``relucent.colors(data)``
   * - ``compactify=False`` / ``True``
     - ``compactify="truncate"`` / ``"borel_moore"`` (``"one_point"`` is unchanged)
   * - ``relucent.utils``
     - split up: ``encode_ss`` and ``flip_ss_at_shi`` are in :mod:`relucent.core.ss`; ``mlp``,
       ``torch_mlp``, ``set_seeds``, ``split_sequential``, ``add_output_relu`` are in
       :mod:`relucent.model.builders` (and at the top level, as before)

Other changes
-------------

* **Search results.** ``bfs``, ``dfs``, ``random_walk`` and ``searcher`` return a
  :class:`~relucent.search.exploration.SearchResult` dataclass instead of a dict keyed by
  display strings, and take their options as named, keyword-only arguments.
* **Verbosity.** Every long-running function takes ``verbose: int | None`` (0 quiet, 1
  progress bars and summaries, 2 debug detail; ``None`` uses ``config.VERBOSE``). This
  replaces ``verbose: bool`` flags and ``hamming_astar(show_pbar=)``.
* **Configuration.** ``import relucent`` and ``Complex(net)`` no longer write tolerances into
  the global config. ``relucent.config.numeric_tolerances``, ``Complex(auto_tolerances=)`` and
  ``RELUCENT_SKIP_NUMERIC_BOOTSTRAP`` are gone: geometric decisions are certified against
  each row's float64 error instead. Solver-tuning settings moved to
  :mod:`relucent.config.advanced`. ``BOUNDARY_MIP_CUT_ORDER`` and
  ``BOUNDARY_MIP_BULK_NOGOOD_EMIT`` are read from ``RELUCENT_``-prefixed environment
  variables, like every other setting.
* **Networks.** ``Complex.net`` is the converted
  :class:`~relucent.model.model.ReLUNetwork`; the model you passed is ``Complex.source_model``.
  ``mlp()`` always returns a NumPy ``ReLUNetwork``; ``torch_mlp()`` builds the PyTorch module.
* **Conversion.** :func:`~relucent.model.convert_model.convert` checks every ``torch.nn.Module`` against its own
  forward pass and raises if the result differs, so a ``forward`` with a skip connection is
  refused instead of converted to a different function. ``Conv2d`` with dilation, groups,
  a non-zero padding mode or ``padding="same"`` raises instead of converting wrongly.
* **Boundedness.** ``Polyhedron.finite`` is decided from the recession cone. In 0.9 a cell
  with a lower-dimensional recession cone (a half-infinite prism) was reported bounded;
  ``center`` and ``inradius`` are still the Chebyshev values, which can be finite for an
  unbounded cell. ``Polyhedron.feasible`` no longer needs ``finite``.
* **Save files.** :meth:`~relucent.core.complex.Complex.save` records
  ``complete``/``verified``, so a loaded complex is ready for topology without
  ``set_exploration_state``. Files from 0.9 still load, without that state.
* **Polyhedron construction.** Everything after ``ss`` is keyword-only, and unknown keywords
  raise ``TypeError``. Sign sequences are stored 1-D, and ``Polyhedron.ss`` is read-only (it is
  the cell's identity).
* **Polyhedron properties.** Every computed property is cached the same way, and an empty cell
  answers ``None``: ``interior_point``, ``interior_point_norm``, ``center``, ``inradius``,
  ``vertices``, ``halfspace_intersection``, ``convex_hull`` and ``volume``. ``volume`` is
  ``None`` (was ``-1``) when the cell is empty or Qhull fails, and ``inf`` only for an unbounded
  cell. ``halfspace_intersection`` returns ``None`` instead of raising when Qhull gives none.
  ``compute_geometric_properties`` raises ``ValueError`` for a name outside
  ``Polyhedron.GEOMETRY_PROPERTIES`` (``"halfspaces_np"`` is no longer accepted; use
  ``"halfspaces"``).
* **Removed.** ``Polyhedron.num_dead_relus``, ``Polyhedron.num_faces`` (use ``num_shis``),
  ``Polyhedron.hyperplanes`` (use ``equalities``), the ``strict`` and ``new_method`` options
  of the SHI computation, ``Complex.partial_derivative_on_1cell`` (use
  :func:`relucent.topology.morse.partial_derivative_on_1cell`), ``Complex.get_boundary_edges``
  and the ``Complex`` wrappers around meta-graph helpers (use :mod:`relucent.graph.meta_graph`).
* **Dependencies.** pandas, scikit-learn, Pillow, matplotlib, pyvis and kaleido are no longer
  installed. PyTorch is optional (``pip install "relucent[torch]"``).
