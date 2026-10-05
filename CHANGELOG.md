# Changelog

relucent follows [semantic versioning](https://semver.org) from 1.0. What counts as public
API is listed in [Stability and migrating to 1.0](https://bl-ake.github.io/relucent/migrating.html).

## 1.0.0

First stable release. Code written for 0.9 needs changes; the migration page has a table of
every rename.

### Breaking

- Methods that return a computed result are named for it: `Complex.dual_graph()`,
  `meta_graph()`, `chain_complex()`, `betti_numbers()`, `persistent_homology()`,
  `critical_points()`, `boundary_cells()`, `boundary_complex()`;
  `Polyhedron.neighbor()`, `face()`, `bounded_halfspaces()`, `bounded_vertices()`,
  `find_interior_point()`, `compute_geometric_properties()`;
  `relucent.geometry.halfspaces()` and `shis()` (all were `get_*`).
- Search functions return a `SearchResult` dataclass and take keyword-only options.
- `compactify` takes `"truncate"`, `"borel_moore"` or `"one_point"`.
- One verbosity convention: `verbose: int | None` everywhere.
- `relucent.utils` is split into `relucent.core.ss`, `relucent.model.builders` and private
  modules; `mlp()` always returns a NumPy `ReLUNetwork` (`torch_mlp()` builds PyTorch).
- Configuration is no longer rewritten by `import relucent` or `Complex(net)`;
  `config.numeric_tolerances` and `Complex(auto_tolerances=)` are removed, and solver tuning
  moved to `relucent.config.advanced` (unstable).
- `Polyhedron.hs` / `ch` are `halfspace_intersection` / `convex_hull`, and `plot_cells()` /
  `plot_graph()` are `plot(plot_mode=...)`.
- Removed: `Polyhedron.num_dead_relus`, `num_faces` (use `num_shis`), `hyperplanes` (use
  `equalities`); the SHI options `strict` and `new_method`.
- Dropped dependencies: pandas, scikit-learn, Pillow, matplotlib, pyvis, kaleido. PyTorch is
  an optional extra.

### Fixed

- `Polyhedron.finite` reported cells with a lower-dimensional recession cone (half-infinite
  prisms) as bounded, and `vertices`/`volume` followed. It is now decided from the recession
  cone, certified in float64 or decided exactly.
- `convert()` failed on every convolutional `nn.Sequential` with a `Flatten`, and silently
  converted a `forward` with a skip connection into a different function. It now checks
  every `nn.Module` against its forward pass, and raises for `Conv2d` dilation, groups,
  padding modes and `padding="same"`. It no longer draws from the global torch RNG.
- `Complex.save`/`load` keep `complete`/`verified`, so a loaded complex runs topology
  directly. Files carry a format version.
- `contract()` returned the 1-cells instead of the codimension-one cells above 2D.
- The GF(2) C backend is cached per CPU in a user cache directory and built atomically.

### Changed

- Boundary discovery (`discover_boundary_complex`) is marked experimental: it can miss
  boundary components and still report the result complete. Use `bfs()` and
  `boundary_complex(i)` for a verified boundary.
- Searching no longer computes `finite` (only the Chebyshev LP it needs).
- Dependency floors are lowered from the newest releases to numpy 2.0, scipy 1.13,
  networkx 3.0 and tqdm 4.60 (plotly 5.20 and gurobipy 12 as before). A CI job runs the
  tests at exactly these versions on Python 3.11.
- The wheel and sdist include the license text.
