## Source code structure

This folder contains the implementation of the `relucent` Python package.

### Public surface

- **`__init__.py`**: Lazy public API (`Complex`, `Polyhedron`, `mlp`, plotting helpers, …)
- **`config.py`**: Global configuration and numeric defaults (`import relucent.config as cfg`)

### Core data model

- **`model.py`**: Canonical network representation (`ReLUNetwork`, `LinearLayer`, `ReLULayer`, `FlattenLayer`)
- **`convert_model.py`**: Converts PyTorch models to the canonical format (Conv2d, AvgPool2d, etc.)
- **`ss.py`**: Sign-sequence indexing (`SSManager`, `encode_ss`)
- **`poly.py`**: `Polyhedron` — cell identity, geometry caches, and user-facing region API
- **`complex.py`**: `Complex` — container, search orchestration, dual/chain/meta-graph entry points, Betti/PH wrappers, save/load

### Behavior modules (called by `Polyhedron` / `Complex`)

- **`calculations.py`**: Gurobi/Qhull/SHI routines that take a `Polyhedron` (imported eagerly from `poly.py`)
- **`search.py`**: BFS/DFS/random-walk, `parallel_add`, A* pathfinding, worker entry points
- **`worker_context.py`**: Multiprocessing worker state (`set_worker_context`, `get_worker_context`)
- **`complex_graph.py`**: Dual-graph contraction and network surgery helpers
- **`meta_graph.py`**: Meta-graph face discovery, boundedness classification, truncation/compactification
- **`topology.py`**: Betti numbers over GF(2) from meta-graph boundary maps
- **`filtration.py`**: Filtration values on meta-graph cells
- **`persistence.py`**: Persistent homology (column reduction on filtration boundary matrix)
- **`vis.py`**: Plotly plotting (`plot_complex`, `plot_polyhedron`, …)

### Utilities and internals

- **`utils.py`**: Gurobi env, `mlp`, queues, reproducibility helpers
- **`_torch_compat.py`**, **`_logging.py`**, **`_gf2.py`** / **`_gf2_rank.c`**: optional torch shim, logging, GF(2) rank backend

Modules prefixed with `_` are implementation details and are not re-exported from the top-level package.

### Import conventions

- Prefer **`Complex` methods** for common workflows (`bfs`, `get_betti_numbers`, `plot`).
- Lower-level submodules (`relucent.topology`, `relucent.meta_graph`, `relucent.filtration`, …) are importable for research tooling but not part of the lazy top-level API.
- Search workers read **`worker_context.get_worker_context()`**; they do not import `complex` for module globals.
