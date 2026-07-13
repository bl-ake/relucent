## Source code structure

The `relucent` package is organized into domain subpackages.

### Layout

| Subpackage | Modules | Role |
|------------|---------|------|
| **`core/`** | `complex`, `poly`, `ss`, `errors` | `Complex`, `Polyhedron`, sign-sequence indexing, domain exceptions |
| **`model/`** | `model`, `convert_model` | Canonical `ReLUNetwork` and PyTorch conversion |
| **`geometry/`** | `calculations` | Gurobi/Qhull/SHI routines for `Polyhedron` geometry |
| **`search/`** | `engine`, `exploration`, `worker_context`, `boundary_*` | BFS/DFS, boundary discovery, multiprocessing workers |
| **`graph/`** | `incidence`, `meta_graph`, `complex_graph` | Dual graph, meta-graph, cubical incidence |
| **`topology/`** | `betti`, `filtration`, `persistence`, `morse`, `_gf2` | Betti numbers, filtrations, persistent homology |
| **`verify/`** | `certify` | Certification and arrangement verification |
| **`vis/`** | (package `__init__`) | Plotly plotting |
| **`config/`** | (package `__init__`), `numeric_tolerances` | Tunables and automatic tolerance scaling |
| **`utils/`** | (package `__init__`) | Gurobi env, `mlp`, queues, reproducibility helpers |
| **`_internal/`** | `logging`, `torch_compat`, `network_scale` | Private implementation details |

### Public surface

- **`relucent`**: Lazy public API (`Complex`, `Polyhedron`, `mlp`, plotting helpers, …)
- **`relucent.config`**: Global configuration (`import relucent.config as cfg`)

### Import conventions

- Prefer **`Complex` methods** for common workflows (`bfs`, `get_betti_numbers`, `plot`, `certify`).
- Import from subpackages directly, e.g. `relucent.core.complex`, `relucent.graph.incidence`, `relucent.topology.betti`.
- Search workers read **`relucent.search.worker_context.get_worker_context()`**; they do not import `complex` for module globals.

Modules prefixed with `_` (or living under `_internal/`) are implementation details and are not re-exported from the top-level package.
