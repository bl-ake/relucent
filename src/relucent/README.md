## Source code structure

The `relucent` package is organized into domain subpackages.

### Layout

| Subpackage | Modules | Role |
|------------|---------|------|
| **`core/`** | `complex`, `poly`, `ss`, `errors` | `Complex`, `Polyhedron`, sign-sequence indexing, domain exceptions |
| **`model/`** | `model`, `convert_model` | Canonical `ReLUNetwork` and PyTorch conversion |
| **`geometry/`** | `calculations`, `slicing` | Gurobi/Qhull/SHI routines for `Polyhedron` geometry; affine slices of a complex |
| **`search/`** | `engine`, `exploration`, `worker_context`, `boundary_search`, `boundary_mip`, `boundary_exclusion_trie` | BFS/DFS, boundary discovery, multiprocessing workers |
| **`graph/`** | `incidence`, `vertex_star`, `meta_graph`, `boundary`, `complex_graph` | Dual graph, chain complex from vertex stars, meta-graph, one neuron's boundary cells, dual-graph recovery and neuron deletion |
| **`topology/`** | `betti`, `filtration`, `persistence`, `morse`, `_gf2` | Betti numbers, filtrations, persistent homology, PL Morse critical points |
| **`verify/`** | `certify` | Certification and arrangement verification |
| **`vis/`** | (package `__init__`) | Plotly plotting |
| **`config/`** | (package `__init__`), `advanced` | Settings; `advanced` holds unstable solver-tuning knobs |
| **`utils/`** | (package `__init__`) | Gurobi env, `mlp`, queues, reproducibility helpers |
| **`_internal/`** | `logging`, `torch_compat`, `network_scale`, `rounding`, `exact` | Private implementation details: float64 error bounds and exact rational fallbacks for geometric decisions |

### Public surface

- **`relucent`**: Lazy public API (`Complex`, `Polyhedron`, `SearchResult`, errors, `mlp`, plotting helpers, …)
- **`relucent.topology`**: Betti numbers, filtrations, persistent homology
- **`relucent.config`**: Global configuration (`import relucent.config as cfg`); `relucent.config.advanced` for unstable solver knobs

### Import conventions

- Prefer **`Complex` methods** for common workflows (`bfs`, `get_betti_numbers`, `plot`, `certify`). They are thin wrappers; the algorithms live in the subpackage modules (e.g. `graph.meta_graph.build_meta_graph`, `graph.vertex_star.build_chain_complex`), which take the complex as their first argument.
- Import from subpackages directly, e.g. `relucent.core.complex`, `relucent.graph.incidence`, `relucent.topology.betti`.
- Search workers read **`relucent.search.worker_context.get_worker_context()`**; they do not import `complex` for module globals.

Modules prefixed with `_` (or living under `_internal/`) are implementation details and are not re-exported from the top-level package.
