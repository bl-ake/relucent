## Source Code Structure
This folder contains the implementation of the `relucent` Python package.

- **`__init__.py`**: Public API exports (what you can `from relucent import ...`)
- **`config.py`**: Global configuration and numeric defaults
- **`model.py`**: `torch.nn.Module` wrappers/utilities (e.g. `NN`, `get_mlp_model`)
- **`convert_model.py`**: Utilities to convert supported `torch.nn` layers to linear layers
- **`ss.py`**: Data structures for storing large numbers of sign-sequence vectors
- **`poly.py`**: Polyhedron-level computations (e.g. boundaries, neighbors, volume)
- **`complex.py`**: Polyhedral complex computations (e.g. search, connectivity/dual graph)
- **`search.py`**: Search algorithms/utilities used for exploring the complex
- **`calculations.py`**: Lower-level numeric/geometry routines used across the codebase
- **`utils.py`**: Misc helpers (env, reproducibility utilities, sequential splitting, etc.)
- **`vis.py`**: Visualization utilities (Plotly/matplotlib helpers, colors, plotting)