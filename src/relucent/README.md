## Source Code Structure
This folder contains the implementation of the `relucent` Python package.

- **`__init__.py`**: Public API exports (what you can `from relucent import ...`)
- **`config.py`**: Global configuration and numeric defaults
- **`model.py`**: Canonical network representation (`ReLUNetwork`, `LinearLayer`, `ReLULayer`, `FlattenLayer`)
- **`convert_model.py`**: Converts PyTorch models to the canonical format (handles Conv2d, AvgPool2d, etc.)
- **`ss.py`**: Data structures for storing large numbers of sign-sequence vectors
- **`poly.py`**: Polyhedron-level computations (e.g. boundaries, neighbors, volume)
- **`complex.py`**: Polyhedral complex computations (e.g. search, connectivity/dual graph)
- **`search.py`**: Search algorithms/utilities used for exploring the complex
- **`calculations.py`**: Lower-level numeric/geometry routines used across the codebase
- **`utils.py`**: Misc helpers (env, reproducibility utilities, `mlp`, sequential splitting, etc.)
- **`vis.py`**: Visualization utilities (Plotly/matplotlib helpers, colors, plotting)