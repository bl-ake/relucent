- Betti Number Checks
- Could be something weird in degree zero
- Rework `_shi_variable_bounds_to_try()`
- fix torch support for model.py
- project lower-dim polyhedra for SHI calculations
- Standardize code for `Polyhedron`'s property computations / accesses.
- Make it so that geometry calculation errors don't interrupt the searcher/adder.
- Make SHI propagation more efficient by only flipping SHIs
- Boundary discovery (`discover_boundary_complex`): make MIP pricing's infeasibility a real
  certificate (a box that provably meets every component; a per-unit margin instead of one
  network-wide `BOUNDARY_MIP_EPS`), then drop its experimental status.
- different CPWL activations
- maxout layers (max layers)
