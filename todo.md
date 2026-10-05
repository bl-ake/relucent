- Betti Number Checks
- Rework `_shi_variable_bounds_to_try()`
- Make SHI propagation more efficient by only flipping SHIs
- Boundary discovery (`discover_boundary_complex`): make MIP pricing's infeasibility a real
  certificate (a box that provably meets every component; a per-unit margin instead of one
  network-wide `BOUNDARY_MIP_EPS`), then drop its experimental status.
- different CPWL activations
- maxout layers (max layers)
