"""Polyhedron geometry: halfspaces, SHIs, and Qhull routines."""

from .calculations import (
    adjacent_polyhedra,
    compute_properties,
    get_hs,
    get_shis,
    solve_radius,
)

__all__ = [
    "adjacent_polyhedra",
    "compute_properties",
    "get_hs",
    "get_shis",
    "solve_radius",
]
