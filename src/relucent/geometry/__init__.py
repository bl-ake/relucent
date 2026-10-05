"""Polyhedron geometry: halfspaces, SHIs, boundedness, and Qhull routines."""

from .calculations import (
    adjacent_polyhedra,
    certified_bounded,
    compute_properties,
    halfspaces,
    shis,
    solve_radius,
)

__all__ = [
    "adjacent_polyhedra",
    "certified_bounded",
    "compute_properties",
    "halfspaces",
    "shis",
    "solve_radius",
]
