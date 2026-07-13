"""Vertices from :meth:`Complex.contract` expand to all maximal cells (get_ssr-style).

Relucent should satisfy the same combinatorics on vertices obtained from a complete dual
graph via repeated :meth:`~relucent.core.complex.Complex.contract`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable

import numpy as np
import pytest

from relucent import Complex, mlp, set_seeds
from relucent.config import update_settings
from relucent.search.exploration import explore_for_topology


def _ss_tuple(ss: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(ss, dtype=int).ravel())


def _sign_patterns(input_dim: int) -> list[tuple[int, ...]]:
    if input_dim <= 0:
        return [()]
    if input_dim == 1:
        return [(1,), (-1,)]
    out: list[tuple[int, ...]] = []
    for tail in _sign_patterns(input_dim - 1):
        out.append(tail + (1,))
        out.append(tail + (-1,))
    return out


def expand_vertex_to_maximal(vertex_ss: tuple[int, ...], input_dim: int) -> list[tuple[int, ...]]:
    """Mirror ``canonicalpoly2.0/polyhedra/cx.get_ssr`` on one vertex row."""
    arr = np.asarray(vertex_ss, dtype=np.int8).reshape(1, -1)
    locs = np.flatnonzero(arr.ravel() == 0)
    if len(locs) != input_dim:
        return []
    out: list[tuple[int, ...]] = []
    for sign in _sign_patterns(input_dim):
        temp = arr.copy()
        temp[0, locs] = np.asarray(sign, dtype=np.int8)
        out.append(_ss_tuple(temp))
    return out


def _populate_2d_tiny(cplx: Complex) -> None:
    for start in (
        np.array([[0.1, 0.1]]),
        np.array([[-0.5, 0.3]]),
        np.array([[0.2, -0.4]]),
    ):
        with contextlib.suppress(ValueError):
            cplx.bfs(start=start, max_polys=3000)
    explore_for_topology(cplx, np.zeros((1, 2), dtype=np.float64), max_polys=3000)


@pytest.mark.parametrize(
    ("widths", "seed", "populate"),
    [
        pytest.param([2, 3, 1], 1, "2d_tiny", id="2d_tiny"),
        pytest.param([2, 4, 4, 1], 2, "uniform", id="deep_2441_seed2"),
    ],
)
def test_contract_vertices_expand_to_maximal_cells(
    widths: list[int],
    seed: int,
    populate: str,
) -> None:
    """Each 0-cell from ``get_chain_complex`` yields ``2^d`` feasible maximal cells."""
    set_seeds(seed)
    net = mlp(widths=widths, add_last_relu=True, init="uniform")
    cplx = Complex(net)
    input_dim = widths[0]
    expected_per_vertex = 2**input_dim

    if populate == "2d_tiny":
        _populate_2d_tiny(cplx)
    else:
        explore_for_topology(cplx, np.zeros((1, input_dim), dtype=np.float64), max_polys=10000)

    top_dim = cplx.dim
    top_cells = {_ss_tuple(p.ss_np) for p in cplx if int(p.dim) == top_dim}
    chain = cplx.get_chain_complex(verbose=False)
    vertices = list(chain[-1])
    assert vertices and int(vertices[0].dim) == 0

    expanded: set[tuple[int, ...]] = set()
    for vtx in vertices:
        vertex_ss = _ss_tuple(vtx.ss_np)
        maximal = expand_vertex_to_maximal(vertex_ss, input_dim)
        assert len(maximal) == expected_per_vertex
        for ss in maximal:
            poly = cplx.ss2poly(np.asarray(ss, dtype=np.int8).reshape(1, -1), check_exists=False)
            poly.get_interior_point()
            expanded.add(ss)

    assert expanded == top_cells


def test_tolerance_sweep_does_not_revive_encoding_phantom() -> None:
    """A canonical-only phantom SS stays infeasible under loose relucent tolerances."""
    # Sign pattern for mlp([2, 4, 4, 1], seed=2): feasible in canonical get_ssr but
    # empty for relucent's ReLU sign semantics.
    phantom = (-1, -1, -1, 1, -1, -1, 1, 1, -1)
    set_seeds(2)
    cplx = Complex(mlp([2, 4, 4, 1], add_last_relu=True, init="uniform"))
    explore_for_topology(cplx, np.zeros((1, 2), dtype=np.float64), max_polys=10000)

    defaults = {
        "TOL_HALFSPACE_CONTAINMENT": 1e-6,
        "TOL_HALFSPACE_NORMAL": 1e-12,
        "TOL_DEAD_RELU": 1e-8,
        "MAX_RADIUS": 100.0,
        "TOL_SHI_HYPERPLANE": 1e-6,
    }
    sweeps: dict[str, Iterable[float]] = {
        "TOL_HALFSPACE_CONTAINMENT": (1e-6, 1e-4, 1e-2),
        "TOL_HALFSPACE_NORMAL": (1e-12, 1e-8, 1e-4),
        "TOL_DEAD_RELU": (1e-8, 1e-4, 1e-2),
        "MAX_RADIUS": (100.0, 1e6, 1e8),
        "TOL_SHI_HYPERPLANE": (1e-6, 1e-3),
    }

    for param, values in sweeps.items():
        for value in values:
            update_settings(**defaults)
            update_settings(**{param: value})
            with pytest.raises(ValueError, match="infeasible"):
                cplx.ss2poly(
                    np.asarray(phantom, dtype=np.int8).reshape(1, -1),
                    check_exists=False,
                ).get_interior_point()
    update_settings(**defaults)
