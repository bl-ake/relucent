"""Complex.boundary_component: one component of a bent hyperplane, by slice BFS."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch
import torch.nn as nn

from relucent import Complex, add_output_relu, set_seeds, torch_mlp
from relucent.config import update_settings
from relucent.search.exploration import explore_for_topology
from tests.helpers import cover_by_components, summed_betti

update_settings(VERBOSE=0)


def _add_points(cplx: Complex, pts: np.ndarray) -> None:
    for x in np.asarray(pts, dtype=np.float64):
        ss = cplx.point2ss(x.reshape(1, -1))
        if (np.asarray(ss) == 0).any():
            continue
        cplx.add_point(x.reshape(1, -1), check_exists=True)


def _diamond_boundary_model_l1_ball(radius: float = 1.0) -> nn.Sequential:
    fc0 = nn.Linear(2, 6, bias=False, dtype=torch.float64)
    base = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    fc0.weight.data[:] = base + 1e-3 * torch.randn_like(base)

    fc1 = nn.Linear(6, 2, bias=False, dtype=torch.float64)
    fc1.weight.data[:] = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    fc2 = nn.Linear(2, 1, bias=True, dtype=torch.float64)
    fc2.weight.data[:] = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    fc2.bias.data[:] = torch.tensor([-float(radius)], dtype=torch.float64)

    return nn.Sequential(fc0, nn.ReLU(), fc1, fc2, nn.ReLU())


def _line_boundary_model() -> nn.Sequential:
    fc = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    return nn.Sequential(fc, nn.ReLU())


def _mlp_tiny_model() -> nn.Sequential:
    return add_output_relu(torch_mlp(widths=[3, 5, 5, 1]))


def _populate_diamond(cplx: Complex) -> None:
    thetas = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    _add_points(cplx, np.vstack([0.9 * dirs, 1.1 * dirs, np.random.randn(200, 2)]))


def _populate_line(cplx: Complex) -> None:
    xs = np.linspace(-2.0, 2.0, 25)
    ys = np.linspace(-2.0, 2.0, 25)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    eps = 1e-2
    left = grid.copy()
    left[:, 0] = -eps
    right = grid.copy()
    right[:, 0] = eps
    _add_points(cplx, np.vstack([left, right, np.random.randn(200, 2)]))


def _nonzero(betti: dict[int, int]) -> dict[int, int]:
    return {int(k): int(v) for k, v in betti.items() if v}


def _reference(model: nn.Sequential, populate, start: np.ndarray) -> tuple[Complex, int]:
    cplx = Complex(model)
    populate(cplx)
    explore_for_topology(cplx, start)
    shi = cplx.n - 1
    return cplx.boundary_complex(shi, verbose=False), shi


@pytest.mark.parametrize("nworkers", [1])
def test_boundary_component_diamond_is_the_whole_boundary(seeded: int, nworkers: int):
    set_seeds(seeded)
    model = _diamond_boundary_model_l1_ball(radius=1.0)
    ref, shi = _reference(model, _populate_diamond, np.array([0.1, 0.2]))
    (comp,) = cover_by_components(ref, model, shi, nworkers=nworkers)
    assert {p.tag for p in comp} == {p.tag for p in ref}
    assert ref.betti_numbers(verify_chain_complex=True) == comp.betti_numbers(verify_chain_complex=True)


@pytest.mark.parametrize("nworkers", [1])
def test_boundary_component_line(seeded: int, nworkers: int):
    set_seeds(seeded)
    model = _line_boundary_model()
    ref, shi = _reference(model, _populate_line, np.array([0.5, 0.0]))
    (comp,) = cover_by_components(ref, model, shi, nworkers=nworkers)
    assert ref.betti_numbers() == comp.betti_numbers()
    ref_bm = ref.betti_numbers(compactify="borel_moore", reduced=True)
    assert ref_bm == comp.betti_numbers(compactify="borel_moore", reduced=True)


@pytest.mark.parametrize("nworkers", [1])
def test_boundary_components_cover_mlp_tiny(seeded: int, nworkers: int):
    """Components started from uncovered cells tile the full boundary, and their Betti numbers add up."""
    set_seeds(seeded)
    model = _mlp_tiny_model()
    cplx = Complex(model)
    explore_for_topology(cplx)
    shi = cplx.n - 1
    ref = cplx.boundary_complex(shi, verbose=False)
    if len(ref) == 0:
        pytest.skip(f"no boundary at seed {seeded}")
    components = cover_by_components(ref, model, shi, nworkers=nworkers)
    assert summed_betti(components) == _nonzero(ref.betti_numbers())


def test_boundary_component_rejects_a_start_off_the_hyperplane(seeded: int):
    set_seeds(seeded)
    model = _line_boundary_model()
    cplx = Complex(model)
    shi = cplx.n - 1
    with pytest.raises(ValueError, match="only zero"):
        cplx.boundary_component(shi, np.array([1], dtype=np.int8), verbose=0)


def test_boundary_component_rejects_a_start_that_is_not_a_cell(seeded: int):
    """A sign sequence with the right zero but no ambient cofaces on both sides is not a start."""
    set_seeds(seeded)
    fc0 = nn.Linear(2, 2, bias=True, dtype=torch.float64)
    fc0.weight.data[:] = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    fc0.bias.data[:] = torch.tensor([0.0, -1.0], dtype=torch.float64)
    model = nn.Sequential(fc0, nn.ReLU())
    cplx = Complex(model)
    # x = 0 (unit 0's hyperplane) lies where unit 1 is negative, never positive.
    with pytest.raises(ValueError, match="not a cell"):
        cplx.boundary_component(0, np.array([0, 1], dtype=np.int8), verbose=0)


def test_bfs_fixed_matches_boundary_component(seeded: int):
    set_seeds(seeded)
    model = _diamond_boundary_model_l1_ball(radius=1.0)
    ref, shi = _reference(model, _populate_diamond, np.array([0.1, 0.2]))
    start = min(ref, key=lambda p: p.tag).ss_np
    via_bfs = Complex(model)
    result = via_bfs.bfs(start, fixed=[shi], nworkers=1, verbose=0)
    assert result.complete and via_bfs.verified
    via_component = Complex(model).boundary_component(shi, start, nworkers=1, verbose=0)
    assert {p.tag for p in via_bfs} == {p.tag for p in via_component}


def test_searcher_fixed_two_neurons_tiles_their_intersection(seeded: int):
    """Holding two neurons at zero explores where their bent hyperplanes cross (curves in d = 3)."""
    set_seeds(seeded)
    model = add_output_relu(torch_mlp(widths=[3, 5, 5, 1]))
    cplx = Complex(model)
    explore_for_topology(cplx)
    edges = cplx.chain_complex()[2]  # the 1-cells
    zero_sets = Counter(tuple(np.flatnonzero(np.asarray(p.ss_np).ravel() == 0).tolist()) for p in edges)
    fixed = max((z for z in zero_sets if len(z) == 2), key=lambda z: (zero_sets[z], z))
    ref = {p.tag: p for p in edges if tuple(np.flatnonzero(np.asarray(p.ss_np).ravel() == 0).tolist()) == fixed}
    remaining, covered = dict(ref), set()
    while remaining:
        start = min(remaining)
        comp = Complex(model)
        result = comp.bfs(remaining[start].ss_np, fixed=fixed, nworkers=1, verbose=0)
        tags = {p.tag for p in comp}
        assert result.complete and comp.verified
        assert start in tags and tags <= set(ref) and not tags & covered
        covered |= tags
        for t in tags:
            remaining.pop(t)
    assert covered == set(ref)


def test_searcher_rejects_a_nonzero_fixed_sign(seeded: int):
    set_seeds(seeded)
    model = _diamond_boundary_model_l1_ball(radius=1.0)
    ref, shi = _reference(model, _populate_diamond, np.array([0.1, 0.2]))
    start = min(ref, key=lambda p: p.tag).ss_np
    nonzero = int(np.flatnonzero(np.asarray(start).ravel() != 0)[0])
    with pytest.raises(ValueError, match="nonzero sign is not supported"):
        Complex(model).bfs(start, fixed=[shi, nonzero], nworkers=1, verbose=0)
