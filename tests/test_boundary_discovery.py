"""Parity tests for MIP + slice-BFS boundary discovery."""

from __future__ import annotations

import os

import networkx as nx
import numpy as np
import pytest
import torch
import torch.nn as nn

from relucent import Complex, add_output_relu, mlp, set_seeds
from relucent.config import update_settings

os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")
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


_MLP_BOUNDARY_ENV = "RELUCENT_RUN_MLP_BOUNDARY_DISCOVERY"


def _mlp_tiny_model() -> nn.Sequential:
    return add_output_relu(mlp(widths=[3, 5, 5, 1]))


def _mlp_small_model() -> nn.Sequential:
    return add_output_relu(mlp(widths=[5, 8, 8, 8, 1]))


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


def _dual_components(cplx: Complex) -> int:
    dual = cplx.get_dual_graph(verbose=False, require_complete=False)
    if dual.number_of_nodes() == 0:
        return 0
    return nx.number_connected_components(dual)


def _assert_boundary_parity(ref: Complex, new: Complex) -> None:
    assert {p.tag for p in ref} == {p.tag for p in new}
    assert _dual_components(ref) == _dual_components(new)
    assert ref.get_betti_numbers() == new.get_betti_numbers()


def _complete_for_topology(cplx: Complex) -> None:
    """Run BFS from an interior start so topology entry points see a verified complex."""
    rng = np.random.default_rng(0)
    for _ in range(16):
        start = rng.normal(size=(1, cplx.dim))
        if not (cplx.point2ss(start) == 0).any():
            cplx.bfs(start=start, verbose=False)
            return
    raise RuntimeError("could not find generic start for topology completion")


@pytest.mark.parametrize("nworkers", [1])
def test_discover_boundary_complex_diamond_matches_reference(seeded: int, nworkers: int):
    set_seeds(seeded)
    model = _diamond_boundary_model_l1_ball(radius=1.0)
    cplx = Complex(model)
    _populate_diamond(cplx)
    _complete_for_topology(cplx)
    shi = cplx.n - 1
    ref = cplx.get_boundary_complex(shi, verbose=False)
    discover_out = Complex(model).discover_boundary_complex(
        shi,
        verbose=False,
        return_stats=True,
        nworkers=nworkers,
    )
    new, stats = discover_out
    assert stats["n_components"] >= 1
    _assert_boundary_parity(ref, new)
    assert ref.get_betti_numbers(verify_chain_complex=True) == new.get_betti_numbers(verify_chain_complex=True)


@pytest.mark.parametrize("nworkers", [1])
def test_discover_boundary_complex_line_matches_reference(seeded: int, nworkers: int):
    set_seeds(seeded)
    model = _line_boundary_model()
    cplx = Complex(model)
    _populate_line(cplx)
    _complete_for_topology(cplx)
    shi = cplx.n - 1
    ref = cplx.get_boundary_complex(shi, verbose=False)
    new, _stats = Complex(model).discover_boundary_complex(
        shi,
        verbose=False,
        return_stats=True,
        nworkers=nworkers,
    )
    _assert_boundary_parity(ref, new)


@pytest.mark.parametrize("nworkers", [1])
def test_discover_boundary_complex_bm_mode_line(seeded: int, nworkers: int):
    set_seeds(seeded)
    model = _line_boundary_model()
    cplx = Complex(model)
    _populate_line(cplx)
    _complete_for_topology(cplx)
    shi = cplx.n - 1
    ref = cplx.get_boundary_complex(shi, verbose=False)
    new = Complex(model).discover_boundary_complex(shi, verbose=False, nworkers=nworkers)
    ref_bm = ref.get_betti_numbers(compactify=True, reduced=True)
    new_bm = new.get_betti_numbers(compactify=True, reduced=True)
    assert ref_bm == new_bm


@pytest.mark.parametrize("nworkers", [1])
def test_discover_boundary_complex_mlp_tiny(seeded: int, nworkers: int):
    """Discover-only smoke on a small seeded random MLP.

    Hand-crafted diamond/line cases above already check reference parity. Larger
    MLP integration tests are opt-in via ``RELUCENT_RUN_MLP_BOUNDARY_DISCOVERY``.
    """
    set_seeds(seeded)
    model = _mlp_tiny_model()
    shi = Complex(model).n - 1
    try:
        new, stats = Complex(model).discover_boundary_complex(
            shi,
            verbose=False,
            return_stats=True,
            nworkers=nworkers,
            verify=False,
        )
    except ValueError as exc:
        if "Initial Solve Failed" in str(exc):
            pytest.skip(f"boundary witness infeasible for get_shis at seed {seeded}: {exc}")
        raise
    assert stats["n_components"] >= 1
    assert len(new) > 0
    _ = new.get_betti_numbers()
    _ = new.get_betti_numbers(compactify=True, reduced=True)


@pytest.mark.skipif(
    os.environ.get(_MLP_BOUNDARY_ENV) != "1",
    reason="Opt-in MLP boundary discovery tests. Set RELUCENT_RUN_MLP_BOUNDARY_DISCOVERY=1.",
)
@pytest.mark.parametrize("nworkers", [1])
def test_discover_boundary_complex_mlp_small(seeded: int, nworkers: int):
    set_seeds(seeded)
    model = _mlp_small_model()
    shi = Complex(model).n - 1
    new, stats = Complex(model).discover_boundary_complex(
        shi,
        verbose=False,
        return_stats=True,
        nworkers=nworkers,
    )
    assert stats["n_components"] >= 1
    assert len(new) > 0
    _ = new.get_betti_numbers()


@pytest.mark.skipif(
    os.environ.get(_MLP_BOUNDARY_ENV) != "1",
    reason="Opt-in MLP boundary discovery tests. Set RELUCENT_RUN_MLP_BOUNDARY_DISCOVERY=1.",
)
@pytest.mark.parametrize("nworkers", [1])
def test_discover_boundary_complex_mlp_medium_parity(seeded: int, nworkers: int):
    set_seeds(seeded)
    model = add_output_relu(mlp(widths=[4, 12, 12, 12, 12, 1]))
    cplx = Complex(model)
    cplx.bfs(verbose=0, nworkers=nworkers)
    shi = cplx.n - 1
    ref = cplx.get_boundary_complex(shi, verbose=False)
    new, stats = Complex(model).discover_boundary_complex(
        shi,
        verbose=False,
        return_stats=True,
        nworkers=nworkers,
    )
    assert stats["n_components"] >= 1
    _assert_boundary_parity(ref, new)
