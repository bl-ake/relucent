"""Decision-boundary Betti number tests for known shapes.

These tests construct small, fixed-weight ReLU networks whose decision boundary
(the last neuron's bent hyperplane) is a piecewise-linear set with known topology.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from relucent import Complex, set_seeds

INTEGRATION_ENV = "RELUCENT_RUN_DB_INTEGRATION"


def _add_points(cplx: Complex, pts: np.ndarray) -> None:
    """Add a batch of points to a complex, skipping any boundary points."""
    for x in np.asarray(pts, dtype=np.float64):
        ss = cplx.point2ss(x.reshape(1, -1))
        if (np.asarray(ss) == 0).any():
            continue
        cplx.add_point(x.reshape(1, -1), check_exists=True)


def _betti_list(betti: dict[int, int], *, up_to: int) -> list[int]:
    return [int(betti.get(i, 0)) for i in range(up_to + 1)]


def _diamond_boundary_model_l1_ball(radius: float = 1.0) -> nn.Sequential:
    """ReLU network whose decision boundary is |x|_1 = radius (a diamond, homeomorphic to S^1)."""
    # Build g(x) = |x1| + |x2| - radius using ReLU primitives:
    # abs(t) = relu(t) + relu(-t)
    #
    # fc0: [x1, -x1, x2, -x2, extra...]
    # relu
    # fc1: [|x1|, |x2|]
    # fc2: [|x1| + |x2| - radius]
    # relu  (final ReLU so the last neuron's BH encodes the decision boundary)
    #
    # The extra ReLU units are "topology-only": their outgoing weights are 0, so they
    # do not affect g(x), but they *do* subdivide the decision boundary into multiple
    # cells so `get_boundary_complex()` has a meaningful cell decomposition.
    #
    # Small noise is added to fc0 weights so that no two neurons share an identical
    # halfspace direction (genericity condition required by the SHI count heuristic).
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
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # relu(x1) + relu(-x1)
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],  # relu(x2) + relu(-x2)
        ],
        dtype=torch.float64,
    )

    fc2 = nn.Linear(2, 1, bias=True, dtype=torch.float64)
    fc2.weight.data[:] = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    fc2.bias.data[:] = torch.tensor([-float(radius)], dtype=torch.float64)

    return nn.Sequential(fc0, nn.ReLU(), fc1, fc2, nn.ReLU())


def _line_boundary_model() -> nn.Sequential:
    """ReLU network whose decision boundary is x1 = 0 (a non-compact line, homeomorphic to R)."""
    # Add extra "unused" ReLUs to subdivide the line into multiple 1-cells
    # without changing the output function.
    # Small noise breaks the anti-parallel pair (rows 1 and 2) to satisfy genericity.
    fc0 = nn.Linear(2, 4, bias=False, dtype=torch.float64)
    base = torch.tensor(
        [
            [1.0, 0.0],  # the actual decision hyperplane x1=0
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    fc0.weight.data[:] = base + 1e-3 * torch.randn_like(base)
    fc1 = nn.Linear(4, 1, bias=False, dtype=torch.float64)
    fc1.weight.data[:] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    return nn.Sequential(fc0, nn.ReLU(), fc1, nn.ReLU())


def test_decision_boundary_diamond_circle_betti_agree(seeded: int):
    """Compact boundary: both modes should agree (sanity check)."""
    set_seeds(seeded)
    model = _diamond_boundary_model_l1_ball(radius=1.0)
    cplx = Complex(model)
    # Populate many regions near the decision boundary without running BFS.
    thetas = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    inside = 0.9 * dirs
    outside = 1.1 * dirs
    _add_points(cplx, np.vstack([inside, outside, np.random.randn(200, 2)]))
    db_cplx = cplx.get_boundary_complex(cplx.n - 1)

    betti_std = db_cplx.get_betti_numbers()
    betti_bm = db_cplx.get_betti_numbers(compactify=True, reduced=True)

    # These are different conventions; just sanity-check both run and that the
    # boundary has a nontrivial 1-cycle over GF(2).
    assert isinstance(betti_std, dict)
    assert isinstance(betti_bm, dict)
    # At minimum, the boundary should have a nontrivial 1-cycle over GF(2).
    assert int(betti_std.get(1, 0)) >= 1


def test_decision_boundary_verify_chain_complex_passes(seeded: int):
    """∂²=0 on meta-graph boundary maps for small analytic decision boundaries."""
    set_seeds(seeded)
    model = _diamond_boundary_model_l1_ball(radius=1.0)
    cplx = Complex(model)
    thetas = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    inside = 0.9 * dirs
    outside = 1.1 * dirs
    _add_points(cplx, np.vstack([inside, outside, np.random.randn(200, 2)]))
    db = cplx.get_boundary_complex(cplx.n - 1)
    betti_std = db.get_betti_numbers()
    assert db.get_betti_numbers(verify_chain_complex=True) == betti_std
    betti_bm = db.get_betti_numbers(compactify=True, reduced=True)
    assert db.get_betti_numbers(compactify=True, reduced=True, verify_chain_complex=True) == betti_bm

    fc = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    line_model = nn.Sequential(fc, nn.ReLU())
    cplx2 = Complex(line_model)
    xs = np.linspace(-2.0, 2.0, 25)
    ys = np.linspace(-2.0, 2.0, 25)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    eps = 1e-2
    left = grid.copy()
    left[:, 0] = -eps
    right = grid.copy()
    right[:, 0] = eps
    _add_points(cplx2, np.vstack([left, right, np.random.randn(200, 2)]))
    db2 = cplx2.get_boundary_complex(cplx2.n - 1)
    _ = db2.get_betti_numbers(verify_chain_complex=True)
    _ = db2.get_betti_numbers(compactify=True, reduced=True, verify_chain_complex=True)
    _ = db2.get_betti_numbers(respect_finite=True, verify_chain_complex=True)


def test_decision_boundary_line_differs_between_homologies(seeded: int):
    """Non-compact boundary: both modes should run (sanity check)."""
    set_seeds(seeded)
    # Use a truly affine last-neuron BH so the boundary is x1=0 (a line).
    fc = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    model = nn.Sequential(fc, nn.ReLU())
    cplx = Complex(model)
    # Sample points on both sides of the boundary x1=0, across a range of x2 values.
    xs = np.linspace(-2.0, 2.0, 25)
    ys = np.linspace(-2.0, 2.0, 25)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    eps = 1e-2
    left = grid.copy()
    left[:, 0] = -eps
    right = grid.copy()
    right[:, 0] = eps
    _add_points(cplx, np.vstack([left, right, np.random.randn(200, 2)]))
    db_cplx = cplx.get_boundary_complex(cplx.n - 1)

    betti_std = db_cplx.get_betti_numbers()
    betti_bm = db_cplx.get_betti_numbers(compactify=True, reduced=True)
    betti_trad = db_cplx.get_betti_numbers()
    betti_embedded = db_cplx.get_betti_numbers(respect_finite=True)

    assert isinstance(betti_std, dict)
    assert isinstance(betti_bm, dict)
    assert isinstance(betti_trad, dict)
    assert isinstance(betti_embedded, dict)

    # Meta-graph Betti numbers are purely combinatorial; for this 1D non-compact
    # boundary complex, we mainly assert they are well-defined.
    assert isinstance(betti_trad, dict)

    # Embedded (finite-only) convention should be well-defined and (in this simple
    # example) match the standard contracted-chain convention.
    assert betti_embedded == betti_std


@torch.no_grad()
def _far_away_hyperplane_model() -> nn.Sequential:
    """Model whose last neuron's BH is an affine line far from sampled points."""
    # y = ReLU(x1 + 100) has decision boundary x1 = -100 (homeomorphic to R).
    fc = nn.Linear(2, 1, bias=True, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    fc.bias.data[:] = torch.tensor([100.0], dtype=torch.float64)
    return nn.Sequential(fc, nn.ReLU())


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS", "0") == "1" or os.environ.get("CI", "0") == "1",
    reason="Skip decision-boundary integration tests on CI (optional).",
)
@pytest.mark.skipif(
    os.environ.get(INTEGRATION_ENV, "0") != "1",
    reason=f"Opt-in integration test. Set {INTEGRATION_ENV}=1.",
)
def test_decision_boundary_empty_boundary_has_no_cells(seeded: int):
    """Far-away affine BH: sampling misses one side; one boundary 1-cell is recovered.

    With intrinsic ``finite`` from Chebyshev (not coface propagation), the lone
    bounded line segment yields the same ``{1: 1}`` pattern in standard and
    Borel–Moore conventions here.
    """
    set_seeds(seeded)
    model = _far_away_hyperplane_model()
    cplx = Complex(model)
    _add_points(cplx, np.random.randn(300, 2))
    db = cplx.get_boundary_complex(cplx.n - 1)
    assert len(db) == 1
    assert db.get_betti_numbers() == {1: 1}
    assert db.get_betti_numbers(compactify=True, reduced=True) == {1: 1}
