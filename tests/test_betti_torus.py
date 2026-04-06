"""Integration test for torus decision-boundary Betti numbers."""

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch import optim

from relucent import Complex, get_mlp_model, set_seeds

BUNDLED_TORUS_CHECKPOINT = Path(__file__).parent / "data" / "torus_boundary_model_seed2.pt"


def _make_sphere(
    dim: int,
    n: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    temppts = torch.normal(0.0, 1.0, (n, dim), device=device, dtype=dtype)
    points0 = 2 * temppts / torch.linalg.norm(temppts, dim=1).reshape(n, 1)
    scatter = 0.05 * torch.normal(0.0, 1.0, (n, 1), device=device, dtype=dtype)
    points0 = points0 * scatter + points0

    points1 = 0.2 * torch.normal(0.0, 1.0, (n, dim), device=device, dtype=dtype)
    points = torch.vstack([points0, points1])
    labels = torch.hstack(
        [
            torch.zeros(n, device=device, dtype=dtype),
            torch.ones(n, device=device, dtype=dtype),
        ]
    ).reshape(2 * n, 1)
    return points, labels


def _make_torus(
    n: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    thetas = 2 * torch.pi * torch.rand(n, device=device, dtype=dtype)
    phis = 2 * torch.pi * torch.rand(n, device=device, dtype=dtype)

    xs = 4 * torch.cos(thetas) - 2 * torch.cos(thetas) * torch.cos(phis)
    ys = 4 * torch.sin(thetas) - 2 * torch.sin(thetas) * torch.cos(phis)
    zs = 2 * torch.sin(phis)
    pts0 = torch.vstack([xs, ys, zs]).T

    pts1, _ = _make_sphere(2, n, device=device, dtype=dtype)
    pts1 = 2 * torch.hstack([pts1[0:n], torch.zeros((n, 1), device=device, dtype=dtype)])

    points = torch.vstack([pts0, pts1])
    labels = torch.hstack(
        [
            torch.zeros(n, device=device, dtype=dtype),
            torch.ones(n, device=device, dtype=dtype),
        ]
    ).reshape(2 * n, 1)
    return points, labels


def _train_torus_model(steps: int = 40_000) -> torch.nn.Module:
    """Train the torus classifier from canonicalpoly2.0's notebook recipe."""
    model = get_mlp_model(widths=[3, 15, 15, 1], add_last_relu=False)
    model.to("cpu")
    criterion = nn.BCELoss()
    lr = 0.01
    sample_size = 30
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    valpoints, vallabels = _make_torus(1_000)
    valpoints *= 0.25

    for training_step in range(steps):
        points, labels = _make_torus(sample_size)
        points *= 0.25

        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(torch.sigmoid(outputs), labels)
        loss.backward()
        optimizer.step()

        if training_step % 5_000 == 0 and training_step > 0:
            lr *= 0.95
            sample_size += 5
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            with torch.no_grad():
                _ = criterion(torch.sigmoid(model(valpoints)), vallabels)

    return model


def _model_with_db_relu(model: torch.nn.Module) -> torch.nn.Module:
    """Create a topology model with final ReLU so DB is represented as a BH."""
    topo_model = get_mlp_model(widths=[3, 15, 15, 1], add_last_relu=True)
    src_state = model.state_dict()
    dst_state = topo_model.state_dict()
    for key, value in src_state.items():
        if key in dst_state:
            dst_state[key] = value
    topo_model.load_state_dict(dst_state)
    return topo_model


@pytest.mark.skipif(
    os.environ.get("RELUCENT_RUN_TORUS_TEST", "0") != "1",
    reason=("Integration test is opt-in. Set RELUCENT_RUN_TORUS_TEST=1 (uses bundled checkpoint by default)."),
)
def test_torus_decision_boundary_betti_numbers(seeded):
    """Torus decision boundary should have Betti numbers [1, 2, 1]."""
    set_seeds(seeded)

    checkpoint = os.environ.get("RELUCENT_TORUS_CHECKPOINT")
    checkpoint_path = Path(checkpoint) if checkpoint is not None else BUNDLED_TORUS_CHECKPOINT
    model: torch.nn.Module

    if checkpoint_path.exists():
        model = get_mlp_model(widths=[3, 15, 15, 1], add_last_relu=False)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    elif os.environ.get("RELUCENT_TORUS_TRAIN", "0") == "1":
        steps = int(os.environ.get("RELUCENT_TORUS_TRAIN_STEPS", "40000"))
        model = _train_torus_model(steps=steps)
    else:
        pytest.skip(
            "No torus checkpoint found and training disabled. "
            "Restore bundled checkpoint, provide RELUCENT_TORUS_CHECKPOINT, "
            "or set RELUCENT_TORUS_TRAIN=1."
        )

    topo_model = _model_with_db_relu(model)
    cplx = Complex(topo_model)

    # Rarely, a random start can land exactly on a BH; retry a few times.
    start = None
    for _ in range(10):
        candidate = torch.randn((1, 3), dtype=next(topo_model.parameters()).dtype)
        ss = cplx.point2ss(candidate)
        if not (ss == 0).any():
            start = candidate
            break
    if start is None:
        pytest.skip("Could not sample a non-boundary start point for BFS.")

    cplx.bfs(start=start, nworkers=1, verbose=0)
    db_cplx = cplx.get_boundary_complex(cplx.n - 1)
    betti = db_cplx.get_betti_numbers()
    betti_list = [betti.get(i, 0) for i in range(3)]

    assert betti_list == [1, 2, 1]
