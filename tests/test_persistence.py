"""Tests for filtrations and persistent homology."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn

from relucent import Complex, set_seeds, vis
from relucent.filtration import (
    LogitSublevelFiltration,
    NeuronActivationFiltration,
    TrainingDistanceFiltration,
    lower_star_extension,
)
from relucent.persistence import (
    PersistenceDiagram,
    PersistencePair,
    _gf2_column_reduce_persistence,
    betti_curve,
    compute_persistent_homology,
)
from tests.conftest import explore_for_topology


def _triangle_persistence() -> tuple[list[set[int]], list[float], list[int], list[str]]:
    """Triangle: 3 vertices (dim 0), 3 edges (dim 1); no 2-cell. Filtration 0 then 1."""
    boundaries = [
        set(),
        set(),
        set(),
        {0, 1},
        {1, 2},
        {0, 2},
    ]
    filtration = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    dimensions = [0, 0, 0, 1, 1, 1]
    keys = ["v0", "v1", "v2", "e01", "e12", "e02"]
    return boundaries, filtration, dimensions, keys


def test_column_reduction_triangle_h0():
    """Three H0 classes born at vertices; two die when edges appear; one essential H0 and H1."""
    b, f, d, keys = _triangle_persistence()
    pairs = _gf2_column_reduce_persistence(b, f, d, keys)
    finite_h0 = [p for p in pairs if p.dimension == 0 and math.isfinite(p.death)]
    essential = [p for p in pairs if not math.isfinite(p.death)]
    assert len(finite_h0) == 2
    assert all(p.death == 1.0 for p in finite_h0)
    assert len(essential) == 2
    assert sum(1 for p in essential if p.dimension == 0) == 1
    assert sum(1 for p in essential if p.dimension == 1) == 1


def test_lower_star_extension_max_over_faces():
    meta = nx.MultiDiGraph()
    meta.add_node("a", dim=0, ss=np.array([[1]], dtype=np.int8))
    meta.add_node("b", dim=0, ss=np.array([[0]], dtype=np.int8))
    meta.add_node("e", dim=1, ss=np.array([[0]], dtype=np.int8))
    meta.add_edge("e", "a", shi=0)
    meta.add_edge("e", "b", shi=0)
    raw = {"a": 1.0, "b": 2.0, "e": 0.5}
    ext = lower_star_extension(meta, raw)
    assert ext["e"] == 2.0


def _diamond_boundary_model(radius: float = 1.0) -> nn.Sequential:
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


def _add_points(cplx: Complex, pts: np.ndarray) -> None:
    for x in np.asarray(pts, dtype=np.float64):
        ss = cplx.point2ss(x.reshape(1, -1))
        if (np.asarray(ss) == 0).any():
            continue
        cplx.add_point(x.reshape(1, -1), check_exists=True)


def test_neuron_activation_filtration_on_diamond(seeded: int):
    """Activation filtration is combinatorial and yields a valid persistence diagram."""
    set_seeds(seeded)
    model = _diamond_boundary_model()
    cplx = Complex(model)
    thetas = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    _add_points(cplx, np.vstack([0.9 * dirs, 1.1 * dirs, np.random.randn(80, 2)]))
    db = cplx.get_boundary_complex(cplx.n - 1)
    assert len(db) > 0

    p0 = db.index2poly[0]
    shi = int(p0.zero_indices[0]) if p0.zero_indices.size else 0
    fil = NeuronActivationFiltration(shi=shi, target=0)
    diagram = compute_persistent_homology(db, fil, verbose=False)
    assert isinstance(diagram, PersistenceDiagram)
    assert diagram.cell_filtration
    assert all(math.isfinite(v) or v in (0.0, 1.0) for v in diagram.cell_filtration.values())


def test_logit_filtration_and_betti_curve(seeded: int):
    set_seeds(seeded)
    model = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 1), nn.ReLU())
    cplx = Complex(model)
    explore_for_topology(cplx, np.array([0.0]))
    fil = LogitSublevelFiltration(binary=False)
    diagram = compute_persistent_homology(cplx, fil)
    thresholds = np.unique(sorted(diagram.cell_filtration.values()))
    curve = betti_curve(diagram, thresholds)
    assert 0 in curve
    assert curve[0].shape == thresholds.shape


def test_plot_persistence_diagram():
    """Persistence diagram plotting produces a Plotly figure with expected traces."""
    diagram = PersistenceDiagram(
        pairs=(
            PersistencePair(0, 0.0, 1.0),
            PersistencePair(1, 0.5, 2.0),
            PersistencePair(1, 1.0, float("inf")),
        ),
        cell_filtration={"a": 0.0, "b": 2.0},
    )
    fig = vis.plot_persistence_diagram(diagram)
    assert isinstance(fig, go.Figure)
    assert len(list(fig.data)) >= 2
    fig2 = diagram.plot(title=None)
    assert fig2.layout.title.text is None


def test_training_distance_filtration(seeded: int):
    set_seeds(seeded)
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    cplx = Complex(model)
    train = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    explore_for_topology(cplx, train[0])
    fil = TrainingDistanceFiltration(train)
    diagram = compute_persistent_homology(cplx, fil)
    assert len(diagram.pairs) >= 0
    assert all(v >= 0.0 for v in diagram.cell_filtration.values())
