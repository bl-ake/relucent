"""Tests for 1-dimensional boundary genericity and cubical dual-graph construction."""

from __future__ import annotations

import os

import networkx as nx
import numpy as np
import pytest
import torch
import torch.nn as nn

from relucent import Complex, NonGenericArrangementError, set_seeds
from relucent.exploration import explore_for_topology
from tests.test_betti_decision_boundaries import (
    _add_points,
    _diamond_boundary_model_l1_ball,
)

os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")


def _degenerate_v_boundary_model() -> nn.Sequential:
    """Hidden and output hyperplanes concur at the origin (non-generic init)."""
    fc0 = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    fc0.weight.data[:] = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    fc1 = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc1.weight.data[:] = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    return nn.Sequential(fc0, nn.ReLU(), fc1, nn.ReLU())


def test_boundary_generic_diamond_dual_graph_connected(seeded: int) -> None:
    set_seeds(seeded)
    model = _diamond_boundary_model_l1_ball(radius=1.0)
    cplx = Complex(model)
    thetas = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    inside = 0.9 * dirs
    outside = 1.1 * dirs
    _add_points(cplx, np.vstack([inside, outside, np.random.randn(80, 2)]))
    explore_for_topology(cplx, np.array([0.1, 0.2]))

    db = cplx.get_boundary_complex(cplx.n - 1)
    G = db.get_dual_graph(verbose=False)
    assert G.number_of_nodes() >= 2
    assert G.number_of_edges() >= 1
    assert nx.number_connected_components(G) == 1
    from relucent.incidence import certify_dual_graph

    certify_dual_graph(G, db)


def test_boundary_degenerate_v_raises_non_generic(
    seeded: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from relucent import config as rel_cfg

    monkeypatch.setattr(rel_cfg, "CAREFUL_MODE", False)
    monkeypatch.setenv("RELUCENT_CAREFUL_MODE", "0")

    set_seeds(seeded)
    model = _degenerate_v_boundary_model()
    cplx = Complex(model)
    cplx.bfs(start=np.array([[0.5, -0.3]], dtype=np.float64), max_polys=32, verbose=False, verify=False)
    cplx.set_exploration_state(complete=True, verified=True)
    db = Complex(model)
    for poly in cplx.get_boundary_cells(cplx.n - 1, verify=False):
        db.add_polyhedron(poly, check_exists=False)

    with pytest.raises(NonGenericArrangementError, match="distinct geometric|geometric endpoint"):
        db.verify_arrangement_genericity()


def test_one_dim_dual_graph_edges_match_shared_endtags(seeded: int) -> None:
    """Each dual edge corresponds to exactly one shared combinatorial 0-face tag."""
    set_seeds(seeded)
    model = _diamond_boundary_model_l1_ball(radius=1.0)
    cplx = Complex(model)
    thetas = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    _add_points(cplx, np.vstack([0.9 * dirs, 1.1 * dirs, np.random.randn(80, 2)]))
    explore_for_topology(cplx, np.array([0.1, 0.2]))

    db = cplx.get_boundary_complex(cplx.n - 1)
    G = db.get_dual_graph(verbose=False)
    if G.number_of_edges() == 0:
        pytest.skip("boundary exploration produced no dual edges")
    from relucent.incidence import certify_dual_graph

    certify_dual_graph(G, db)
