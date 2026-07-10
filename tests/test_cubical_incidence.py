"""Tests for the cubical incidence engine (dual graph, meta-graph consistency)."""

from __future__ import annotations

import os
import time

import networkx as nx
import numpy as np
import torch

from relucent import Complex, mlp, set_seeds
from relucent import meta_graph as mg
from relucent.exploration import explore_for_topology
from relucent.utils import encode_ss
from tests.test_betti_decision_boundaries import (
    _add_points,
    _diamond_boundary_model_l1_ball,
)

os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")


def test_face_tag_and_flip_tag_roundtrip() -> None:
    ss = np.array([[1, -1, 0]], dtype=np.int8)
    shi = 0
    ft = mg.face_tag(ss, shi)
    flip = mg.flip_tag(ss, shi)
    ss_zero = ss.copy()
    ss_zero.ravel()[shi] = 0
    assert ft == encode_ss(ss_zero)
    assert flip == encode_ss(np.array([[-1, -1, 0]], dtype=np.int8))


def test_dual_edges_match_flip_neighbors_on_diamond_boundary(seeded: int) -> None:
    set_seeds(seeded)
    model = _diamond_boundary_model_l1_ball(radius=1.0)
    cplx = Complex(model)
    thetas = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    _add_points(cplx, np.vstack([0.9 * dirs, 1.1 * dirs, np.random.randn(80, 2)]))
    explore_for_topology(cplx, np.array([0.1, 0.2]))

    db = cplx.get_boundary_complex(cplx.n - 1)
    G = db.get_dual_graph(verbose=False)
    assert G.number_of_edges() >= 1
    from relucent.incidence import certify_dual_graph

    certify_dual_graph(G, db)
    assert nx.number_connected_components(G) == 1


def test_cubical_dual_graph_on_boundary_has_edges(seeded: int) -> None:
    """Cubical 1D dual graph connects subdivided diamond boundary cells."""
    set_seeds(seeded)
    model = _diamond_boundary_model_l1_ball(radius=1.0)
    cplx = Complex(model)
    thetas = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    _add_points(cplx, np.vstack([0.9 * dirs, 1.1 * dirs, np.random.randn(80, 2)]))
    explore_for_topology(cplx, np.array([0.1, 0.2]))

    db = cplx.get_boundary_complex(cplx.n - 1)
    G = db.get_dual_graph(verbose=False)
    assert G.number_of_edges() >= 1
    assert nx.number_connected_components(G) == 1


def test_meta_graph_dual_graph_top_dim_consistency(seeded: int) -> None:
    """Stress: top-dimensional dual edges align with meta-graph coface pairs."""
    from collections import defaultdict

    set_seeds(seeded)
    net = mlp(widths=[2, 4, 4, 1], add_last_relu=True, init="uniform")
    cplx = Complex(net)
    start = torch.randn(2, dtype=torch.float64)
    explore_for_topology(cplx, start.numpy(), max_polys=500)

    meta = cplx.get_meta_graph(verbose=False)
    top_dim = max(int(p.dim) for p in cplx)
    dual = cplx.get_dual_graph(verbose=False)

    face_to_tops: dict[bytes, list[bytes]] = defaultdict(list)
    for u, v, _data in meta.edges(data=True):
        du = int(meta.nodes[u].get("dim", -1))
        dv = int(meta.nodes[v].get("dim", -1))
        if du == top_dim and dv == top_dim - 1:
            face_to_tops[v].append(u)

    meta_pairs = {tuple(sorted(pair)) for pair in face_to_tops.values() if len(pair) == 2}
    dual_pairs = {tuple(sorted((u.tag, v.tag))) for u, v in dual.edges()}

    assert dual_pairs <= meta_pairs


def test_cubical_dual_graph_build_is_fast_enough(seeded: int) -> None:
    """Smoke benchmark: cubical dual-graph build stays under a loose time budget."""
    set_seeds(seeded)
    net = mlp(widths=[2, 8, 4, 1], add_last_relu=True)
    cplx = Complex(net)
    explore_for_topology(cplx, np.zeros(2), max_polys=800)
    top = [p for p in cplx if p.dim == cplx.dim]
    tags = {p.tag for p in top}

    t0 = time.perf_counter()
    for _ in range(5):
        mg.dual_edges_top_dim(top, tags)
    elapsed = (time.perf_counter() - t0) / 5.0
    assert elapsed < 2.0, f"cubical dual_edges_top_dim too slow: {elapsed:.3f}s"
