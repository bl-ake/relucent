from __future__ import annotations

import numpy as np
import torch

from relucent import Complex, set_seeds


def _add_points(cplx: Complex, pts: np.ndarray) -> None:
    """Add a batch of points to a complex, skipping any boundary points."""
    for x in np.asarray(pts, dtype=np.float64):
        ss = cplx.point2ss(x.reshape(1, -1))
        if (np.asarray(ss) == 0).any():
            continue
        cplx.add_point(x.reshape(1, -1), check_exists=True)


def test_meta_graph_has_all_dims_and_face_edges(seeded: int):
    set_seeds(seeded)

    # Reuse the simple "x1=0" decision-boundary setup from the boundary Betti tests:
    # the boundary complex is 1D (a line) in ambient R^2.
    import torch.nn as nn

    fc = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    model = nn.Sequential(fc, nn.ReLU())

    cplx = Complex(model)

    xs = np.linspace(-2.0, 2.0, 11)
    ys = np.linspace(-2.0, 2.0, 11)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    eps = 1e-2
    left = grid.copy()
    left[:, 0] = -eps
    right = grid.copy()
    right[:, 0] = eps
    _add_points(cplx, np.vstack([left, right, np.random.randn(200, 2)]))

    # Ensure we have both sides so the boundary complex has a nontrivial 1D decomposition.
    cplx._dual_graph = cplx.get_dual_graph(auto_add=True, verbose=False)
    db = cplx.get_boundary_complex(cplx.n - 1)

    chain = db.get_chain_complex(verbose=False)
    assert len(chain) >= 1

    meta = db.get_meta_graph(enrich=True, verbose=False)

    # Meta-graph should contain every cell across the contraction chain.
    expected_nodes = {p.tag for cc in chain for p in cc}
    assert set(meta.nodes) >= expected_nodes

    # Every edge should decrease dimension by exactly 1.
    for u, v, data in meta.edges(data=True):
        assert int(meta.nodes[u]["dim"]) == int(meta.nodes[v]["dim"]) + 1
        assert "shi" in data

    # Enrichment should ensure `finite` and `shis` are present as node attributes.
    for _n, attrs in meta.nodes(data=True):
        assert "finite" in attrs
        assert "shis" in attrs
        assert isinstance(attrs["shis"], list)
