from __future__ import annotations

import numpy as np
import torch

from relucent import Complex, set_seeds
from relucent.complex import TRUNCATION_META_SHI


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


def test_meta_graph_truncate_augmented_ss_bounded_subcomplex(seeded: int):
    """Boundary complex of a 1-neuron network: ``truncate`` augments ss and duplicates unbounded cells.

    The boundary cell (the ReLU hyperplane x1=0 in R^2) is a 1-D line, which is
    geometrically unbounded.  ``truncate=True`` therefore adds a ``("trunc", tag)``
    duplicate for it, exactly as it does for unbounded cells in the full complex.
    We check that:
    - every non-truncation node has its sign sequence extended with a trailing 1,
    - every positive-dim unbounded cell gets a ``("trunc", tag)`` copy with trailing 0.
    """
    set_seeds(seeded)
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

    db = cplx.get_boundary_complex(cplx.n - 1)

    meta_plain = db.get_meta_graph(enrich=True, verbose=False, truncate=False)
    meta_tr = db.get_meta_graph(enrich=True, verbose=False, truncate=True)

    # Every original (non-truncation) node has its sign sequence extended with a trailing 1.
    for n in meta_plain.nodes():
        assert n in meta_tr.nodes(), f"original node {n!r} missing from truncated graph"
        ss0 = np.asarray(meta_plain.nodes[n]["ss"])
        sst = np.asarray(meta_tr.nodes[n]["ss"])
        assert sst.shape == (ss0.shape[0], ss0.shape[1] + 1)
        assert int(sst.flat[-1]) == 1

    # Every positive-dim unbounded cell gets a truncation duplicate with trailing 0.
    unbounded = {n for n, a in meta_plain.nodes(data=True) if a.get("finite") is False and int(a.get("dim", -1)) > 0}
    for n in unbounded:
        dup = ("trunc", n)
        assert dup in meta_tr.nodes(), f"expected truncation duplicate {dup!r} for unbounded node {n!r}"
        ssd = np.asarray(meta_tr.nodes[dup]["ss"])
        assert int(ssd.flat[-1]) == 0


def test_meta_graph_truncate_unbounded_duplication_and_links(seeded: int):
    """Half-plane activation regions: duplicates mirror the unbounded induced subgraph."""
    set_seeds(seeded)
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

    meta_plain = cplx.get_meta_graph(enrich=True, verbose=False, truncate=False)
    meta_tr = cplx.get_meta_graph(enrich=True, verbose=False, truncate=True)

    ub = {n for n, a in meta_plain.nodes(data=True) if a.get("finite", None) is False}
    assert ub, "expected unbounded meta nodes in this construction"

    for n in meta_tr.nodes():
        if isinstance(n, tuple):
            continue
        sst = np.asarray(meta_tr.nodes[n]["ss"])
        assert int(sst.flat[-1]) == 1

    for n in ub:
        if int(meta_plain.nodes[n]["dim"]) <= 0:
            continue
        dk = ("trunc", n)
        assert dk in meta_tr.nodes
        assert int(meta_tr.nodes[dk]["dim"]) == int(meta_plain.nodes[n]["dim"]) - 1
        ssd = np.asarray(meta_tr.nodes[dk]["ss"])
        assert int(ssd.flat[-1]) == 0
        assert meta_tr.has_edge(n, dk)
        trunc_shis = [ed.get("shi") for _u, _v, _k, ed in meta_tr.out_edges(n, keys=True, data=True) if _v == dk]
        assert TRUNCATION_META_SHI in trunc_shis

    for u, v, d in meta_plain.edges(data=True):
        if u in ub and v in ub:
            tu, tv = ("trunc", u), ("trunc", v)
            assert meta_tr.has_edge(tu, tv)
            shis_dup = [ed.get("shi") for _u, _v, _k, ed in meta_tr.out_edges(tu, keys=True, data=True) if _v == tv]
            assert d.get("shi") in shis_dup
