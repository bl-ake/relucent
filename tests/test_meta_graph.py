from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from relucent import Complex, Polyhedron, mlp, set_seeds
from relucent import meta_graph as mg
from relucent.calculations import get_shis
from relucent.complex import TRUNCATION_META_SHI
from relucent.exploration import explore_for_topology
from relucent.utils import get_env


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
    explore_for_topology(cplx, np.array([0.1, 0.2]))
    db = cplx.get_boundary_complex(cplx.n - 1)

    chain = db.get_chain_complex(verbose=False)
    assert len(chain) >= 1

    meta = db.get_meta_graph(verbose=False)

    # Meta-graph should contain every feasible cell across the contraction chain.
    expected_nodes = {p.tag for cc in chain for p in cc if not (p.dim > 0 and p.finite is None)}
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

    explore_for_topology(cplx, np.array([0.1, 0.2]))
    db = cplx.get_boundary_complex(cplx.n - 1)

    meta_plain = db.get_meta_graph(verbose=False)
    meta_tr = meta_plain.copy()
    Complex.truncate_meta_graph(meta_tr)

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
    explore_for_topology(cplx, np.array([0.5, 0.0]))

    meta_plain = cplx.get_meta_graph(verbose=False)
    meta_tr = meta_plain.copy()
    Complex.truncate_meta_graph(meta_tr)

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

    for n, a in meta_tr.nodes(data=True):
        if int(a.get("dim", -1)) != 1:
            continue
        zero_faces = [v for _u, v, _ in meta_tr.out_edges(n, data=True) if int(meta_tr.nodes[v].get("dim", -1)) == 0]
        assert len(zero_faces) == 2, f"1-cell {n!r} should have two 0-face endpoints, got {zero_faces!r}"

    for u, v, d in meta_plain.edges(data=True):
        if u in ub and v in ub:
            tu, tv = ("trunc", u), ("trunc", v)
            assert meta_tr.has_edge(tu, tv)
            shis_dup = [ed.get("shi") for _u, _v, _k, ed in meta_tr.out_edges(tu, keys=True, data=True) if _v == tv]
            assert d.get("shi") in shis_dup


def _cells_from_dual_graph_propagation(cplx: Complex) -> dict[bytes, Polyhedron]:
    """Map tag -> polyhedron for every cell in the contraction chain.

    Each contraction step calls :meth:`Complex.get_dual_graph` and propagates
    SHIs downward via :meth:`Complex.contract` (intersecting coface SHIs).
    """
    chain = cplx.get_chain_complex(verbose=False)
    return {p.tag: p for cc in chain for p in cc}


def _lp_shis(poly: Polyhedron, env) -> list[int] | None:
    """Supporting hyperplane indices from a fresh ``get_shis`` LP solve."""
    poly._shis = None
    kwargs: dict[str, Any] = {"env": env, "strict": False}
    if poly.bound is not None:
        kwargs["bound"] = float(poly.bound)
    try:
        return sorted(int(s) for s in get_shis(poly, **kwargs))
    except ValueError:
        return None


def _finite_from_dual_graph_propagation(
    cells: dict[bytes, Polyhedron],
    face_edges: list[tuple[bytes, bytes, int]],
    top_dim: int,
) -> dict[bytes, bool]:
    """Boundedness propagated upward from 1-cells via 0-face incidence + face sweep."""
    finite: dict[bytes, bool] = {}
    for dim in range(1, top_dim + 1):
        for tag, poly in cells.items():
            if poly.dim != dim:
                continue
            if dim == 1:
                if poly.finite is None:
                    continue
                zero_faces = {
                    dst for src, dst, _ in face_edges if src == tag and cells.get(dst) is not None and cells[dst].dim == 0
                }
                finite[tag] = len(zero_faces) >= 2
                continue
            faces = [dst for src, dst, _ in face_edges if src == tag and dst in finite]
            if not faces:
                continue
            if any(not finite[f] for f in faces):
                finite[tag] = False
            else:
                finite[tag] = True
    return finite


def test_classify_one_cells_finite_from_face_edges_empty_shis_two_zero_faces() -> None:
    """1-cells with empty ``_shis`` are bounded when two 0-faces appear in face edges."""
    halfspaces = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    ss_z0 = np.array([[1, 0, 1, 0]], dtype=np.int8)
    ss_z1 = np.array([[1, 1, 0, 0]], dtype=np.int8)
    ss_seg = np.array([[1, 0, 0, 0]], dtype=np.int8)
    p0 = Polyhedron(None, ss_z0, halfspaces=halfspaces, finite=True)
    p1 = Polyhedron(None, ss_z1, halfspaces=halfspaces, finite=True)
    seg = Polyhedron(None, ss_seg, halfspaces=halfspaces, finite=True)
    seg._shis = []
    seg._finite_computed = False
    seg._finite = None

    by_dim = {0: [p0, p1], 1: [seg]}
    edges_by_dim = {1: ([(seg.tag, p0.tag, 1), (seg.tag, p1.tag, 2)], [])}

    n = mg.classify_one_cells_finite_from_face_edges(by_dim, edges_by_dim)[0]
    assert n == 1
    assert seg._finite is True


def test_classify_one_cells_finite_from_face_edges_infeasible_left_none() -> None:
    """Infeasible 1-cells stay ``_finite is None`` (excluded from truncation)."""
    halfspaces = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    ss_seg = np.array([[1, 0, 0, 0]], dtype=np.int8)
    seg = Polyhedron(None, ss_seg, halfspaces=halfspaces, finite=None)
    seg._shis = []
    seg._finite_computed = False

    by_dim = {1: [seg]}
    edges_by_dim: dict[int, tuple[list[tuple[bytes, bytes, int]], list[bytes]]] = {1: ([], [])}

    n = mg.classify_one_cells_finite_from_face_edges(
        by_dim,
        edges_by_dim,
        geometric_infeasible={seg.tag},
    )[0]
    assert n == 1
    assert seg._finite is None


def test_geometric_infeasible_one_cells_absorbs_near_zero_inradius_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Near-zero negative Chebyshev radii classify phantom 1-cells as infeasible."""
    halfspaces = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    ss_seg = np.array([[1, 0, 0, 0]], dtype=np.int8)
    seg = Polyhedron(None, ss_seg, halfspaces=halfspaces, finite=None)
    seg._shis = []
    seg._finite_computed = False

    def _raise_inradius() -> tuple[None, None]:
        raise ValueError("Inradius -8.7393e-07")

    monkeypatch.setattr(seg, "get_center_inradius", _raise_inradius)

    by_dim = {1: [seg]}
    edges_by_dim: dict[int, tuple[list[tuple[bytes, bytes, int]], list[bytes]]] = {1: ([], [])}

    infeasible = mg.geometric_infeasible_one_cells(by_dim, edges_by_dim)
    assert infeasible == {seg.tag}


@pytest.mark.filterwarnings("ignore:Working with k<d polyhedron\\.:UserWarning")
def test_meta_graph_shis_match_cubical_derivation(seeded: int) -> None:
    """Meta-graph node ``shis`` match :func:`~relucent.meta_graph.cubical_cell_shis` per slice."""
    set_seeds(seeded)
    net = mlp(widths=[4, 5, 5, 1], add_last_relu=True)
    cplx = Complex(net)
    start = torch.randn(4, dtype=torch.float64)
    explore_for_topology(cplx, start.numpy(), max_polys=5000)

    chain = cplx.get_chain_complex(verbose=False)
    by_dim = {int(cc.index2poly[0].dim): cc for cc in chain if len(cc)}
    dim_neighbor_tags = {k: {p.tag for p in cc} for k, cc in by_dim.items()}

    meta = cplx.get_meta_graph(verbose=False)
    mismatches: list[str] = []

    for tag, attrs in meta.nodes(data=True):
        dim = int(attrs["dim"])
        if dim <= 0:
            continue
        poly = attrs.get("poly")
        if poly is None:
            continue
        neighbor_tags = dim_neighbor_tags.get(dim, set())
        expected = mg.cubical_cell_shis(poly.ss_np, neighbor_tags=neighbor_tags)
        if int(poly.dim) == 1 and poly.halfspaces is not None:
            expected = [s for s in expected if poly.is_shi_face_feasible(int(s))]
        meta_shis = sorted(int(s) for s in attrs["shis"])
        if meta_shis != expected:
            mismatches.append(f"dim={dim} tag={tag!r}: meta={meta_shis} cubical={expected}")

    assert not mismatches, (
        f"Meta-graph SHIs disagree with cubical derivation for {len(mismatches)} cells:\n"
        + "\n".join(mismatches[:20])
        + ("\n..." if len(mismatches) > 20 else "")
    )


@pytest.mark.filterwarnings("ignore:Working with k<d polyhedron\\.:UserWarning")
def test_top_dim_lp_shis_subset_of_cubical_derivation(seeded: int) -> None:
    """LP SHI facets on top-dimensional cells lie in the cubical flip-neighbor set."""
    set_seeds(seeded)
    net = mlp(widths=[4, 5, 5, 1], add_last_relu=True)
    cplx = Complex(net)
    start = torch.randn(4, dtype=torch.float64)
    explore_for_topology(cplx, start.numpy(), max_polys=5000)

    top = [p for p in cplx if p.dim == cplx.dim]
    neighbor_tags = {p.tag for p in top}
    env = get_env()
    mismatches: list[str] = []

    for poly in top:
        lp_shis = _lp_shis(poly, env)
        if lp_shis is None:
            continue
        cubical = mg.cubical_cell_shis(poly.ss_np, neighbor_tags=neighbor_tags)
        extra = set(lp_shis) - set(cubical)
        if extra:
            mismatches.append(f"tag={poly.tag!r}: lp={lp_shis} cubical={cubical} extra={sorted(extra)}")

    assert not mismatches, (
        f"LP SHIs not contained in cubical derivation for {len(mismatches)} top cells:\n"
        + "\n".join(mismatches[:20])
        + ("\n..." if len(mismatches) > 20 else "")
    )


def test_meta_graph_finite_matches_dual_graph_propagation(seeded: int) -> None:
    """Meta-graph ``finite`` matches upward propagation from 1-cells on the dual graph."""
    set_seeds(seeded)
    net = mlp(widths=[4, 5, 5, 1], add_last_relu=True)
    cplx = Complex(net)
    start = torch.randn(4, dtype=torch.float64)
    explore_for_topology(cplx, start.numpy(), max_polys=5000)

    meta = cplx.get_meta_graph(verbose=False)
    dual_cells = _cells_from_dual_graph_propagation(cplx)
    top_dim = max(int(a["dim"]) for _, a in meta.nodes(data=True))
    face_edges = [(u, v, int(d["shi"])) for u, v, d in meta.edges(data=True)]
    expected_finite = _finite_from_dual_graph_propagation(dual_cells, face_edges, top_dim)

    mismatches: list[str] = []
    for tag, attrs in meta.nodes(data=True):
        dim = int(attrs["dim"])
        if dim <= 0 or dim >= top_dim:
            continue
        poly = attrs.get("poly")
        if poly is None or tag not in expected_finite:
            continue
        if poly._finite != expected_finite[tag]:
            mismatches.append(f"dim={dim} tag={tag!r}: meta_finite={poly._finite} dual_graph_finite={expected_finite[tag]}")

    assert not mismatches, (
        "Meta-graph finite disagrees with dual-graph upward propagation for "
        + f"{len(mismatches)} cells:\n"
        + "\n".join(mismatches[:20])
        + ("\n..." if len(mismatches) > 20 else "")
    )


@pytest.mark.filterwarnings("ignore:Working with k<d polyhedron\\.:UserWarning")
def test_meta_graph_verify_incidence(seeded: int) -> None:
    """``verify=True`` checks incidence-engine consistency without mutating the graph."""
    set_seeds(seeded)
    net = mlp(widths=[4, 5, 5, 1], add_last_relu=True)
    cplx = Complex(net)
    start = torch.randn(4, dtype=torch.float64)
    explore_for_topology(cplx, start.numpy(), max_polys=5000)

    meta = cplx.get_meta_graph(verify=True, verbose=False)
    assert meta.number_of_nodes() > 0
    assert meta.number_of_edges() > 0
