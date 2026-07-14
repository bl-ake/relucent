from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pytest
import torch

from relucent import Complex, Polyhedron, mlp, set_seeds
from relucent.core.errors import NonGenericArrangementError
from relucent.geometry.calculations import get_shis
from relucent.graph import meta_graph as mg
from relucent.graph.incidence import face_tag
from relucent.search.exploration import explore_for_topology
from relucent.topology import ChainComplexInconsistent, get_betti_numbers
from relucent.utils import encode_ss, get_env


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
    mg.verify_meta_graph_one_cells(meta)

    # Meta-graph should contain feasible cells; phantoms and their cofaces are excluded.
    meta_tags = set(meta.nodes)
    for cc in chain:
        for p in cc:
            if p.dim > 0 and p.finite is None:
                assert p.tag not in meta_tags
    assert meta_tags  # non-empty

    # Every edge should decrease dimension by exactly 1.
    for u, v, data in meta.edges(data=True):
        assert int(meta.nodes[u]["dim"]) == int(meta.nodes[v]["dim"]) + 1
        assert "shi" in data

    # Enrichment should ensure `finite` and `shis` are present as node attributes.
    for _n, attrs in meta.nodes(data=True):
        assert "finite" in attrs
        assert "shis" in attrs
        assert isinstance(attrs["shis"], list)


def _expected_cap_tags(ss_ext: np.ndarray, n_caps: int) -> list[bytes]:
    """Byte tags of truncation cap cells for an extended sign sequence."""
    t1_shi, t2_shi = mg._truncation_bit_indices(ss_ext)
    cap_bits = [t1_shi] if n_caps == 1 else [t1_shi, t2_shi]
    return [face_tag(ss_ext, int(bit_shi)) for bit_shi in cap_bits[:n_caps]]


def test_meta_graph_truncate_augmented_ss_bounded_subcomplex(seeded: int):
    """Boundary complex of a 1-neuron network: truncation extends ss and materializes cap cells.

    The boundary cell (the ReLU hyperplane x1=0 in R^2) is a 1-D line with no combinatorial
    0-faces, so bilateral truncation adds two cap 0-cells as ordinary byte-tagged nodes.
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

    assert not any(isinstance(n, tuple) for n in meta_tr.nodes()), "truncation uses byte tags only"

    # Every original cell is retained under its extended tag.
    unbounded = {n for n, a in meta_plain.nodes(data=True) if a.get("finite") is False and int(a.get("dim", -1)) > 0}
    cache: dict[Any, int] = {}
    for n in unbounded:
        _ = mg._open_cap_count(n, meta_plain, unbounded, cache)

    for n, attrs in meta_plain.nodes(data=True):
        ss0 = np.asarray(attrs["ss"])
        n_caps = cache.get(n, 0) if n in unbounded else 0
        t1, t2 = (1, 1) if n_caps >= 2 else (1, 0)
        ss_ext = mg._ss_with_truncation_bits(ss0, t1, t2)
        ext_tag = encode_ss(np.asarray(ss_ext, dtype=np.int8))
        assert ext_tag in meta_tr.nodes(), f"extended node for {n!r} missing from truncated graph"
        sst = np.asarray(meta_tr.nodes[ext_tag]["ss"])
        assert sst.shape == (ss0.shape[0], ss0.shape[1] + 2)
        assert int(sst.flat[-2]) == 1

    for n in unbounded:
        ss0 = np.asarray(meta_plain.nodes[n]["ss"])
        n_caps = cache[n]
        t1, t2 = (1, 1) if n_caps >= 2 else (1, 0)
        ss_ext = mg._ss_with_truncation_bits(ss0, t1, t2)
        for cap_index, cap_tag in enumerate(_expected_cap_tags(ss_ext, n_caps)):
            assert cap_tag in meta_tr.nodes(), f"expected cap {cap_tag!r} for unbounded node {n!r}"
            assert int(meta_tr.nodes[cap_tag]["dim"]) == int(meta_plain.nodes[n]["dim"]) - 1
            expected_ss = mg._cap_sign_sequence(ss_ext, cap_index=cap_index, n_caps=n_caps)
            assert np.array_equal(np.asarray(meta_tr.nodes[cap_tag]["ss"]), expected_ss)


def test_meta_graph_truncate_unbounded_duplication_and_links(seeded: int):
    """Half-plane activation regions: caps at infinity and combinatorial face edges."""
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
    mg.verify_meta_graph_one_cells(meta_plain)
    meta_tr = meta_plain.copy()
    Complex.truncate_meta_graph(meta_tr)

    assert not any(isinstance(n, tuple) for n in meta_tr.nodes()), "truncation uses byte tags only"

    ub = {n for n, a in meta_plain.nodes(data=True) if a.get("finite", None) is False}
    assert ub, "expected unbounded meta nodes in this construction"

    for n, attrs in meta_plain.nodes(data=True):
        ss0 = np.asarray(attrs["ss"])
        n_caps = mg._open_cap_count(n, meta_plain, ub, {}) if n in ub and int(attrs.get("dim", -1)) > 0 else 0
        t1, t2 = (1, 1) if n_caps >= 2 else (1, 0)
        ss_ext = mg._ss_with_truncation_bits(ss0, t1, t2)
        ext_tag = encode_ss(np.asarray(ss_ext, dtype=np.int8))
        assert ext_tag in meta_tr.nodes
        assert int(np.asarray(meta_tr.nodes[ext_tag]["ss"]).flat[-2]) == 1

    for n in ub:
        if int(meta_plain.nodes[n]["dim"]) <= 0:
            continue
        cache: dict[Any, int] = {}
        n_caps = mg._open_cap_count(n, meta_plain, ub, cache)
        if n_caps <= 0:
            continue
        ss0 = np.asarray(meta_plain.nodes[n]["ss"])
        t1, t2 = (1, 1) if n_caps >= 2 else (1, 0)
        ss_ext = mg._ss_with_truncation_bits(ss0, t1, t2)
        ext_tag = encode_ss(np.asarray(ss_ext, dtype=np.int8))
        for cap_tag in _expected_cap_tags(ss_ext, n_caps):
            assert cap_tag in meta_tr.nodes
            assert meta_tr.has_edge(ext_tag, cap_tag)
            trunc_shis = [
                shi
                for _u, v, _k, ed in meta_tr.out_edges(ext_tag, keys=True, data=True)
                if v == cap_tag
                if (shi := ed.get("shi")) is not None
            ]
            assert trunc_shis, f"expected truncation incidence from {ext_tag!r} to {cap_tag!r}"
            assert all(int(s) >= 0 for s in trunc_shis)

    for n, a in meta_tr.nodes(data=True):
        if int(a.get("dim", -1)) != 1:
            continue
        if a.get("finite") is None:
            # Phantom 1-cells (empty geometry) are excluded from homology incidence.
            continue
        zero_faces = [v for _u, v, _ in meta_tr.out_edges(n, data=True) if int(meta_tr.nodes[v].get("dim", -1)) == 0]
        assert 1 <= len(zero_faces) <= 2, (
            f"1-cell {n!r} should have 1 or 2 0-face endpoints after truncation, got {zero_faces!r}"
        )

    get_betti_numbers(meta_tr, verify_chain_complex=True, verify_connected_components=False)


def test_meta_graph_truncated_satisfies_chain_complex(seeded: int) -> None:
    """Truncated boundary meta-graph must satisfy ``∂²=0`` after phantom exclusion."""
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
    db = cplx.get_boundary_complex(cplx.n - 1)

    meta = db.get_meta_graph(verbose=False)
    Complex.truncate_meta_graph(meta)
    get_betti_numbers(meta, verify_chain_complex=True)


def test_truncation_cap_functor_mixed_boundary_2_cell() -> None:
    """Mixed-boundary 2-cells get their own sphere-cut cap; facet incidence is preserved."""
    import networkx as nx

    meta = nx.MultiDiGraph()

    def _add(ss: np.ndarray, dim: int, finite: bool | None) -> bytes:
        tag = encode_ss(ss)
        meta.add_node(
            tag,
            dim=dim,
            ss=ss,
            finite=finite,
            shis=list(mg.ss_nonzero_indices(ss)),
        )
        return tag

    ss_sheet = np.array([[1, 1, 1]], dtype=np.int8)
    ss_bounded = np.array([[0, 1, 1]], dtype=np.int8)
    ss_unbounded = np.array([[1, 0, 1]], dtype=np.int8)
    sheet = _add(ss_sheet, 2, False)
    bounded = _add(ss_bounded, 1, True)
    unbounded = _add(ss_unbounded, 1, False)
    z_a = _add(np.array([[0, 0, 1]], dtype=np.int8), 0, True)
    z_b = _add(np.array([[0, 1, 0]], dtype=np.int8), 0, True)
    z_c = _add(np.array([[0, 0, 0]], dtype=np.int8), 0, True)
    meta.add_edges_from(
        [
            (sheet, bounded, {"shi": 0}),
            (sheet, unbounded, {"shi": 1}),
            (bounded, z_a, {"shi": 1}),
            (bounded, z_b, {"shi": 2}),
            (unbounded, z_c, {"shi": 0}),
        ]
    )

    ub = {n for n, a in meta.nodes(data=True) if a.get("finite") is False}
    cap_cache: dict[Any, int] = {}
    assert mg._open_cap_count(sheet, meta, ub, cap_cache) == 1
    assert mg._open_cap_count(unbounded, meta, ub, cap_cache) == 1

    meta_tr = meta.copy()
    Complex.truncate_meta_graph(meta_tr)

    # Derive expected trunc pads from open-cap counts (not hardcoded bits).
    sheet_n = cap_cache[sheet]
    ray_n = cap_cache[unbounded]
    t1_s, t2_s = (1, 1) if sheet_n >= 2 else (1, 0)
    t1_r, t2_r = (1, 1) if ray_n >= 2 else (1, 0)
    sheet_ext = encode_ss(mg._ss_with_truncation_bits(ss_sheet, t1_s, t2_s))
    unbounded_ext = encode_ss(mg._ss_with_truncation_bits(ss_unbounded, t1_r, t2_r))
    t1_shi, t2_shi = mg._truncation_bit_indices(np.asarray(meta_tr.nodes[sheet_ext]["ss"]))
    trunc_bit_shis = {t1_shi, t2_shi}
    cap_one_faces = [
        v
        for _, v, ed in meta_tr.out_edges(sheet_ext, data=True)
        if int(meta_tr.nodes[v].get("dim", -1)) == 1 and int(ed.get("shi", -1)) in trunc_bit_shis
    ]
    assert len(cap_one_faces) == sheet_n, f"expected {sheet_n} truncation-bit cap 1-face(s) on 2-cell, got {cap_one_faces!r}"

    sheet_cap_1 = cap_one_faces[0]
    unbounded_cap_0 = _expected_cap_tags(np.asarray(meta_tr.nodes[unbounded_ext]["ss"]), ray_n)[0]
    assert meta_tr.has_edge(sheet_cap_1, unbounded_cap_0), (
        f"cap of unbounded facet {unbounded_cap_0!r} should be facet of cap {sheet_cap_1!r}"
    )


class _FinitePolyStub:
    """Minimal poly stub so sidedness filtering treats the cell as a real network facet."""

    _finite: bool | None = False
    halfspaces = None


def test_open_cap_count_raises_on_unanchored_sidedness_disagreement() -> None:
    """Unanchored k-cells with disagreeing unbounded facets are non-generic (poly-free)."""
    import networkx as nx

    meta = nx.MultiDiGraph()

    def _add(ss: np.ndarray, dim: int, finite: bool | None) -> bytes:
        tag = encode_ss(ss)
        meta.add_node(
            tag,
            dim=dim,
            ss=ss,
            finite=finite,
            shis=list(mg.ss_nonzero_indices(ss)),
        )
        return tag

    sheet = _add(np.array([[1, 1, 1, 1]], dtype=np.int8), 2, False)
    ray = _add(np.array([[1, 0, 1, 1]], dtype=np.int8), 1, False)
    line = _add(np.array([[1, 1, 0, 1]], dtype=np.int8), 1, False)
    z_ray = _add(np.array([[0, 0, 1, 1]], dtype=np.int8), 0, True)
    meta.add_edges_from(
        [
            (sheet, ray, {"shi": 1}),
            (sheet, line, {"shi": 2}),
            (ray, z_ray, {"shi": 0}),
        ]
    )

    ub = {n for n, a in meta.nodes(data=True) if a.get("finite") is False}
    with pytest.raises(NonGenericArrangementError, match="disagree on truncation sidedness"):
        mg._open_cap_count(sheet, meta, ub, {})


def test_open_cap_count_anchored_with_only_bi_infinite_facets_gets_one_cap() -> None:
    """Anchored cell + only bi-infinite line facets ⇒ n_caps=1 (not 0).

    Production meta-graphs attach ``poly`` to 1-cells, so sidedness filtering drops
    bi-infinite lines. That filter must not be reused as an existence test for caps.
    """
    import networkx as nx

    meta = nx.MultiDiGraph()

    def _add(
        ss: np.ndarray,
        dim: int,
        finite: bool | None,
        *,
        poly: Any | None = None,
        comb_n_zero_faces: int | None = None,
    ) -> bytes:
        tag = encode_ss(ss)
        attrs: dict[str, Any] = {
            "dim": dim,
            "ss": ss,
            "finite": finite,
            "shis": list(mg.ss_nonzero_indices(ss)),
            "poly": poly,
        }
        if comb_n_zero_faces is not None:
            attrs["comb_n_zero_faces"] = comb_n_zero_faces
        meta.add_node(tag, **attrs)
        return tag

    # 2-cell with a bounded 1-face (anchor) and a bi-infinite line facet (poly set, 0 zeros).
    sheet = _add(np.array([[1, 1, 1]], dtype=np.int8), 2, False)
    bounded = _add(np.array([[0, 1, 1]], dtype=np.int8), 1, True)
    line = _add(
        np.array([[1, 0, 1]], dtype=np.int8),
        1,
        False,
        poly=_FinitePolyStub(),
        comb_n_zero_faces=0,
    )
    z_a = _add(np.array([[0, 0, 1]], dtype=np.int8), 0, True)
    z_b = _add(np.array([[0, 1, 0]], dtype=np.int8), 0, True)
    meta.add_edges_from(
        [
            (sheet, bounded, {"shi": 0}),
            (sheet, line, {"shi": 1}),
            (bounded, z_a, {"shi": 1}),
            (bounded, z_b, {"shi": 2}),
        ]
    )

    ub = {n for n, a in meta.nodes(data=True) if a.get("finite") is False}
    # Sidedness filter drops the bi-infinite line; raw facets remain nonempty.
    assert mg._facets_for_sidedness_propagation(sheet, meta, ub, 1) == []
    assert mg._raw_unbounded_facets(sheet, meta, ub, 1) == [line]
    assert mg._open_cap_count(sheet, meta, ub, {}) == 1
    assert mg._open_cap_count(line, meta, ub, {}) == 2

    # Truncation must assign a sphere-cut (not raise in careful mode). Homology of
    # this minimal hand-built graph need not be ∂²=0 (line–sheet trunc-bit mismatch
    # is intentional; full witnesses live in shi-regression / Synthetic_Progress).
    meta_tr = meta.copy()
    Complex.truncate_meta_graph(meta_tr)
    sheet_ext = encode_ss(mg._ss_with_truncation_bits(np.array([[1, 1, 1]], dtype=np.int8), 1, 0))
    assert sheet_ext in meta_tr.nodes
    t1_shi, _t2 = mg._truncation_bit_indices(np.asarray(meta_tr.nodes[sheet_ext]["ss"]))
    cap = face_tag(np.asarray(meta_tr.nodes[sheet_ext]["ss"]), int(t1_shi))
    assert cap in meta_tr.nodes, "anchored sheet must materialize its trunc-cap"
    assert meta_tr.has_edge(sheet_ext, cap)


def test_open_cap_count_with_poly_filters_bi_infinite_from_unanchored_inheritance() -> None:
    """With ``poly`` set, unanchored all-bi-infinite sheets default to one local cap."""
    import networkx as nx

    meta = nx.MultiDiGraph()

    def _add(ss: np.ndarray, dim: int, finite: bool | None) -> bytes:
        tag = encode_ss(ss)
        meta.add_node(
            tag,
            dim=dim,
            ss=ss,
            finite=finite,
            shis=list(mg.ss_nonzero_indices(ss)),
            poly=_FinitePolyStub() if dim == 1 else None,
            comb_n_zero_faces=0 if dim == 1 else None,
        )
        return tag

    sheet = _add(np.array([[1, 1, 1]], dtype=np.int8), 2, False)
    line_a = _add(np.array([[0, 1, 1]], dtype=np.int8), 1, False)
    line_b = _add(np.array([[1, 0, 1]], dtype=np.int8), 1, False)
    meta.add_edges_from([(sheet, line_a, {"shi": 0}), (sheet, line_b, {"shi": 1})])

    ub = {n for n, a in meta.nodes(data=True) if a.get("finite") is False}
    # Inheritance filter empty → default local cap (does not raise disagreement).
    assert mg._facets_for_sidedness_propagation(sheet, meta, ub, 1) == []
    assert mg._open_cap_count(sheet, meta, ub, {}) == 1
    assert mg._open_cap_count(line_a, meta, ub, {}) == 2
    assert mg._open_cap_count(line_b, meta, ub, {}) == 2


def _truncation_handbuilt_node(dim: int, ss: list[int], finite: bool | None) -> dict[str, Any]:
    ss_arr = np.array([ss], dtype=np.int8)
    return {"dim": dim, "ss": ss_arr, "finite": finite, "shis": list(mg.ss_nonzero_indices(ss_arr))}


def _isolated_ray_meta() -> nx.MultiDiGraph[Any]:
    """Single ray with one bounded endpoint (minimal ray-capping case)."""
    meta: nx.MultiDiGraph[Any] = nx.MultiDiGraph()
    meta.add_node("ray", **_truncation_handbuilt_node(1, [0, 1], False))
    meta.add_node("z0", **_truncation_handbuilt_node(0, [0, 0], True))
    meta.add_edge("ray", "z0", shi=1)
    return meta


def _mixed_boundary_wedge_meta() -> nx.MultiDiGraph[Any]:
    """Half-infinite strip: bounded edge AB plus rays at A and B."""
    meta: nx.MultiDiGraph[Any] = nx.MultiDiGraph()
    meta.add_node("P", **_truncation_handbuilt_node(2, [1, 1, 1], False))
    meta.add_node("AB", **_truncation_handbuilt_node(1, [0, 1, 1], True))
    meta.add_node("rayA", **_truncation_handbuilt_node(1, [1, 0, 1], False))
    meta.add_node("rayB", **_truncation_handbuilt_node(1, [1, 1, 0], False))
    meta.add_node("A", **_truncation_handbuilt_node(0, [0, 0, 1], True))
    meta.add_node("B", **_truncation_handbuilt_node(0, [0, 1, 0], True))
    meta.add_edges_from(
        [
            ("P", "AB", {"shi": 0}),
            ("P", "rayA", {"shi": 1}),
            ("P", "rayB", {"shi": 2}),
            ("AB", "A", {"shi": 1}),
            ("AB", "B", {"shi": 2}),
            ("rayA", "A", {"shi": 0}),
            ("rayB", "B", {"shi": 0}),
        ]
    )
    return meta


def test_truncate_caps_an_isolated_ray() -> None:
    """A ray with one bounded endpoint gains one cap 0-cell and becomes a segment."""
    meta = _isolated_ray_meta()
    n_before = meta.number_of_nodes()

    Complex.truncate_meta_graph(meta)

    assert meta.number_of_nodes() == n_before + 1, "expected exactly one new cap 0-cell"

    (ray_tag,) = [n for n, a in meta.nodes(data=True) if int(a.get("dim", -1)) == 1]
    zero_faces = [v for _u, v, _k in meta.out_edges(ray_tag, keys=True) if int(meta.nodes[v]["dim"]) == 0]
    assert len(zero_faces) == 2, "capped ray should have two combinatorial 0-faces"
    assert meta.nodes[ray_tag]["finite"] is True, "capped ray should be reclassified as bounded"


def test_truncate_closes_mixed_boundary_cell_into_a_disk() -> None:
    """Truncating a half-infinite strip yields a closed disk with trivial homology."""
    meta = _mixed_boundary_wedge_meta()
    Complex.truncate_meta_graph(meta)

    unbounded_after = [n for n, a in meta.nodes(data=True) if a.get("finite") is False]
    assert not unbounded_after, f"truncation should leave no unbounded cells, got {unbounded_after!r}"

    try:
        betti = get_betti_numbers(meta, verify_chain_complex=True, verify_connected_components=False)
    except ChainComplexInconsistent as exc:
        pytest.fail(f"truncated meta-graph is not a valid chain complex (∂²≠0): {exc}")

    assert betti == {0: 1}, f"truncated strip should be a contractible disk, got {betti!r}"


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
                n_zero = len(zero_faces)
                n_shis = len(poly._shis or [])
                assert 0 <= n_shis <= 2, f"1-cell {tag!r}: expected 0–2 flip SHIs, got {n_shis}"
                assert 0 <= n_zero <= 2, f"1-cell {tag!r}: expected 0–2 combinatorial 0-faces, got {n_zero}"
                finite[tag] = n_zero >= 2
                if finite[tag]:
                    assert n_shis >= 2, f"bounded 1-cell {tag!r} must list two flip SHIs"
                elif n_shis >= 2:
                    assert n_zero < 2, f"unbounded 1-cell {tag!r} cannot have two 0-faces and two flip SHIs"
                continue
            faces = [dst for src, dst, _ in face_edges if src == tag and dst in finite]
            if not faces:
                continue
            if any(not finite[f] for f in faces):
                finite[tag] = False
            else:
                finite[tag] = True
    return finite


def test_propagate_infeasible_exclusion_expands_to_cofaces() -> None:
    """Cofaces of phantom faces must be excluded so ``∂²=0`` is preserved."""
    from relucent.graph.incidence import propagate_infeasible_exclusion

    bad = b"\x01"
    one = b"\x02"
    two = b"\x03"
    edges_by_dim = {
        2: ([(two, one, 0), (two, bad, 1)], []),
        3: ([(b"\x04", two, 0)], []),
    }
    excluded = propagate_infeasible_exclusion({bad}, edges_by_dim)
    assert bad in excluded
    assert two in excluded
    assert b"\x04" in excluded
    assert one not in excluded


def test_classify_finite_combinatorial_fixed_point() -> None:
    """A single ascending pass can leave k-cells pending; fixed point resolves them."""
    from relucent.graph.incidence import classify_finite_combinatorial

    def _poly(dim: int, tag_byte: int) -> Polyhedron:
        ss = np.zeros((1, 4), dtype=np.int8)
        p = Polyhedron(None, ss, halfspaces=None, finite=None)
        p._finite_computed = False
        p._finite = None
        object.__setattr__(p, "dim", dim)  # type: ignore[misc]
        p.tag = bytes([tag_byte])  # type: ignore[misc]
        return p

    one_ub = _poly(1, 1)
    one_ub._finite = False
    one_ub._finite_computed = True
    one_bd = _poly(1, 2)
    one_bd._finite = True
    one_bd._finite_computed = True

    face_c = _poly(2, 3)
    face_b = _poly(2, 4)
    coface_y = _poly(3, 5)

    lookup = {
        one_ub.tag: one_ub,
        one_bd.tag: one_bd,
        face_c.tag: face_c,
        face_b.tag: face_b,
        coface_y.tag: coface_y,
    }
    by_dim = {1: [one_ub, one_bd], 2: [face_c, face_b], 3: [coface_y]}
    edges_by_dim = {
        2: (
            [
                (face_c.tag, one_ub.tag, 0),
                (face_b.tag, one_bd.tag, 1),
                (face_b.tag, face_c.tag, 2),
            ],
            [],
        ),
        3: ([(coface_y.tag, face_b.tag, 0)], []),
    }

    n = classify_finite_combinatorial(by_dim, lookup, edges_by_dim)
    assert n >= 2
    assert face_c._finite is False
    assert face_b._finite is False
    assert coface_y._finite is False


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
    """Meta-graph node ``shis`` match :func:`~relucent.graph.meta_graph.cubical_cell_shis` per slice."""
    set_seeds(seeded)
    net = mlp(widths=[4, 5, 5, 1], add_last_relu=True)
    cplx = Complex(net)
    start = torch.randn(4, dtype=torch.float64)
    explore_for_topology(cplx, start.numpy(), max_polys=5000)

    chain = cplx.get_chain_complex(verbose=False)
    by_dim = {int(cc.index2poly[0].dim): cc for cc in chain if len(cc)}
    dim_neighbor_tags = {k: {p.tag for p in cc} for k, cc in by_dim.items()}

    meta = cplx.get_meta_graph(verbose=False)
    mg.verify_meta_graph_one_cells(meta)
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
    mg.verify_meta_graph_one_cells(meta)
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
