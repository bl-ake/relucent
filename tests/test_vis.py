"""Unit tests for relucent.vis plotting helpers and dispatchers."""

from __future__ import annotations

import warnings
from types import MethodType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from relucent.poly import Polyhedron

import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pytest

from relucent import vis
from relucent.complex import Complex
from relucent.model import LinearLayer, ReLUNetwork
from relucent.poly import Polyhedron


def _tiny_nn(input_dim: int) -> ReLUNetwork:
    # Identity map to keep graph plotting helpers simple and deterministic.
    w = np.eye(input_dim, dtype=np.float64)
    b = np.zeros((1, input_dim), dtype=np.float64)
    return ReLUNetwork([LinearLayer(weight=w, bias=b)], input_shape=(input_dim,))


def _poly_with_vertices(
    *,
    ambient_dim: int,
    ss_np: np.ndarray,
    verts: np.ndarray | None,
    halfspaces_np: np.ndarray | None = None,
    center: np.ndarray | None = None,
    interior_point: np.ndarray | None = None,
    Wl2: float = 1.0,
    net: ReLUNetwork | None = None,
) -> Polyhedron:
    # Create a real Polyhedron with a dummy halfspace array of the right ambient dimension.
    dummy_hs = np.zeros((max(1, ss_np.shape[1]), ambient_dim + 1), dtype=float)
    p = Polyhedron(net or _tiny_nn(ambient_dim), ss_np, halfspaces=dummy_hs)

    # Seed caches used by plotting helpers.
    p._w = np.zeros((ambient_dim, ambient_dim), dtype=float)
    if halfspaces_np is not None:
        p._halfspaces_np = halfspaces_np
    p._center = center
    p._interior_point = interior_point
    p._Wl2 = Wl2

    # Avoid heavy geometry computations by providing bounded vertices directly.
    def _get_bounded_vertices(_self: Polyhedron, _bound: float) -> np.ndarray | None:
        return verts

    p.get_bounded_vertices = MethodType(_get_bounded_vertices, p)
    return p


def _complex_with_polys(dim: int, polys: list[Polyhedron]) -> Complex:
    """Build a real Complex containing the provided polyhedra."""
    cx = Complex(_tiny_nn(dim))
    for p in polys:
        cx.add_polyhedron(p, check_exists=False)
    return cx


def test_get_colors_empty_and_basic():
    assert vis.get_colors([]) == []
    colors = vis.get_colors([0.0, 1.0, 2.0])
    assert len(colors) == 3
    assert all(c.startswith("#") and len(c) == 7 for c in colors)


def test_bounded_plot_geometry_point_segment_polygon():
    point_poly = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[0, 0, 1]]),
        verts=np.array([[1.0, 2.0], [1.0, 2.0]]),
    )
    result = vis.bounded_plot_geometry(point_poly, bound=10.0)
    assert result is not None
    kind, verts = result
    assert kind == "point"
    assert verts.shape == (1, 2)

    seg_poly = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[0, 1, 1]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    )
    result = vis.bounded_plot_geometry(seg_poly, bound=10.0)
    assert result is not None
    kind, verts = result
    assert kind == "segment"
    assert verts.shape == (2, 2)

    poly_poly = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[1, 1, 1]]),
        verts=np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [0.5, 0.2]]),
    )
    result = vis.bounded_plot_geometry(poly_poly, bound=10.0)
    assert result is not None
    kind, verts = result
    assert kind == "polygon"
    assert verts.shape[1] == 2


def test_poly_traces_3d_complex_variants():
    bad = _poly_with_vertices(ambient_dim=2, ss_np=np.array([[1]]), verts=np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError):
        vis._poly_traces_3d_complex(bad)

    point = _poly_with_vertices(
        ambient_dim=3,
        ss_np=np.array([[1, 1]]),
        verts=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    )
    out = vis._poly_traces_3d_complex(point, color="blue")
    assert len(out) == 1 and isinstance(out[0], go.Scatter3d)
    assert out[0].mode == "markers"

    seg = _poly_with_vertices(
        ambient_dim=3,
        ss_np=np.array([[1, 1]]),
        verts=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    out = vis._poly_traces_3d_complex(seg, color="green")
    assert len(out) == 1 and isinstance(out[0], go.Scatter3d)
    assert out[0].mode == "lines"

    face = _poly_with_vertices(
        ambient_dim=3,
        ss_np=np.array([[1, 1]]),
        verts=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    out = vis._poly_traces_3d_complex(face, color="red", filled=True)
    assert len(out) == 1 and isinstance(out[0], go.Mesh3d)


def test_poly_traces_2d_complex_and_halfspaces():
    poly = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[1, 1]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        halfspaces_np=np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]),
    )
    traces = vis._poly_traces_2d_complex(poly, plot_halfspaces=True)
    assert len(traces) >= 3
    assert isinstance(traces[0], go.Scatter)

    bad = _poly_with_vertices(
        ambient_dim=3,
        ss_np=np.array([[1]]),
        verts=np.array([[0.0, 0.0, 0.0]]),
    )
    with pytest.raises(ValueError):
        vis._poly_traces_2d_complex(bad)


def test_poly_traces_2d_graph_variants_and_exception_path():
    poly = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[1]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        net=_tiny_nn(2),
    )
    out = vis._poly_traces_2d_graph(poly)
    assert out is not None and "mesh" in out and "outline" in out

    seg = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[1]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0]]),
        net=_tiny_nn(2),
    )
    out = vis._poly_traces_2d_graph(seg)
    assert out is not None and "outline" in out

    bad = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[1]]),
        verts=np.array([[0.0, 0.0]]),
        net=_tiny_nn(2),
    )
    # Missing device/dtype and call -> warning path.
    object.__setattr__(bad, "_net", cast(Any, SimpleNamespace()))
    with warnings.catch_warnings(record=True) as rec:
        out = vis._poly_traces_2d_graph(bad)
    assert out is None
    assert isinstance(rec, list)


def test_plot_polyhedron_dispatch_and_validation():
    poly2 = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[1]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        net=_tiny_nn(2),
    )
    cells = vis.plot_polyhedron(poly2, plot_mode="cells")
    graph = vis.plot_polyhedron(poly2, plot_mode="graph")
    assert isinstance(cells, list)
    assert isinstance(graph, dict) or graph is None

    poly_bad = _poly_with_vertices(ambient_dim=4, ss_np=np.array([[1]]), verts=np.ones((2, 4)))
    with pytest.raises(ValueError):
        vis.plot_polyhedron(poly_bad, plot_mode="cells")
    with pytest.raises(ValueError):
        vis.plot_polyhedron(poly2, plot_mode="unknown")  # pyright: ignore[reportCallIssue,reportArgumentType]


def test_color_helpers_and_highlight():
    p1 = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[1]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0]]),
        Wl2=1.0,
    )
    p2 = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[1]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0]]),
        Wl2=2.0,
    )
    cpx = _complex_with_polys(2, [p1, p2])
    wl2_colors = vis._per_poly_colors(cpx, [p1, p2], color="Wl2", remap_equitable=True)
    assert len(wl2_colors) == 2

    assert vis._highlight("blue", "a", None) == "blue"
    assert vis._highlight("blue", "a", {"a"}) == "red"


def test_equitable_colors_uses_plotly_scheme_for_degree_at_most_10(monkeypatch):
    nodes = list(range(6))
    dual = nx.Graph()
    dual.add_nodes_from(nodes)
    # Star graph: max degree is 5 -> should use Plotly qualitative scheme.
    dual.add_edges_from((0, i) for i in range(1, 6))
    seen: dict[str, int] = {}

    def fake_equitable_color(graph: nx.Graph[object], num_colors: int) -> dict[object, int]:
        seen["num_colors"] = num_colors
        return {node: i % num_colors for i, node in enumerate(graph.nodes)}

    monkeypatch.setattr(nx.algorithms.coloring, "equitable_color", fake_equitable_color)
    colors = vis._equitable_colors(dual, nodes, remap=False)
    assert seen["num_colors"] == min(len(px.colors.qualitative.Plotly), len(nodes))
    assert len(colors) == len(nodes)
    assert all(c in px.colors.qualitative.Plotly for c in colors)


def test_equitable_colors_uses_light24_without_red_for_degree_11_to_23(monkeypatch):
    nodes = list(range(13))
    dual = nx.Graph()
    dual.add_nodes_from(nodes)
    # Star graph: max degree is 12 -> should use Light24 minus the first color.
    dual.add_edges_from((0, i) for i in range(1, 13))
    seen: dict[str, int] = {}

    def fake_equitable_color(graph: nx.Graph[object], num_colors: int) -> dict[object, int]:
        seen["num_colors"] = num_colors
        return {node: i % num_colors for i, node in enumerate(graph.nodes)}

    monkeypatch.setattr(nx.algorithms.coloring, "equitable_color", fake_equitable_color)
    colors = vis._equitable_colors(dual, nodes, remap=False)
    light24_without_red = px.colors.qualitative.Light24[1:]
    assert seen["num_colors"] == min(len(light24_without_red), len(nodes))
    assert len(colors) == len(nodes)
    assert all(c in light24_without_red for c in colors)
    assert px.colors.qualitative.Light24[0] not in colors


def test_equitable_colors_falls_back_to_scheme_for_degree_above_23():
    nodes = list(range(25))
    dual = nx.Graph()
    dual.add_nodes_from(nodes)
    # Star graph: max degree is 24 -> fallback branch.
    dual.add_edges_from((0, i) for i in range(1, 25))

    colors = vis._equitable_colors(dual, nodes, remap=False)
    assert len(colors) == len(nodes)
    assert all(c in px.colors.qualitative.Plotly for c in colors)


def test_equitable_colors_falls_back_to_random_scheme_when_equitable_fails(monkeypatch):
    nodes = list(range(6))
    dual = nx.Graph()
    dual.add_nodes_from(nodes)
    dual.add_edges_from((0, i) for i in range(1, 6))

    def fail_equitable_color(*_args: object, **_kwargs: object) -> dict[object, int]:
        raise nx.NetworkXAlgorithmError("equitable coloring failed")

    monkeypatch.setattr(nx.algorithms.coloring, "equitable_color", fail_equitable_color)
    colors = vis._equitable_colors(dual, nodes, remap=False)
    assert len(colors) == len(nodes)
    assert all(c in px.colors.qualitative.Plotly for c in colors)


def test_plot_polyhedron_hide_unbounded_flag(monkeypatch):
    poly = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[1]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        net=_tiny_nn(2),
    )
    poly._finite = False
    poly._finite_computed = True

    # Ensure no plotting backend gets called when hide_unbounded=True.
    monkeypatch.setattr(
        vis,
        "_poly_traces_2d_complex",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("called")),
    )
    monkeypatch.setattr(
        vis,
        "_poly_traces_2d_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("called")),
    )

    assert vis.plot_polyhedron(poly, plot_mode="cells", hide_unbounded=True) == []
    assert vis.plot_polyhedron(poly, plot_mode="graph", hide_unbounded=True) is None


def test_plot_complex_hide_unbounded_filters_regions(monkeypatch):
    p_bounded = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[1, 0]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        interior_point=np.array([0.1, 0.1]),
        net=_tiny_nn(2),
    )
    p_bounded._finite = True
    p_bounded._finite_computed = True

    p_unbounded = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[0, 1]]),
        verts=np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
        net=_tiny_nn(2),
    )
    p_unbounded._finite = False
    p_unbounded._finite_computed = True

    c2 = _complex_with_polys(2, [p_bounded, p_unbounded])
    c2.get_dual_graph = MethodType(  # type: ignore[method-assign]
        lambda _self, **_kwargs: nx.Graph([(p_bounded, p_unbounded)]),
        c2,
    )

    # Keep this test isolated from bound-intersection internals and expensive plotting.
    monkeypatch.setattr(vis, "_poly_intersects_plot_bound", lambda *_args, **_kwargs: True)

    def fake_plot_cells(_self: Polyhedron, **_kwargs: object) -> list[go.Scatter]:
        return [go.Scatter(x=[0.0, 1.0], y=[0.0, 1.0], mode="lines")]

    p_bounded.plot_cells = MethodType(fake_plot_cells, p_bounded)  # type: ignore[method-assign]
    p_unbounded.plot_cells = MethodType(fake_plot_cells, p_unbounded)  # type: ignore[method-assign]

    fig_default = cast(go.Figure, vis.plot_complex(c2, plot_mode="cells", bound=10.0))
    fig_hidden = cast(go.Figure, vis.plot_complex(c2, plot_mode="cells", hide_unbounded=True, bound=10.0))
    default_data = cast(tuple[Any, ...], fig_default.data)
    hidden_data = cast(tuple[Any, ...], fig_hidden.data)
    assert len(default_data) == 2
    assert len(hidden_data) == 1


def test_complex_figure_builders_and_plot_complex_dispatch():
    def _assert_equitable_warning_policy(
        rec: list[warnings.WarningMessage],
        dual_graph: nx.Graph[Polyhedron],
    ) -> None:
        max_degree = max((deg for _, deg in dual_graph.degree()), default=0)
        saw_equitable_warning = any("Equitable coloring could not be found" in str(w.message) for w in rec)
        if saw_equitable_warning and max_degree > 23:
            pytest.fail("Unexpected equitable-coloring warning for a high-degree dual graph " + f"(max degree {max_degree}).")

    p1 = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[1, 0]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        center=np.array([0.2, 0.2]),
        interior_point=np.array([0.2, 0.2]),
        net=_tiny_nn(2),
    )
    p2 = _poly_with_vertices(
        ambient_dim=2,
        ss_np=np.array([[0, 1]]),
        verts=np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
        center=np.array([1.2, 1.2]),
        interior_point=np.array([1.2, 1.2]),
        net=_tiny_nn(2),
    )
    c2 = _complex_with_polys(2, [p1, p2])
    # Make the dual graph deterministic and independent of expensive adjacency logic.
    c2.get_dual_graph = MethodType(  # type: ignore[method-assign]
        lambda _self, **_kwargs: nx.Graph([(p1, p2)]),
        c2,
    )
    c2_graph = c2.get_dual_graph()
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        fig2 = vis._complex_figure_2d_cells(c2, label_regions=True, highlight_regions={p1})
    _assert_equitable_warning_policy(rec2, c2_graph)
    assert isinstance(fig2, go.Figure)
    data2 = fig2.data
    assert isinstance(data2, tuple)
    assert len(data2) >= 2

    p3 = _poly_with_vertices(
        ambient_dim=3,
        ss_np=np.array([[1, 1, 1]]),
        verts=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        center=np.array([0.1, 0.1, 0.0]),
        interior_point=np.array([0.1, 0.1, 0.0]),
        net=_tiny_nn(3),
    )
    c3 = _complex_with_polys(3, [p3])
    c3.get_dual_graph = MethodType(  # type: ignore[method-assign]
        lambda _self, **_kwargs: nx.Graph([(p3, p3)]),
        c3,
    )
    c3_graph = c3.get_dual_graph()
    with warnings.catch_warnings(record=True) as rec3:
        warnings.simplefilter("always")
        fig3 = vis._complex_figure_3d_cells(c3, label_regions=True, fill_mode="filled")
    _assert_equitable_warning_policy(rec3, c3_graph)
    assert isinstance(fig3, go.Figure)
    data3 = fig3.data
    assert isinstance(data3, tuple)
    assert len(data3) >= 1

    p1.plot_graph = MethodType(  # type: ignore[method-assign]
        lambda _self, **_kwargs: {"outline": go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode="lines")},
        p1,
    )
    p2.plot_graph = MethodType(  # type: ignore[method-assign]
        lambda _self, **_kwargs: {"mesh": go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1], k=[2])},
        p2,
    )
    with warnings.catch_warnings(record=True) as recg:
        warnings.simplefilter("always")
        figg = vis._complex_figure_graph(c2, project=None)
    _assert_equitable_warning_policy(recg, c2_graph)
    assert isinstance(figg, go.Figure)
    datag = figg.data
    assert isinstance(datag, tuple)
    assert len(datag) >= 2

    with warnings.catch_warnings(record=True) as rec_cells:
        warnings.simplefilter("always")
        out_cells = vis.plot_complex(c2, plot_mode="cells")
    _assert_equitable_warning_policy(rec_cells, c2_graph)
    with warnings.catch_warnings(record=True) as rec_graph:
        warnings.simplefilter("always")
        out_graph = vis.plot_complex(c2, plot_mode="graph")
    _assert_equitable_warning_policy(rec_graph, c2_graph)
    assert isinstance(out_cells, go.Figure)
    assert isinstance(out_graph, go.Figure)
    with pytest.raises(ValueError):
        vis.plot_complex(_complex_with_polys(4, []), plot_mode="cells")
    with pytest.raises(ValueError):
        vis.plot_complex(c2, plot_mode="unknown")  # type: ignore[arg-type]
