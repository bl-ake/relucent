"""Unit tests for relucent.vis plotting helpers and dispatchers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from relucent.poly import Polyhedron

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import pytest
import torch

from relucent import vis


@dataclass(eq=False)
class FakePoly:
    ambient_dim: int
    W_shape0: int
    dim: int
    ss_np: np.ndarray
    verts: np.ndarray | None
    center: np.ndarray | None = None
    interior_point: np.ndarray | None = None
    Wl2: float = 1.0
    halfspaces_np: np.ndarray | None = None

    def __post_init__(self):
        self.W = np.zeros((self.W_shape0, self.W_shape0))
        self.net: Any = SimpleNamespace(device="cpu", dtype=torch.float32)
        self._graph_return: dict[str, go.Mesh3d | go.Scatter3d] | go.Scatter3d | None = None

    def __str__(self) -> str:
        return f"poly-{id(self)}"

    def get_bounded_vertices(self, _bound: float):
        return self.verts

    def _get_bounded_plot_geometry(self, bound: float):
        return vis.bounded_plot_geometry(cast("Polyhedron", self), bound)

    def plot_cells(self, **kwargs):
        return vis.plot_polyhedron(cast("Polyhedron", self), plot_mode="cells", **kwargs)

    def plot_graph(self, **kwargs):
        if self._graph_return is not None:
            return self._graph_return
        return vis.plot_polyhedron(cast("Polyhedron", self), plot_mode="graph", **kwargs)


class TinyNet:
    device = "cpu"
    dtype = torch.float32

    def __call__(self, x):
        # Returns (N, 2) so vis._poly_traces_2d_graph can index [:, 1]
        return torch.stack((x[:, 0], x[:, 0] + x[:, 1]), dim=1)


class FakeComplex:
    def __init__(self, dim: int, polys: list[FakePoly]):
        self.dim = dim
        self._polys = polys
        self.net = TinyNet()

    def __iter__(self):
        return iter(self._polys)

    def get_dual_graph(self):
        G = nx.Graph()
        for i, p in enumerate(self._polys):
            G.add_node(p)
            if i > 0:
                G.add_edge(self._polys[i - 1], p)
        return G


def test_get_colors_empty_and_basic():
    assert vis.get_colors([]) == []
    colors = vis.get_colors([0.0, 1.0, 2.0])
    assert len(colors) == 3
    assert all(c.startswith("#") and len(c) == 7 for c in colors)


def test_bounded_plot_geometry_point_segment_polygon():
    point_poly = FakePoly(
        ambient_dim=2,
        W_shape0=2,
        dim=2,
        ss_np=np.array([[0, 0, 1]]),
        verts=np.array([[1.0, 2.0], [1.0, 2.0]]),
    )
    result = vis.bounded_plot_geometry(cast("Polyhedron", point_poly), bound=10.0)
    assert result is not None
    kind, verts = result
    assert kind == "point"
    assert verts.shape == (1, 2)

    seg_poly = FakePoly(
        ambient_dim=2,
        W_shape0=2,
        dim=2,
        ss_np=np.array([[0, 1, 1]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    )
    result = vis.bounded_plot_geometry(cast("Polyhedron", seg_poly), bound=10.0)
    assert result is not None
    kind, verts = result
    assert kind == "segment"
    assert verts.shape == (2, 2)

    poly_poly = FakePoly(
        ambient_dim=2,
        W_shape0=2,
        dim=2,
        ss_np=np.array([[1, 1, 1]]),
        verts=np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [0.5, 0.2]]),
    )
    result = vis.bounded_plot_geometry(cast("Polyhedron", poly_poly), bound=10.0)
    assert result is not None
    kind, verts = result
    assert kind == "polygon"
    assert verts.shape[1] == 2


def test_poly_traces_3d_complex_variants():
    bad = FakePoly(ambient_dim=2, W_shape0=2, dim=2, ss_np=np.array([[1]]), verts=np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError):
        vis._poly_traces_3d_complex(cast("Polyhedron", bad))

    point = FakePoly(
        ambient_dim=3,
        W_shape0=3,
        dim=3,
        ss_np=np.array([[1, 1]]),
        verts=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    )
    out = vis._poly_traces_3d_complex(cast("Polyhedron", point), color="blue")
    assert len(out) == 1 and isinstance(out[0], go.Scatter3d)
    assert out[0].mode == "markers"

    seg = FakePoly(
        ambient_dim=3,
        W_shape0=3,
        dim=3,
        ss_np=np.array([[1, 1]]),
        verts=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    out = vis._poly_traces_3d_complex(cast("Polyhedron", seg), color="green")
    assert len(out) == 1 and isinstance(out[0], go.Scatter3d)
    assert out[0].mode == "lines"

    face = FakePoly(
        ambient_dim=3,
        W_shape0=3,
        dim=3,
        ss_np=np.array([[1, 1]]),
        verts=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    out = vis._poly_traces_3d_complex(cast("Polyhedron", face), color="red", filled=True)
    assert len(out) == 1 and isinstance(out[0], go.Mesh3d)


def test_poly_traces_2d_complex_and_halfspaces():
    poly = FakePoly(
        ambient_dim=2,
        W_shape0=2,
        dim=2,
        ss_np=np.array([[1, 1]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        halfspaces_np=np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]),
    )
    traces = vis._poly_traces_2d_complex(cast("Polyhedron", poly), plot_halfspaces=True)
    assert len(traces) >= 3
    assert isinstance(traces[0], go.Scatter)

    bad = FakePoly(
        ambient_dim=3,
        W_shape0=3,
        dim=3,
        ss_np=np.array([[1]]),
        verts=np.array([[0.0, 0.0, 0.0]]),
    )
    with pytest.raises(ValueError):
        vis._poly_traces_2d_complex(cast("Polyhedron", bad))


def test_poly_traces_2d_graph_variants_and_exception_path():
    poly = FakePoly(
        ambient_dim=2,
        W_shape0=2,
        dim=2,
        ss_np=np.array([[1]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )
    poly.net = TinyNet()
    out = vis._poly_traces_2d_graph(cast("Polyhedron", poly))
    assert out is not None and "mesh" in out and "outline" in out

    seg = FakePoly(
        ambient_dim=2,
        W_shape0=2,
        dim=2,
        ss_np=np.array([[1]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0]]),
    )
    seg.net = TinyNet()
    out = vis._poly_traces_2d_graph(cast("Polyhedron", seg))
    assert out is not None and "outline" in out

    bad = FakePoly(
        ambient_dim=2,
        W_shape0=2,
        dim=2,
        ss_np=np.array([[1]]),
        verts=np.array([[0.0, 0.0]]),
    )
    bad.net = SimpleNamespace()  # missing device/dtype and call -> warning path
    with warnings.catch_warnings(record=True) as rec:
        out = vis._poly_traces_2d_graph(cast("Polyhedron", bad))
    assert out is None
    assert isinstance(rec, list)


def test_plot_polyhedron_dispatch_and_validation():
    poly2 = FakePoly(
        ambient_dim=2,
        W_shape0=2,
        dim=2,
        ss_np=np.array([[1]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )
    poly2.net = TinyNet()
    cells = vis.plot_polyhedron(cast("Polyhedron", poly2), plot_mode="cells")
    graph = vis.plot_polyhedron(cast("Polyhedron", poly2), plot_mode="graph")
    assert isinstance(cells, list)
    assert isinstance(graph, dict) or graph is None

    poly_bad = FakePoly(ambient_dim=4, W_shape0=4, dim=4, ss_np=np.array([[1]]), verts=np.ones((2, 4)))
    with pytest.raises(ValueError):
        vis.plot_polyhedron(cast("Polyhedron", poly_bad), plot_mode="cells")
    with pytest.raises(ValueError):
        vis.plot_polyhedron(poly2, plot_mode="unknown")  # type: ignore[arg-type]


def test_color_helpers_and_highlight(monkeypatch):
    polys = [SimpleNamespace(Wl2=1.0), SimpleNamespace(Wl2=2.0)]
    cpx = SimpleNamespace(get_dual_graph=lambda: nx.Graph())
    wl2_colors = vis._per_poly_colors(cpx, polys, color="Wl2", remap_equitable=True)  # type: ignore[arg-type]
    assert len(wl2_colors) == 2

    monkeypatch.setattr(
        nx.algorithms.coloring, "equitable_color", lambda *_args, **_kwargs: (_ for _ in ()).throw(Exception("x"))
    )
    fallback = vis._equitable_colors(nx.Graph(), [0, 1, 2], remap=False)
    assert len(fallback) == 3

    assert vis._highlight("blue", "a", None) == "blue"
    assert vis._highlight("blue", "a", {"a"}) == "red"


def test_complex_figure_builders_and_plot_complex_dispatch():
    p1 = FakePoly(
        ambient_dim=2,
        W_shape0=2,
        dim=2,
        ss_np=np.array([[1, 0]]),
        verts=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        center=np.array([0.2, 0.2]),
        interior_point=np.array([0.2, 0.2]),
    )
    p2 = FakePoly(
        ambient_dim=2,
        W_shape0=2,
        dim=2,
        ss_np=np.array([[0, 1]]),
        verts=np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
        center=np.array([1.2, 1.2]),
        interior_point=np.array([1.2, 1.2]),
    )
    p1.net = TinyNet()
    p2.net = TinyNet()
    c2 = FakeComplex(2, [p1, p2])
    fig2: go.Figure = vis._complex_figure_2d_cells(c2, label_regions=True, highlight_regions={p1})  # type: ignore[arg-type]
    assert isinstance(fig2, go.Figure)
    assert len(fig2.data) >= 2  # type: ignore[arg-type]

    p3 = FakePoly(
        ambient_dim=3,
        W_shape0=3,
        dim=3,
        ss_np=np.array([[1, 1, 1]]),
        verts=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        center=np.array([0.1, 0.1, 0.0]),
        interior_point=np.array([0.1, 0.1, 0.0]),
    )
    c3 = FakeComplex(3, [p3])
    fig3: go.Figure = vis._complex_figure_3d_cells(c3, label_regions=True, fill_mode="filled")  # type: ignore[arg-type]
    assert isinstance(fig3, go.Figure)
    assert len(fig3.data) >= 1  # type: ignore[arg-type]

    p1._graph_return = {"outline": go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode="lines")}
    p2._graph_return = {"mesh": go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1], k=[2])}
    figg: go.Figure = vis._complex_figure_graph(c2, project=None)  # type: ignore[arg-type]
    assert isinstance(figg, go.Figure)
    assert len(figg.data) >= 2  # type: ignore[arg-type]

    out_cells = vis.plot_complex(c2, plot_mode="cells")  # type: ignore[arg-type]
    out_graph = vis.plot_complex(c2, plot_mode="graph")  # type: ignore[arg-type]
    assert isinstance(out_cells, go.Figure)
    assert isinstance(out_graph, go.Figure)
    with pytest.raises(ValueError):
        vis.plot_complex(FakeComplex(4, []), plot_mode="cells")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        vis.plot_complex(c2, plot_mode="unknown")  # type: ignore[arg-type]
