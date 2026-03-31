import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from scipy.spatial import ConvexHull
from tqdm.auto import tqdm

from relucent.config import (
    DEFAULT_COMPLEX_PLOT_BOUND,
    DEFAULT_PLOT_BOUND,
    DEFAULT_PYVIS_SAVE_FILE,
    MAX_IMAGES_PYVIS,
    MAX_NUM_EXAMPLES_PYVIS,
    PIE_LABEL_DISTANCE,
    PLOT_DEFAULT_MAXCOORD,
    PLOT_MARGIN_FACTOR,
    TOL_NEARLY_VERTICAL,
)

if TYPE_CHECKING:
    from relucent.complex import Complex
    from relucent.poly import Polyhedron

__all__ = [
    "data_graph",
    "get_colors",
    "plot_complex",
    "plot_polyhedron",
]


def get_colors(data: Sequence[float], cmap: str = "viridis", **kwargs: Any) -> list[str]:
    """Map numeric values to hex color strings via a colormap."""
    if not data:
        return []
    a = np.asarray(data)
    a = a - np.min(a)
    am = np.max(a)
    a = a / (am if am > 0 else 1)
    a = colormaps[cmap](a)
    a = (a * 255).astype(int)
    return [f"#{x[0]:02x}{x[1]:02x}{x[2]:02x}" for x in a]


def data_graph(
    node_df: Any,
    edge_df: Any,
    dataset: Any | None = None,
    draw_function: Callable[..., Any] = lambda x, **__: x,
    class_labels: bool | None = True,
    node_title_formatter: Callable[[int, Mapping[str, Any]], str] = lambda i, row: (
        row["title"] if "title" in row else str(row)
    ),
    node_label_formatter: Callable[[int, Mapping[str, Any]], str] = lambda i, row: (
        row["label"] if "label" in row else str(i)
    ),
    node_size_formatter: Callable[[Mapping[str, Any]], int] = lambda row: row.get("size", 10),
    edge_title_formatter: Callable[[Mapping[str, Any]], str] = lambda row: row.get("title", ""),
    edge_label_formatter: Callable[[Mapping[str, Any]], str] = lambda row: row.get("label", ""),
    edge_value_formatter: Callable[[Mapping[str, Any]], float | int] = lambda row: row.get("value", 1),
    max_images: int = MAX_IMAGES_PYVIS,
    max_num_examples: int = MAX_NUM_EXAMPLES_PYVIS,
    save_file: str = DEFAULT_PYVIS_SAVE_FILE,
) -> None:
    """Create an interactive pyvis graph from dataframes of nodes and edges."""

    from pyvis.network import Network  # type: ignore

    class_labels_list: list = []
    if class_labels is True and dataset is not None:
        class_labels_list = torch.unique(torch.tensor([dataset[i][1] for i in range(len(dataset))])).tolist()

    G = nx.Graph()
    bar = tqdm(node_df.iterrows(), total=len(node_df), desc="Adding Nodes")
    for i, row in bar:
        if i < max_images:
            num_examples = min(len(row["data"]), max_num_examples) + (class_labels is not False)
            num_rows = np.ceil(np.sqrt(num_examples)).astype(int)
            num_cols = num_examples // num_rows
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
            axs = axs.ravel() if isinstance(axs, np.ndarray) and num_rows > 1 else [axs]
            for j, ax in enumerate(axs[:-1]):
                ax.axis("equal")
                ax.set_axis_off()
                if j <= num_examples:
                    data = row["data"][j]
                    draw_function(data=data, ax=ax)

            if class_labels and "class_proportions" in row:
                axs[-1].pie(
                    row["class_proportions"],
                    labeldistance=PIE_LABEL_DISTANCE,
                    labels=class_labels_list,
                )
            axs[-1].axis("equal")
            axs[-1].set_axis_off()

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            img = Image.frombytes("RGBA", canvas.get_width_height(), bytes(canvas.buffer_rgba()))
            plt.close(fig)
            img.convert("RGB").save(f"images/{i}.png")

        G.add_node(
            i,
            title=node_title_formatter(i, row),
            label=node_label_formatter(i, row),
            image=f"images/{i}.png",
            shape="image",
            size=node_size_formatter(row),
            **{k: str(v) for k, v in row.items() if k not in ["label", "title", "size", "image", "data"]},
        )
    pbar = tqdm(edge_df.iterrows(), total=len(edge_df), desc="Adding Edges")
    for (A, B), row in pbar:
        G.add_edge(
            A,
            B,
            title=edge_title_formatter(row),
            label=edge_label_formatter(row),
            value=edge_value_formatter(row),
        )
        bar.set_postfix({"Nodes": G.number_of_nodes(), "Edges": G.number_of_edges()})
    print(f"Number of Nodes: {G.number_of_nodes()}\nNumber of Edges: {G.number_of_edges()}")

    nt = Network(height="1000px", width="100%")
    nt.from_nx(G)
    nt.show_buttons()
    nt.toggle_physics(False)
    nt.save_graph(save_file)


# --- Polyhedron geometry & traces (used by ``Polyhedron`` methods) -----------------


def bounded_plot_geometry(poly: "Polyhedron", bound: float) -> tuple[str, np.ndarray] | None:
    """Classify bounded 2D geometry for plotting as polygon, segment, or point."""
    vertices = poly.get_bounded_vertices(bound)
    if vertices is None or vertices.size == 0:
        return None
    if vertices.shape[1] != 2:
        return None

    num_zeros = int(np.sum(poly.ss_np == 0))
    if num_zeros >= 2:
        point = vertices.mean(axis=0, keepdims=True)
        return "point", point
    if num_zeros == 1:
        if vertices.shape[0] == 1:
            return "point", vertices
        centered = vertices - vertices.mean(axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            direction = vh[0]
            t = vertices @ direction
            p_min = vertices[np.argmin(t)]
            p_max = vertices[np.argmax(t)]
            segment = np.stack([p_min, p_max], axis=0)
            return "segment", segment
        except Exception:
            uniq = np.unique(vertices, axis=0)
            if uniq.shape[0] == 1:
                return "point", uniq
            return "segment", uniq[:2]

    if poly.dim < 2:
        uniq = np.unique(vertices, axis=0)
        if uniq.shape[0] == 1:
            return "point", uniq
        if uniq.shape[0] == 2:
            return "segment", uniq
        return "polygon", uniq
    try:
        hull = ConvexHull(vertices)
        boundary = vertices[hull.vertices]
        return "polygon", boundary
    except Exception:
        uniq = np.unique(vertices, axis=0)
        if uniq.shape[0] == 1:
            return "point", uniq
        if uniq.shape[0] == 2:
            return "segment", uniq
        return "polygon", uniq


def _poly_traces_3d_complex(
    poly: "Polyhedron",
    showlegend: bool = False,
    bound: float = DEFAULT_PLOT_BOUND,
    filled: bool = False,
    **kwargs: Any,
) -> list[go.Mesh3d | go.Scatter3d]:
    if poly.ambient_dim != 3:
        raise ValueError("Polyhedron must have ambient dimension 3 to plot 3D complex")
    base_kwargs: dict[str, Any] = dict(kwargs)
    line_color = base_kwargs.pop("color", None)
    traces: list[go.Mesh3d | go.Scatter3d] = []
    vertices = poly.get_bounded_vertices(bound)
    if vertices is None or vertices.size == 0 or vertices.shape[1] != 3:
        return traces

    centered = vertices - vertices.mean(axis=0, keepdims=True)
    try:
        _, s, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        s = np.array([])
        vh = np.zeros((0, 3))
    if s.size == 0 or np.all(s < 1e-12):
        point_kwargs = dict(base_kwargs)
        if line_color is not None:
            marker = dict(point_kwargs.get("marker", {}))
            marker.setdefault("color", line_color)
            point_kwargs["marker"] = marker
        traces.append(
            go.Scatter3d(
                x=[vertices[0, 0]],
                y=[vertices[0, 1]],
                z=[vertices[0, 2]],
                mode="markers",
                showlegend=showlegend,
                **point_kwargs,
            )
        )
        return traces

    tol = 1e-6 * s[0]
    eff_dim = int(np.sum(s > tol))
    if eff_dim == 1:
        direction = vh[0]
        t = vertices @ direction
        p_min = vertices[np.argmin(t)]
        p_max = vertices[np.argmax(t)]
        seg = np.stack([p_min, p_max], axis=0)
        line_kwargs = dict(base_kwargs)
        if line_color is not None:
            line = dict(line_kwargs.get("line", {}))
            line.setdefault("color", line_color)
            line_kwargs["line"] = line
        traces.append(
            go.Scatter3d(
                x=seg[:, 0],
                y=seg[:, 1],
                z=seg[:, 2],
                mode="lines",
                showlegend=showlegend,
                **line_kwargs,
            )
        )
        return traces

    if eff_dim == 2:
        basis = vh[:2]
        coords_2d = vertices @ basis.T
        if poly.dim < 2:
            order = np.arange(vertices.shape[0])
        else:
            try:
                hull_2d = ConvexHull(coords_2d)
                order = hull_2d.vertices
            except Exception:
                order = np.arange(vertices.shape[0])
        ordered = vertices[order]
        x = ordered[:, 0].tolist() + [ordered[0, 0]]
        y = ordered[:, 1].tolist() + [ordered[0, 1]]
        z = ordered[:, 2].tolist() + [ordered[0, 2]]
        face_line_kwargs = dict(base_kwargs)
        if line_color is not None:
            line = dict(face_line_kwargs.get("line", {}))
            line.setdefault("color", line_color)
            face_line_kwargs["line"] = line
        if filled and line_color is not None:
            n = len(ordered)
            i_vals = [0] * (n - 2)
            j_vals = list(range(1, n - 1))
            k_vals = list(range(2, n))
            mesh_kwargs = {k: v for k, v in base_kwargs.items() if k not in ("line", "marker")}
            traces.append(
                go.Mesh3d(
                    x=ordered[:, 0].tolist(),
                    y=ordered[:, 1].tolist(),
                    z=ordered[:, 2].tolist(),
                    i=i_vals,
                    j=j_vals,
                    k=k_vals,
                    color=line_color,
                    opacity=0.7,
                    showlegend=showlegend,
                    **mesh_kwargs,
                )
            )
        else:
            traces.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", showlegend=showlegend, **face_line_kwargs))
        return traces

    if poly.dim < 2:
        return traces
    try:
        hull = ConvexHull(vertices)
        edge_to_facets: dict[tuple[int, int], list[int]] = {}
        for fi, simplex in enumerate(hull.simplices):
            a, b, c = simplex
            for u, v in ((a, b), (a, c), (b, c)):
                if u == v:
                    continue
                key = (u, v) if u < v else (v, u)
                edge_to_facets.setdefault(key, []).append(fi)

        planes = hull.equations
        normals = planes[:, :3]
        offsets = planes[:, 3]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals_n = normals / norms
        offsets_n = offsets / norms.ravel()

        def facets_coplanar(facet_indices: list[int], atol: float = 1e-6) -> bool:
            if not facet_indices:
                return True
            f0 = facet_indices[0]
            n0 = normals_n[f0]
            d0 = offsets_n[f0]
            for fi in facet_indices[1:]:
                ni = normals_n[fi]
                di = offsets_n[fi]
                if np.abs(np.dot(n0, ni)) < 1.0 - atol:
                    return False
                if np.abs(d0 - di) > atol:
                    return False
            return True

        edges: set[tuple[int, int]] = set()
        for edge, facet_indices in edge_to_facets.items():
            if not facets_coplanar(facet_indices):
                edges.add(edge)

        if filled and line_color is not None:
            mesh_kwargs = {k: v for k, v in base_kwargs.items() if k not in ("line", "marker")}
            traces.append(
                go.Mesh3d(
                    x=vertices[:, 0].tolist(),
                    y=vertices[:, 1].tolist(),
                    z=vertices[:, 2].tolist(),
                    i=hull.simplices[:, 0].tolist(),
                    j=hull.simplices[:, 1].tolist(),
                    k=hull.simplices[:, 2].tolist(),
                    color=line_color,
                    opacity=0.7,
                    showlegend=showlegend,
                    **mesh_kwargs,
                )
            )
        else:
            edge_line_kwargs = dict(base_kwargs)
            if line_color is not None:
                line = dict(edge_line_kwargs.get("line", {}))
                line.setdefault("color", line_color)
                edge_line_kwargs["line"] = line

            xs: list[float] = []
            ys: list[float] = []
            zs: list[float] = []
            for edge_u, edge_v in edges:
                seg = vertices[[edge_u, edge_v]]
                xs.extend([float(seg[0, 0]), float(seg[1, 0]), float("nan")])
                ys.extend([float(seg[0, 1]), float(seg[1, 1]), float("nan")])
                zs.extend([float(seg[0, 2]), float(seg[1, 2]), float("nan")])
            traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", showlegend=showlegend, **edge_line_kwargs))
    except Exception as e:
        warnings.warn(f"Error while computing 3D mesh for polyhedron {poly}: {e}")
    return traces


def _poly_traces_2d_complex(
    poly: "Polyhedron",
    fill: str = "toself",
    showlegend: bool = False,
    bound: float = DEFAULT_PLOT_BOUND,
    plot_halfspaces: bool = False,
    halfspace_shade: bool = True,
    **kwargs: Any,
) -> list[go.Scatter]:
    if poly.W.shape[0] != 2:
        raise ValueError("Polyhedron must be 2D to plot")
    traces: list[go.Scatter] = []
    geom = poly._get_bounded_plot_geometry(bound)
    if geom is not None:
        kind, verts = geom
        if kind == "polygon":
            x = verts[:, 0].tolist() + [verts[0, 0]]
            y = verts[:, 1].tolist() + [verts[0, 1]]
            traces.append(go.Scatter(x=x, y=y, fill=fill, showlegend=showlegend, **kwargs))
        elif kind == "segment":
            x = verts[:, 0].tolist()
            y = verts[:, 1].tolist()
            seg_kwargs = {k: v for k, v in kwargs.items() if k != "mode"}
            traces.append(go.Scatter(x=x, y=y, mode="lines", fill=None, showlegend=showlegend, **seg_kwargs))
        elif kind == "point":
            x = verts[:, 0].tolist()
            y = verts[:, 1].tolist()
            point_kwargs = {k: v for k, v in kwargs.items() if k != "mode"}
            traces.append(go.Scatter(x=x, y=y, mode="markers", fill=None, showlegend=showlegend, **point_kwargs))

    if plot_halfspaces:
        W = poly.halfspaces_np[:, :-1]
        b = poly.halfspaces_np[:, -1]
        bounds = (-bound, bound)
        for i in range(W.shape[0]):
            w = W[i]
            if np.abs(w[1]) < TOL_NEARLY_VERTICAL:
                x_line = -b[i] / w[0] if np.abs(w[0]) >= TOL_NEARLY_VERTICAL else 0.0
                xs = [x_line, x_line]
                ys = [bounds[0], bounds[1]]
                halfspace_shade_this = False
            else:
                halfspace_shade_this = halfspace_shade
                y0 = (-b[i] - w[0] * bounds[0]) / w[1]
                y1 = (-b[i] - w[0] * bounds[1]) / w[1]
                if halfspace_shade_this:
                    outer = max(bounds[1], y0, y1) if w[1] < 0 else min(bounds[0], y0, y1)
                    xs = [bounds[0], bounds[0], bounds[1], bounds[1], bounds[0]]
                    ys = [outer, y0, y1, outer, outer]
                else:
                    xs = [bounds[0], bounds[1]]
                    ys = [y0, y1]
            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    name=f"Halfspace {i}",
                    fill="toself" if halfspace_shade_this else None,
                    visible="legendonly",
                    showlegend=True,
                )
            )
    return traces


def _poly_traces_2d_graph(
    poly: "Polyhedron",
    fill: str = "toself",
    showlegend: bool = False,
    bound: float = DEFAULT_PLOT_BOUND,
    project: float | None = None,
    **kwargs: Any,
) -> dict[str, go.Mesh3d | go.Scatter3d] | None:
    _ = fill
    if poly.W.shape[0] != 2:
        raise ValueError("Polyhedron must be 2D to plot")
    geom = poly._get_bounded_plot_geometry(bound)
    if geom is None:
        return None
    kind, verts = geom
    try:
        x = verts[:, 0].tolist()
        y = verts[:, 1].tolist()
        z = (
            (
                poly.net(torch.tensor([x, y], device=poly.net.device, dtype=poly.net.dtype).T)
                .detach()
                .cpu()
                .numpy()
                .squeeze()[:, 1]
            )
            if project is None
            else [project] * len(x)
        )
        if kind == "polygon":
            x_closed = x + [x[0]]
            y_closed = y + [y[0]]
            z_closed = z + [z[0]]
            mesh = go.Mesh3d(x=x_closed, y=y_closed, z=z_closed, alphahull=-1, lighting=dict(ambient=1), **kwargs)
            scatter = go.Scatter3d(
                x=x_closed,
                y=y_closed,
                z=z_closed,
                mode="lines",
                showlegend=False,
                line=dict(width=5, color="black"),
                visible=False,
            )
            return {"mesh": mesh, "outline": scatter}
        if kind == "segment":
            seg_kwargs = {k: v for k, v in kwargs.items() if k != "mode"}
            scatter = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                showlegend=False,
                line=dict(width=5, color="black"),
                **seg_kwargs,
            )
            return {"outline": scatter}
        if kind == "point":
            point_kwargs = {k: v for k, v in kwargs.items() if k != "mode"}
            scatter = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                showlegend=False,
                marker=dict(size=4, color="black"),
                **point_kwargs,
            )
            return {"outline": scatter}
    except Exception as e:
        warnings.warn(f"Error while plotting polyhedron: {e}")
    return None


@overload
def plot_polyhedron(
    poly: "Polyhedron",
    *,
    plot_mode: Literal["cells"],
    **kwargs: Any,
) -> list[go.Scatter] | list[go.Mesh3d | go.Scatter3d]: ...


@overload
def plot_polyhedron(
    poly: "Polyhedron",
    *,
    plot_mode: Literal["graph"],
    **kwargs: Any,
) -> dict[str, go.Mesh3d | go.Scatter3d] | None: ...


def plot_polyhedron(
    poly: "Polyhedron",
    *,
    plot_mode: Literal["cells", "graph"],
    **kwargs: Any,
) -> list[go.Mesh3d | go.Scatter3d] | list[go.Scatter] | dict[str, go.Mesh3d | go.Scatter3d] | None:
    """Build plotly traces for one polyhedron.

    * ``cells`` — cell in input space: uses ``poly.ambient_dim`` to choose 2D (``go.Scatter``)
      vs 3D (``go.Mesh3d`` / ``go.Scatter3d``) traces (supports ambient dimensions 2 and 3).
    * ``graph`` — 2D cell lifted through the network (mesh/outline dict or similar).

    Uses ``plot_mode`` (keyword-only) so Plotly kwargs such as ``mode=\"lines\"`` are not ambiguous.
    """
    if plot_mode == "cells":
        ad = poly.ambient_dim
        if ad == 2:
            kw2 = {k: v for k, v in kwargs.items() if k not in _POLY_CELLS_2D_EXCLUDE}
            return _poly_traces_2d_complex(poly, **kw2)
        if ad == 3:
            kw3 = {k: v for k, v in kwargs.items() if k not in _POLY_CELLS_3D_EXCLUDE}
            return _poly_traces_3d_complex(poly, **kw3)
        raise ValueError(f"plot_polyhedron(..., plot_mode='cells') supports ambient_dim 2 or 3, got {ad}")
    if plot_mode == "graph":
        return _poly_traces_2d_graph(poly, **kwargs)
    raise ValueError(f"Unknown plot_mode {plot_mode!r}; expected 'cells' or 'graph'.")


# --- Complex figures ----------------------------------------------------------------

# Keyword args only used by the 3D cell figure / 3D poly traces; must not reach 2D Scatter paths.
_CELLS_3D_ONLY_KEYS = frozenset({"fill_mode", "show_axes", "filled"})
_CELLS_2D_ONLY_KEYS = frozenset({"ss_name"})

# Polyhedron.plot_cells forwards both 2D- and 3D-only kwargs; strip before each trace builder.
_POLY_CELLS_2D_EXCLUDE = frozenset({"filled"})
# 2D go.Scatter only; invalid on go.Scatter3d / go.Mesh3d (see Polyhedron.plot_cells default fill=...).
_POLY_CELLS_3D_EXCLUDE = frozenset({"plot_halfspaces", "halfspace_shade", "fill", "fillcolor", "ss_name"})


def _equitable_colors(
    dual: nx.Graph,
    polys: list[Any],
    *,
    remap: bool,
) -> list[str]:
    color_scheme = px.colors.qualitative.Plotly
    n = len(polys)
    try:
        coloring = nx.algorithms.coloring.equitable_color(dual, min(len(color_scheme), n))
        if remap:
            remap_d: dict[int, int] = {}
            idx = 0
            for p in polys:
                if coloring[p] not in remap_d:
                    remap_d[coloring[p]] = idx
                    idx += 1
            return [color_scheme[remap_d[coloring[i]]] for i in polys]
        return [color_scheme[coloring[i]] for i in polys]
    except Exception:
        return [color_scheme[i % len(color_scheme)] for i in range(n)]


def _per_poly_colors(cpx: "Complex", polys: list[Any], color: str | None, *, remap_equitable: bool) -> list[str]:
    if color == "Wl2":
        return get_colors([poly.Wl2 for poly in polys])
    return _equitable_colors(cpx.get_dual_graph(), polys, remap=remap_equitable)


def _highlight(c: str, poly: Any, highlight_regions: Iterable[Any] | None) -> str:
    if highlight_regions is None:
        return c
    if poly in highlight_regions or str(poly) in highlight_regions:
        return "red"
    return c


def _ensure_minimum_plotted_polyhedra(total: int, plotted: int, context: str) -> None:
    if total == 0:
        return
    print(f"Plotted {plotted / total * 100:.2f}% of polyhedra within the bounds")
    minimum_required = (total + 1) // 2
    if plotted < minimum_required:
        raise RuntimeError(
            f"Unable to plot at least half of polyhedra for {context}: "
            f"plotted {plotted} of {total} intersecting the plot bounds (minimum required {minimum_required})."
        )


def _poly_intersects_plot_bound(poly: Any, bound: float) -> bool:
    """True if ``poly`` intersects the axis-aligned bounding hypercube of half-width ``bound``.

    Matches the feasibility check used before ``get_bounded_vertices`` / cell plotting: polyhedra
    that miss the box entirely are excluded from the success threshold denominator.

    Uses :meth:`~relucent.poly.Polyhedron.get_bounded_halfspaces` when available (cheap feasibility
    only). Otherwise falls back to :meth:`~relucent.poly.Polyhedron.get_bounded_vertices` so test
    doubles and stubs without halfspace helpers still work.
    """
    get_hs = getattr(poly, "get_bounded_halfspaces", None)
    if callable(get_hs):
        try:
            get_hs(bound)
            return True
        except ValueError:
            return False
    verts = poly.get_bounded_vertices(bound)
    return verts is not None and getattr(verts, "size", 0) > 0


def _complex_figure_2d_cells(
    cpx: "Complex",
    *,
    label_regions: bool = False,
    color: str | None = None,
    highlight_regions: Iterable[Any] | None = None,
    ss_name: bool = False,
    bound: float = DEFAULT_COMPLEX_PLOT_BOUND,
    **kwargs: Any,
) -> go.Figure:
    if cpx.dim != 2:
        raise ValueError(f"2D cell figure requires complex.dim == 2, got {cpx.dim}")
    fig = go.Figure()
    polys = list(cpx)
    colors = _per_poly_colors(cpx, polys, color, remap_equitable=True)
    eligible_polys = 0
    plotted_polys = 0
    for c, poly in tqdm(zip(colors, polys, strict=True), desc="Plotting Polyhedra", total=len(polys), delay=1):
        if not _poly_intersects_plot_bound(poly, bound):
            continue
        eligible_polys += 1
        c = _highlight(c, poly, highlight_regions)
        name = f"{poly.ss_np.ravel().astype(int).tolist()}" if ss_name else f"{poly}"
        try:
            traces = poly.plot_cells(
                name=name,
                fillcolor=c,
                line_color="black",
                mode="lines",
                bound=bound,
                **kwargs,
            )
        except Exception as e:
            warnings.warn(f"Error while plotting polyhedron {poly}: {e}")
            print("AHHHHHH", e)
            continue
        if len(traces) == 0:
            warnings.warn(f"No traces generated while plotting polyhedron {poly}.")
            continue
        plotted_polys += 1
        for trace in traces:
            fig.add_trace(trace)
        if label_regions and poly.center is not None:
            fig.add_trace(
                go.Scatter(x=[poly.center[0]], y=[poly.center[1]], mode="text", text=str(poly), showlegend=False)
            )
    _ensure_minimum_plotted_polyhedra(eligible_polys, plotted_polys, "2D complex cell plot")
    interior_points = [np.max(np.abs(p.interior_point)) for p in cpx if p.interior_point is not None]
    maxcoord = (
        (np.mean(interior_points) * PLOT_MARGIN_FACTOR)
        if len(interior_points) > 0
        else min(PLOT_DEFAULT_MAXCOORD, bound if bound else PLOT_DEFAULT_MAXCOORD)
    )
    fig.update_layout(
        showlegend=True,
        plot_bgcolor="white",
        xaxis=dict(range=(-maxcoord, maxcoord)),
        yaxis_scaleanchor="x",
        yaxis=dict(range=(-maxcoord, maxcoord)),
    )
    return fig


def _complex_figure_3d_cells(
    cpx: "Complex",
    *,
    label_regions: bool = False,
    color: str | None = None,
    highlight_regions: Iterable[Any] | None = None,
    show_axes: bool = False,
    fill_mode: str = "wireframe",
    **kwargs: Any,
) -> go.Figure:
    if cpx.dim != 3:
        raise ValueError("Complex must have 3D input to plot 3D complex")
    bound_effective = kwargs.get("bound", DEFAULT_PLOT_BOUND)
    fig = go.Figure()
    polys = list(cpx)
    colors = _per_poly_colors(cpx, polys, color, remap_equitable=True)
    eligible_polys = 0
    plotted_polys = 0
    for c, poly in tqdm(zip(colors, polys), desc="Plotting 3D Polyhedra", total=len(polys), delay=1):
        if not _poly_intersects_plot_bound(poly, bound_effective):
            continue
        eligible_polys += 1
        is_highlighted = highlight_regions is not None and (poly in highlight_regions or str(poly) in highlight_regions)
        c = "red" if is_highlighted else c
        filled = is_highlighted or fill_mode == "filled"
        try:
            traces = poly.plot_cells(showlegend=False, color=c, filled=filled, **kwargs)
        except Exception as e:
            warnings.warn(f"Error while plotting polyhedron {poly}: {e}")
            continue
        if len(traces) == 0:
            warnings.warn(f"No traces generated while plotting polyhedron {poly}.")
            continue
        plotted_polys += 1
        for trace in traces:
            fig.add_trace(trace)
        if label_regions and poly.center is not None and poly.center.shape[0] >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=[poly.center[0]],
                    y=[poly.center[1]],
                    z=[poly.center[2]],
                    mode="text",
                    text=str(poly),
                    showlegend=False,
                )
            )
    _ensure_minimum_plotted_polyhedra(eligible_polys, plotted_polys, "3D complex cell plot")
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=show_axes),
            yaxis=dict(visible=show_axes),
            zaxis=dict(visible=show_axes),
        ),
    )
    return fig


def _complex_figure_graph(
    cpx: "Complex",
    *,
    label_regions: bool = False,
    color: str | None = None,
    highlight_regions: Iterable[Any] | None = None,
    show_axes: bool = False,
    project: Any = True,
    **kwargs: Any,
) -> go.Figure:
    bound_effective = kwargs.get("bound", DEFAULT_PLOT_BOUND)
    fig = go.Figure()
    polys = list(cpx)
    colors = _per_poly_colors(cpx, polys, color, remap_equitable=False)
    outlines: list[go.Mesh3d | go.Scatter3d] = []
    meshes: list[go.Mesh3d | go.Scatter3d] = []
    eligible_polys = 0
    plotted_polys = 0
    for c, poly in tqdm(zip(colors, polys), desc="Plotting Polyhedra", total=len(polys), delay=1):
        if not _poly_intersects_plot_bound(poly, bound_effective):
            continue
        eligible_polys += 1
        c = _highlight(c, poly, highlight_regions)
        poly_plotted = False
        try:
            p_plot = poly.plot_graph(name=f"{poly}", color=c, **kwargs)
            if p_plot is not None:
                if isinstance(p_plot, dict):
                    if "mesh" in p_plot:
                        meshes.append(p_plot["mesh"])
                        poly_plotted = True
                    if "outline" in p_plot:
                        outlines.append(p_plot["outline"])
                        poly_plotted = True
                else:
                    fig.add_trace(p_plot)
                    poly_plotted = True
            if project is not None:
                p_plot = poly.plot_graph(name=f"{poly}", color=c, project=project, **kwargs)
                if p_plot is not None:
                    if isinstance(p_plot, dict):
                        if "mesh" in p_plot:
                            meshes.append(p_plot["mesh"])
                            poly_plotted = True
                        if "outline" in p_plot:
                            outlines.append(p_plot["outline"])
                            poly_plotted = True
                    else:
                        fig.add_trace(p_plot)
                        poly_plotted = True
        except Exception as e:
            warnings.warn(f"Error while plotting polyhedron {poly}: {e}")
            continue
        if poly_plotted:
            plotted_polys += 1
        else:
            warnings.warn(f"No traces generated while plotting polyhedron {poly}.")
        if label_regions and poly.center is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[poly.center[0]],
                    y=[poly.center[1]],
                    z=[
                        cpx.net(torch.tensor(poly.center, device=cpx.net.device, dtype=cpx.net.dtype).T)
                        .detach()
                        .cpu()
                        .numpy()
                        .squeeze()
                        .ravel()[:, 0]
                    ],
                    mode="text",
                    text=str(poly),
                    showlegend=False,
                )
            )
    _ensure_minimum_plotted_polyhedra(eligible_polys, plotted_polys, "complex graph plot")
    for outline in outlines:
        fig.add_trace(outline)
    for mesh in meshes:
        fig.add_trace(mesh)
    maxcoord = (
        np.median([np.max(np.abs(p.interior_point)) for p in cpx if p.interior_point is not None]) * PLOT_MARGIN_FACTOR
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=(-maxcoord, maxcoord), visible=show_axes),
            yaxis=dict(range=(-maxcoord, maxcoord), visible=show_axes),
            zaxis=dict(visible=show_axes),
        ),
    )
    return fig


def plot_complex(
    cpx: "Complex",
    *,
    plot_mode: Literal["cells", "graph"],
    **kwargs: Any,
) -> go.Figure:
    """Plot an entire ``Complex`` as one Plotly figure.

    * ``cells`` — cells in input space; 2D vs 3D layout is chosen from ``cpx.dim`` (network input
      dimension), which must be 2 or 3.
    * ``graph`` — 2D cells with third coordinate from the network (optional projected copy).
    """
    if plot_mode == "cells":
        d = cpx.dim
        if d == 2:
            kw2 = {k: v for k, v in kwargs.items() if k not in _CELLS_3D_ONLY_KEYS}
            return _complex_figure_2d_cells(cpx, **kw2)
        if d == 3:
            kw3 = {k: v for k, v in kwargs.items() if k not in _CELLS_2D_ONLY_KEYS}
            return _complex_figure_3d_cells(cpx, **kw3)
        raise ValueError(f"plot_complex(..., plot_mode='cells') supports complex.dim 2 or 3, got {d}")
    if plot_mode == "graph":
        return _complex_figure_graph(cpx, **kwargs)
    raise ValueError(f"Unknown plot_mode {plot_mode!r}; expected 'cells' or 'graph'.")
