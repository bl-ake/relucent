"""Filtrations on ReLU polyhedral cell complexes.

Filtration values are attached combinatorially to meta-graph cells wherever possible.
Geometric evaluation (interior points, distances to training data) is optional and
localized to the built-in filtrations that need it.

Unbounded cells and truncation caps. After truncation
(:func:`~relucent.graph.meta_graph.truncate_meta_graph`) each unbounded cell has caps: the
faces left where a large sphere of radius ``R`` cuts it. A cap shares its parent cell's
``poly``, so its value cannot come from that polyhedron's interior point. The built-in
filtrations give caps their value in the limit ``R -> inf`` instead (see
:class:`AffineOutputFiltration` and :class:`TrainingDistanceFiltration`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from relucent.core.poly import Polyhedron
    from relucent.model.model import ReLUNetwork

__all__ = [
    "AffineOutputFiltration",
    "ConstantFiltration",
    "Filtration",
    "LogitSublevelFiltration",
    "NeuronActivationFiltration",
    "TrainingDistanceFiltration",
    "lower_star_extension",
]


class Filtration(ABC):
    """Base class for filtrations on cells of a :class:`~relucent.core.complex.Complex`.

    Subclasses implement :meth:`raw_cell_value` for each meta-graph node. Values are
    then promoted to a lower-star (sublevel-set) filtration on the face poset via
    :func:`lower_star_extension` when ``lower_star`` is True (the default).
    """

    lower_star: bool = True

    @abstractmethod
    def raw_cell_value(self, tag: Any, attrs: dict[str, Any]) -> float:
        """Filtration value before lower-star extension (if enabled)."""

    def values_for_meta(self, meta: Any) -> dict[Any, float]:
        """Evaluate :meth:`raw_cell_value` on every node of ``meta``."""
        out: dict[Any, float] = {}
        for tag, attrs in meta.nodes(data=True):
            if int(attrs.get("dim", -1)) < 0:
                continue
            poly = attrs.get("poly")
            if poly is not None and getattr(poly, "finite", True) is None:
                continue
            out[tag] = float(self.raw_cell_value(tag, attrs))
        return out


class ConstantFiltration(Filtration):
    """Every cell enters the filtration at the same value.

    Use with ``lower_star=False`` so :func:`~relucent.topology.persistence.betti_at_filtration_end`
    matches :meth:`~relucent.core.complex.Complex.betti_numbers` on the same complex.
    """

    lower_star: bool = False

    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def raw_cell_value(self, tag: Any, attrs: dict[str, Any]) -> float:
        return self.value


def lower_star_extension(meta: Any, raw_values: dict[Any, float]) -> dict[Any, float]:
    """Promote vertex values to all cells by ``f(σ) = max_{τ face of σ} f(τ)``.

    Processes cells in increasing dimension so each value is the maximum of its
    own raw value and all codimension-one faces already processed.
    """
    by_dim: dict[int, list[Any]] = {}
    for tag, attrs in meta.nodes(data=True):
        k = int(attrs.get("dim", -1))
        if k < 0 or tag not in raw_values:
            continue
        by_dim.setdefault(k, []).append(tag)

    extended = dict(raw_values)
    for k in sorted(by_dim.keys()):
        if k == 0:
            continue
        for tag in by_dim[k]:
            val = extended.get(tag, float("-inf"))
            for _u, face, _ in meta.out_edges(tag, data=True):
                if face in extended:
                    val = max(val, extended[face])
            extended[tag] = float(val)
    return extended


@dataclass
class _CellRoles:
    """How each meta-graph node relates to the geometry of the complex.

    ``caps`` maps each truncation cap to the cell it caps (they share one ``poly``),
    ``vertices`` lists the 0-cells that are not caps, ``rays`` maps each unbounded 1-cell
    with one endpoint to that endpoint, and ``unpointed`` lists the cells (not caps) of
    positive dimension with no vertex among their faces, i.e. cells that contain a line.
    """

    caps: dict[Any, Any] = field(default_factory=dict)
    vertices: list[Any] = field(default_factory=list)
    rays: dict[Any, Any] = field(default_factory=dict)
    unpointed: list[Any] = field(default_factory=list)


def _feasible(attrs: dict[str, Any]) -> bool:
    poly = attrs.get("poly")
    return poly is None or getattr(poly, "finite", True) is not None


def _cell_roles(meta: Any) -> _CellRoles:
    """Classify the nodes of a (possibly truncated) meta-graph; see :class:`_CellRoles`."""
    roles = _CellRoles()
    by_poly: dict[int, list[Any]] = {}
    for n, attrs in meta.nodes(data=True):
        if int(attrs.get("dim", -1)) < 0 or not _feasible(attrs):
            continue
        poly = attrs.get("poly")
        if poly is not None:
            by_poly.setdefault(id(poly), []).append(n)
    for nodes in by_poly.values():
        if len(nodes) < 2:
            continue
        top = max(nodes, key=lambda n: int(meta.nodes[n]["dim"]))
        top_dim = int(meta.nodes[top]["dim"])
        for n in nodes:
            if int(meta.nodes[n]["dim"]) == top_dim - 1:
                roles.caps[n] = top

    by_dim: dict[int, list[Any]] = {}
    for n, attrs in meta.nodes(data=True):
        k = int(attrs.get("dim", -1))
        if n in roles.caps or k < 0 or not _feasible(attrs):
            continue
        by_dim.setdefault(k, []).append(n)
        if k == 0:
            roles.vertices.append(n)
        elif k == 1:
            ends = [
                v
                for _u, v, _ in meta.out_edges(n, data=True)
                if int(meta.nodes[v].get("dim", -1)) == 0 and v not in roles.caps
            ]
            poly = attrs.get("poly")
            if len(ends) == 1 and poly is not None and getattr(poly, "finite", None) is False:
                roles.rays[n] = ends[0]

    # A polyhedron is pointed exactly when it has a vertex, and a cell's vertices are reached
    # through its faces.
    pointed = set(roles.vertices)
    for k in sorted(by_dim):
        if k == 0:
            continue
        for n in by_dim[k]:
            if any(v in pointed for _u, v, _ in meta.out_edges(n, data=True)):
                pointed.add(n)
            else:
                roles.unpointed.append(n)
    return roles


def _cell_point(attrs: dict[str, Any]) -> np.ndarray:
    poly: Polyhedron | None = attrs.get("poly")
    x = None if poly is None else poly.interior_point
    if x is None:
        raise ValueError("Cannot evaluate filtration: a cell has no interior point (compute geometry first).")
    return np.asarray(x, dtype=np.float64).reshape(-1)


class AffineOutputFiltration(Filtration):
    """Sublevel-set filtration of a scalar network output that is affine on every cell.

    Each cell enters at the supremum of the output over the closed cell:

    * a vertex at its value;
    * a ray ``{v + s r : s >= 0}`` (an unbounded 1-cell) at its limit as ``s -> inf``, which is
      ``+inf`` or ``-inf`` by the sign of the slope along ``r``, or the value at ``v`` when the
      output is constant along the ray (slope within ``flat_tol``);
    * a truncation cap of a ray (a point at infinity) at the same limit;
    * every other cell at the largest value among its faces (lower-star extension).

    On a truncated complex this is the sublevel filtration of the output restricted to a
    ball of radius ``R``, in the limit ``R -> inf``: cells that only enter far out enter at
    ``±inf``. For a bounded cell the supremum of an affine function is attained at a vertex,
    and for a pointed unbounded cell it is ``+inf`` exactly when the function increases along
    one of its unbounded edges, so the lower-star rule gives each cell its supremum.
    Homology at a finite threshold ``t`` is therefore that of ``{x : output(x) <= t}``.

    The output is evaluated by running the network on points (vertices and points along
    rays), not from per-cell affine maps. Cells that contain a whole line (possible only
    when the first layer has rank below the input dimension) are not supported.

    Args:
        row: Output coordinate (default: the last).
        combine: ``"last"`` reads ``row``; ``"diff"`` reads output 0 minus output 1.
        pre_activation: Read the output of the last linear layer, before any trailing ReLU
            (as added by :func:`~relucent.model.builders.add_output_relu`).
        flat_tol: A ray counts as flat when its two sampled values differ by at most
            ``flat_tol * max(1, |g(v)|, |g(q)|)``.
    """

    lower_star: bool = True
    combine: Literal["last", "diff"]

    def __init__(
        self,
        *,
        row: int | None = None,
        combine: Literal["last", "diff"] = "last",
        pre_activation: bool = False,
        flat_tol: float = 1e-9,
    ) -> None:
        self.row = row
        self.combine = combine
        self.pre_activation = pre_activation
        self.flat_tol = float(flat_tol)

    def raw_cell_value(self, tag: Any, attrs: dict[str, Any]) -> float:
        raise NotImplementedError(f"{type(self).__name__} assigns values to the whole meta-graph (values_for_meta)")

    def evaluate(self, net: ReLUNetwork, points: np.ndarray) -> np.ndarray:
        """The scalar output at each row of ``points``."""
        from relucent.model.model import LinearLayer, ReLUNetwork

        layers = list(net.layers.values())
        stop = len(layers)
        if self.pre_activation:
            linear = [i for i, layer in enumerate(layers) if isinstance(layer, LinearLayer)]
            if not linear:
                raise ValueError("network has no linear layer")
            stop = linear[-1] + 1
        x: Any = np.asarray(points, dtype=np.float64).reshape((-1, *net.input_shape))
        for layer in layers[:stop]:
            x = ReLUNetwork._apply_layer(layer, x)
        out = np.asarray(x, dtype=np.float64).reshape(len(points), -1)
        if self.combine == "diff":
            if out.shape[1] < 2:
                raise ValueError(f"combine='diff' (binary=True) needs two outputs, the network has {out.shape[1]}")
            return out[:, 0] - out[:, 1]
        return out[:, -1 if self.row is None else int(self.row)]

    def values_for_meta(self, meta: Any) -> dict[Any, float]:
        roles = _cell_roles(meta)
        if roles.unpointed:
            raise ValueError(
                f"{len(roles.unpointed)} cells have no vertex (they contain a whole line); {type(self).__name__} "
                + "needs every cell to be pointed (first-layer weights of rank equal to the input dimension)"
            )
        nets = (a["poly"]._net for _n, a in meta.nodes(data=True) if a.get("poly") is not None)
        net = next(nets, None)
        if net is None:
            return {}

        vpos = {n: _cell_point(meta.nodes[n]) for n in roles.vertices}
        far: dict[Any, np.ndarray] = {}
        for ray, v in roles.rays.items():
            direction = _cell_point(meta.nodes[ray]) - vpos[v]
            length = float(np.linalg.norm(direction))
            if length == 0.0:
                raise ValueError("a ray's interior point coincides with its endpoint")
            far[ray] = vpos[v] + (1.0 + float(np.linalg.norm(vpos[v]))) * direction / length

        keys = list(vpos) + list(far)
        pts = np.array([*vpos.values(), *far.values()], dtype=np.float64).reshape(len(keys), -1)
        g = dict(zip(keys, self.evaluate(net, pts) if keys else [], strict=True))

        raw: dict[Any, float] = {}
        for n, attrs in meta.nodes(data=True):
            if int(attrs.get("dim", -1)) >= 0 and _feasible(attrs):
                raw[n] = float("-inf")
        for v in roles.vertices:
            raw[v] = float(g[v])
        for ray, v in roles.rays.items():
            at_v, at_q = float(g[v]), float(g[ray])
            if abs(at_q - at_v) <= self.flat_tol * max(1.0, abs(at_v), abs(at_q)):
                raw[ray] = at_v
            else:
                raw[ray] = float("inf") if at_q > at_v else float("-inf")
        for cap, parent in roles.caps.items():
            if int(meta.nodes[cap]["dim"]) == 0:
                raw[cap] = raw[parent]
        return raw


class LogitSublevelFiltration(AffineOutputFiltration):
    """Sublevel-set filtration of the logit: the output of the network's last linear layer.

    A trailing ReLU (as added by :func:`~relucent.model.builders.add_output_relu` to put the
    decision boundary in the complex) is ignored, so on such a complex this filters by the
    decision function ``g`` itself, whose zero set is the boundary. ``binary=True`` reads
    logit 0 minus logit 1 of a two-logit network. See :class:`AffineOutputFiltration` for how
    vertices, rays and caps get their values.
    """

    def __init__(self, *, row: int | None = None, binary: bool = False, flat_tol: float = 1e-9) -> None:
        super().__init__(row=row, combine="diff" if binary else "last", pre_activation=True, flat_tol=flat_tol)


class NeuronActivationFiltration(Filtration):
    """Filtration by whether a ReLU unit has a prescribed sign on a cell.

    Purely combinatorial: reads the cell sign sequence from the meta-graph. Cells
    matching ``target`` receive value ``0`` and enter the filtration first; non-matching
    cells receive ``1``.
    """

    lower_star: bool = True

    def __init__(
        self,
        shi: int,
        *,
        target: Literal[-1, 0, 1] = 1,
        active_value: float = 0.0,
        inactive_value: float = 1.0,
    ) -> None:
        if int(target) not in (-1, 0, 1):
            raise ValueError(f"target must be -1, 0 or 1, got {target!r}")
        self.shi = int(shi)
        self.target = int(target)
        self.active_value = float(active_value)
        self.inactive_value = float(inactive_value)

    def raw_cell_value(self, tag: Any, attrs: dict[str, Any]) -> float:
        ss = attrs.get("ss")
        if ss is None:
            return self.inactive_value
        sign = int(np.asarray(ss).reshape(-1)[self.shi])
        return self.active_value if sign == self.target else self.inactive_value


class TrainingDistanceFiltration(Filtration):
    """Filtration by distance from a cell representative point to training data.

    A bounded cell is scored by the distance at one interior point (and, with
    ``lower_star``, the maximum over its faces), which approximates its farthest point.
    Without ``lower_star`` a face can score higher than a cell it bounds, which
    :func:`~relucent.topology.persistence.compute_persistent_homology` rejects.
    Unbounded cells lie arbitrarily far from the data and enter at ``+inf``, and so do
    truncation caps, which carry the ``poly`` of the unbounded cell they cap.
    """

    lower_star: bool

    def __init__(
        self,
        training_points: np.ndarray,
        *,
        ord: int | float = 2,
        lower_star: bool = True,
    ) -> None:
        pts = np.asarray(training_points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.ndim != 2:
            raise ValueError("training_points must be a 2D array (n_samples, n_features)")
        self.training_points = pts
        self.ord = ord
        self.lower_star = lower_star

    def _distance_to_training(self, x: np.ndarray) -> float:
        diff = self.training_points - np.asarray(x).reshape(1, -1)
        if self.ord == np.inf:
            return float(np.min(np.max(np.abs(diff), axis=1)))
        return float(np.min(np.linalg.norm(diff, ord=self.ord, axis=1)))

    def raw_cell_value(self, tag: Any, attrs: dict[str, Any]) -> float:
        poly = attrs.get("poly")
        if poly is None or getattr(poly, "finite", None) is False:
            return float("inf")
        x = poly.interior_point
        if x is None:
            return float("inf")
        return self._distance_to_training(x)
