"""Filtrations on ReLU polyhedral cell complexes.

Filtration values are attached combinatorially to meta-graph cells wherever possible.
Geometric evaluation (interior points, distances to training data) is optional and
localized to the built-in filtrations that need it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from relucent._torch_compat import TORCH_AVAILABLE, torch

if TYPE_CHECKING:
    from relucent.poly import Polyhedron

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
    """Base class for filtrations on cells of a :class:`~relucent.complex.Complex`.

    Subclasses implement :meth:`raw_cell_value` for each meta-graph node. Values are
    then promoted to a lower-star (sublevel-set) filtration on the face poset via
    :func:`lower_star_extension` when :attr:`lower_star` is True (the default).
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
            out[tag] = float(self.raw_cell_value(tag, attrs))
        return out


class ConstantFiltration(Filtration):
    """Every cell enters the filtration at the same value.

    Use with ``lower_star=False`` so :func:`~relucent.persistence.betti_at_filtration_end`
    matches :meth:`~relucent.complex.Complex.get_betti_numbers` on the same complex.
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


def _poly_from_attrs(attrs: dict[str, Any]) -> Polyhedron | None:
    poly = attrs.get("poly")
    return poly if poly is not None else None


def _affine_output_at_representative(
    poly: Polyhedron,
    *,
    row: int | None = None,
    combine: Literal["last", "diff"] = "last",
) -> float:
    """Evaluate a network output coordinate on one interior point (minimal geometry)."""
    w = poly.W
    b = poly.b
    if isinstance(w, np.ndarray):
        w_np = w
        b_np = np.asarray(b).reshape(-1)
    elif TORCH_AVAILABLE and isinstance(w, torch.Tensor):
        w_np = w.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy().reshape(-1) if isinstance(b, torch.Tensor) else np.asarray(b).reshape(-1)
    else:
        raise TypeError(f"Unsupported affine map type: {type(w)}")

    x = poly.interior_point
    if x is None:
        raise ValueError("Cannot evaluate filtration: polyhedron has no interior point.")
    out = w_np @ np.asarray(x).reshape(-1) + b_np
    if combine == "diff" and out.shape[0] >= 2:
        return float(out[0] - out[1])
    idx = int(out.shape[0] - 1) if row is None else int(row)
    return float(out[idx])


class AffineOutputFiltration(Filtration):
    """Filtration by an affine output functional ``w^T x + b`` on each cell.

    Uses the per-region affine map ``Polyhedron.W``, ``Polyhedron.b`` (from sign
    sequences) and one interior point per cell. By default, :attr:`lower_star` extends
    values to higher cells by max over faces (sublevel-set convention on the face
    poset). Set ``lower_star=False`` to rank each cell only by its representative point.
    """

    lower_star: bool
    combine: Literal["last", "diff"]

    def __init__(
        self,
        *,
        row: int | None = None,
        combine: Literal["last", "diff"] = "last",
        lower_star: bool = True,
    ) -> None:
        self.row = row
        self.combine = combine
        self.lower_star = lower_star

    def raw_cell_value(self, tag: Any, attrs: dict[str, Any]) -> float:
        poly = _poly_from_attrs(attrs)
        if poly is None:
            return float("inf")
        return _affine_output_at_representative(poly, row=self.row, combine=self.combine)


class LogitSublevelFiltration(AffineOutputFiltration):
    """Sublevel-set filtration induced by a scalar logit (last output or class difference).

    By default (``lower_star=True``), raw logit values on 0-cells are extended by max
    over faces so a cell enters the sublevel set when all its faces are present. Set
    ``lower_star=False`` to rank each cell only by the logit at one interior point.
    """

    def __init__(
        self,
        *,
        row: int | None = None,
        binary: bool = False,
        lower_star: bool = True,
    ) -> None:
        combine: Literal["last", "diff"] = "diff" if binary else "last"
        super().__init__(row=row, combine=combine, lower_star=lower_star)


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
        self.shi = int(shi)
        self.target = int(target)
        self.active_value = float(active_value)
        self.inactive_value = float(inactive_value)

    def raw_cell_value(self, tag: Any, attrs: dict[str, Any]) -> float:
        ss = attrs.get("ss")
        if ss is None:
            return self.inactive_value
        arr = np.asarray(ss)
        sign = int(arr[0, self.shi])
        return self.active_value if sign == self.target else self.inactive_value


class TrainingDistanceFiltration(Filtration):
    """Filtration by distance from a cell representative point to training data.

    Uses one interior point per cell and ``numpy`` norms—no explicit Voronoi or
    distance-transform geometry. By default, :attr:`lower_star` extends each cell's
    value by max over faces (sublevel-set convention).
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
        poly = _poly_from_attrs(attrs)
        if poly is None:
            return float("inf")
        x = poly.interior_point
        if x is None:
            return float("inf")
        return self._distance_to_training(x)
