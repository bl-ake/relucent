"""Persistent homology over GF(2) on ReLU cell complexes.

Homology is computed from the meta-graph face poset using column reduction of the
filtration boundary matrix (algebraic), matching the incidence convention in
:mod:`relucent.topology`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from relucent.complex import Complex
    from relucent.filtration import Filtration

__all__ = [
    "PersistencePair",
    "PersistenceDiagram",
    "betti_at_filtration_end",
    "betti_curve",
    "compute_persistent_homology",
]


def _verbose_line(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"relucent.persistence: {msg}", file=sys.stderr, flush=True)


@dataclass(frozen=True, slots=True)
class PersistencePair:
    """Birth–death pair for a homology class over GF(2).

    Attributes:
        dimension: Homological degree of the born feature.
        birth: Filtration value at which the class is born.
        death: Filtration value at which the class dies (``math.inf`` if essential).
        birth_cell: Meta-graph node key for the birth simplex.
        death_cell: Meta-graph node key for the death simplex (``None`` if essential).
    """

    dimension: int
    birth: float
    death: float
    birth_cell: Any = None
    death_cell: Any = None


@dataclass(frozen=True, slots=True)
class PersistenceDiagram:
    """Persistence diagram bundled with the filtration values used."""

    pairs: tuple[PersistencePair, ...]
    cell_filtration: dict[Any, float]

    def plot(
        self,
        *,
        max_death: float | None = None,
        show_diagonal: bool = True,
        title: str | None = "Persistence diagram",
        **kwargs: Any,
    ) -> Any:
        """Plot this diagram via :func:`relucent.vis.plot_persistence_diagram`.

        Returns:
            A :class:`plotly.graph_objects.Figure`.
        """
        from relucent.vis import plot_persistence_diagram

        return plot_persistence_diagram(
            self,
            max_death=max_death,
            show_diagonal=show_diagonal,
            title=title,
            **kwargs,
        )


def _symmetric_diff(a: set[int], b: set[int]) -> set[int]:
    return a.symmetric_difference(b)


def _lowest_row(col: set[int]) -> int | None:
    return max(col) if col else None


def _gf2_column_reduce_persistence(
    boundaries: list[set[int]],
    filtration: list[float],
    dimensions: list[int],
    cell_keys: list[Any],
) -> list[PersistencePair]:
    """Left-to-right column reduction (mod 2); row/column indices index the same cells."""
    n = len(boundaries)
    low: dict[int, int] = {}
    essential: list[int] = []

    for j in range(n):
        col = set(boundaries[j])
        while col:
            r = _lowest_row(col)
            assert r is not None
            if r not in low:
                low[r] = j
                break
            col = _symmetric_diff(col, boundaries[low[r]])
        else:
            essential.append(j)

    pairs: list[PersistencePair] = []
    for r, j in low.items():
        if r == j:
            continue
        birth = float(filtration[r])
        death = float(filtration[j])
        if death < birth:
            continue
        pairs.append(
            PersistencePair(
                dimension=int(dimensions[r]),
                birth=birth,
                death=death,
                birth_cell=cell_keys[r],
                death_cell=cell_keys[j],
            )
        )

    death_columns = {j for r, j in low.items() if r != j}
    paired_birth_rows = {r for r, j in low.items() if r != j and filtration[j] >= filtration[r]}
    for j in essential:
        if j in death_columns or j in paired_birth_rows:
            continue
        pairs.append(
            PersistencePair(
                dimension=int(dimensions[j]),
                birth=float(filtration[j]),
                death=float("inf"),
                birth_cell=cell_keys[j],
                death_cell=None,
            )
        )

    return pairs


def _face_incidence_counts(
    meta: Any,
    index_of: dict[Any, int],
    *,
    compactify: bool,
) -> dict[Any, int]:
    """Count cofaces for each (k-1)-cell when ``compactify`` is enabled."""
    if not compactify:
        return {}
    inc: dict[Any, int] = {}
    for _u, v, _ in meta.edges(data=True):
        if _u not in index_of or v not in index_of:
            continue
        inc[v] = inc.get(v, 0) + 1
    return inc


def _cell_boundary_faces(
    meta: Any,
    cell_key: Any,
    index_of: dict[Any, int],
    *,
    compactify: bool,
    inc_count: dict[Any, int],
) -> set[int]:
    """Indices of codimension-one faces of ``cell_key`` in the filtration ordering."""
    faces: set[int] = set()
    for _u, v, _ in meta.out_edges(cell_key, data=True):
        if v not in index_of:
            continue
        if compactify and inc_count.get(v, 2) < 2:
            continue
        faces.add(index_of[v])
    return faces


def compute_persistent_homology(
    cplx: Complex,
    filtration: Filtration,
    *,
    compactify: bool = False,
    respect_finite: bool = False,
    lower_star: bool | None = None,
    verbose: bool = False,
) -> PersistenceDiagram:
    """Compute persistence pairs for a filtration on a :class:`~relucent.complex.Complex`.

    Cells are ordered by ``(filtration value, dimension, stable node key)``. Boundary
    columns are built from meta-graph face incidences (same convention as
    :func:`relucent.topology.get_betti_numbers`).

    Args:
        cplx: Polyhedral complex from breadth-first search / exploration.
        filtration: Filtration assigning a real value to each cell.
        compactify: If True, drop face incidences with fewer than two cofaces (Borel–Moore
            style), matching :func:`~relucent.topology.get_betti_numbers`.
        respect_finite: Restrict to cells with ``finite is True`` on the meta-graph.
        lower_star: If True, extend values to higher cells by
            ``f(σ) = max_{τ ≤ σ} f(τ)`` (sublevel-set convention). If None, use
            ``filtration.lower_star`` (default True for built-in filtrations).
        verbose: Progress lines on stderr.

    Returns:
        :class:`PersistenceDiagram` with finite and essential pairs over GF(2).
    """
    if len(cplx) == 0:
        return PersistenceDiagram(pairs=(), cell_filtration={})

    _verbose_line(verbose, "building meta-graph …")
    meta = cplx.get_meta_graph(verbose=verbose)
    if not compactify and not respect_finite:
        from relucent.complex import Complex

        Complex.truncate_meta_graph(meta)

    raw_values = filtration.values_for_meta(meta)
    use_lower_star = filtration.lower_star if lower_star is None else lower_star
    if use_lower_star:
        from relucent.filtration import lower_star_extension

        cell_values = lower_star_extension(meta, raw_values)
    else:
        cell_values = raw_values

    nodes: list[tuple[Any, int, float]] = []
    for n, attrs in meta.nodes(data=True):
        k = int(attrs.get("dim", -1))
        if k < 0:
            continue
        if respect_finite and attrs.get("finite", None) is not True:
            continue
        if n not in cell_values:
            continue
        nodes.append((n, k, float(cell_values[n])))

    if not nodes:
        return PersistenceDiagram(pairs=(), cell_filtration={})

    nodes.sort(key=lambda t: (t[2], t[1], repr(t[0])))
    cell_keys = [t[0] for t in nodes]
    dimensions = [t[1] for t in nodes]
    filt = [t[2] for t in nodes]
    index_of = {key: i for i, key in enumerate(cell_keys)}

    inc_count = _face_incidence_counts(meta, index_of, compactify=compactify)

    boundaries: list[set[int]] = []
    for key in cell_keys:
        boundaries.append(
            _cell_boundary_faces(
                meta,
                key,
                index_of,
                compactify=compactify,
                inc_count=inc_count,
            )
        )

    _verbose_line(verbose, f"reducing boundary matrix ({len(cell_keys)} cells) …")
    pairs = _gf2_column_reduce_persistence(boundaries, filt, dimensions, cell_keys)
    _verbose_line(verbose, f"done ({len(pairs)} pairs)")

    return PersistenceDiagram(pairs=tuple(pairs), cell_filtration=dict(cell_values))


def betti_at_filtration_end(
    diagram: PersistenceDiagram,
    *,
    margin: float = 1.0,
) -> dict[int, int]:
    """Betti numbers after all cells in ``diagram`` have entered the filtration.

    Evaluates :func:`betti_curve` at ``max(cell_filtration) + margin``, which matches
    the homology of the full meta-graph complex when paired with a constant filtration.

    Args:
        diagram: Persistence diagram from :func:`compute_persistent_homology`.
        margin: Added to the maximum cell filtration value to form the evaluation
            threshold (must be positive if all values are equal).

    Returns:
        ``{k: β_k}`` including dimensions with ``β_k = 0``.
    """
    if not diagram.cell_filtration:
        return {}
    t_end = float(max(diagram.cell_filtration.values())) + float(margin)
    curve = betti_curve(diagram, [t_end])
    if not curve:
        return {}
    max_dim = max(curve.keys())
    return {k: int(curve[k][0]) for k in range(max_dim + 1)}


def betti_curve(
    diagram: PersistenceDiagram,
    thresholds: np.ndarray | list[float],
    *,
    max_dim: int | None = None,
) -> dict[int, np.ndarray]:
    """Betti numbers β_k at each filtration threshold (right-continuous counting).

    Args:
        diagram: Output of :func:`compute_persistent_homology`.
        thresholds: Filtration values at which to evaluate β_k.
        max_dim: Maximum homological dimension to report (default: max in pairs).

    Returns:
        ``{k: array of length len(thresholds)}`` counting classes with birth ≤ t < death.
    """
    ts = np.asarray(thresholds, dtype=np.float64)
    if ts.size == 0:
        return {}

    pairs = diagram.pairs
    if max_dim is None:
        dims = [p.dimension for p in pairs]
        max_dim = max(dims) if dims else 0

    out: dict[int, np.ndarray] = {}
    for k in range(max_dim + 1):
        beta = np.zeros(ts.shape, dtype=np.int64)
        for p in pairs:
            if p.dimension != k:
                continue
            born_ok = ts >= p.birth
            alive = born_ok & (ts < p.death) if np.isfinite(p.death) else born_ok
            beta += alive.astype(np.int64)
        out[k] = beta
    return out
