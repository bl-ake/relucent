"""Certification of polyhedral complexes.

:func:`certify_complex` checks the invariants the topology code relies on. Each
:class:`CertifyLevel` includes the one before it and fails closed:

- ``COMBINATORIAL``: dual-graph SHI symmetry, cubical face-tag consistency, and
  contracted-slice SHI checks.
- ``COMPLETE``: also checks LP flip-neighbor completeness on a fully explored
  ambient complex (every geometric facet has a neighbor in the complex).
- ``GEOMETRIC``: also recomputes, with a fresh LP, every cached ``_shis`` that
  ``calculations.shis`` didn't compute on its cell.

Chain-complex and meta-graph building only need ``COMBINATORIAL``; the other two
are optional checks for search and geometry.

With ``repair=True`` (the default), :func:`~relucent.graph.incidence.build_dual_graph`
resyncs each top cell's ``_shis`` from the dual graph before checking. Nothing else
is repaired: a failure means the complex needs more exploration.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from relucent._internal.logging import logger, progress, with_verbosity
from relucent._internal.parallel import process_aware_cpu_count, worker_pool
from relucent.core.errors import IncompleteDualGraphError, NonGenericArrangementError, ShiProofError
from relucent.core.poly import Polyhedron
from relucent.core.ss import encode_ss, flip_ss_at_shi
from relucent.graph.incidence import (
    certify_dual_graph,
    face_tag,
    ss_nonzero_indices,
    verify_contracted_shis,
    verify_flip_shi_symmetry,
)
from relucent.search.worker_context import get_worker_context, set_worker_context

if TYPE_CHECKING:
    from relucent.core.complex import Complex

__all__ = [
    "CertifyLevel",
    "certify_complex",
    "verify_arrangement_genericity",
    "verify_boundary_cell",
    "verify_lp_flip_neighbors_in_complex",
]


class CertifyLevel(StrEnum):
    """Cumulative certification strength; each level implies the ones before it."""

    COMBINATORIAL = "combinatorial"
    COMPLETE = "complete"
    GEOMETRIC = "geometric"


_LEVEL_RANK: dict[CertifyLevel, int] = {
    CertifyLevel.COMBINATORIAL: 0,
    CertifyLevel.COMPLETE: 1,
    CertifyLevel.GEOMETRIC: 2,
}


@with_verbosity
def certify_complex(
    cplx: Complex,
    *,
    level: CertifyLevel = CertifyLevel.COMPLETE,
    repair: bool = True,
    graph: nx.Graph[Polyhedron] | None = None,
    record_state: bool = False,
    verbose: int | None = None,
) -> None:
    """Run the certification pipeline up to ``level``; sets ``cplx._verified`` on success.

    Args:
        cplx: The complex to certify.
        level: How strong a certification to run (see module docstring).
        repair: When True (default) and ``graph`` is not given, resync top-cell
            ``_shis`` from the freshly built combinatorial dual graph before
            checking anything. This is the only repair relucent performs.
        graph: A pre-built dual graph to certify against, e.g. one already
            constructed (and possibly repaired) by the caller. When omitted,
            one is built via :meth:`~relucent.core.complex.Complex.dual_graph`.
        record_state: When True, also update
            :meth:`~relucent.core.complex.Complex.set_exploration_state` so callers
            do not need a separate state write.
        verbose: Output level: ``0`` quiet, ``1`` progress bar, ``2`` per-stage detail.
            ``None`` uses :data:`relucent.config.VERBOSE`.

    Raises:
        ShiFlipInvariantError, DualGraphAsymmetricEdgeError, CubicalConsistencyError:
            Combinatorial invariants are violated.
        IncompleteDualGraphError: ``level >= COMPLETE`` and a geometric facet has
            no same-dimension neighbor in the complex.
        ShiProofError: ``level == GEOMETRIC`` and a cached ``_shis`` list not computed by
            ``calculations.shis`` on its cell does not match a fresh LP recompute.
    """
    del verbose  # applied by @with_verbosity
    if len(cplx) == 0:
        if record_state:
            complete = True if cplx._complete is None else bool(cplx._complete)
            cplx.set_exploration_state(complete=complete, verified=True)
        else:
            cplx._verified = True
        return

    top_dim = max(int(p.dim) for p in cplx)
    is_contracted_slice = top_dim != int(cplx.dim)

    g = graph if graph is not None else cplx.dual_graph(repair=repair)

    logger.debug("certify_complex: flip-SHI symmetry ...")
    t_stage = time.perf_counter()
    verify_flip_shi_symmetry(cplx)
    logger.debug("certify_complex: flip-SHI symmetry finished in %.1fs", time.perf_counter() - t_stage)

    logger.debug("certify_complex: dual-graph certification ...")
    t_stage = time.perf_counter()
    certify_dual_graph(g, cplx)
    logger.debug("certify_complex: dual-graph certification finished in %.1fs", time.perf_counter() - t_stage)

    if is_contracted_slice:
        logger.debug("certify_complex: contracted-slice SHI checks ...")
        t_stage = time.perf_counter()
        verify_contracted_shis(cplx)
        logger.debug("certify_complex: contracted-slice SHI checks finished in %.1fs", time.perf_counter() - t_stage)

    if _LEVEL_RANK[level] >= _LEVEL_RANK[CertifyLevel.COMPLETE] and cplx.complete is True:
        logger.debug("certify_complex: LP facet completeness certification ...")
        t_stage = time.perf_counter()
        verify_lp_flip_neighbors_in_complex(cplx)  # skip expensive LPs on partial complexes
        logger.debug("certify_complex: LP facet completeness finished in %.1fs", time.perf_counter() - t_stage)

    if _LEVEL_RANK[level] >= _LEVEL_RANK[CertifyLevel.GEOMETRIC]:
        for poly in cplx:
            if poly._shis is not None:
                verify_shi_geometry(poly)

    if record_state:
        complete = True if cplx._complete is None else bool(cplx._complete)
        cplx.set_exploration_state(complete=complete, verified=True)
    else:
        cplx._verified = True


# ---------------------------------------------------------------------------
# LP facet completeness (ambient top cells only)
# ---------------------------------------------------------------------------


def _iter_top_dim_polys(cplx: Complex, top_dim: int) -> Iterable[Polyhedron]:
    """Yield top-dimensional cells in stable complex iteration order."""
    for poly in cplx:
        if int(poly.dim) == top_dim:
            yield poly


def _poly_has_strict_cached_shis(poly: Polyhedron) -> bool:
    """Whether ``poly._shis`` is this cell's certified facet list, so needs no recompute.

    That is, :func:`~relucent.geometry.calculations.shis` computed it on this cell.
    """
    return poly._shis is not None and bool(getattr(poly, "_shis_strict", False))


def _verify_lp_neighbors_for_poly(
    poly: Polyhedron,
    *,
    top_tags: frozenset[bytes],
    bound: float,
) -> tuple[str | None, list[str]]:
    """Return a per-poly SHI recompute error or missing-neighbor diagnostics."""
    from relucent.geometry import calculations

    try:
        lp_shis = calculations.shis(poly, bound=float(bound))
    except ValueError as exc:
        return str(exc), []
    return None, _missing_lp_neighbors_for_shis(poly, shis=lp_shis, top_tags=top_tags)


def _missing_lp_neighbors_for_shis(
    poly: Polyhedron,
    *,
    shis: Iterable[int],
    top_tags: frozenset[bytes],
) -> list[str]:
    """Return missing-neighbor diagnostics for a trusted SHI list."""
    missing: list[str] = []
    ss = np.asarray(poly.ss_np, dtype=np.int8)
    for shi in shis:
        shi_i = int(shi)
        if int(ss[shi_i]) == 0:
            continue  # inactive hyperplane on this cell
        neighbor_ss = flip_ss_at_shi(ss, shi_i)
        if encode_ss(neighbor_ss) not in top_tags:
            missing.append(f"LP facet shi={shi_i} on {poly!r} has no neighbor in complex")
    return missing


def _verify_lp_neighbors_worker(
    task: tuple[np.ndarray, float, frozenset[bytes]],
) -> tuple[str, str | None, list[str]]:
    """Recompute LP SHIs for one top cell inside a worker process."""
    ss, bound, top_tags = task
    ctx = get_worker_context()
    poly = Polyhedron(ctx.net, ss, bound=float(bound))
    err, missing = _verify_lp_neighbors_for_poly(poly, top_tags=top_tags, bound=float(bound))
    return repr(poly), err, missing


def verify_lp_flip_neighbors_in_complex(cplx: Complex, *, nworkers: int | None = None) -> None:
    """Every LP facet on a top cell must flip to a same-dimension neighbor in the complex."""
    from relucent._internal.network_scale import default_polyhedron_bound

    if len(cplx) == 0:
        return
    top_dim = max(int(p.dim) for p in cplx)
    if top_dim != int(cplx.dim):
        return  # contracted slices skip ambient LP completeness
    top_polys = list(_iter_top_dim_polys(cplx, top_dim))
    top_tags = frozenset(poly.tag for poly in top_polys)
    tasks: list[tuple[np.ndarray, float, frozenset[bytes]]] = []
    serial_tasks: list[tuple[Polyhedron, float]] = []
    for poly in top_polys:
        bound = poly.bound
        if bound is None:
            bound = default_polyhedron_bound(cplx._net)
        if bound is None:
            continue  # no bound -> can't run facet LP
        bound_f = float(bound)
        if _poly_has_strict_cached_shis(poly):
            serial_tasks.append((poly, bound_f))
        else:
            tasks.append((np.asarray(poly.ss_np, dtype=np.int8), bound_f, top_tags))

    total_cells = len(serial_tasks) + len(tasks)
    if total_cells == 0:
        return

    # Use one Gurobi thread per worker to avoid multiplying solver threads by process count.
    requested_workers = nworkers or process_aware_cpu_count() or 1
    worker_count = max(1, min(requested_workers, len(tasks))) if tasks else 1
    if tasks:
        logger.debug(
            "verify_lp_flip_neighbors_in_complex: certifying %d top cells "
            + "(%d trusted cached SHIs, %d LP recertify, %d worker%s)",
            total_cells,
            len(serial_tasks),
            len(tasks),
            worker_count,
            "" if worker_count == 1 else "s",
        )
    else:
        logger.debug(
            "verify_lp_flip_neighbors_in_complex: certifying %d top cells " + "(trusted cached SHIs only, no LP recertify)",
            total_cells,
        )
    t_verify = time.perf_counter()
    missing: list[str] = []
    pbar = progress(
        desc="Certifying LP facets",
        total=total_cells,
        mininterval=1,
    )
    if serial_tasks:
        for poly, _bound in serial_tasks:
            assert poly._shis is not None
            missing.extend(_missing_lp_neighbors_for_shis(poly, shis=poly._shis, top_tags=top_tags))
            pbar.update()
    if tasks and worker_count == 1:
        for ss, bound, _top_tags in tasks:
            poly = Polyhedron(cplx._net, ss, bound=bound)
            err, poly_missing = _verify_lp_neighbors_for_poly(poly, top_tags=top_tags, bound=bound)
            if err is not None:
                pbar.close()
                raise IncompleteDualGraphError(
                    "Dual graph completeness could not be certified from LP facets: "
                    + f"failed to recompute SHIs on {poly!r}: {err}"
                )
            missing.extend(poly_missing)
            pbar.update()
    elif tasks:
        with worker_pool(
            worker_count,
            initializer=set_worker_context,
            initargs=(cplx._net, False, 1),
        ) as pool:
            for poly_repr, err, poly_missing in pool.imap_unordered(_verify_lp_neighbors_worker, tasks):
                if err is not None:
                    pbar.close()
                    raise IncompleteDualGraphError(
                        "Dual graph completeness could not be certified from LP facets: "
                        + f"failed to recompute SHIs on {poly_repr}: {err}"
                    )
                missing.extend(poly_missing)
                pbar.update()
    pbar.close()

    logger.debug(
        "verify_lp_flip_neighbors_in_complex: certified %d top cells in %.1fs",
        total_cells,
        time.perf_counter() - t_verify,
    )

    if missing:
        missing.sort()
        raise IncompleteDualGraphError(
            "Dual graph is incomplete relative to LP facets: "
            + f"{len(missing)} missing neighbor(s). "
            + missing[0]
            + (" ..." if len(missing) > 1 else "")
        )


# ---------------------------------------------------------------------------
# Per-cell geometric certification
# ---------------------------------------------------------------------------


def verify_shi_geometry(poly: Polyhedron, *, bound: float | None = None) -> None:
    """Recompute SHIs and require the cached list to match.

    A cached list that :func:`~relucent.geometry.calculations.shis` computed on this cell (``_shis_strict``) is already its
    certified facet list, so only lists assigned some other way (propagated from the dual graph or
    a coface) are recomputed.
    """
    from relucent.geometry import calculations

    if poly._shis is None:
        raise ShiProofError(f"Polyhedron {poly!r} has no cached _shis.")
    if _poly_has_strict_cached_shis(poly):
        return
    if bound is None:
        bound = poly.bound
    if bound is None:
        from relucent._internal.network_scale import default_polyhedron_bound

        if poly._net is None:
            raise ShiProofError(f"Polyhedron {poly!r} has no network for bound estimation.")
        bound = default_polyhedron_bound(poly._net)
    fresh = calculations.shis(poly, bound=float(bound))
    if set(fresh) != set(poly._shis):
        raise ShiProofError(f"Cached _shis {sorted(poly._shis)} != recomputed {sorted(fresh)} on {poly!r}.")


def verify_boundary_cell(poly: Polyhedron, boundary_shi: int) -> None:
    """Both ambient cofaces of a boundary top cell must be nonempty."""
    from relucent.search.boundary_search import _both_ambient_cofaces_feasible

    if not _both_ambient_cofaces_feasible(poly, boundary_shi):
        raise ValueError(f"Boundary cell {poly!r} fails ambient coface feasibility at shi={boundary_shi}.")


# ---------------------------------------------------------------------------
# Geometric genericity (1-dimensional arrangements)
# ---------------------------------------------------------------------------


def _one_cell_endpoint_map(poly: Polyhedron) -> dict[bytes, tuple[int, np.ndarray, float]]:
    """Map combinatorial 0-face tags to ``(witness shi, point, error radius)`` for a 1-cell.

    Each endpoint is certified by ``Polyhedron._halfspace_point_with_error()``: strictly inside
    every other row of the cell beyond its float64 error, or the call raises.
    """
    if int(poly.dim) != 1:
        return {}
    out: dict[bytes, tuple[int, np.ndarray, float]] = {}
    ss = np.asarray(poly.ss_np)
    hs = np.asarray(poly.halfspaces_np)
    err = np.asarray(poly.halfspaces_err_np)
    for shi in ss_nonzero_indices(ss):
        shi_i = int(shi)
        active = np.array(list(poly.zero_indices) + [shi_i], dtype=np.intp)
        sol = poly._halfspace_point_with_error(hs, active, err, poly._exact_rows)
        if sol is None:
            continue
        pt, pt_err = sol
        out[face_tag(ss, shi_i)] = (shi_i, np.asarray(pt, dtype=np.float64).reshape(-1), float(pt_err))
    return out


def verify_arrangement_genericity(polys: Iterable[Polyhedron]) -> None:
    """Geometric check for 1-dimensional arrangements (transversality).

    Combinatorial 0-face endpoints on a 1-cell must be geometrically distinct, endpoints that
    share a combinatorial tag must be the same point, and endpoints with different tags must be
    different points. Two points are the same or different only when their distance is beyond
    or within the sum of their float64 error radii; anything in between raises. Violations mean
    the underlying hyperplane arrangement is not generic (hyperplanes concur at a point).
    """
    from scipy.spatial import KDTree

    cells = [p for p in polys if int(p.dim) == 1]
    if not cells:
        return

    endpoint_maps = {p.tag: _one_cell_endpoint_map(p) for p in cells}

    for poly in cells:
        if len(endpoint_maps[poly.tag]) > 2:
            raise NonGenericArrangementError(
                f"1-cell {poly!r} has {len(endpoint_maps[poly.tag])} certified endpoints; a segment has at most 2"
            )

    # Every endpoint of every cell, deduplicated by (cell, tag).
    tags: list[bytes] = []
    points: list[np.ndarray] = []
    radii: list[float] = []
    cells_by_tag: dict[bytes, list[bytes]] = defaultdict(list)
    for cell_tag, ep_map in endpoint_maps.items():
        for tag, (_shi, pt, pt_err) in ep_map.items():
            tags.append(tag)
            points.append(pt)
            radii.append(pt_err)
            cells_by_tag[tag].append(cell_tag)
    if not points:
        return
    pts = np.vstack(points)
    rad = np.asarray(radii, dtype=np.float64)
    # Candidate pairs within the largest possible "same point" distance; each then judged exactly.
    for i, j in KDTree(pts).query_pairs(r=2.0 * float(rad.max()) + 1e-300, output_type="ndarray"):
        dist = float(np.linalg.norm(pts[i] - pts[j]))
        close = dist <= rad[i] + rad[j]
        if tags[i] != tags[j] and close:
            raise NonGenericArrangementError(
                "two different 0-faces meet at the same point to within float64 error "
                + "(non-transversal junction: hyperplanes concur at a vertex)"
            )
    # Same tag must be the same point: compare each copy to the first.
    first: dict[bytes, int] = {}
    for k, tag in enumerate(tags):
        if tag not in first:
            first[tag] = k
            continue
        i = first[tag]
        if float(np.linalg.norm(pts[i] - pts[k])) > rad[i] + rad[k]:
            raise NonGenericArrangementError("one 0-face tag is reached at two different points; adjacency is ambiguous")
    # Two 1-cells may share at most one endpoint.
    shared: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for owners in cells_by_tag.values():
        owners = sorted(set(owners))
        for a in range(len(owners)):
            for b in range(a + 1, len(owners)):
                shared[(owners[a], owners[b])] += 1
    for (left, right), n_shared in shared.items():
        if n_shared > 1:
            raise NonGenericArrangementError(
                f"1-cells {left!r} and {right!r} share {n_shared} combinatorial 0-face tags; adjacency is ambiguous."
            )
