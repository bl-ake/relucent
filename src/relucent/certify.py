"""Certification pipeline for polyhedral complexes.

:func:`certify_complex` is the single entry point for checking that a
:class:`~relucent.complex.Complex` satisfies the invariants topology routines
rely on. Each :class:`CertifyLevel` is cumulative and fails closed:

- ``COMBINATORIAL``: dual-graph SHI symmetry and cubical face-tag consistency,
  plus contracted-slice SHI checks on chain-complex slices.
- ``COMPLETE``: adds LP flip-neighbor completeness on a fully explored ambient
  complex (every geometric facet has a same-dimension neighbor in the complex).
- ``GEOMETRIC``: adds a fresh LP recompute of ``_shis`` on every cached cell.

Repair is conservative and explicit: when ``repair=True`` (the default),
:func:`~relucent.incidence.build_dual_graph` resyncs each top cell's ``_shis``
from the combinatorial dual graph before any check runs. No other repair is
attempted -- a certification failure means the complex needs more exploration,
not automatic correction.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

import relucent.config as cfg
from relucent._logging import logger
from relucent.errors import IncompleteDualGraphError, NonGenericArrangementError, ShiProofError
from relucent.incidence import (
    certify_dual_graph,
    face_tag,
    ss_nonzero_indices,
    verify_contracted_shis,
    verify_flip_shi_symmetry,
)
from relucent.poly import Polyhedron
from relucent.utils import encode_ss, flip_ss_at_shi, get_mp_context, process_aware_cpu_count
from relucent.worker_context import get_worker_context, set_worker_context

if TYPE_CHECKING:
    from relucent.complex import Complex

__all__ = [
    "CertifyLevel",
    "certify_complex",
    "verify_arrangement_genericity",
    "verify_boundary_cell",
    "verify_lp_flip_neighbors_in_complex",
    "verify_shi_geometry",
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


def certify_complex(
    cplx: Complex,
    *,
    level: CertifyLevel = CertifyLevel.COMPLETE,
    repair: bool = True,
    graph: nx.Graph[Polyhedron] | None = None,
    record_state: bool = False,
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
            one is built via :meth:`~relucent.complex.Complex.get_dual_graph`.
        record_state: When True, also update
            :meth:`~relucent.complex.Complex.set_exploration_state` so callers
            do not need a separate state write.

    Raises:
        ShiFlipInvariantError, DualGraphAsymmetricEdgeError, CubicalConsistencyError:
            Combinatorial invariants are violated.
        IncompleteDualGraphError: ``level >= COMPLETE`` and a geometric facet has
            no same-dimension neighbor in the complex.
        ShiProofError: ``level == GEOMETRIC`` and a cached ``_shis`` list does not
            match a fresh LP recompute.
    """
    if len(cplx) == 0:
        if record_state:
            complete = True if cplx._complete is None else bool(cplx._complete)
            cplx.set_exploration_state(complete=complete, verified=True)
        else:
            cplx._verified = True
        return

    top_dim = max(int(p.dim) for p in cplx)
    is_contracted_slice = top_dim != int(cplx.dim)

    g = graph if graph is not None else cplx.get_dual_graph(verbose=False, repair=repair)

    logger.info("certify_complex: flip-SHI symmetry ...")
    t_stage = time.perf_counter()
    verify_flip_shi_symmetry(cplx)
    logger.info("certify_complex: flip-SHI symmetry finished in %.1fs", time.perf_counter() - t_stage)

    logger.info("certify_complex: dual-graph certification ...")
    t_stage = time.perf_counter()
    certify_dual_graph(g, cplx)
    logger.info("certify_complex: dual-graph certification finished in %.1fs", time.perf_counter() - t_stage)

    if is_contracted_slice:
        logger.info("certify_complex: contracted-slice SHI checks ...")
        t_stage = time.perf_counter()
        verify_contracted_shis(cplx)
        logger.info("certify_complex: contracted-slice SHI checks finished in %.1fs", time.perf_counter() - t_stage)

    if _LEVEL_RANK[level] >= _LEVEL_RANK[CertifyLevel.COMPLETE] and cplx.complete is True:
        logger.info("certify_complex: LP facet completeness certification ...")
        t_stage = time.perf_counter()
        verify_lp_flip_neighbors_in_complex(cplx)  # skip expensive LPs on partial complexes
        logger.info("certify_complex: LP facet completeness finished in %.1fs", time.perf_counter() - t_stage)

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
    """Whether ``poly._shis`` came from a strict SHI solve and can be trusted directly."""
    return poly._shis is not None and bool(getattr(poly, "_shis_strict", False))


def _verify_lp_neighbors_for_poly(
    poly: Polyhedron,
    *,
    top_tags: frozenset[bytes],
    bound: float,
) -> tuple[str | None, list[str]]:
    """Return a per-poly SHI recompute error or missing-neighbor diagnostics."""
    from relucent.calculations import get_shis

    try:
        lp_shis = get_shis(poly, bound=float(bound), strict=True)
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
        if int(ss.ravel()[shi_i]) == 0:
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
    from relucent._network_scale import default_polyhedron_bound

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
        logger.info(
            "verify_lp_flip_neighbors_in_complex: certifying %d top cells "
            + "(%d trusted cached SHIs, %d LP recertify, %d worker%s)",
            total_cells,
            len(serial_tasks),
            len(tasks),
            worker_count,
            "" if worker_count == 1 else "s",
        )
    else:
        logger.info(
            "verify_lp_flip_neighbors_in_complex: certifying %d top cells " + "(trusted cached SHIs only, no LP recertify)",
            total_cells,
        )
    t_verify = time.perf_counter()
    missing: list[str] = []
    pbar = tqdm(
        desc="verify_lp_flip_neighbors_in_complex",
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
        with get_mp_context().Pool(
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

    logger.info(
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
    """Recompute SHIs and require the cached list to match."""
    from relucent.calculations import get_shis

    if poly._shis is None:
        raise ShiProofError(f"Polyhedron {poly!r} has no cached _shis.")
    if bound is None:
        bound = poly.bound
    if bound is None:
        from relucent._network_scale import default_polyhedron_bound

        if poly._net is None:
            raise ShiProofError(f"Polyhedron {poly!r} has no network for bound estimation.")
        bound = default_polyhedron_bound(poly._net)
    fresh = get_shis(poly, bound=float(bound), strict=True)
    if set(fresh) != set(poly._shis):
        raise ShiProofError(f"Cached _shis {sorted(poly._shis)} != recomputed {sorted(fresh)} on {poly!r}.")


def verify_boundary_cell(poly: Polyhedron, boundary_shi: int) -> None:
    """Both ambient cofaces of a boundary top cell must be nonempty."""
    from relucent.boundary_search import _both_ambient_cofaces_feasible

    if not _both_ambient_cofaces_feasible(poly, boundary_shi):
        raise ValueError(f"Boundary cell {poly!r} fails ambient coface feasibility at shi={boundary_shi}.")


# ---------------------------------------------------------------------------
# Geometric genericity (1-dimensional arrangements)
# ---------------------------------------------------------------------------


def _quantize_point_for_genericity(pt: np.ndarray) -> tuple[int, ...]:
    vec = np.asarray(pt, dtype=np.float64).ravel()
    scale = max(1.0, float(np.max(np.abs(vec))))
    step = float(cfg.TOL_INTERIOR_VERIFY) * scale
    return tuple(int(round(float(x) / step)) for x in vec)


def _one_cell_endpoint_map(poly: Polyhedron) -> dict[bytes, tuple[int, np.ndarray]]:
    """Map combinatorial 0-face tags to ``(witness shi, geometric point)`` for a 1-cell."""
    if int(poly.dim) != 1:
        return {}
    out: dict[bytes, tuple[int, np.ndarray]] = {}
    ss = np.asarray(poly.ss_np)
    hs = np.asarray(poly.halfspaces_np)
    for shi in ss_nonzero_indices(ss):
        shi_i = int(shi)
        active = np.array(list(poly.zero_indices) + [shi_i], dtype=np.intp)
        pt = poly._halfspace_point(hs, active)
        if pt is None:
            continue
        out[face_tag(ss, shi_i)] = (shi_i, np.asarray(pt, dtype=np.float64).reshape(-1))
    return out


def verify_arrangement_genericity(polys: Iterable[Polyhedron]) -> None:
    """Geometric check for 1-dimensional arrangements (transversality).

    Combinatorial 0-face endpoints on a 1-cell must be geometrically distinct,
    and geometrically coincident endpoints across different 1-cells must share
    a combinatorial 0-face tag. Violations mean the underlying hyperplane
    arrangement is not generic (hyperplanes concur at a point).
    """
    cells = [p for p in polys if int(p.dim) == 1]
    if not cells:
        return

    endpoint_maps = {p.tag: _one_cell_endpoint_map(p) for p in cells}

    for poly in cells:
        ep_map = endpoint_maps[poly.tag]
        if not ep_map:
            continue
        geom_buckets: dict[tuple[int, ...], list[bytes]] = defaultdict(list)
        for tag, (_shi, pt) in ep_map.items():
            geom_buckets[_quantize_point_for_genericity(pt)].append(tag)
        if len(geom_buckets) < len(ep_map):
            raise NonGenericArrangementError(
                "1-cell "
                + f"{poly!r} has {len(ep_map)} combinatorial endpoint(s) but only "
                + f"{len(geom_buckets)} distinct geometric location(s); hyperplanes likely "
                + "concur at a vertex. Try a later training epoch or a non-degenerate "
                + "initialization."
            )

    for i, left in enumerate(cells):
        left_map = endpoint_maps[left.tag]
        left_geom = {_quantize_point_for_genericity(pt) for _shi, pt in left_map.values()}
        left_tags = set(left_map)
        for right in cells[i + 1 :]:
            right_map = endpoint_maps[right.tag]
            right_geom = {_quantize_point_for_genericity(pt) for _shi, pt in right_map.values()}
            right_tags = set(right_map)
            shared_geom = left_geom & right_geom
            shared_tags = left_tags & right_tags
            if not shared_geom:
                continue
            if len(shared_tags) == 0:
                raise NonGenericArrangementError(
                    "1-cells "
                    + f"{left!r} and {right!r} share a geometric endpoint but no combinatorial "
                    + "0-face tag (non-transversal junction). Try a later training epoch or "
                    + "more exploration."
                )
            if len(shared_tags) > 1:
                raise NonGenericArrangementError(
                    "1-cells "
                    + f"{left!r} and {right!r} share {len(shared_tags)} combinatorial 0-face "
                    + "tags; adjacency is ambiguous."
                )
