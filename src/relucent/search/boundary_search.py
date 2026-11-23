"""Slices of the complex: cells on one or more bent hyperplanes, explored without the ambient complex.

A slice holds the cells whose sign sequence is zero at a fixed set of neurons (one neuron's bent
hyperplane, or where several cross). :meth:`~relucent.core.complex.Complex.searcher` explores
one by its ``fixed`` argument, starting from a cell the caller supplies, and certifies it with
:func:`~relucent.search.exploration.finalize_boundary_complex`. :func:`boundary_component` does
this for one neuron. A slice search finds only the connected component holding its start and says
nothing about other components; for a whole bent hyperplane, explore the input space
(:meth:`~relucent.core.complex.Complex.bfs`) and then call
:meth:`~relucent.core.complex.Complex.boundary_complex`.

This module also holds the helpers the slice search and its finalize step share.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

import relucent.config as cfg
from relucent._internal.logging import logger, with_verbosity
from relucent._internal.network_scale import default_polyhedron_bound
from relucent._internal.parallel import process_aware_cpu_count, worker_pool
from relucent.core.errors import IncompleteDualGraphError
from relucent.core.poly import Polyhedron
from relucent.graph.incidence import ss_nonzero_indices

if TYPE_CHECKING:
    from relucent.core.complex import Complex
    from relucent.model.model import ReLUNetwork

__all__ = [
    "boundary_component",
]


def _as_fixed(fixed: int | Iterable[int]) -> tuple[int, ...]:
    """Sorted, de-duplicated fixed indices."""
    if isinstance(fixed, (int, np.integer)):
        return (int(fixed),)
    return tuple(sorted({int(j) for j in fixed}))


def _is_slice_ss(ss: np.ndarray, fixed: int | Iterable[int]) -> bool:
    """True when ``ss`` is zero exactly at ``fixed``: a top cell of that slice."""
    row = np.asarray(ss, dtype=np.int8).ravel()
    return set(np.flatnonzero(row == 0).tolist()) == set(_as_fixed(fixed))


def _ambient_lifts(poly: Polyhedron, fixed: int | Iterable[int]) -> list[Polyhedron]:
    """Every cell got by giving the ``fixed`` (zero) entries of ``poly`` signs ``±1``."""
    net = poly._net
    if net is None:
        raise ValueError("slice cell missing network reference for ambient lift")
    idx = list(_as_fixed(fixed))
    lifts = []
    for signs in itertools.product((1, -1), repeat=len(idx)):
        ss = np.asarray(poly.ss_np, dtype=np.int8).copy()
        ss[idx] = signs
        lifts.append(Polyhedron(net, ss))
    return lifts


def _all_ambient_lifts_feasible(poly: Polyhedron, fixed: int | Iterable[int]) -> bool:
    """True when every sign choice on the ``fixed`` entries gives a nonempty cell.

    That is what makes a slice cell a real face of the ambient complex: in a generic arrangement,
    the ``k`` hyperplanes through it cut its neighbourhood into all ``2^k`` sign regions. A sign
    pattern that is feasible but fails this (a phantom) is not a cell.
    """
    return all(p.feasible for p in _ambient_lifts(poly, fixed))


def _both_ambient_cofaces_feasible(poly: Polyhedron, boundary_shi: int) -> bool:
    """True when lifts to ``ss[boundary_shi]=±1`` are both nonempty."""
    return _all_ambient_lifts_feasible(poly, (boundary_shi,))


def _ambient_boundary_metadata_for_cell(
    poly: Polyhedron,
    fixed: int | Iterable[int],
    *,
    bound: float | None = None,
) -> tuple[list[int], Any, np.ndarray, np.ndarray | None]:
    """``(_shis, halfspaces, halfspaces_err, halfspaces_ss)`` for a slice top cell.

    ``_shis`` are all nonzero sign-sequence crossings on the slice (finalized by
    :func:`~relucent.graph.incidence.set_contracted_shis`); the rows are those of the ambient
    coface with every ``fixed`` entry set to ``+1``, matching :meth:`Complex.boundary_cells`.
    """
    fixed = _as_fixed(fixed)
    ss = np.asarray(poly.ss_np, dtype=np.int8)
    if any(int(ss[j]) != 0 for j in fixed):
        raise ValueError(f"Expected ss = 0 at {list(fixed)} on a slice cell, got {ss!r}")
    del bound  # the lift's rows do not depend on the LP box
    shis = sorted(int(s) for s in ss_nonzero_indices(ss))
    lift = _ambient_lifts(poly, fixed)[0]  # all fixed entries +1
    return shis, lift.halfspaces, lift.halfspaces_err_np, lift.halfspaces_rows_ss


def _set_coface_rows(poly: Polyhedron, halfspaces: Any, halfspaces_err: np.ndarray, halfspaces_ss: np.ndarray | None) -> None:
    """Give a slice cell its ambient coface's rows, with their float64 error scale."""
    poly._halfspaces = halfspaces
    poly._halfspaces_np = None
    poly._halfspaces_err = halfspaces_err
    poly._halfspaces_ss = halfspaces_ss
    poly._halfspaces_own = False


def _apply_ambient_boundary_shis(
    cx: Complex,
    fixed: int | Iterable[int],
    *,
    bound: float | None = None,
    nworkers: int | None = None,
) -> None:
    """Assign slice ``_shis`` and ambient coface rows to every top slice cell before dual-graph finalize."""
    if len(cx) == 0:
        return
    fixed = _as_fixed(fixed)
    if bound is None:
        bound = cfg.DEFAULT_SEARCH_BOUND if cx._net is None else default_polyhedron_bound(cx._net)
    nw = nworkers or process_aware_cpu_count() or 1
    polys = list(cx)
    if nw <= 1 or len(polys) < 32:
        for poly in polys:
            shis, halfspaces, halfspaces_err, halfspaces_ss = _ambient_boundary_metadata_for_cell(poly, fixed, bound=bound)
            poly._shis = shis
            poly._shis_strict = False  # slice crossings, not this cell's certified LP facets
            _set_coface_rows(poly, halfspaces, halfspaces_err, halfspaces_ss)
        return

    from relucent.search.worker_context import set_worker_context

    tasks = [(poly.ss_np, poly.tag) for poly in polys]
    tag_to_poly = {poly.tag: poly for poly in polys}
    with worker_pool(nw, initializer=set_worker_context, initargs=(cx._net, False)) as pool:
        results = pool.starmap(partial(_ambient_coface_shis_worker, fixed=fixed, bound=bound), tasks)
    for tag, shis, halfspaces, halfspaces_err, halfspaces_ss in results:
        tag_to_poly[tag]._shis = shis
        tag_to_poly[tag]._shis_strict = False  # slice crossings, not this cell's certified LP facets
        _set_coface_rows(tag_to_poly[tag], halfspaces, halfspaces_err, halfspaces_ss)
    logger.debug("slice finalize: ambient coface _shis for %d cells (%d workers)", len(polys), nw)


def _ambient_coface_shis_worker(
    ss: np.ndarray,
    tag: bytes,
    *,
    fixed: tuple[int, ...],
    bound: float | None,
) -> tuple[bytes, list[int], Any, np.ndarray, np.ndarray | None]:
    from relucent.search.worker_context import get_worker_context

    ctx = get_worker_context()
    poly = Polyhedron(ctx.net, ss)
    shis, halfspaces, halfspaces_err, halfspaces_ss = _ambient_boundary_metadata_for_cell(poly, fixed, bound=bound)
    return tag, shis, halfspaces, halfspaces_err, halfspaces_ss


def _slice_start(net: ReLUNetwork, start: Polyhedron | np.ndarray, fixed: tuple[int, ...]) -> Polyhedron:
    """``start`` as a validated top cell of the slice zero at ``fixed``."""
    poly = start if isinstance(start, Polyhedron) else Polyhedron(net, np.asarray(start, dtype=np.int8).reshape(-1))
    if poly._net is None:
        poly._net = net
    zeros = sorted(np.flatnonzero(np.asarray(poly.ss_np).ravel() == 0).tolist())
    if zeros != list(fixed):
        if set(fixed) - set(zeros):
            raise ValueError(
                f"start must be 0 at every fixed index {list(fixed)}; it is nonzero at {sorted(set(fixed) - set(zeros))} "
                + "(fixing a nonzero sign is not supported)"
            )
        raise ValueError(f"start must be zero only at the fixed indices {list(fixed)}; got zeros at {zeros}")
    if not poly.feasible or not _all_ambient_lifts_feasible(poly, fixed):
        raise ValueError(f"start {poly.ss_np!r} is not a cell of the slice where neurons {list(fixed)} are 0")
    return poly


@with_verbosity
def boundary_component(
    net: ReLUNetwork,
    boundary_shi: int,
    start: Polyhedron | np.ndarray,
    *,
    bound: float | None = None,
    nworkers: int | None = None,
    verify: bool = True,
    verbose: int | None = None,
    **kwargs: Any,
) -> Complex:
    """The connected component of neuron ``boundary_shi``'s bent hyperplane that contains ``start``.

    A breadth-first :meth:`~relucent.core.complex.Complex.searcher` with ``fixed=[boundary_shi]``.
    Only the component holding ``start`` is found; nothing is implied about other components.

    Args:
        net: Canonical :class:`~relucent.model.model.ReLUNetwork`.
        boundary_shi: Global supporting-hyperplane index of the neuron whose bent hyperplane to explore.
        start: A top cell of that hyperplane: a :class:`~relucent.core.poly.Polyhedron` or a sign
            sequence whose only zero is at ``boundary_shi``, with both ambient cofaces nonempty.
        bound: Gurobi box bound for SHI LPs; defaults to ``default_polyhedron_bound()``.
        nworkers: Worker process count.
        verify: When True (default), require a complete search and certify the component.
        verbose: Output level: ``0`` quiet, ``1`` progress bars and summaries, ``2`` debug detail.
            ``None`` uses :data:`relucent.config.VERBOSE`.
        **kwargs: Forwarded to :meth:`~relucent.core.complex.Complex.searcher`.

    Returns:
        A new :class:`~relucent.core.complex.Complex` of the component's top cells.

    Raises:
        ValueError: If ``start`` is not a top cell of the hyperplane.
        IncompleteDualGraphError: If ``verify`` and the search stopped before exhausting the component.
    """
    del verbose  # applied by @with_verbosity
    from relucent.core.complex import Complex

    shi = int(boundary_shi)
    poly = start if isinstance(start, Polyhedron) else Polyhedron(net, np.asarray(start, dtype=np.int8).reshape(-1))
    if not _is_slice_ss(poly.ss_np, (shi,)):
        raise ValueError(f"start must have ss[{shi}]=0 as its only zero entry; got {poly.ss_np!r}")
    component = Complex(net)
    result = component.searcher(poly, fixed=(shi,), bound=bound, nworkers=nworkers, verify=verify, **kwargs)
    if verify and not result.complete:
        raise IncompleteDualGraphError("boundary component search stopped before exhausting the component")
    return component
