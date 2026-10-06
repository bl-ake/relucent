"""Recover the chain complex from verified vertices' local stars.

Masden (2022), Theorem 20: for a generic, supertransversal network the sign-sequence
complex ``S(F)`` is a pure, ambient-dimensional cubical complex. Every vertex has
exactly ``ambient_dim`` zero sign entries (Lemma 16), and once one is verified, every
sign assignment on those entries (others held fixed) is a real cell of ``C(F)``
(Lemma 18). So there's no need to rediscover neighboring top cells or verify cubes.

This replaces ``graph.covectors``, which needed BFS to have found the full ``2^c``
cube of top cells around a vertex. That's more than the theorem needs, and it silently
dropped cells that provably exist.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import combinations, product
from typing import TYPE_CHECKING, Any, cast

import numpy as np

import relucent.config as cfg
from relucent._internal.logging import logger, with_verbosity
from relucent._internal.parallel import get_mp_context, process_aware_cpu_count, worker_pool
from relucent.core.ss import encode_ss
from relucent.graph import incidence

if TYPE_CHECKING:
    import networkx as nx

    from relucent.core.complex import Complex
    from relucent.core.poly import Polyhedron
    from relucent.model.model import ReLUNetwork

__all__ = [
    "VertexRecord",
    "build_chain_complex",
    "cells_from_vertices",
    "expand_vertex_star",
    "find_vertices",
    "recover_cells_from_vertices",
    "verified_vertices",
    "vertex_polyhedron",
]

# Each worker needs at least this many candidates to be worth its startup cost.
# Below this, verification runs serially. Above it, the worker count scales with the
# candidate count (see `_verify_candidates_parallel`) instead of always using
# `nworkers`. Measured: 1,132 candidates got slower with 32 workers (0.54s -> 1.42s);
# 193k and 731k got ~2x faster. This sits between the two.
MIN_CANDIDATES_PER_WORKER = 4096
PARALLEL_VERIFY_MIN_CANDIDATES = 2 * MIN_CANDIDATES_PER_WORKER


@dataclass(frozen=True)
class VertexRecord:
    """A verified vertex, with everything needed to expand its local star.

    Fields:
        ss: Sign sequence of the vertex (zeros mark the hyperplanes through it).
        point: Coordinates of the vertex in input space.
        witness_tag: Tag of a top-dimensional cell known to contain this vertex.
        varying_shis: Hyperplane indices that vary among the cells around the vertex.
    """

    ss: np.ndarray
    point: np.ndarray
    witness_tag: bytes
    varying_shis: tuple[int, ...]

    @property
    def tag(self) -> bytes:
        """Bytes-encoded sign sequence, the key used by :class:`~relucent.core.ss.SSManager`."""
        return encode_ss(self.ss)


def _coface_incident_shis(graph: nx.Graph[Polyhedron], poly: Polyhedron) -> set[int]:
    """Hyperplanes with a real (dual-graph-verified) flip-neighbor edge from ``poly``."""
    return {int(data["shi"]) for _u, _v, data in graph.edges(poly, data=True) if data.get("shi") is not None}


def _root_candidate_combos(
    root: Polyhedron, incident: list[int], *, ambient_dim: int, n_more: int, screen: bool
) -> list[tuple[int, ...]]:
    """The ``n_more``-subsets of ``incident`` to zero in ``root``'s sign sequence for its candidates."""
    root_ss = np.asarray(root.ss_np, dtype=np.int8)
    n_fixed_zeros = int(np.count_nonzero(root_ss.ravel() == 0))
    if n_fixed_zeros + n_more != ambient_dim:
        # Lemma 16: a `top_dim`-cell has exactly `ambient_dim - top_dim`
        # zero entries; reaching a full vertex needs exactly `top_dim`
        # more. A mismatch means `root` is not actually a `top_dim`-cell.
        return []
    if n_more == 0:
        return [()]
    if len(incident) < n_more:
        return []
    combos = list(combinations(incident, n_more))
    if screen:
        fixed = np.flatnonzero(root_ss.ravel() == 0)
        zeros = np.hstack((np.broadcast_to(fixed, (len(combos), fixed.size)), np.asarray(combos, dtype=np.intp)))
        excluded = _provably_not_vertices(root, zeros)
        combos = [combo for combo, out in zip(combos, excluded.tolist(), strict=True) if not out]
    return combos


# Roots and their incident hyperplanes while a fork-started pool screens candidates: inherited
# by the forked workers, never pickled. None otherwise.
_screen_state: tuple[list[Polyhedron], list[list[int]], int, int] | None = None
PARALLEL_SCREEN_MIN_ROOTS = 2048


def _screen_roots_chunk(lo: int, hi: int) -> list[tuple[int, tuple[int, ...]]]:
    """Worker: surviving ``(root index, combo)`` pairs for roots ``lo:hi`` of :data:`_screen_state`."""
    assert _screen_state is not None
    roots, incidents, ambient_dim, n_more = _screen_state
    out: list[tuple[int, tuple[int, ...]]] = []
    for k in range(lo, hi):
        root = roots[k]
        cached = root._get_cached_halfspaces_np() is not None
        combos = _root_candidate_combos(root, incidents[k], ambient_dim=ambient_dim, n_more=n_more, screen=True)
        out.extend((k, combo) for combo in combos)
        if not cached:
            # Rows computed here would only fill this worker's copy of the complex.
            root._halfspaces = root._halfspaces_np = root._halfspaces_err = root._w = root._b = None
    return out


def _generate_vertex_candidates(
    top_cells: Iterable[Polyhedron],
    graph: nx.Graph[Polyhedron],
    *,
    ambient_dim: int,
    top_dim: int,
    screen: bool = False,
    nworkers: int = 1,
) -> dict[bytes, tuple[Polyhedron, np.ndarray, tuple[int, ...]]]:
    """Enumerate every candidate vertex sign-sequence reachable from ``top_cells``.

    A cell of dimension ``top_dim`` already has ``ambient_dim - top_dim`` zero
    entries (Lemma 16); reaching a full vertex needs exactly ``top_dim`` more,
    chosen from the cell's *incident* dual-graph directions (real flip
    neighbors — not every nonzero sign entry). This is pure combinatorics (no
    verification); ``find_vertices`` checks each candidate afterward.

    Deduped by tag: when several top cells reach the same candidate, only the
    first one seen is kept as its witness. Which witness is used doesn't
    affect the verification result (Theorem 20), so this is safe -- it just
    avoids redundant checks of the same candidate from different cofaces.

    With ``screen``, each root's candidates are first screened together by
    :func:`_provably_not_vertices`, and only the survivors (a few per root) are materialized and
    deduped. Since the verdict does not depend on the witness, a candidate screened out from one
    root would be screened out from any other, so no record of screened-out ones is kept. On
    large complexes the screening runs across ``nworkers`` forked processes; the survivors are
    deduped here in root order, as sequentially.
    """
    global _screen_state
    n_more = top_dim
    roots = list(top_cells)
    incidents = [sorted(_coface_incident_shis(graph, root)) for root in roots]
    pairs: Iterable[tuple[int, tuple[int, ...]]]
    pool = None
    pending: dict[bytes, tuple[Polyhedron, np.ndarray, tuple[int, ...]]] = {}
    try:
        ctx = get_mp_context()
        if screen and nworkers > 1 and len(roots) >= PARALLEL_SCREEN_MIN_ROOTS and ctx.get_start_method() == "fork":
            _screen_state = (roots, incidents, ambient_dim, n_more)
            pool = worker_pool(nworkers)
            chunk = max(len(roots) // (nworkers * 8), 1)
            ranges = [(lo, min(lo + chunk, len(roots))) for lo in range(0, len(roots), chunk)]
            pairs = (pair for part in pool.imap(_screen_roots_chunk_star, ranges) for pair in part)
        else:
            pairs = (
                (k, combo)
                for k, root in enumerate(roots)
                for combo in _root_candidate_combos(root, incidents[k], ambient_dim=ambient_dim, n_more=n_more, screen=screen)
            )
        for k, combo in pairs:
            candidate = np.asarray(roots[k].ss_np, dtype=np.int8).copy()
            flat = candidate.reshape(-1)
            for shi in combo:
                flat[shi] = 0
            pending.setdefault(encode_ss(candidate), (roots[k], candidate, combo))
    finally:
        if pool is not None:
            pool.shutdown()
        _screen_state = None
    return pending


def _screen_roots_chunk_star(bounds: tuple[int, int]) -> list[tuple[int, tuple[int, ...]]]:
    return _screen_roots_chunk(*bounds)


def _provably_not_vertices(root: Polyhedron, zeros: np.ndarray) -> np.ndarray:
    """For each row of ``zeros`` (index sets to zero in ``root``'s sign sequence), whether the
    candidate is provably not a vertex: a batched float64 check.

    Most candidates are not vertices (on real checkpoints ~50 per vertex): the point where their
    zeroed rows vanish lies clearly outside another row of the witness cell. This is the same
    rigorous test :meth:`Polyhedron.verify_vertex_covector` applies first
    (:func:`relucent._internal.rounding.solve_equalities` then ``classify_rows``), vectorized over
    the candidates: True only when the exact point provably violates some row. Candidates it
    cannot rule out, including those whose rows are dependent within their error, are False and
    left for the full check.
    """
    from relucent._internal import rounding

    m = zeros.shape[0]
    excluded = np.zeros(m, dtype=bool)
    if m == 0 or (root._net is None and root._get_cached_halfspaces_np() is None):
        return excluded  # rows unavailable here: leave these candidates to the full check
    n = int(np.asarray(root.ss_np).size)
    h = np.asarray(root.halfspaces_np, dtype=np.float64)[:n]
    e = np.asarray(root.halfspaces_err_np, dtype=np.float64)[:n]
    dim = h.shape[1] - 1
    if zeros.shape[1] != dim:
        return excluded
    live = ~np.all(h[:, :-1] == 0.0, axis=1)  # dead units' constant rows are not hyperplanes
    a, rhs, ea = h[zeros, :-1], -h[zeros, -1], e[zeros]
    da = np.sqrt(np.sum(ea[:, :, :-1] ** 2, axis=(1, 2)))
    # A lower bound on the smallest singular value, 1 / ||A^-1||_F, cheaper than an SVD; the
    # computed inverse is accurate to cond * eps relative, so deflating it by 1e-6 keeps it a
    # bound wherever cond < 1e8, and the rest are left to the full check.
    with np.errstate(all="ignore"):
        try:
            inv_norm = np.linalg.norm(np.linalg.inv(a), axis=(1, 2))
        except np.linalg.LinAlgError:
            return excluded
        cond = inv_norm * np.linalg.norm(a, axis=(1, 2))
        smin = 1.0 / (inv_norm * (1.0 + 1e-6))
    ok = np.isfinite(cond) & (cond < 1e8) & (smin > 2.0 * da) & np.all(live[zeros], axis=1)
    if not np.any(ok):
        return excluded
    a, rhs, ea, da, smin, zeros_ok = a[ok], rhs[ok], ea[ok], da[ok], smin[ok], zeros[ok]
    x = np.linalg.solve(a, rhs[..., None])[..., 0]
    ax = np.abs(x)
    resid = np.abs(np.einsum("mij,mj->mi", a, x) - rhs)
    slack = 1.0 + 4.0 * rounding.EPS
    hz_abs = np.abs(h[zeros_ok]) * (2.0 * rounding.gamma(dim + 1))
    err_rows = (np.einsum("mij,mj->mi", ea[:, :, :-1], ax) + ea[:, :, -1]) * slack
    err_rows += (np.einsum("mij,mj->mi", hz_abs[:, :, :-1], ax) + hz_abs[:, :, -1]) * slack
    x_err = np.linalg.norm(resid + err_rows, axis=1) / (smin - da)
    values = x @ h[:, :-1].T + h[:, -1]
    bound = (ax @ e[:, :-1].T + e[:, -1]) * slack
    bound += x_err[:, None] * (np.linalg.norm(h[:, :-1], axis=1) + np.linalg.norm(e[:, :-1], axis=1))[None, :]
    outside = (values > bound) & live[None, :]
    outside[np.arange(zeros_ok.shape[0])[:, None], zeros_ok] = False
    excluded[ok] = np.any(outside, axis=1)
    return excluded


def _verify_candidate_chunk(
    chunk: list[tuple[bytes, Polyhedron, np.ndarray]],
    net: ReLUNetwork,
) -> list[tuple[bytes, np.ndarray | None]]:
    """Verify one chunk of candidates in a worker process.

    Calls :meth:`Polyhedron.verify_vertex_covector` verbatim (no reimplemented
    math) so this can never silently drift from the sequential path.
    """
    results: list[tuple[bytes, np.ndarray | None]] = []
    for tag, root, candidate_ss in chunk:
        if root._net is None:
            root._net = net  # pickling drops it; exact fallback needs it to rebuild the rows
        results.append((tag, root.verify_vertex_covector(candidate_ss)))
    return results


# Candidates while a fork-started pool verifies them: inherited by the workers, never pickled.
_verify_state: list[tuple[bytes, Polyhedron, np.ndarray]] | None = None


def _verify_range(bounds: tuple[int, int]) -> list[tuple[bytes, np.ndarray]]:
    """Worker: ``(tag, point)`` for the verified vertices among ``_verify_state[lo:hi]``."""
    assert _verify_state is not None
    out: list[tuple[bytes, np.ndarray]] = []
    for tag, root, candidate in _verify_state[bounds[0] : bounds[1]]:
        cached = root._get_cached_halfspaces_np() is not None
        point = root.verify_vertex_covector(candidate)
        if point is not None:
            out.append((tag, point))
        if not cached:
            # Rows computed here would only fill this worker's copy of the complex.
            root._halfspaces = root._halfspaces_np = root._halfspaces_err = root._w = root._b = None
    return out


def _verify_candidates_parallel(
    pending: dict[bytes, tuple[Polyhedron, np.ndarray, tuple[int, ...]]],
    *,
    net: ReLUNetwork,
    nworkers: int,
) -> dict[bytes, np.ndarray]:
    """Verify every pending candidate across a worker pool; returns tag -> point for hits.

    Chunked in the same (root-grouped) order ``pending`` was built in, so most
    of a root's candidates land in one chunk -- pickle's object memoization
    then serializes each shared root only once per chunk, not once per
    candidate, keeping the per-task payload close to one copy per unique root.
    """
    global _verify_state
    items = list(pending.items())
    n = len(items)
    # Scale workers to the work: a candidate count just past the gate in
    # `find_vertices` should get a couple of workers, not 32 sitting idle.
    # `effective_nworkers` is always >= 2 here.
    effective_nworkers = min(nworkers, max(1, n // MIN_CANDIDATES_PER_WORKER))
    ctx = get_mp_context()
    if ctx.get_start_method() == "fork":
        # Forked workers inherit the candidates and their roots (network attached): send them
        # index ranges only, instead of pickling every root with its rows.
        chunk = max(n // (effective_nworkers * 4), 1)
        verified_fork: dict[bytes, np.ndarray] = {}
        try:
            _verify_state = [(tag, root, candidate) for tag, (root, candidate, _combo) in items]
            with worker_pool(effective_nworkers) as pool:
                for part in pool.imap_unordered(_verify_range, [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]):
                    verified_fork.update(part)
        finally:
            _verify_state = None
        return verified_fork

    # `Polyhedron.__reduce__` drops `_net` when pickling, so lazily-computed properties
    # that need it must be resolved here, before a root crosses the Pool boundary.
    # `halfspaces_np` goes first: `ambient_dim` falls back to `self.halfspaces`, which
    # reuses `_halfspaces_np` if cached instead of recomputing via `_net`. Only the exact
    # fallback (rebuilding a root's rows when float64 can't decide) still needs `net`.
    seen_roots: set[int] = set()
    for root, _candidate, _combo in pending.values():
        if id(root) in seen_roots:
            continue
        seen_roots.add(id(root))
        _ = root.halfspaces_np
        root._ambient_dim = int(root.ambient_dim)

    chunk_size = max(n // (effective_nworkers * 4), 1)
    chunks = [
        [(tag, root, candidate) for tag, (root, candidate, _combo) in items[i : i + chunk_size]]
        for i in range(0, n, chunk_size)
    ]
    verified: dict[bytes, np.ndarray] = {}
    with worker_pool(effective_nworkers) as pool:
        for chunk_results in pool.starmap(
            _verify_candidate_chunk,
            [(chunk, net) for chunk in chunks],
        ):
            for tag, point in chunk_results:
                if point is not None:
                    verified[tag] = point
    return verified


def find_vertices(
    top_cells: Iterable[Polyhedron],
    graph: nx.Graph[Polyhedron],
    *,
    ambient_dim: int,
    top_dim: int,
    verify_vertex: Callable[[Polyhedron, np.ndarray], np.ndarray | None],
    net: ReLUNetwork | None = None,
    nworkers: int = 1,
    screen: bool = False,
) -> dict[bytes, VertexRecord]:
    """Seed every candidate vertex reachable from ``top_cells``, and verify it.

    With ``screen`` (only when ``verify_vertex`` is :meth:`Polyhedron.verify_vertex_covector`),
    candidates are first screened in batches per witness by :func:`_provably_not_vertices`,
    which drops only those that check would reject.

    Candidate generation (:func:`_generate_vertex_candidates`) is always
    sequential -- it's cheap combinatorics. Verification of each candidate
    (typically :meth:`Polyhedron.verify_vertex_covector`: one float64
    equality solve plus a check of every other row) is what dominates
    runtime on large complexes, since it runs once per candidate. Each
    candidate's check is independent of every other's, so when ``net`` is
    supplied and there are enough candidates to be worth Pool startup cost,
    verification is farmed out across ``nworkers`` processes, which run
    :meth:`Polyhedron.verify_vertex_covector` itself instead of ``verify_vertex``.
    The parallel path is therefore only for that ``verify_vertex``; the parallel
    and sequential paths must otherwise produce identical results.
    """
    pending = _generate_vertex_candidates(
        top_cells, graph, ambient_dim=ambient_dim, top_dim=top_dim, screen=screen, nworkers=nworkers
    )

    use_parallel = net is not None and nworkers > 1 and len(pending) >= PARALLEL_VERIFY_MIN_CANDIDATES

    vertices: dict[bytes, VertexRecord] = {}
    if use_parallel:
        assert net is not None
        verified_points = _verify_candidates_parallel(pending, net=net, nworkers=nworkers)
        for tag, (root, candidate, combo) in pending.items():
            point = verified_points.get(tag)
            if point is None:
                continue
            varying = tuple(sorted(combo))
            vertices[tag] = VertexRecord(ss=candidate, point=point, witness_tag=root.tag, varying_shis=varying)
    else:
        for tag, (root, candidate, combo) in pending.items():
            point = verify_vertex(root, candidate)
            if point is None:
                continue
            varying = tuple(sorted(combo))
            vertices[tag] = VertexRecord(ss=candidate, point=point, witness_tag=root.tag, varying_shis=varying)
    return vertices


def expand_vertex_star(vertex: VertexRecord) -> Iterable[np.ndarray]:
    """Every cell in a verified vertex's local star (Theorem 20).

    Varies each of ``vertex.varying_shis`` (the ``top_dim`` coordinates that
    distinguish this vertex from its witness top cell) independently over
    ``{-1, 0, 1}``, holding every other coordinate — including the witness's
    own already-zero entries — fixed. This stays within the ambient
    ``top_dim`` of the complex being recovered (e.g. a boundary sub-complex's
    always-zero coordinate is never perturbed), and produces cells of every
    dimension from 0 (the vertex itself) up to ``top_dim``.
    """
    base = np.asarray(vertex.ss, dtype=np.int8)
    shis = vertex.varying_shis
    for signs in product((-1, 0, 1), repeat=len(shis)):
        ss = base.copy()
        flat = ss.reshape(-1)
        for shi, sign in zip(shis, signs, strict=True):
            flat[shi] = sign
        yield ss


def recover_cells_from_vertices(
    top_cells: Iterable[Polyhedron],
    graph: nx.Graph[Polyhedron],
    *,
    ambient_dim: int,
    top_dim: int,
    verify_vertex: Callable[[Polyhedron, np.ndarray], np.ndarray | None],
    net: ReLUNetwork | None = None,
    nworkers: int = 1,
) -> tuple[dict[int, dict[bytes, np.ndarray]], dict[bytes, VertexRecord]]:
    """Recover every cell reachable from a finite, verified vertex.

    Returns ``(cells_by_dim, vertices)``: ``cells_by_dim[k]`` maps cell tag to
    sign sequence for every recovered ``k``-cell (``0 <= k <= top_dim``);
    ``vertices`` maps vertex tag to :class:`VertexRecord` (interior point,
    witness top cell) for materialization.

    Every generated cell of dimension ``k >= 1`` has, by construction, at
    least one verified vertex among its own faces (its generating vertex,
    reached by zeroing all of ``varying_shis``) — a cell with every endpoint
    unverifiable can never be produced, so no separate "cascade drop" pass is
    needed the way the old cubical-star reconstruction required.

    ``net``/``nworkers`` are forwarded to :func:`find_vertices`
    to parallelize candidate verification on large complexes; see its docstring.
    """
    vertices = find_vertices(
        top_cells,
        graph,
        ambient_dim=ambient_dim,
        top_dim=top_dim,
        verify_vertex=verify_vertex,
        net=net,
        nworkers=nworkers,
    )
    return cells_from_vertices(vertices, top_dim=top_dim), vertices


def cells_from_vertices(vertices: dict[bytes, VertexRecord], *, top_dim: int) -> dict[int, dict[bytes, np.ndarray]]:
    """Every cell in the verified vertices' local stars, by dimension: tag -> sign sequence."""
    cells_by_dim: dict[int, dict[bytes, np.ndarray]] = {k: {} for k in range(top_dim + 1)}
    for vertex in vertices.values():
        for ss in expand_vertex_star(vertex):
            zero_count = int(np.count_nonzero(ss.reshape(-1)[list(vertex.varying_shis)] == 0))
            dim = top_dim - zero_count
            cells_by_dim[dim][encode_ss(ss)] = ss
    return cells_by_dim


def vertex_polyhedron(source: Complex, cplx: Complex, vertex: VertexRecord) -> Polyhedron:
    """Add a verified vertex to ``cplx`` as a 0-cell carrying its witness's rows and its point."""
    ambient_dim = int(source.dim)
    witness = source.tag2poly[vertex.witness_tag]
    poly = cplx.add_ss(
        vertex.ss,
        ambient_dim=ambient_dim,
        halfspaces=witness.halfspaces,
        halfspaces_err=witness.halfspaces_err_np,
        halfspaces_ss=witness.halfspaces_rows_ss,
        rows_data=witness._rows_data,
        finite=True,
    )
    poly._interior_point = vertex.point
    return poly


def verified_vertices(cplx: Complex) -> tuple[int, dict[bytes, VertexRecord]]:
    """``(top_dim, vertices)``: every verified vertex of this complete, verified complex.

    The vertices of :func:`build_chain_complex` (its 0-cells), without the rest of the chain.
    """
    from relucent.core.poly import Polyhedron

    top_dim = max(int(p.dim) for p in cplx)
    top_cells = [p for p in cplx if int(p.dim) == top_dim]
    graph = cast(Any, cplx.dual_graph(require_complete=False))
    incidence.certify_dual_graph(graph, cplx, top_dim=top_dim)

    # Candidate-vertex verification dominates runtime on large complexes (see
    # find_vertices). Passing net lets it run across a worker pool.
    vertices = find_vertices(
        top_cells,
        graph,
        net=cplx._net,
        nworkers=process_aware_cpu_count() or 1,
        ambient_dim=int(cplx.dim),
        top_dim=top_dim,
        verify_vertex=Polyhedron.verify_vertex_covector,
        screen=True,
    )
    return top_dim, vertices


@with_verbosity
def build_chain_complex(source: Complex, verbose: int | None = None) -> list[Complex]:
    """Recover the chain complex directly from verified vertices' local stars.

    Masden (2022), Theorem 20: the sign-sequence complex is a pure,
    ambient-dimensional cubical complex, so once a vertex (exactly
    ``ambient_dim`` zero sign entries, Lemma 16) is verified, *every* cell
    in its local star is algebraically guaranteed to be present (Lemma
    18's sign-product semigroup) — no independent rediscovery of
    neighboring top-dimensional cells, dual-graph cube verification, or
    coverage heuristic is required. See this module's docstring.

    Candidate vertices receive one float64 equality solve followed by a
    check against every other row of their witness cell
    (:meth:`Polyhedron.verify_vertex_covector`);
    no facet or boundedness LP is used here. Every recovered cell of
    dimension ``k >= 1`` has, by construction, at least one verified
    vertex among its own faces (its generating vertex), so a cell can
    never end up with every endpoint unverifiable.

    Raises:
        CubicalConsistencyError: If the labeled top-cell graph is not cubical.
    """
    del verbose  # applied by @with_verbosity
    source.assert_topology_ready()
    if len(source) == 0:
        return [source]
    ambient_dim = int(source.dim)
    top_dim, vertices = verified_vertices(source)
    cells_by_dim = cells_from_vertices(vertices, top_dim=top_dim)
    vertex_points = {tag: v.point for tag, v in vertices.items()}

    chain: list[Complex] = [source]
    for dim in range(top_dim - 1, -1, -1):
        recovered = cells_by_dim.get(dim, {})
        if not recovered:
            continue
        cplx = source._empty_like()
        ordered_tags = sorted(recovered)
        if dim == 1:
            ordered_tags.sort(
                key=lambda tag: (
                    -sum(
                        incidence.face_tag(recovered[tag], shi) in vertex_points
                        for shi in incidence.ss_nonzero_indices(recovered[tag])
                    ),
                    tag,
                )
            )
        elif dim == 0 and chain and len(chain[-1]) > 0 and int(chain[-1].index2poly[0].dim) == 1:
            endpoint_order: list[bytes] = []
            seen_endpoints: set[bytes] = set()
            for one_cell in chain[-1]:
                for shi in one_cell._covector_endpoint_shis or []:
                    endpoint_tag = incidence.face_tag(one_cell.ss_np, shi)
                    if endpoint_tag in recovered and endpoint_tag not in seen_endpoints:
                        endpoint_order.append(endpoint_tag)
                        seen_endpoints.add(endpoint_tag)
            ordered_tags = endpoint_order + [tag for tag in ordered_tags if tag not in seen_endpoints]
        for tag in ordered_tags:
            ss = recovered[tag]
            kwargs: dict[str, Any] = {
                "ambient_dim": ambient_dim,
            }
            point = vertex_points.get(tag)
            if dim == 0:
                vertex_polyhedron(source, cplx, vertices[tag])
                continue
            if dim == 1:
                candidate_by_shi = {shi: incidence.face_tag(ss, shi) for shi in incidence.ss_nonzero_indices(ss)}
                kwargs["covector_endpoint_shis"] = sorted(
                    shi for shi, face in candidate_by_shi.items() if face in vertex_points
                )
            poly = cplx.add_ss(ss, **kwargs)
            if point is not None:
                poly._interior_point = point

        if len(cplx) == 0:
            continue
        incidence.set_contracted_shis(cplx)
        if cfg.CAREFUL_MODE:
            incidence.verify_contracted_shis(cplx)
        cplx.set_exploration_state(complete=True, verified=True)
        chain.append(cplx)

    logger.debug("Chain: %s", ", ".join([f"{len(c)} {c.index2poly[0].dim}-cells" for c in chain]))
    return chain
