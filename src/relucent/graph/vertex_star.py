"""Recover the chain complex directly from verified vertices' local stars.

Masden, *Algorithmic Determination of the Combinatorial Structure of the Linear
Regions of ReLU Neural Networks* (2022), Theorem 20: for a generic,
supertransversal network the sign-sequence complex ``S(F)`` is a **pure,
ambient-dimensional cubical complex**. Every vertex has exactly ``ambient_dim``
zero sign entries (Lemma 16), and once such a vertex is verified, *every*
sign assignment on those zero entries — holding all other entries fixed — is a
real, present cell of ``C(F)`` (Lemma 18's sign-product semigroup). No
independent rediscovery of neighboring top-dimensional cells, dual-graph cube
verification, or coverage heuristic is required for correctness.

This replaces the previous ``graph.covectors`` approach, which only recovered
a cell when the *complete* ``2^c`` cube of top-dimensional cofaces had been
independently discovered by BFS — strictly more than the theorem requires, and
the root cause of cells that provably exist being silently dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import combinations, product
from typing import TYPE_CHECKING

import numpy as np

from relucent.utils import encode_ss, get_mp_context

if TYPE_CHECKING:
    import networkx as nx

    from relucent.core.poly import Polyhedron
    from relucent.model.model import ReLUNetwork

__all__ = ["VertexRecord", "find_vertices", "expand_vertex_star", "recover_cells_from_vertices"]

# Each worker in the pool should have at least this many candidates to verify, or
# it's not worth the process it runs in. Below one worker's worth (i.e. fewer than
# PARALLEL_VERIFY_MIN_CANDIDATES candidates total), verification falls back to the
# sequential `verify_vertex` loop entirely; above it, the *number of workers actually
# used* scales with candidate count (see `_verify_candidates_parallel`) rather than
# always spinning up all of `nworkers` -- Pool(32) startup cost is roughly constant
# regardless of workload, so a candidate count just over the gate should use a
# couple of workers, not 32. Calibrated against real checkpoints: 1,132 candidates
# regressed under a 32-worker pool (0.54s serial -> 1.42s parallel, all overhead);
# 193k and 731k candidates each got ~2x faster. This sits with margin below the
# smallest win and well above the regression.
MIN_CANDIDATES_PER_WORKER = 4096
PARALLEL_VERIFY_MIN_CANDIDATES = 2 * MIN_CANDIDATES_PER_WORKER


@dataclass(frozen=True)
class VertexRecord:
    """A verified vertex, with everything needed to expand its local star."""

    ss: np.ndarray
    point: np.ndarray
    witness_tag: bytes
    varying_shis: tuple[int, ...]

    @property
    def tag(self) -> bytes:
        return encode_ss(self.ss)


def _coface_incident_shis(graph: nx.Graph[Polyhedron], poly: Polyhedron) -> set[int]:
    """Hyperplanes with a real (dual-graph-verified) flip-neighbor edge from ``poly``."""
    return {int(data["shi"]) for _u, _v, data in graph.edges(poly, data=True) if data.get("shi") is not None}


def _generate_vertex_candidates(
    top_cells: Iterable[Polyhedron],
    graph: nx.Graph[Polyhedron],
    *,
    ambient_dim: int,
    top_dim: int,
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
    """
    n_more = top_dim
    pending: dict[bytes, tuple[Polyhedron, np.ndarray, tuple[int, ...]]] = {}
    for root in top_cells:
        root_ss = np.asarray(root.ss_np, dtype=np.int8)
        n_fixed_zeros = int(np.count_nonzero(root_ss.ravel() == 0))
        if n_fixed_zeros + n_more != ambient_dim:
            # Lemma 16: a `top_dim`-cell has exactly `ambient_dim - top_dim`
            # zero entries; reaching a full vertex needs exactly `top_dim`
            # more. A mismatch means `root` is not actually a `top_dim`-cell.
            continue
        incident = sorted(_coface_incident_shis(graph, root))
        if n_more == 0:
            combos: Iterable[tuple[int, ...]] = [()]
        elif len(incident) < n_more:
            continue
        else:
            combos = combinations(incident, n_more)
        for combo in combos:
            candidate = root_ss.copy()
            flat = candidate.reshape(-1)
            for shi in combo:
                flat[shi] = 0
            tag = encode_ss(candidate)
            if tag in pending:
                continue
            pending[tag] = (root, candidate, combo)
    return pending


def _verify_candidate_chunk(
    chunk: list[tuple[bytes, Polyhedron, np.ndarray]],
    net: ReLUNetwork,
    sign_margin: float,
) -> list[tuple[bytes, np.ndarray | None]]:
    """Verify one chunk of candidates in a worker process.

    Calls :meth:`Polyhedron.verify_vertex_covector` verbatim (no reimplemented
    math) so this can never silently drift from the sequential path. The only
    thing that needs reconstructing per-worker is ``point2preactivations``:
    it's a pure function of ``net`` (see ``Complex.preactivation_iterator``,
    which depends only on ``self._net`` and a layer list derived from it), so
    a throwaway ``Complex(net)`` here is equivalent to the caller's own complex.
    """
    from relucent.core.complex import Complex

    worker_complex = Complex(net)

    def _point2preactivations(x: np.ndarray) -> np.ndarray:
        return np.asarray(worker_complex.point2preactivations(x))

    results: list[tuple[bytes, np.ndarray | None]] = []
    for tag, root, candidate_ss in chunk:
        point = root.verify_vertex_covector(
            candidate_ss,
            point2preactivations=_point2preactivations,
            sign_margin=sign_margin,
        )
        results.append((tag, point))
    return results


def _verify_candidates_parallel(
    pending: dict[bytes, tuple[Polyhedron, np.ndarray, tuple[int, ...]]],
    *,
    net: ReLUNetwork,
    sign_margin: float,
    nworkers: int,
) -> dict[bytes, np.ndarray]:
    """Verify every pending candidate across a worker pool; returns tag -> point for hits.

    Chunked in the same (root-grouped) order ``pending`` was built in, so most
    of a root's candidates land in one chunk -- pickle's object memoization
    then serializes each shared root only once per chunk, not once per
    candidate, keeping the per-task payload close to one copy per unique root.
    """
    # `Polyhedron.__reduce__` deliberately drops `_net` when pickling (each worker's
    # copy of every root would otherwise duplicate the whole network) -- so any
    # lazily-computed property that needs it must be resolved here, in the process
    # that still has it, before a root crosses the Pool boundary. `halfspaces_np`
    # must come first: `ambient_dim`'s fallback path reads `self.halfspaces`, whose own
    # fallback (self._halfspaces is None) checks `self._halfspaces_np` next -- if that's
    # already cached (as it will be, immediately below) it's reused instead of
    # recomputing via `_net`. Only `point2preactivations`, evaluated at each candidate's
    # solved point, genuinely needs a live network afterward -- that's what `net` is for.
    seen_roots: set[int] = set()
    for root, _candidate, _combo in pending.values():
        if id(root) in seen_roots:
            continue
        seen_roots.add(id(root))
        _ = root.halfspaces_np
        root._ambient_dim = int(root.ambient_dim)

    items = list(pending.items())
    n = len(items)
    # Scale workers to work available rather than always using all of `nworkers`:
    # Pool startup cost is roughly per-worker, so a candidate count just past the
    # gate in `find_vertices` should get a couple of workers, not (say) 32 of them
    # sitting mostly idle. `find_vertices` only calls this once past the gate, so
    # `effective_nworkers` here is always >= 2.
    effective_nworkers = min(nworkers, max(1, n // MIN_CANDIDATES_PER_WORKER))
    chunk_size = max(n // (effective_nworkers * 4), 1)
    chunks = [
        [(tag, root, candidate) for tag, (root, candidate, _combo) in items[i : i + chunk_size]]
        for i in range(0, n, chunk_size)
    ]
    verified: dict[bytes, np.ndarray] = {}
    with get_mp_context().Pool(effective_nworkers) as pool:
        for chunk_results in pool.starmap(
            _verify_candidate_chunk,
            [(chunk, net, sign_margin) for chunk in chunks],
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
    sign_margin: float | None = None,
    nworkers: int = 1,
) -> dict[bytes, VertexRecord]:
    """Seed every candidate vertex reachable from ``top_cells``, and verify it.

    Candidate generation (:func:`_generate_vertex_candidates`) is always
    sequential -- it's cheap combinatorics. Verification of each candidate
    (typically :meth:`Polyhedron.verify_vertex_covector`: one float64
    equality solve plus a strict forward-sign check) is what dominates
    runtime on large complexes, since it runs once per candidate. Each
    candidate's check is independent of every other's, so when ``net`` and
    ``sign_margin`` are supplied and there are enough candidates to be worth
    Pool startup cost, verification is farmed out across ``nworkers``
    processes instead of running through ``verify_vertex`` one at a time.
    ``net``/``sign_margin`` must agree with what ``verify_vertex`` itself
    would compute (``get_chain_complex`` constructs both from the same
    complex, so this always holds for its one caller); the parallel and
    sequential paths must otherwise produce identical results.
    """
    pending = _generate_vertex_candidates(top_cells, graph, ambient_dim=ambient_dim, top_dim=top_dim)

    use_parallel = (
        net is not None
        and sign_margin is not None
        and nworkers > 1
        and len(pending) >= PARALLEL_VERIFY_MIN_CANDIDATES
    )

    vertices: dict[bytes, VertexRecord] = {}
    if use_parallel:
        assert net is not None and sign_margin is not None
        verified_points = _verify_candidates_parallel(pending, net=net, sign_margin=sign_margin, nworkers=nworkers)
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
    sign_margin: float | None = None,
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

    ``net``/``sign_margin``/``nworkers`` are forwarded to :func:`find_vertices`
    to parallelize candidate verification on large complexes; see its docstring.
    """
    vertices = find_vertices(
        top_cells,
        graph,
        ambient_dim=ambient_dim,
        top_dim=top_dim,
        verify_vertex=verify_vertex,
        net=net,
        sign_margin=sign_margin,
        nworkers=nworkers,
    )
    cells_by_dim: dict[int, dict[bytes, np.ndarray]] = {k: {} for k in range(top_dim + 1)}
    for vertex in vertices.values():
        for ss in expand_vertex_star(vertex):
            zero_count = int(np.count_nonzero(ss.reshape(-1)[list(vertex.varying_shis)] == 0))
            dim = top_dim - zero_count
            cells_by_dim[dim][encode_ss(ss)] = ss
    return cells_by_dim, vertices
