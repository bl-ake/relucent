from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import networkx as nx
import numpy as np
import pytest

from relucent import Complex
from relucent.core.errors import CubicalAmbiguityError
from relucent.core.poly import Polyhedron
from relucent.graph.covectors import enumerate_covectors, sign_intersection
from relucent.search.exploration import explore_for_topology
from relucent.utils import mlp, set_seeds


def _square_tope_graph() -> tuple[list[Polyhedron], nx.Graph[Polyhedron]]:
    cells = [Polyhedron(None, np.array([[a, b, 1]], dtype=np.int8), dim=2, _ambient_dim=2) for a in (-1, 1) for b in (-1, 1)]
    by_sign = {tuple(p.ss_np.ravel()): p for p in cells}
    graph: nx.Graph[Polyhedron] = nx.Graph()
    graph.add_nodes_from(cells)
    for signs, cell in by_sign.items():
        for shi in (0, 1):
            flipped = list(signs)
            flipped[shi] *= -1
            neighbor = by_sign[tuple(flipped)]
            graph.add_edge(cell, neighbor, shi=shi)
    return cells, graph


def test_sign_intersection_retains_constants_and_zeros_variation() -> None:
    rows = np.array(
        [
            [-1, -1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )
    assert np.array_equal(sign_intersection(rows), np.array([[0, 0, 1]], dtype=np.int8))


def test_enumerate_covectors_recovers_square_face_lattice() -> None:
    cells, graph = _square_tope_graph()
    by_dim = enumerate_covectors(cells, graph, ambient_dim=2)

    assert {dim: len(found) for dim, found in by_dim.items()} == {0: 1, 1: 4, 2: 4}
    vertex = next(iter(by_dim[0].values()))
    assert vertex.zero_shis == (0, 1)
    assert len(vertex.coface_tags) == 4

    edge = next(iter(by_dim[1].values()))
    vertex_poly = Polyhedron(None, vertex.ss, finite=True)
    edge_poly = Polyhedron(None, edge.ss, finite=True)
    composed = np.asarray((vertex_poly * edge_poly).ss)
    assert bool(np.all(composed == np.asarray(edge_poly.ss)))


def test_enumerate_covectors_preserves_fixed_slice_zero() -> None:
    cells = [
        Polyhedron(None, np.array([[0, a, b, 1]], dtype=np.int8), dim=2, _ambient_dim=3) for a in (-1, 1) for b in (-1, 1)
    ]
    graph: nx.Graph[Polyhedron] = nx.Graph()
    graph.add_nodes_from(cells)
    by_sign = {tuple(p.ss_np.ravel()): p for p in cells}
    for signs, cell in by_sign.items():
        for shi in (1, 2):
            flipped = list(signs)
            flipped[shi] *= -1
            graph.add_edge(cell, by_sign[tuple(flipped)], shi=shi)
    by_dim = enumerate_covectors(cells, graph, ambient_dim=3, top_dim=2)

    vertex = next(iter(by_dim[0].values()))
    assert vertex.zero_shis == (0, 1, 2)


def test_enumerate_covectors_rejects_duplicate_edge_label() -> None:
    cells, graph = _square_tope_graph()
    root = cells[0]
    other_neighbor = next(p for p in cells if p is not root and not graph.has_edge(root, p))
    graph.add_edge(root, other_neighbor, shi=0)

    with pytest.raises(CubicalAmbiguityError, match="multiple incident edges"):
        enumerate_covectors(cells, graph, ambient_dim=2)


def test_verify_vertex_covector_uses_only_nonzero_sign_margin() -> None:
    halfspaces = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0],
        ]
    )
    coface = Polyhedron(
        None,
        np.array([[1, 1, 1]], dtype=np.int8),
        halfspaces=halfspaces,
        dim=2,
        _ambient_dim=2,
    )
    vertex_ss = np.array([[0, 0, 1]], dtype=np.int8)

    point = coface.verify_vertex_covector(
        vertex_ss,
        point2preactivations=lambda _x: np.array([[0.0, 0.0, 2.0]]),
        sign_margin=1e-7,
    )
    assert point is not None
    assert np.array_equal(point, np.zeros(2))

    rejected = coface.verify_vertex_covector(
        vertex_ss,
        point2preactivations=lambda _x: np.array([[0.0, 0.0, -2.0]]),
        sign_margin=1e-7,
    )
    assert rejected is None


def test_verify_vertex_covector_skips_dead_relu_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spurious ±1 on a vanishing normal must not reject a genuine vertex."""
    import relucent.config as cfg

    monkeypatch.setattr(cfg, "TOL_DEAD_RELU", 1e-8)
    halfspaces = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1e-15, -1e-15, 2e-15],  # dead / vanishing hyperplane
            [-1.0, -1.0, -1.0],
        ]
    )
    coface = Polyhedron(
        None,
        np.array([[1, 1, -1, 1]], dtype=np.int8),
        halfspaces=halfspaces,
        dim=2,
        _ambient_dim=2,
    )
    vertex_ss = np.array([[0, 0, -1, 1]], dtype=np.int8)

    # Dead coord preactivation is below sign_margin; live coords match.
    point = coface.verify_vertex_covector(
        vertex_ss,
        point2preactivations=lambda _x: np.array([[0.0, 0.0, 1e-14, 2.0]]),
        sign_margin=1e-7,
    )
    assert point is not None
    assert np.array_equal(point, np.zeros(2))

    # A live hyperplane with the wrong tiny preactivation is still virtual → reject.
    rejected = coface.verify_vertex_covector(
        vertex_ss,
        point2preactivations=lambda _x: np.array([[0.0, 0.0, 1e-14, 1e-14]]),
        sign_margin=1e-7,
    )
    assert rejected is None


def test_default_chain_and_meta_graph_do_not_call_lp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_seeds(4)
    cplx = Complex(mlp(widths=[2, 3, 1], add_last_relu=True, init="uniform"))
    explore_for_topology(cplx, np.zeros(2), max_polys=1000, nworkers=1)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("topology builder called an LP routine")

    monkeypatch.setattr("relucent.geometry.calculations.get_shis", fail)
    monkeypatch.setattr("relucent.core.poly.get_shis", fail)
    monkeypatch.setattr("relucent.verify.certify.verify_lp_flip_neighbors_in_complex", fail)
    # Chebyshev may run for zero-face 1-cells in geometric_infeasible_one_cells; SHI LPs must not.

    chain = cplx.get_chain_complex()
    meta = cplx.get_meta_graph()
    assert chain
    assert meta.number_of_nodes() > 0


def test_get_chain_complex_skips_1cells_with_only_rejected_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Virtual vertices must not leave poisoned 1-cells that wipe nonempty cofaces."""
    from relucent.graph import covectors, incidence
    from relucent.utils import encode_ss

    set_seeds(4)
    cplx = Complex(mlp(widths=[2, 4, 1], add_last_relu=True, init="uniform"))
    explore_for_topology(cplx, np.zeros(2), max_polys=2000, nworkers=1)

    top_dim = max(int(p.dim) for p in cplx)
    top_cells = [p for p in cplx if int(p.dim) == top_dim]
    graph = cplx.get_dual_graph(verbose=False, require_complete=False)
    cells_by_dim = covectors.enumerate_covectors(top_cells, graph, ambient_dim=int(cplx.dim), top_dim=top_dim)
    assert cells_by_dim.get(0), "expected at least one combinatorial vertex"
    reject_tag = next(iter(cells_by_dim[0]))

    dim0_tags = set(cells_by_dim.get(0, {}))
    expected_skip: set[bytes] = set()
    for tag, cell in cells_by_dim.get(1, {}).items():
        candidates = {incidence.face_tag(cell.ss, shi) for shi in incidence.ss_nonzero_indices(cell.ss)}
        in_dim0 = candidates & dim0_tags
        if in_dim0 and in_dim0 <= {reject_tag}:
            expected_skip.add(tag)
    assert expected_skip, "fixture must yield a 1-cell whose only dim-0 endpoints are rejected"

    original = Polyhedron.verify_vertex_covector

    def patched(
        self: Polyhedron,
        vertex_ss: np.ndarray,
        *,
        point2preactivations: Callable[[np.ndarray], np.ndarray],
        sign_margin: float,
    ) -> np.ndarray | None:
        tag = encode_ss(np.asarray(vertex_ss, dtype=np.int8))
        if tag == reject_tag:
            return None
        return original(
            self,
            vertex_ss,
            point2preactivations=point2preactivations,
            sign_margin=sign_margin,
        )

    monkeypatch.setattr(Polyhedron, "verify_vertex_covector", patched)

    chain = cplx.get_chain_complex(verbose=False)
    one_cplx = next((c for c in chain if len(c) and int(c.index2poly[0].dim) == 1), None)
    present_one = {p.tag for p in one_cplx} if one_cplx is not None else set()
    assert expected_skip.isdisjoint(present_one)

    zero_cplx = next((c for c in chain if len(c) and int(c.index2poly[0].dim) == 0), None)
    if zero_cplx is not None:
        assert reject_tag not in {p.tag for p in zero_cplx}

    meta = cplx.get_meta_graph(verbose=False)
    for p in top_cells:
        assert p.tag in meta.nodes, "nonempty top cells must survive virtual-edge omission"


def test_get_meta_graph_unions_chebyshev_phantom_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty covector-infeasible set must not disable geometric_infeasible_one_cells."""
    from relucent.graph import incidence

    set_seeds(4)
    cplx = Complex(mlp(widths=[2, 3, 1], add_last_relu=True, init="uniform"))
    explore_for_topology(cplx, np.zeros(2), max_polys=1000, nworkers=1)
    cplx.get_chain_complex()

    calls: list[int] = []
    real = incidence.geometric_infeasible_one_cells

    def wrapped(
        by_dim: Mapping[int, Iterable[Polyhedron]],
        edges_by_dim: dict[int, tuple[list[tuple[bytes, bytes, int]], list[bytes]]],
    ) -> set[bytes]:
        calls.append(1)
        return real(by_dim, edges_by_dim)

    monkeypatch.setattr(incidence, "geometric_infeasible_one_cells", wrapped)
    meta = cplx.get_meta_graph(verbose=False)
    assert calls, "get_meta_graph must still run Chebyshev phantom scan"
    assert meta.number_of_nodes() > 0
