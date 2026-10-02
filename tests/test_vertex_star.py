from __future__ import annotations

from collections.abc import Iterable, Mapping

import networkx as nx
import numpy as np
import pytest

from relucent import Complex
from relucent.core.poly import Polyhedron
from relucent.core.ss import encode_ss
from relucent.graph.vertex_star import expand_vertex_star, find_vertices, recover_cells_from_vertices
from relucent.model.builders import set_seeds, torch_mlp
from relucent.search.exploration import explore_for_topology


def _square_tope_graph() -> tuple[list[Polyhedron], nx.Graph[Polyhedron]]:
    cells = [Polyhedron(None, np.array([[a, b, 1]], dtype=np.int8), ambient_dim=2) for a in (-1, 1) for b in (-1, 1)]
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


def _accept_all(_root: Polyhedron, _candidate: np.ndarray) -> np.ndarray | None:
    return np.zeros(2)


def test_find_vertices_recovers_the_shared_square_vertex() -> None:
    cells, graph = _square_tope_graph()
    vertices = find_vertices(cells, graph, ambient_dim=2, top_dim=2, verify_vertex=_accept_all)

    assert len(vertices) == 1
    vertex = next(iter(vertices.values()))
    assert np.array_equal(vertex.ss.ravel(), np.array([0, 0, 1], dtype=np.int8))
    assert vertex.varying_shis == (0, 1)
    assert vertex.witness_tag in {c.tag for c in cells}


def test_find_vertices_rejects_candidates_verify_vertex_rejects() -> None:
    cells, graph = _square_tope_graph()
    vertices = find_vertices(cells, graph, ambient_dim=2, top_dim=2, verify_vertex=lambda _r, _c: None)
    assert vertices == {}


def test_expand_vertex_star_covers_every_dimension_up_to_top_dim() -> None:
    cells, _graph = _square_tope_graph()
    vertex_ss = np.array([[0, 0, 1]], dtype=np.int8)
    from relucent.graph.vertex_star import VertexRecord

    vertex = VertexRecord(ss=vertex_ss, point=np.zeros(2), witness_tag=cells[0].tag, varying_shis=(0, 1))
    by_zero_count: dict[int, set[bytes]] = {0: set(), 1: set(), 2: set()}
    for ss in expand_vertex_star(vertex):
        flat = ss.reshape(-1)
        zero_count = int(np.count_nonzero(flat[[0, 1]] == 0))
        by_zero_count[zero_count].add(encode_ss(ss))

    assert len(by_zero_count[2]) == 1  # the vertex itself
    assert len(by_zero_count[1]) == 4  # the four edges out of it
    assert len(by_zero_count[0]) == 4  # the four surrounding 2-cells


def test_recover_cells_from_vertices_recovers_square_face_lattice() -> None:
    cells, graph = _square_tope_graph()
    cells_by_dim, vertices = recover_cells_from_vertices(cells, graph, ambient_dim=2, top_dim=2, verify_vertex=_accept_all)

    assert {dim: len(found) for dim, found in cells_by_dim.items()} == {0: 1, 1: 4, 2: 4}
    assert len(vertices) == 1


def test_default_chain_and_meta_graph_do_not_call_lp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_seeds(4)
    cplx = Complex(torch_mlp(widths=[2, 3, 1], add_last_relu=True, init="uniform"))
    explore_for_topology(cplx, np.zeros(2), max_polys=1000, nworkers=1)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("topology builder called an LP routine")

    monkeypatch.setattr("relucent.geometry.calculations.get_shis", fail)
    monkeypatch.setattr("relucent.core.poly.get_shis", fail)
    monkeypatch.setattr("relucent.verify.certify.verify_lp_flip_neighbors_in_complex", fail)
    # Chebyshev may run for zero-face 1-cells in geometric_infeasible_one_cells; SHI LPs must not.

    chain = cplx.chain_complex()
    meta = cplx.meta_graph()
    assert chain
    assert meta.number_of_nodes() > 0


def test_get_meta_graph_unions_chebyshev_phantom_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty vertex-star-infeasible set must not disable geometric_infeasible_one_cells."""
    from relucent.graph import incidence

    set_seeds(4)
    cplx = Complex(torch_mlp(widths=[2, 3, 1], add_last_relu=True, init="uniform"))
    explore_for_topology(cplx, np.zeros(2), max_polys=1000, nworkers=1)
    cplx.chain_complex()

    calls: list[int] = []
    real = incidence.geometric_infeasible_one_cells

    def wrapped(
        by_dim: Mapping[int, Iterable[Polyhedron]],
        edges_by_dim: dict[int, tuple[list[tuple[bytes, bytes, int]], list[bytes]]],
    ) -> set[bytes]:
        calls.append(1)
        return real(by_dim, edges_by_dim)

    monkeypatch.setattr(incidence, "geometric_infeasible_one_cells", wrapped)
    meta = cplx.meta_graph(verbose=False)
    assert calls, "meta_graph must still run Chebyshev phantom scan"
    assert meta.number_of_nodes() > 0


def _data_cell(rows: list[list[float]], ss: list[int]) -> Polyhedron:
    hs = np.array(rows, dtype=np.float64)
    return Polyhedron(None, np.array([ss], dtype=np.int8), halfspaces=hs, ambient_dim=2)


def test_verify_vertex_covector_decides_from_the_cell_rows() -> None:
    """The vertex is kept iff every other row of the cell is strictly negative there."""
    inside = _data_cell([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, -1.0, -1.0]], [1, 1, 1])
    point = inside.verify_vertex_covector(np.array([[0, 0, 1]], dtype=np.int8))
    assert point is not None
    assert np.array_equal(point, np.zeros(2))

    outside = _data_cell([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, -1.0, 1.0]], [1, 1, 1])
    assert outside.verify_vertex_covector(np.array([[0, 0, 1]], dtype=np.int8)) is None


def test_verify_vertex_covector_skips_dead_units() -> None:
    """A dead unit's row (exactly zero normal) is a constant, not a hyperplane: it takes no part."""
    cell = _data_cell([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0], [-1.0, -1.0, -1.0]], [1, 1, 1, 1])
    point = cell.verify_vertex_covector(np.array([[0, 0, 1, 1]], dtype=np.int8))
    assert point is not None
    # A tiny but nonzero normal is a real (far-away) hyperplane and is judged like any other row.
    tiny = _data_cell([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1e-15, -1e-15, 2e-15], [-1.0, -1.0, -1.0]], [1, 1, 1, 1])
    assert tiny.verify_vertex_covector(np.array([[0, 0, 1, 1]], dtype=np.int8)) is None


def test_verify_vertex_covector_uses_exact_arithmetic_inside_rounding() -> None:
    """Far out, a row's float64 value is inside its rounding error: the exact rows decide."""
    ulp = float(np.spacing(2e12))
    # Vertex at (1e12, 1e12); the third row is exactly -ulp there, far below float64's resolution
    # of that evaluation, so only exact arithmetic can keep the vertex.
    kept = _data_cell([[1.0, 0.0, -1e12], [0.0, 1.0, -1e12], [-1.0, -1.0, 2e12 - ulp]], [1, 1, 1])
    point = kept.verify_vertex_covector(np.array([[0, 0, 1]], dtype=np.int8))
    assert point is not None
    np.testing.assert_array_equal(point, np.array([1e12, 1e12]))

    # Exactly concurrent hyperplanes: a non-generic arrangement, reported as such.
    from relucent import NonGenericArrangementError

    concurrent = _data_cell([[1.0, 0.0, -1e12], [0.0, 1.0, -1e12], [-1.0, -1.0, 2e12]], [1, 1, 1])
    with pytest.raises(NonGenericArrangementError):
        concurrent.verify_vertex_covector(np.array([[0, 0, 1]], dtype=np.int8))
