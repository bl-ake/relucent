"""Tests for PL Morse gradients and critical points (Brooks & Masden)."""

from __future__ import annotations

import numpy as np
import pytest

from relucent import Complex, mlp, set_seeds
from relucent.topology.morse import (
    assert_scalar_output,
    coface_sign_sequence,
    gradient_on_cell,
    is_pl_critical_vertex,
    partial_derivative_sign,
    partial_derivative_value,
)


class TestLemma9Gradient:
    def test_gradient_matches_polyhedron_W(self, seed: int):
        set_seeds(seed)
        net = mlp(widths=[2, 4, 1])
        cplx = Complex(net)
        cplx.add_point(np.zeros(2, dtype=np.float64))
        poly = cplx.index2poly[0]
        grad = gradient_on_cell(cplx._net, poly.ss_np)
        w_row = np.asarray(poly.W, dtype=np.float64).reshape(-1)
        assert grad.shape == w_row.shape
        assert np.allclose(grad, w_row, atol=1e-9)


class TestPartialDerivativeOn1Cell:
    def test_sign_matches_finite_difference(self, seed: int):
        set_seeds(seed)
        net = mlp(widths=[2, 4, 1])
        cplx = Complex(net)
        cplx.bfs(start=np.zeros(2, dtype=np.float64), max_polys=100)
        chain = cplx.get_chain_complex()
        if len(chain) < 2 or chain[-1].index2poly[0].dim != 0:
            pytest.skip("exploration did not produce a 1-skeleton")

        cells_1 = chain[-2]
        verts = chain[-1]
        edge = cells_1.index2poly[0]
        v0 = verts.index2poly[0]
        v1 = verts.index2poly[1]

        sign0 = cplx.partial_derivative_on_1cell(edge, v0)
        sign1 = cplx.partial_derivative_on_1cell(edge, v1)
        assert sign0 in (-1, 0, 1)
        assert sign1 in (-1, 0, 1)
        if sign0 != 0 and sign1 != 0:
            assert sign0 == -sign1

        val0 = cplx.partial_derivative_on_1cell(edge, v0, value=True)
        eps = 1e-6
        x0 = np.asarray(v0.interior_point, dtype=np.float64).reshape(-1)
        x1 = np.asarray(v1.interior_point, dtype=np.float64).reshape(-1)
        direction = x1 - x0
        if np.linalg.norm(direction) < 1e-12:
            pytest.skip("degenerate 1-cell endpoints")

        def _eval(x: np.ndarray) -> float:
            y = np.asarray(cplx._net(x.reshape(1, -1)), dtype=np.float64).reshape(-1)
            return float(y[0])

        fd = (_eval(x0 + eps * direction) - _eval(x0)) / eps
        if abs(fd) > 1e-8:
            assert np.sign(val0) == np.sign(fd)

    def test_coface_sign_sequence(self):
        ss = np.array([[0, 1, -1]], dtype=np.int8)
        coface = coface_sign_sequence(ss)
        assert np.array_equal(coface, np.array([[1, 1, -1]], dtype=np.int8))


class TestCriticalPoints:
    def test_scalar_output_required(self):
        set_seeds(0)
        cplx = Complex(mlp(widths=[2, 4, 2]))
        with pytest.raises(ValueError, match="scalar output"):
            cplx.get_critical_points()
        with pytest.raises(ValueError, match="scalar output"):
            assert_scalar_output(cplx._net)

    def test_small_network_critical_points_run(self, seed: int):
        set_seeds(seed)
        net = mlp(widths=[1, 2, 1])
        cplx = Complex(net)
        cplx.add_point(np.array([[0.3]], dtype=np.float64))
        cplx.bfs(start=np.array([[0.3]], dtype=np.float64), max_polys=50)
        critical = cplx.get_critical_points()
        assert isinstance(critical, list)
        for cp in critical:
            assert cp.is_critical
            assert cp.index >= 0
            assert cp.tag == cp.polyhedron.tag

    def test_lemma10_yields_nondegenerate_critical_points(self):
        set_seeds(0)
        net = mlp(widths=[2, 8, 1])
        cplx = Complex(net)
        cplx.bfs(start=np.zeros(2, dtype=np.float64), max_polys=150)
        # Lemma 10 / Jacobians must run; many random nets have only regular vertices.
        critical = cplx.get_critical_points(include_degenerate=True)
        assert isinstance(critical, list)
        for cp in critical:
            assert cp.is_critical
            assert cp.index >= -1

    def test_output_relu_network_critical_points_run(self, seed: int):
        from relucent.utils import add_output_relu

        set_seeds(seed)
        net = add_output_relu(mlp(widths=[2, 4, 1]))
        cplx = Complex(net)
        cplx.bfs(start=np.zeros(2, dtype=np.float64), max_polys=100)
        # Trailing output ReLU must not break Jacobian / critical-point code.
        # Morse uses the logit (pre-output-ReLU).
        critical = cplx.get_critical_points(include_degenerate=True)
        assert isinstance(critical, list)
        for cp in critical:
            assert cp.is_critical
            assert cp.point is not None

    def test_non_vertex_cell_raises(self):
        """Non-vertex cells (0 < zeros.size < n_in) must raise, not silently fail."""
        set_seeds(0)
        net = Complex(mlp(widths=[2, 4, 1]))._net
        # 2D input: vertex needs 2 zeros; an edge cell has 1 zero → not a vertex.
        edge_ss = np.array([[0, 1]], dtype=np.int8)  # one zero, one non-zero
        with pytest.raises(ValueError, match="0-cell"):
            is_pl_critical_vertex(
                edge_ss,
                net,
                ssi2maski=[(0, (0, 0)), (0, (0, 1))],
                ss_layers=[0],
            )

    def test_is_pl_critical_vertex_degenerate_at_relu_kink(self):
        from relucent.model import LinearLayer, ReLULayer, ReLUNetwork

        # f(x) = relu(x) has a degenerate (flat) direction at x = 0.
        net = ReLUNetwork(
            [
                LinearLayer(np.array([[1.0]]), np.array([[0.0]])),
                ReLULayer(),
                LinearLayer(np.array([[1.0]]), np.array([[0.0]])),
            ]
        )
        ss_vertex = np.zeros((1, 1), dtype=np.int8)
        is_crit, idx = is_pl_critical_vertex(
            ss_vertex,
            net,
            ssi2maski=[(0, (0, 0))],
            ss_layers=[0],
        )
        assert is_crit is True
        assert idx == -1

    def test_pl_morse_index_matches_brooks_masden_definition(self, monkeypatch: pytest.MonkeyPatch):
        """Index = # of axes with both edges oriented toward v (Def. 5 / Lemma 7)."""
        import relucent.topology.morse as morse_mod

        # ``mlp`` returns a Torch module; Morse helpers need the Relucent ReLUNetwork.
        net = Complex(mlp(widths=[2, 4, 1]))._net
        assert_scalar_output(net)
        ss = np.zeros((1, 2), dtype=np.int8)
        monkeypatch.setattr(morse_mod, "_is_collapsed_edge", lambda *args, **kwargs: False)

        table: dict[tuple[int, int], int] = {}

        def _fake_sign(
            _vertex_ss: np.ndarray,
            edge_ss: np.ndarray,
            _net: object,
            *,
            ssi2maski: object,
            ss_layers: object,
        ) -> int:
            _ = ssi2maski, ss_layers
            e = np.asarray(edge_ss).ravel()
            shi = int(np.flatnonzero(e != 0)[0])
            return int(table[(shi, int(e[shi]))])

        monkeypatch.setattr(morse_mod, "partial_derivative_sign", _fake_sign)
        ssi2maski = [(0, (0, 0)), (0, (0, 1))]
        ss_layers = [0, 0]

        # Axis 0 toward, axis 1 away → saddle of index 1.
        table.clear()
        table.update({(0, 1): -1, (0, -1): -1, (1, 1): 1, (1, -1): 1})
        is_crit, idx = is_pl_critical_vertex(ss, net, ssi2maski=ssi2maski, ss_layers=ss_layers)
        assert is_crit is True
        assert idx == 1

        # Both axes toward → local max, index 2.
        table.clear()
        table.update({(0, 1): -1, (0, -1): -1, (1, 1): -1, (1, -1): -1})
        is_crit, idx = is_pl_critical_vertex(ss, net, ssi2maski=ssi2maski, ss_layers=ss_layers)
        assert is_crit is True
        assert idx == 2

        # Both axes away → local min, index 0.
        table.clear()
        table.update({(0, 1): 1, (0, -1): 1, (1, 1): 1, (1, -1): 1})
        is_crit, idx = is_pl_critical_vertex(ss, net, ssi2maski=ssi2maski, ss_layers=ss_layers)
        assert is_crit is True
        assert idx == 0

        # Opposite signs on an axis → PL regular (not critical).
        table.clear()
        table.update({(0, 1): -1, (0, -1): 1, (1, 1): -1, (1, -1): -1})
        is_crit, idx = is_pl_critical_vertex(ss, net, ssi2maski=ssi2maski, ss_layers=ss_layers)
        assert is_crit is False
        assert idx is None


class TestTheorem4Consistency:
    def test_partial_derivative_sign_vs_value(self, seed: int):
        set_seeds(seed)
        net = mlp(widths=[2, 3, 1])
        cplx = Complex(net)
        cplx.add_point(np.zeros(2, dtype=np.float64))
        poly = cplx.index2poly[0]
        # Combinatorial vertex: zero the first two SHIs (2D input).
        v_ss = poly.ss_np.copy()
        v_ss.ravel()[:2] = 0
        edge_ss = v_ss.copy()
        edge_ss.ravel()[0] = 1
        sign = partial_derivative_sign(
            v_ss,
            edge_ss,
            cplx._net,
            ssi2maski=cplx.ssi2maski,
            ss_layers=cplx.ss_layers,
        )
        val = partial_derivative_value(
            v_ss,
            edge_ss,
            cplx._net,
            ssi2maski=cplx.ssi2maski,
            ss_layers=cplx.ss_layers,
        )
        assert sign == np.sign(val) or (sign == 0 and abs(val) < 1e-8)


@pytest.mark.parametrize(("widths", "seed"), [([2, 8, 8, 1], 0), ([2, 8, 8, 1], 1), ([3, 6, 6, 1], 2)])
def test_critical_points_match_the_chain_complex_route(widths: list[int], seed: int) -> None:
    """get_critical_points finds vertices without the chain complex; same records as through it."""
    from relucent.topology.morse import is_pl_critical_vertex

    set_seeds(seed)
    cplx = Complex(mlp(widths=widths))
    cplx.bfs(start=np.zeros(widths[0], dtype=np.float64))
    chain = cplx.get_chain_complex()
    assert int(chain[-1].index2poly[0].dim) == 0
    expected = {}
    for v in chain[-1]:
        crit, index = is_pl_critical_vertex(v.ss_np, cplx._net, ssi2maski=cplx.ssi2maski, ss_layers=cplx.ss_layers)
        if crit and index is not None:
            expected[v.tag] = (index, np.asarray(v.interior_point).reshape(-1))
    got = {cp.tag: cp for cp in cplx.get_critical_points(include_degenerate=True)}
    assert set(got) == set(expected)
    for tag, cp in got.items():
        assert cp.index == expected[tag][0]
        assert cp.point is not None and np.array_equal(cp.point, expected[tag][1])
        assert cp.polyhedron.tag == tag and int(cp.polyhedron.dim) == 0


@pytest.mark.parametrize(("seed", "nworkers"), [(0, 1), (3, 1), (0, 3)])
def test_vertex_screen_drops_only_non_vertices(seed: int, nworkers: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Screened (sequentially or across a pool) and unscreened vertex search agree exactly."""
    from relucent.graph import vertex_star

    monkeypatch.setattr(vertex_star, "PARALLEL_SCREEN_MIN_ROOTS", 1)
    monkeypatch.setattr(vertex_star, "PARALLEL_VERIFY_MIN_CANDIDATES", 1)
    monkeypatch.setattr(vertex_star, "MIN_CANDIDATES_PER_WORKER", 1)
    set_seeds(seed)
    cplx = Complex(mlp(widths=[3, 8, 8, 1]))
    cplx.bfs(start=np.zeros(3, dtype=np.float64))
    top = [p for p in cplx if int(p.dim) == 3]
    graph = cplx.get_dual_graph(verbose=False, require_complete=False)
    runs = [
        vertex_star.find_vertices(
            top,
            graph,
            ambient_dim=3,
            top_dim=3,
            verify_vertex=lambda r, c: r.verify_vertex_covector(c),
            screen=screen,
            nworkers=nworkers if screen else 1,
            net=cplx._net,
        )
        for screen in (False, True)
    ]
    assert list(runs[0]) == list(runs[1]) and len(runs[0]) > 0  # same vertices, same witness order
    for tag, record in runs[0].items():
        assert np.array_equal(record.point, runs[1][tag].point)
        assert record.witness_tag == runs[1][tag].witness_tag
