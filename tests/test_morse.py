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
