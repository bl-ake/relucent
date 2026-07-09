"""Betti numbers from topology vs persistent homology must agree at filtration end.

When all cells share one filtration value (:class:`~relucent.filtration.ConstantFiltration`
with ``lower_star=False``), the sublevel complex at the end of the filtration is the
full meta-graph complex. :func:`~relucent.persistence.betti_at_filtration_end` must then
match :meth:`~relucent.complex.Complex.get_betti_numbers` with the same ``compactify`` /
``respect_finite`` flags.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from relucent import Complex, set_seeds
from relucent.exploration import explore_for_topology
from relucent.filtration import ConstantFiltration
from relucent.persistence import betti_at_filtration_end, compute_persistent_homology
from relucent.topology import get_betti_numbers


def _add_points(cplx: Complex, pts: np.ndarray) -> None:
    for x in np.asarray(pts, dtype=np.float64):
        ss = cplx.point2ss(x.reshape(1, -1))
        if (np.asarray(ss) == 0).any():
            continue
        cplx.add_point(x.reshape(1, -1), check_exists=True)


def _betti_dict_to_list(betti: dict[int, int], *, max_dim: int) -> list[int]:
    return [int(betti.get(k, 0)) for k in range(max_dim + 1)]


def _max_hom_dim(cplx: Complex) -> int:
    if len(cplx) == 0:
        return 0
    return max(int(p.dim) for p in cplx)


def assert_betti_match_topology(
    cplx: Complex,
    *,
    compactify: bool = False,
    respect_finite: bool = False,
    filtration: ConstantFiltration | None = None,
) -> None:
    """``betti_at_filtration_end`` must agree with ``get_betti_numbers`` on ``cplx``."""
    fil = filtration or ConstantFiltration(0.0)
    diagram = compute_persistent_homology(
        cplx,
        fil,
        compactify=compactify,
        respect_finite=respect_finite,
        lower_star=False,
    )
    betti_ph = betti_at_filtration_end(diagram)
    betti_topo = cplx.get_betti_numbers(
        compactify=compactify,
        respect_finite=respect_finite,
    )
    max_dim = max(
        _max_hom_dim(cplx),
        max(betti_topo.keys(), default=0),
        max(betti_ph.keys(), default=0),
        max((p.dimension for p in diagram.pairs), default=0),
    )
    list_topo = _betti_dict_to_list(betti_topo, max_dim=max_dim)
    list_ph = _betti_dict_to_list(betti_ph, max_dim=max_dim)
    assert list_ph == list_topo, (
        f"Betti mismatch (compactify={compactify}, respect_finite={respect_finite}): "
        f"persistence {list_ph} vs topology {list_topo}"
    )


def _diamond_boundary_model(radius: float = 1.0) -> nn.Sequential:
    fc0 = nn.Linear(2, 6, bias=False, dtype=torch.float64)
    base = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    fc0.weight.data[:] = base + 1e-3 * torch.randn_like(base)
    fc1 = nn.Linear(6, 2, bias=False, dtype=torch.float64)
    fc1.weight.data[:] = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    fc2 = nn.Linear(2, 1, bias=True, dtype=torch.float64)
    fc2.weight.data[:] = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    fc2.bias.data[:] = torch.tensor([-float(radius)], dtype=torch.float64)
    return nn.Sequential(fc0, nn.ReLU(), fc1, fc2, nn.ReLU())


def _populate_line_boundary(cplx: Complex, *, seed: int) -> Complex:
    rng = np.random.default_rng(seed)
    xs = np.linspace(-2.0, 2.0, 25)
    ys = np.linspace(-2.0, 2.0, 25)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    eps = 1e-2
    left = grid.copy()
    left[:, 0] = -eps
    right = grid.copy()
    right[:, 0] = eps
    _add_points(cplx, np.vstack([left, right, rng.standard_normal((200, 2))]))
    explore_for_topology(cplx, np.array([0.5, 0.0]))
    return cplx.get_boundary_complex(cplx.n - 1)


def _populate_diamond_boundary(cplx: Complex, *, seed: int) -> Complex:
    rng = np.random.default_rng(seed)
    thetas = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    _add_points(cplx, np.vstack([0.9 * dirs, 1.1 * dirs, rng.standard_normal((100, 2))]))
    explore_for_topology(cplx, np.array([0.1, 0.2]))
    return cplx.get_boundary_complex(cplx.n - 1)


@pytest.mark.parametrize("compactify", [False, True])
def test_constant_filtration_matches_betti_on_diamond_boundary(seeded: int, compactify: bool):
    set_seeds(seeded)
    cplx = _populate_diamond_boundary(Complex(_diamond_boundary_model()), seed=seeded)
    assert len(cplx) > 0
    assert_betti_match_topology(cplx, compactify=compactify)


@pytest.mark.parametrize("compactify", [False, True])
def test_constant_filtration_matches_betti_on_line_boundary(seeded: int, compactify: bool):
    set_seeds(seeded)
    fc = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    db = _populate_line_boundary(Complex(nn.Sequential(fc, nn.ReLU())), seed=seeded)
    assert len(db) > 0
    assert_betti_match_topology(db, compactify=compactify)


def test_line_boundary_beta0_matches_components(seeded: int) -> None:
    """Rank β₀ on truncated line boundary agrees with path-component count."""
    set_seeds(seeded)
    fc = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    db = _populate_line_boundary(Complex(nn.Sequential(fc, nn.ReLU())), seed=seeded)
    betti = db.get_betti_numbers(verify_connected_components=True)
    assert betti.get(0, 0) >= 1


def test_line_boundary_truncated_chain_complex_is_consistent(seeded: int) -> None:
    """Truncated line boundary meta-graph satisfies ∂²=0."""
    set_seeds(seeded)
    fc = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    db = _populate_line_boundary(Complex(nn.Sequential(fc, nn.ReLU())), seed=seeded)
    meta = db.get_meta_graph(verbose=False)
    Complex.truncate_meta_graph(meta)
    get_betti_numbers(meta, verify_chain_complex=True)


def test_constant_filtration_matches_betti_on_small_relu_complex(seeded: int):
    set_seeds(seeded)
    model = nn.Sequential(nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 1), nn.ReLU())
    cplx = Complex(model)
    explore_for_topology(cplx, np.array([0.0]))
    assert len(cplx) > 0
    # Sparse 1D sampling does not match compactified truncation Betti; see boundary tests.
    assert_betti_match_topology(cplx, compactify=False)


def test_betti_curve_end_matches_topology(seeded: int):
    """``betti_curve`` at the terminal threshold matches ``betti_at_filtration_end``."""
    set_seeds(seeded)
    cplx = _populate_diamond_boundary(Complex(_diamond_boundary_model()), seed=seeded)
    diagram = compute_persistent_homology(cplx, ConstantFiltration(0.0), lower_star=False)
    t_end = max(diagram.cell_filtration.values()) + 1.0
    from relucent.persistence import betti_curve

    curve = betti_curve(diagram, [t_end])
    end = betti_at_filtration_end(diagram)
    for k, v in end.items():
        assert int(curve[k][0]) == v
