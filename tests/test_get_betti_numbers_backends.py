"""``get_betti_numbers`` must agree when GF(2) rank uses the C vs Python backends.

The C path is selected inside :func:`~relucent.topology.gf2_rank_boundary` when
``topology._c_backend`` is true (see :data:`~relucent.topology.C_BACKEND_AVAILABLE`).
These tests force each backend and compare full Betti dictionaries on small complexes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

import relucent.topology as topology
from relucent import Complex, set_seeds
from relucent.topology import C_BACKEND_AVAILABLE, get_betti_numbers


def _set_gf2_backend(monkeypatch: pytest.MonkeyPatch, *, use_c: bool) -> None:
    if use_c and not C_BACKEND_AVAILABLE:
        pytest.skip("C GF(2) backend not available (gcc compile/load failed)")
    monkeypatch.setattr(topology, "_c_backend", use_c)


def _betti_for_backend(
    monkeypatch: pytest.MonkeyPatch,
    cplx: Complex,
    *,
    use_c: bool,
    **kwargs: Any,
) -> dict[int, int]:
    _set_gf2_backend(monkeypatch, use_c=use_c)
    return get_betti_numbers(cplx, **kwargs)


def _add_points(cplx: Complex, pts: np.ndarray) -> None:
    for x in np.asarray(pts, dtype=np.float64):
        ss = cplx.point2ss(x.reshape(1, -1))
        if (np.asarray(ss) == 0).any():
            continue
        cplx.add_point(x.reshape(1, -1), check_exists=True)


def _diamond_boundary_model(radius: float = 1.0) -> nn.Sequential:
    fc0 = nn.Linear(2, 6, bias=False, dtype=torch.float64)
    fc0.weight.data[:] = torch.tensor(
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


def _populate_diamond_boundary(seed: int) -> Complex:
    rng = np.random.default_rng(seed)
    cplx = Complex(_diamond_boundary_model())
    thetas = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    _add_points(cplx, np.vstack([0.9 * dirs, 1.1 * dirs, rng.standard_normal((80, 2))]))
    cplx._dual_graph = cplx.get_dual_graph(auto_add=True, verbose=False)
    return cplx.get_boundary_complex(cplx.n - 1)


def _populate_line_boundary(seed: int) -> Complex:
    rng = np.random.default_rng(seed)
    fc = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    cplx = Complex(nn.Sequential(fc, nn.ReLU()))
    xs = np.linspace(-2.0, 2.0, 21)
    ys = np.linspace(-2.0, 2.0, 21)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    eps = 1e-2
    left = grid.copy()
    left[:, 0] = -eps
    right = grid.copy()
    right[:, 0] = eps
    _add_points(cplx, np.vstack([left, right, rng.standard_normal((80, 2))]))
    cplx._dual_graph = cplx.get_dual_graph(auto_add=True, verbose=False)
    return cplx.get_boundary_complex(cplx.n - 1)


def _populate_small_1d_complex(seed: int) -> Complex:
    set_seeds(seed)
    model = nn.Sequential(nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 1), nn.ReLU())
    cplx = Complex(model)
    for x in np.linspace(-2.0, 2.0, 11):
        cplx.add_point(np.array([[x]], dtype=np.float64), check_exists=True)
    return cplx


@pytest.mark.python_gf2
def test_get_betti_numbers_python_backend_smoke(seeded: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pure-Python GF(2) rank path runs end-to-end (no C required)."""
    db = _populate_diamond_boundary(seeded)
    betti = _betti_for_backend(monkeypatch, db, use_c=False)
    assert isinstance(betti, dict)
    assert int(betti.get(1, 0)) >= 1


@pytest.mark.requires_c_gf2
def test_c_gf2_backend_available() -> None:
    """CI on Linux must compile and load ``_gf2_rank.c`` (see workflow job ``gf2-backend``)."""
    assert C_BACKEND_AVAILABLE, "C GF(2) backend failed to compile or load; get_betti_numbers would use slow Python rank only"


@pytest.mark.requires_c_gf2
@pytest.mark.parametrize(
    "build_cplx,kwargs",
    [
        (_populate_diamond_boundary, {}),
        (_populate_diamond_boundary, {"compactify": True, "reduced": True}),
        (_populate_diamond_boundary, {"verify_chain_complex": True}),
        (_populate_line_boundary, {}),
        (_populate_line_boundary, {"compactify": True}),
        (_populate_line_boundary, {"respect_finite": True}),
        (_populate_small_1d_complex, {}),
        (_populate_small_1d_complex, {"compactify": True}),
    ],
)
def test_get_betti_numbers_c_matches_python(
    seeded: int,
    monkeypatch: pytest.MonkeyPatch,
    build_cplx: Callable[[int], Complex],
    kwargs: dict[str, Any],
) -> None:
    """C and Python ``gf2_rank_boundary`` backends yield identical Betti numbers."""
    cplx = build_cplx(seeded)
    assert len(cplx) > 0
    betti_c = _betti_for_backend(monkeypatch, cplx, use_c=True, **kwargs)
    betti_py = _betti_for_backend(monkeypatch, cplx, use_c=False, **kwargs)
    assert betti_c == betti_py, f"C {betti_c} != Python {betti_py} (kwargs={kwargs})"


def test_complex_get_betti_numbers_delegates_to_topology(seeded: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Public :meth:`~relucent.complex.Complex.get_betti_numbers`` matches topology module."""
    db = _populate_diamond_boundary(seeded)
    _set_gf2_backend(monkeypatch, use_c=C_BACKEND_AVAILABLE)
    via_topology = get_betti_numbers(db, compactify=False)
    via_complex = db.get_betti_numbers(compactify=False)
    assert via_topology == via_complex
