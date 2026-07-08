"""
A depth-1 ReLU network (one hidden layer, no output ReLU) induces bent hyperplanes in
input space that coincide with an ordinary hyperplane arrangement when weights are
generic. Buck's partition theorem gives the number of top-dimensional cells
(linear regions) exactly as ``sum_{i=0}^{d} binom(n, i)`` for ``n`` hyperplanes in
``R^d`` (see e.g. Buck 1943).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from relucent import Complex, mlp, set_seeds


def _generic_hyperplane_arrangement_num_d_cells(*, ambient_dim: int, num_hyperplanes: int) -> int:
    """Number of ``ambient_dim``-cells in a generic arrangement of ``num_hyperplanes`` in ``R^ambient_dim``."""
    return sum(math.comb(num_hyperplanes, i) for i in range(ambient_dim + 1))


@pytest.mark.parametrize(
    ("ambient_dim", "num_hidden"),
    [
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 7),
        (4, 6),
    ],
)
def test_single_hidden_layer_bfs_region_count_matches_arrangement_formula(
    seeded: int,
    ambient_dim: int,
    num_hidden: int,
) -> None:
    """Exhaustive BFS on a shallow generic MLP discovers every linear region predicted by Buck's formula."""
    set_seeds(seeded)
    expected = _generic_hyperplane_arrangement_num_d_cells(
        ambient_dim=ambient_dim,
        num_hyperplanes=num_hidden,
    )
    model = mlp(widths=[ambient_dim, num_hidden, 1], add_last_relu=False)
    cplx = Complex(model)
    assert cplx.n == num_hidden

    start = torch.rand(ambient_dim, dtype=torch.float64).numpy()
    ss = cplx.point2ss(start.reshape(1, -1))
    assert not (np.asarray(ss) == 0).any(), "start must lie in a full-dimensional region"

    cplx.bfs(
        start=start,
        max_polys=max(expected + 1, 5000),
        verbose=False,
    )
    assert len(cplx) == expected
