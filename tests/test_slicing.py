"""Tests for relucent.geometry.slicing (Complex.slice_affine)."""

from __future__ import annotations

import numpy as np

from relucent import Complex, set_seeds, torch_mlp


def test_slice_affine_covers_plane_with_parent_sign_sequences() -> None:
    """Every point of a 2-D slice of a 3-D complex lies in the sliced cell of its parent's sign sequence."""
    set_seeds(1)
    cplx = Complex(torch_mlp([3, 6, 1]))
    cplx.bfs(start=np.zeros((1, 3)) + 0.05, nworkers=2, verbose=0)

    x0 = np.array([0.1, -0.2, 0.3])
    V = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    sliced = cplx.slice_affine(x0, V)
    assert sliced.dim == 2
    assert 0 < len(sliced) <= len(cplx)

    sliced_tags = {p.tag for p in sliced}
    rng = np.random.default_rng(0)
    for t in rng.uniform(-3, 3, size=(200, 2)):
        x = (x0 + V @ t).reshape(1, -1)
        assert cplx.point2poly(x).tag in sliced_tags
        halfspaces = sliced[cplx.point2ss(x)].halfspaces_np
        assert np.all(halfspaces[:, :-1] @ t + halfspaces[:, -1] <= 1e-9)
