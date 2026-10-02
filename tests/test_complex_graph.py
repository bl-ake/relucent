"""Tests for relucent.graph.complex_graph (neuron deletion on an explored complex)."""

from __future__ import annotations

import numpy as np
import pytest

from relucent import Complex, mlp, set_seeds
from relucent.graph.complex_graph import without_last_layer_neuron


@pytest.mark.parametrize("neuron_idx", [0, 2])
def test_without_last_layer_neuron_matches_fresh_search(neuron_idx: int) -> None:
    """Deleting a neuron from an explored complex gives the cells a fresh search of the smaller net finds."""
    set_seeds(3)
    cplx = Complex(mlp([2, 5, 4, 1]))
    start = np.zeros((1, 2)) + 0.1
    cplx.bfs(start=start, nworkers=2, verbose=0)

    smaller = without_last_layer_neuron(cplx, neuron_idx)
    fresh = Complex(smaller._net)
    fresh.bfs(start=start, nworkers=2, verbose=0)

    assert len(smaller) < len(cplx)
    assert {p.tag for p in smaller} == {p.tag for p in fresh}
    assert smaller.verified is True
