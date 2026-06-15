"""Regression tests for meta-graph SHI / face-incidence fixes.

Face **edges** in :meth:`~relucent.complex.Complex.get_meta_graph` must use every
nonzero sign-sequence entry (:func:`~relucent.meta_graph.ss_face_crossing_indices`),
not propagated ``_shis`` lists that coface intersection can shrink.

Face **construction** in :meth:`~relucent.complex.Complex.contract` must keep
coface-intersected ``shis`` in kwargs plus flip-neighbor filtering
(:func:`~relucent.meta_graph.filter_complex_shis_by_flip_neighbor`): assigning
full SS flip-neighbor membership to ``_shis`` before further contractions would
add spurious dual-graph edges and break ``∂² = 0``.

Deep uniform MLPs with seeds 2 and 51 exhibited the latter failure under the
incorrect WIP.
"""

from __future__ import annotations

import os

import pytest
import torch

from relucent import Complex, mlp, set_seeds
from tests.conftest import explore_for_topology

os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")


@pytest.mark.parametrize(
    ("name", "architecture", "seed", "chain_sizes", "meta_edges", "betti"),
    [
        (
            "deep_2441_seed2",
            [2, 4, 4, 1],
            2,
            [(43, 2), (75, 1), (33, 0)],
            282,
            {0: 1, 2: 1},
        ),
        (
            "deep_3431_seed51",
            [3, 4, 3, 1],
            51,
            [(65, 3), (151, 2), (119, 1), (32, 0)],
            970,
            {0: 1, 3: 1},
        ),
    ],
)
def test_meta_graph_chain_complex_regression(
    name: str,
    architecture: list[int],
    seed: int,
    chain_sizes: list[tuple[int, int]],
    meta_edges: int,
    betti: dict[int, int],
) -> None:
    """``∂² = 0`` and stable chain / meta-graph statistics on deep-batch witnesses."""
    set_seeds(seed)
    net = mlp(widths=architecture, add_last_relu=True, init="uniform")
    cplx = Complex(net)
    start = torch.randn(architecture[0], dtype=torch.float64)
    explore_for_topology(cplx, start.numpy(), max_polys=10000)

    chain = cplx.get_chain_complex(verbose=False)
    sizes = [(len(cc), int(cc.index2poly[0].dim)) for cc in chain if len(cc)]
    assert sizes == chain_sizes, f"{name}: unexpected chain sizes {sizes!r}"

    meta = cplx.get_meta_graph(verbose=False)
    assert meta.number_of_edges() == meta_edges, f"{name}: edge count"

    got = cplx.get_betti_numbers(
        compactify="one_point",
        reduced=False,
        verify_chain_complex=True,
    )
    assert got == betti, f"{name}: Betti mismatch"
