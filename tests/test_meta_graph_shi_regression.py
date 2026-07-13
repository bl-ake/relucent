"""Regression tests for meta-graph SHI / face-incidence fixes.

Face **edges** in :meth:`~relucent.core.complex.Complex.get_meta_graph` must use every
nonzero sign-sequence entry (:func:`~relucent.graph.meta_graph.ss_nonzero_indices`),
not propagated ``_shis`` lists that can be a strict subset of SS crossings.

Face **construction** in :meth:`~relucent.core.complex.Complex.contract` seeds SHI
candidates from SS crossings and finalizes with
:func:`~relucent.graph.meta_graph.set_contracted_shis` (
:func:`~relucent.graph.meta_graph.cubical_cell_shis`). Using propagated ``_shis`` for
meta-graph face-edge discovery would omit valid faces and break ``∂² = 0``.

Deep uniform MLPs with seeds 2 and 51 exhibited the latter failure under the
incorrect WIP.
"""

from __future__ import annotations

import os

import pytest
import torch

from relucent import Complex, mlp, set_seeds
from relucent.graph import meta_graph as mg
from relucent.search.exploration import explore_for_topology

os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")


@pytest.mark.parametrize(
    ("name", "architecture", "seed", "chain_sizes", "meta_edges", "betti"),
    [
        (
            "deep_2441_seed2",
            [2, 4, 4, 1],
            2,
            [(43, 2), (76, 1), (33, 0)],
            276,
            {0: 1, 1: 1},
        ),
        (
            "deep_3431_seed51",
            [3, 4, 3, 1],
            51,
            [(65, 3), (159, 2), (140, 1), (32, 0)],
            603,
            {0: 1, 1: 16},
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
    explore_for_topology(cplx, start.numpy(), max_polys=10000, nworkers=1)

    chain = cplx.get_chain_complex(verbose=False)
    sizes = [(len(cc), int(cc.index2poly[0].dim)) for cc in chain if len(cc)]
    assert sizes == chain_sizes, f"{name}: unexpected chain sizes {sizes!r}"

    meta = cplx.get_meta_graph(verbose=False)
    mg.verify_meta_graph_one_cells(meta)
    assert meta.number_of_edges() == meta_edges, f"{name}: edge count"

    got = cplx.get_betti_numbers(
        compactify="one_point",
        reduced=False,
        verify_chain_complex=True,
    )
    assert got == betti, f"{name}: Betti mismatch"


@pytest.mark.parametrize(
    ("name", "architecture", "seed", "betti"),
    [
        ("deep_2441_seed2", [2, 4, 4, 1], 2, {0: 1}),
        ("deep_3431_seed51", [3, 4, 3, 1], 51, {0: 1, 1: 1}),
    ],
)
def test_truncated_homology_chain_complex_regression(
    name: str,
    architecture: list[int],
    seed: int,
    betti: dict[int, int],
) -> None:
    """Truncated boundary homology (``compactify=False``) satisfies ``∂² = 0``."""
    set_seeds(seed)
    net = mlp(widths=architecture, add_last_relu=True, init="uniform")
    cplx = Complex(net)
    start = torch.randn(architecture[0], dtype=torch.float64)
    explore_for_topology(cplx, start.numpy(), max_polys=10000, nworkers=1)
    boundary = cplx.get_boundary_complex(cplx.n - 1)

    got = boundary.get_betti_numbers(
        compactify=False,
        reduced=False,
        verify_chain_complex=True,
    )
    assert got == betti, f"{name}: truncated Betti mismatch"
