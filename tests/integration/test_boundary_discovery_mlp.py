"""Heavy MLP boundary-discovery smoke and parity tests (opt-in via RELUCENT_RUN_INTEGRATION)."""

from __future__ import annotations

import networkx as nx
import pytest

from relucent import Complex, add_output_relu, set_seeds, torch_mlp

pytestmark = [pytest.mark.integration]


def _mlp_small_model():
    return add_output_relu(torch_mlp(widths=[5, 8, 8, 8, 1]))


def _dual_components(cplx: Complex) -> int:
    dual = cplx.dual_graph(require_complete=False)
    if dual.number_of_nodes() == 0:
        return 0
    return nx.number_connected_components(dual)


def _assert_boundary_parity(ref: Complex, new: Complex) -> None:
    assert {p.tag for p in ref} == {p.tag for p in new}
    assert _dual_components(ref) == _dual_components(new)
    assert ref.betti_numbers() == new.betti_numbers()


def test_discover_boundary_complex_mlp_small(seeded: int, integration_nworkers: int) -> None:
    set_seeds(seeded)
    model = _mlp_small_model()
    shi = Complex(model).n - 1
    try:
        new, stats = Complex(model).discover_boundary_complex(
            shi,
            verbose=False,
            return_stats=True,
            nworkers=integration_nworkers,
        )
    except ValueError as exc:
        if "Initial Solve Failed" in str(exc):
            pytest.skip(f"boundary witness infeasible for calculations.shis at seed {seeded}: {exc}")
        raise
    assert stats["n_components"] >= 1
    assert len(new) > 0
    _ = new.betti_numbers()


def test_discover_boundary_complex_mlp_medium_parity(seeded: int, integration_nworkers: int) -> None:
    set_seeds(seeded)
    model = add_output_relu(torch_mlp(widths=[4, 12, 12, 12, 12, 1]))
    cplx = Complex(model)
    cplx.bfs(verbose=0, nworkers=integration_nworkers)
    shi = cplx.n - 1
    ref = cplx.boundary_complex(shi, verbose=False)
    new, stats = Complex(model).discover_boundary_complex(
        shi,
        verbose=False,
        return_stats=True,
        nworkers=integration_nworkers,
    )
    assert stats["n_components"] >= 1
    _assert_boundary_parity(ref, new)
