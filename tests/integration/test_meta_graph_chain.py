"""Meta-graph chain-complex verification on a large ambient witness."""

from __future__ import annotations

import pytest

from tests.integration.helpers import (
    boundary_shi_for_spec,
    load_witness_model,
    run_bfs_ambient,
    witness_by_id,
    write_failure_artifacts,
)

pytestmark = [pytest.mark.integration]


def test_meta_graph_chain_complex_non_negative_betti(
    integration_nworkers: int,
    integration_outdir: str,
) -> None:
    spec = witness_by_id("meta_graph_large")
    model = load_witness_model(spec)
    ambient = run_bfs_ambient(model, spec, nworkers=integration_nworkers, verify=True)
    shi = boundary_shi_for_spec(ambient, spec)
    boundary = ambient.get_boundary_complex(shi, verbose=False)

    betti = boundary.get_betti_numbers(
        compactify=False,
        verify_chain_complex=True,
        verbose=False,
    )
    betti_int = {int(k): int(v) for k, v in betti.items()}

    negatives = {k: v for k, v in betti_int.items() if v < 0}
    if negatives:
        write_failure_artifacts(
            integration_outdir,
            spec.id,
            extra={"betti": betti_int, "n_regions": len(ambient), "n_boundary": len(boundary)},
        )
    assert not negatives, f"negative Betti on {spec.id}: {negatives}"

    if spec.expected_truncated_betti is not None:
        assert betti_int == spec.expected_truncated_betti
