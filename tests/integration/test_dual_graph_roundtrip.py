"""Dual-graph recovery round-trip preserves boundary homology."""

from __future__ import annotations

import pytest

from relucent.exploration import finalize_ambient_search
from relucent.verify import verify_shi_flip_symmetry
from tests.integration.helpers import (
    boundary_shi_for_spec,
    export_dual_graph_payload,
    load_witness_model,
    recover_from_dual_payload,
    run_bfs_ambient,
    truncated_betti,
    witness_by_id,
    write_failure_artifacts,
)

pytestmark = [pytest.mark.integration]


def test_dual_graph_recovery_matches_direct_bfs(
    integration_nworkers: int,
    integration_outdir: str,
) -> None:
    spec = witness_by_id("dual_recovery_4392")
    model = load_witness_model(spec)

    direct = run_bfs_ambient(model, spec, nworkers=integration_nworkers, verify=True)
    shi = boundary_shi_for_spec(direct, spec)
    direct_boundary = direct.get_boundary_complex(shi, verbose=False)
    direct_betti = truncated_betti(direct_boundary)

    payload = export_dual_graph_payload(direct)
    recovered = recover_from_dual_payload(model, payload)
    finalize_ambient_search(recovered, complete=True, verify=True)
    verify_shi_flip_symmetry(recovered)

    recovered_boundary = recovered.get_boundary_complex(shi, verbose=False)
    recovered_betti = truncated_betti(recovered_boundary)

    if direct_betti != recovered_betti:
        write_failure_artifacts(
            integration_outdir,
            spec.id,
            betti_full=direct_betti,
            betti_discover=recovered_betti,
            extra={"n_direct": len(direct), "n_recovered": len(recovered)},
        )
    assert direct_betti == recovered_betti, (
        f"dual-graph recovery Betti mismatch: direct={direct_betti} recovered={recovered_betti}"
    )
