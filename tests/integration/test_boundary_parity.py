"""Strict parity between full BFS + get_boundary_complex vs discover_boundary_complex."""

from __future__ import annotations

import pytest

from relucent import Complex
from tests.integration.helpers import (
    boundary_shi_for_spec,
    load_manifest,
    load_witness_model,
    run_bfs_ambient,
    tag_set,
    truncated_betti,
    write_failure_artifacts,
)

pytestmark = [pytest.mark.integration]


_PARITY_SKIP = frozenset({"dual_recovery_4392", "meta_graph_large"})


def _parity_witnesses():
    return [w for w in load_manifest() if w.id not in _PARITY_SKIP]


@pytest.mark.parametrize("spec", _parity_witnesses(), ids=lambda s: s.id)
def test_boundary_method_parity(spec, integration_nworkers: int, integration_outdir: str) -> None:
    model = load_witness_model(spec)
    ambient = run_bfs_ambient(model, spec, nworkers=integration_nworkers, verify=True)
    shi = boundary_shi_for_spec(ambient, spec)

    boundary_full = ambient.get_boundary_complex(shi, verbose=False)
    boundary_disc = Complex(model).discover_boundary_complex(
        shi,
        verbose=False,
        nworkers=integration_nworkers,
    )

    tags_full = tag_set(boundary_full)
    tags_disc = tag_set(boundary_disc)
    betti_full = truncated_betti(boundary_full)
    betti_disc = truncated_betti(boundary_disc)

    if tags_full != tags_disc or betti_full != betti_disc:
        write_failure_artifacts(
            integration_outdir,
            spec.id,
            only_full=tags_full - tags_disc,
            only_discover=tags_disc - tags_full,
            betti_full=betti_full,
            betti_discover=betti_disc,
            extra={
                "n_full": len(tags_full),
                "n_discover": len(tags_disc),
                "n_regions": len(ambient),
            },
        )

    assert tags_full == tags_disc, (
        f"{spec.id}: boundary cell mismatch full={len(tags_full)} discover={len(tags_disc)}; "
        + f"artifacts in {integration_outdir}/{spec.id}"
    )
    assert betti_full == betti_disc, f"{spec.id}: truncated Betti mismatch full={betti_full} discover={betti_disc}"

    if spec.expected_n_boundary_cells is not None:
        assert len(tags_full) == spec.expected_n_boundary_cells
    if spec.expected_truncated_betti is not None:
        assert betti_full == spec.expected_truncated_betti
    if spec.expected_n_regions is not None:
        assert len(ambient) == spec.expected_n_regions
