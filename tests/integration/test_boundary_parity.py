"""Strict parity: components from Complex.boundary_component tile full BFS + boundary_complex."""

from __future__ import annotations

import pytest

from tests.helpers import cover_by_components, summed_betti
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

    boundary_full = ambient.boundary_complex(shi, verbose=False)
    components = cover_by_components(boundary_full, model, shi, nworkers=integration_nworkers)

    tags_full = tag_set(boundary_full)
    tags_comp = set().union(*(tag_set(c) for c in components)) if components else set()
    betti_full = {k: v for k, v in truncated_betti(boundary_full).items() if v}
    betti_comp = summed_betti(components, compactify="truncate", verbose=False)

    if tags_full != tags_comp or betti_full != betti_comp:
        write_failure_artifacts(
            integration_outdir,
            spec.id,
            only_full=tags_full - tags_comp,
            only_discover=tags_comp - tags_full,
            betti_full=betti_full,
            betti_discover=betti_comp,
            extra={
                "n_full": len(tags_full),
                "n_components": len(components),
                "n_regions": len(ambient),
            },
        )

    assert tags_full == tags_comp, (
        f"{spec.id}: boundary cell mismatch full={len(tags_full)} components={len(tags_comp)}; "
        + f"artifacts in {integration_outdir}/{spec.id}"
    )
    assert betti_full == betti_comp, f"{spec.id}: truncated Betti mismatch full={betti_full} components={betti_comp}"

    if spec.expected_n_boundary_cells is not None:
        assert len(tags_full) == spec.expected_n_boundary_cells
    if spec.expected_truncated_betti is not None:
        assert betti_full == spec.expected_truncated_betti
    if spec.expected_n_regions is not None:
        assert len(ambient) == spec.expected_n_regions
