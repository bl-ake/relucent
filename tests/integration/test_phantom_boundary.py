"""Phantom boundary cell regression on bundled ckpt0."""

from __future__ import annotations

import numpy as np
import pytest

from relucent import Complex
from relucent.core.poly import Polyhedron
from tests.integration.helpers import (
    boundary_shi_for_spec,
    load_witness_model,
    output_neuron_shi,
    run_bfs_ambient,
    tag_set,
    witness_by_id,
)

pytestmark = [pytest.mark.integration]

_PHANTOM_SS = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 0], dtype=np.int8).reshape(1, -1)


def test_phantom_ss_rejected_by_ambient_coface_gate() -> None:
    from relucent.search.boundary_search import _both_ambient_cofaces_feasible

    spec = witness_by_id("phantom_ckpt0")
    model = load_witness_model(spec)
    shi = output_neuron_shi(Complex(model))
    poly = Polyhedron(model, _PHANTOM_SS)
    assert poly.feasible
    assert not _both_ambient_cofaces_feasible(poly, shi)


def test_discover_boundary_matches_reference_without_phantom(integration_nworkers: int) -> None:
    spec = witness_by_id("phantom_ckpt0")
    model = load_witness_model(spec)
    ambient = run_bfs_ambient(model, spec, nworkers=integration_nworkers, verify=True)
    shi = boundary_shi_for_spec(ambient, spec)

    ref = ambient.get_boundary_complex(shi, verbose=False)
    new = Complex(model).discover_boundary_complex(shi, verbose=False, nworkers=integration_nworkers)

    assert tag_set(ref) == tag_set(new)
    if spec.expected_n_boundary_cells is not None:
        assert len(new) == spec.expected_n_boundary_cells
    assert _PHANTOM_SS.tobytes() not in tag_set(new)
