"""SHI bound sensitivity: network-scaled bound detects output-neuron facets."""

from __future__ import annotations

import numpy as np
import pytest

from relucent import convert
from relucent.calculations import get_shis
from relucent.numeric_tolerances import apply_tolerances
from relucent.poly import Polyhedron
from tests.integration.helpers import (
    boundary_shi_for_spec,
    default_bound,
    load_witness_model,
    run_bfs_ambient,
    witness_by_id,
)

pytestmark = [pytest.mark.integration]

_LARGE_BOUND = 1e8


def test_network_bound_detects_output_shi_on_unbounded_cofaces(integration_nworkers: int) -> None:
    spec = witness_by_id("shi_bound_5303")
    model = load_witness_model(spec)
    # Network-scaled tolerances (pytest disables Complex auto_tolerances by default).
    apply_tolerances(net=convert(model))
    ambient = run_bfs_ambient(model, spec, nworkers=integration_nworkers, verify=True)
    shi = boundary_shi_for_spec(ambient, spec)
    boundary = ambient.get_boundary_complex(shi, verbose=False)

    net_bound = default_bound(model)
    hits = 0
    for poly in boundary:
        ss = np.asarray(poly.ss_np, dtype=np.int8).reshape(1, -1)
        if ss.ravel()[shi] != 0:
            continue
        ss_pos = ss.copy()
        ss_pos.ravel()[shi] = 1
        ppos = ambient[ss_pos]
        sh_large = get_shis(
            Polyhedron(model, ppos.ss_np, bound=_LARGE_BOUND),
            bound=_LARGE_BOUND,
            escalate_bound=False,
        )
        try:
            sh_net = get_shis(
                Polyhedron(model, ppos.ss_np, bound=net_bound),
                bound=net_bound,
                escalate_bound=False,
            )
        except ValueError as exc:
            if "Initial Solve Failed" in str(exc):
                continue
            raise
        if shi not in sh_large and shi in sh_net:
            hits += 1

    assert hits > 0, (
        f"expected cofaces where bound={_LARGE_BOUND} misses output SHI but " + f"network bound={net_bound:.4g} detects it"
    )
