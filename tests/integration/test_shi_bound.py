"""SHI bound sensitivity: network-scaled bound detects output-neuron facets."""

from __future__ import annotations

import numpy as np
import pytest

from relucent.core.poly import Polyhedron
from relucent.geometry.calculations import get_shis
from tests.integration.helpers import (
    boundary_shi_for_spec,
    default_bound,
    load_witness_model,
    run_bfs_ambient,
    witness_by_id,
)

pytestmark = [pytest.mark.integration]

# Too-small fixed box: many unbounded cofaces miss the output SHI that a
# network-scaled bound recovers. (A huge finite box vs network bound is
# environment-sensitive under pytest/Gurobi.)
_SMALL_BOUND = 10.0


def test_network_bound_detects_output_shi_on_unbounded_cofaces(integration_nworkers: int) -> None:
    spec = witness_by_id("shi_bound_5303")
    model = load_witness_model(spec)
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
        try:
            sh_small = get_shis(
                Polyhedron(model, ppos.ss_np),
                bound=_SMALL_BOUND,
                escalate_bound=False,
            )
            sh_net = get_shis(
                Polyhedron(model, ppos.ss_np),
                bound=net_bound,
                escalate_bound=False,
            )
        except ValueError as exc:
            # Unbounded cofaces can be infeasible inside a tiny fixed box
            # (Gurobi status 3) when escalate_bound=False; skip those.
            if "Initial Solve Failed" not in str(exc):
                raise
            continue
        if shi not in sh_small and shi in sh_net:
            hits += 1

    assert hits > 0, (
        f"expected cofaces where bound={_SMALL_BOUND} misses output SHI but " + f"network bound={net_bound:.4g} detects it"
    )
