"""SHI bound sensitivity: a too-small box refuses rather than missing output-neuron facets."""

from __future__ import annotations

import numpy as np
import pytest

from relucent.core.errors import AmbiguousGeometryError
from relucent.core.poly import Polyhedron
from relucent.geometry.calculations import get_shis
from tests.integration.helpers import (
    boundary_shi_for_spec,
    load_witness_model,
    run_bfs_ambient,
    witness_by_id,
)

pytestmark = [pytest.mark.integration]

# Too-small fixed box: on many unbounded cofaces it hides the output SHI. Facet
# decisions are certified, so get_shis must raise there instead of dropping it.
_SMALL_BOUND = 10.0


def test_small_bound_refuses_instead_of_missing_output_shi(integration_nworkers: int) -> None:
    spec = witness_by_id("shi_bound_5303")
    model = load_witness_model(spec)
    ambient = run_bfs_ambient(model, spec, nworkers=integration_nworkers, verify=True)
    shi = boundary_shi_for_spec(ambient, spec)
    boundary = ambient.boundary_complex(shi, verbose=False)

    n_cofaces = 0
    refused = 0
    for poly in boundary:
        ss = np.asarray(poly.ss_np, dtype=np.int8).reshape(1, -1)
        if ss.ravel()[shi] != 0:
            continue
        ss_pos = ss.copy()
        ss_pos.ravel()[shi] = 1
        ppos = ambient[ss_pos]
        n_cofaces += 1

        sh_default = get_shis(Polyhedron(model, ppos.ss_np))
        assert shi in sh_default, f"default get_shis misses output SHI {shi} on coface {ppos!r}"

        try:
            sh_small = get_shis(
                Polyhedron(model, ppos.ss_np),
                bound=_SMALL_BOUND,
                escalate_bound=False,
            )
        except AmbiguousGeometryError:
            refused += 1
            continue
        except ValueError as exc:
            # Unbounded cofaces can be infeasible inside a tiny fixed box
            # (Gurobi status 3) when escalate_bound=False; skip those.
            if "Initial Solve Failed" not in str(exc):
                raise
            continue
        assert sorted(sh_small) == sorted(sh_default), (
            f"bound={_SMALL_BOUND} answered {sorted(sh_small)} but the default gives {sorted(sh_default)} on {ppos!r}"
        )

    assert n_cofaces > 0
    assert refused > 0, f"expected bound={_SMALL_BOUND} to be too small to certify facets on some of {n_cofaces} cofaces"
