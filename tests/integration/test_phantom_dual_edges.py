"""Phantom dual edges: combinatorial flips whose shared face is geometrically empty."""

from __future__ import annotations

import numpy as np
import pytest

from relucent.core.poly import Polyhedron
from tests.integration.helpers import (
    boundary_shi_for_spec,
    load_witness_model,
    run_bfs_ambient,
    witness_by_id,
)

pytestmark = [pytest.mark.integration]


def _shared_face_empty(u: Polyhedron, shi: int) -> bool:
    ss = np.asarray(u.ss_np, dtype=np.int8).copy()
    ss.ravel()[int(shi)] = 0
    face = Polyhedron(u._net, ss, halfspaces=u.halfspaces)
    try:
        center, inradius = face.get_center_inradius()
    except ValueError as exc:
        return str(exc).startswith("Inradius ")
    return center is None and inradius is None


@pytest.mark.xfail(
    reason=(
        "Boundary dual graphs still admit empty shared-face flips on shi_bound_5303. "
        + "Dropping those edges in dual_edges_flip_neighbors breaks contracted-SHI "
        + "certification (cubical SHIs still list the crossing). Needs a SHI-level "
        + "geometric face filter aligned with verify_contracted_shis."
    ),
    strict=True,
)
def test_shi_bound_dual_graph_rejects_empty_shared_faces(integration_nworkers: int) -> None:
    """Boundary dual graph must drop flips whose shared face is empty on both ends.

    Truncated Betti is asserted via parity / manifest goldens. Meta-graph homology
    does not use dual edges, so dual phantoms and Betti are independent checks —
    filtering dual edges alone does not change truncated β (still {0:1, 1:47} here).
    """
    spec = witness_by_id("shi_bound_5303")
    model = load_witness_model(spec)
    ambient = run_bfs_ambient(model, spec, nworkers=integration_nworkers, verify=True)
    boundary = ambient.get_boundary_complex(boundary_shi_for_spec(ambient, spec), verbose=False)

    graph = boundary.get_dual_graph(verbose=False)
    # True phantoms are empty for both endpoints (halfspaces can disagree one-sided).
    phantom_edges = sum(
        1
        for u, v, data in graph.edges(data=True)
        if _shared_face_empty(u, int(data["shi"])) and _shared_face_empty(v, int(data["shi"]))
    )
    assert phantom_edges == 0, f"dual graph still has {phantom_edges} empty shared-face edges"
