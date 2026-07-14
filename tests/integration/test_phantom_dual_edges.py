"""Phantom dual edges across empty shared faces inflate truncated Betti numbers."""

from __future__ import annotations

import numpy as np
import pytest

from relucent.core.poly import Polyhedron
from tests.integration.helpers import (
    boundary_shi_for_spec,
    load_witness_model,
    run_bfs_ambient,
    truncated_betti,
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


def test_shi_bound_dual_graph_rejects_empty_shared_faces(integration_nworkers: int) -> None:
    """Witness shi_bound_5303 previously admitted phantom dual edges (β₁ → 47)."""
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
    assert truncated_betti(boundary) == {0: 1}
