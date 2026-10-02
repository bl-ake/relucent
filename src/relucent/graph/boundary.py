"""Boundary (level-set) cells of an explored complex: the cells on one neuron's bent hyperplane.

For neuron ``i``, each dual-graph edge labelled ``i`` joins two top cells across a common
facet with ``ss[i] = 0``. Those facets are the top cells of neuron ``i``'s boundary complex.
To find that complex without exploring the whole input space, see
:mod:`relucent.search.boundary_search`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import relucent.config as cfg
import relucent.verify.certify as certify
from relucent._internal.logging import progress, with_verbosity
from relucent.core.errors import DualGraphAsymmetricEdgeError
from relucent.core.poly import Polyhedron
from relucent.graph import incidence
from relucent.verify.certify import CertifyLevel

if TYPE_CHECKING:
    from relucent.core.complex import Complex

__all__ = ["boundary_cells", "boundary_complex", "boundary_edges"]


def boundary_edges(cplx: Complex, i: int) -> set[tuple[Polyhedron, Polyhedron]]:
    """Edges of the dual graph that cross neuron ``i``'s bent hyperplane (``shi == i``)."""
    assert 0 <= i < cplx.n, f"Neuron index out of range: {i} not in [0, {cplx.n})"
    return {(a, b) for a, b, shi in progress(cplx.G.edges(data="shi"), desc="Getting Boundary Edges", delay=1) if shi == i}


def _codim_one_face_kwargs(p1: Polyhedron, shi: int) -> dict[str, Any]:
    """Shared kwargs for :func:`boundary_cells` faces.

    Candidate SHIs are all nonzero sign-sequence crossings on the face (role 1);
    :func:`~relucent.graph.incidence.set_contracted_shis` keeps flip neighbors in the slice.
    Infeasible 1-cell faces are dropped at construction time via
    :meth:`~relucent.core.poly.Polyhedron.is_shi_face_feasible`.

    Meta-graph **face edges** always use :func:`~relucent.graph.incidence.ss_nonzero_indices`.
    The ambient chain complex uses :mod:`relucent.graph.vertex_star` instead of this helper.
    """
    ambient = p1.ambient_dim
    codim = p1.codim + 1
    face_dim = ambient - codim
    shi_i = int(shi)
    face_ss = p1.ss_np.copy()
    face_ss[shi_i] = 0
    candidate_shis = list(incidence.ss_nonzero_indices(face_ss))
    if face_dim == 1 and p1.halfspaces is not None:
        new_ss = p1.ss_np.copy()
        new_ss[shi_i] = 0
        probe = Polyhedron(
            p1._net,
            new_ss,
            halfspaces=p1.halfspaces,
            halfspaces_err=p1.halfspaces_err_np,
            halfspaces_ss=p1.halfspaces_rows_ss,
            rows_data=p1._rows_data,
            ambient_dim=ambient,
        )
        candidate_shis = [s for s in candidate_shis if probe.is_shi_face_feasible(int(s))]
    poly_kwargs: dict[str, Any] = {
        "halfspaces": p1.halfspaces,
        "halfspaces_err": p1.halfspaces_err_np,
        "halfspaces_ss": p1.halfspaces_rows_ss,
        "rows_data": p1._rows_data,
        "shis": candidate_shis,
        "ambient_dim": ambient,
    }
    return poly_kwargs


@with_verbosity
def boundary_cells(cplx: Complex, i: int, *, verify: bool = True, verbose: int | None = None) -> set[Polyhedron]:
    """The (d-1)-cells of ``cplx`` on neuron ``i``'s bent hyperplane.

    Each is the common facet of a dual-graph edge with ``shi == i``. With ``verify``, both
    cofaces must list ``i`` among their SHIs and the face must be feasible on both sides.
    """
    del verbose  # applied by @with_verbosity
    from relucent.search.boundary_search import _both_ambient_cofaces_feasible

    faces = set()
    edges = list(boundary_edges(cplx, i))
    for edge in progress(edges, desc="Getting Boundary Cells", delay=1):
        p1, p2 = edge[0], edge[1]
        shi = int(cplx.G.edges[edge]["shi"])
        if verify and (shi not in p1.shis or shi not in p2.shis):
            raise DualGraphAsymmetricEdgeError(f"Boundary edge shi={shi} on ({p1!r}, {p2!r}) lacks bidirectional SHI support.")
        new_ss = p1.ss_np.copy()
        new_ss[shi] = 0
        p = cplx.ss2poly(
            new_ss,
            check_exists=False,
            **_codim_one_face_kwargs(p1, shi),
        )
        if verify and not _both_ambient_cofaces_feasible(p, i):
            raise ValueError(f"Boundary face {p!r} fails ambient coface feasibility for neuron {i}.")
        faces.add(p)
    return faces


@with_verbosity
def boundary_complex(cplx: Complex, i: int, *, verbose: int | None = None) -> Complex:
    """The certified complex of (d-1)-cells on neuron ``i``'s bent hyperplane.

    Raises:
        IncompleteDualGraphError: If top-dimensional adjacency is incomplete.
        ComplexNotCompleteError: If the input complex is not complete.
        ComplexNotVerifiedError: If the input complex is not verified.
    """
    del verbose  # applied by @with_verbosity
    cplx.assert_topology_ready()
    cplx._dual_graph = cplx.dual_graph(require_complete=True)
    out = cplx._empty_like()
    for poly in progress(
        boundary_cells(cplx, i, verify=True),
        desc="Getting Boundary Complex",
        delay=1,
    ):
        out.add_polyhedron(poly, check_exists=False)
    incidence.set_contracted_shis(out)
    if cfg.CAREFUL_MODE:
        incidence.verify_contracted_shis(out)
    out.verify_arrangement_genericity()

    if len(out) == 0:
        out.set_exploration_state(complete=True, verified=True)
        return out
    out.set_exploration_state(complete=True, verified=False)
    certify.certify_complex(out, level=CertifyLevel.COMPLETE, record_state=True)
    return out
