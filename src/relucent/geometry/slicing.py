"""Intersections of a complex's cells with an affine subspace."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import relucent.config as cfg
from relucent._internal import rounding
from relucent._internal.gurobi import get_env
from relucent.core.errors import AmbiguousGeometryError
from relucent.core.poly import Polyhedron
from relucent.geometry.calculations import solve_radius
from relucent.graph import incidence
from relucent.model.model import LinearLayer, ReLUNetwork

if TYPE_CHECKING:
    from relucent.core.complex import Complex

__all__ = ["slice_complex"]


def slice_complex(cplx: Complex, x0: np.ndarray, V: np.ndarray) -> Complex:
    """Return the non-empty intersections of each cell with an affine subspace.

    The affine subspace is given in the parametric form ``{x0 + V @ t : t in R^k}``,
    where ``x0`` is a base point in input space and the columns of ``V`` span the
    subspace direction. ``k = V.shape[1]`` is the intrinsic dimension of the subspace.

    For a cell with halfspace representation ``Ax + b <= 0``, the intersection in
    parameter space is ``{t : (A @ V) t + (A @ x0 + b) <= 0}``, a polyhedron in ``R^k``.

    Feasibility is tested with :func:`~relucent.geometry.calculations.solve_radius` on the
    sliced rows, whose float64 error is carried from the cell's. An intersection is included
    when it has a verified interior point (or an unbounded radius, meaning the subspace lies
    entirely inside that cell).

    The returned :class:`~relucent.core.complex.Complex` is backed by a stub :class:`~relucent.model.model.ReLUNetwork`
    with ``input_shape=(k,)`` and no ReLU layers, so ``cpx.dim == k`` and
    ``cpx.plot(plot_mode="cells")`` works for ``k`` in ``{2, 3}``.  Each
    :class:`~relucent.core.poly.Polyhedron` in the result carries ``halfspaces = H_slice``
    (shape ``(m, k+1)``) and inherits the sign sequence of its parent cell, which
    keeps tags unique across cells and carries correct codimension information.

    Every row is judged against its own float64 error; an intersection that can't be
    decided raises :class:`~relucent.core.errors.AmbiguousGeometryError`.

    Note:
        This method triggers halfspace computation for any cell that has not yet been
        computed. Pre-populate with
        :meth:`~relucent.core.complex.Complex.compute_geometric_properties` to avoid
        on-demand Gurobi calls.

    Args:
        x0: Base point of the affine subspace, shape ``(d,)``.
        V: Direction matrix, shape ``(d, k)``. Columns need not be orthonormal.
            Pass a 1-D array of shape ``(d,)`` for a line (``k=1``).

    Returns:
        A new complex in ``k``-dimensional parameter space, containing
        one :class:`~relucent.core.poly.Polyhedron` per non-empty intersection.
        It can be plotted directly with its ``plot`` method for ``k`` in ``{2, 3}``.
    """
    env = get_env()

    x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
    # Ensure V is 2-D: a 1-D vector becomes a (d, 1) column
    V_arr = np.asarray(V, dtype=np.float64).reshape(len(x0_arr), -1)
    k = V_arr.shape[1]

    # Stub network: gives out.dim = k with no ReLU layers (no halfspace LPs needed).
    stub_net = ReLUNetwork(
        {"linear": LinearLayer(np.eye(k, dtype=np.float64), np.zeros(k, dtype=np.float64))},
        input_shape=(k,),
    )
    out = type(cplx)(stub_net)

    def _slice_poly_kwargs(parent: Polyhedron, halfspaces: np.ndarray) -> dict[str, Any]:
        err = rounding.slice_error(parent.halfspaces_np, parent.halfspaces_err_np, V_arr, x0_arr)
        kwargs: dict[str, Any] = {"halfspaces": halfspaces, "halfspaces_err": err, "ambient_dim": k}
        if parent._shis is not None:
            kwargs["shis"] = list(parent._shis)
        if parent._finite_computed and parent._finite is True:
            # A slice of a bounded cell is bounded; a slice of an unbounded one may not be.
            kwargs["finite"] = True
        return kwargs

    for poly in cplx:
        H = poly.halfspaces_np  # (m, d+1)
        A = H[:, :-1]  # (m, d)
        b_col = H[:, -1]  # (m,)

        # Substitute x = x0 + V t into each constraint a_i^T x + b_i <= 0
        A_v = A @ V_arr  # (m, k)
        b_v = A @ x0_arr + b_col  # (m,)

        if k == 0:
            # Subspace is a single point; just check containment of x0
            b_err = rounding.slice_error(H, poly.halfspaces_err_np, V_arr, x0_arr)[:, -1]
            if np.any(b_v > b_err):
                continue
            if not np.all(b_v < -b_err):
                raise AmbiguousGeometryError(f"the slice point lies on a row of {poly!r} to within float64 error")
            out.add_polyhedron(
                Polyhedron(
                    stub_net,
                    poly.ss_np,
                    **_slice_poly_kwargs(poly, b_v.reshape(-1, 1)),
                ),
                check_exists=False,
            )
            continue

        H_slice = np.column_stack([A_v, b_v])  # (m, k+1)

        # Chebyshev-center LP on the sliced rows, certified against their own error:
        # a verified interior point or an unbounded radius means the slice is nonempty.
        kwargs = _slice_poly_kwargs(poly, H_slice)
        center, radius = solve_radius(env, H_slice, errors=kwargs["halfspaces_err"])
        if center is not None or radius == float("inf"):
            out.add_polyhedron(
                Polyhedron(stub_net, poly.ss_np, **kwargs),
                check_exists=False,
            )
    if len(out) > 0:
        incidence.set_contracted_shis(out)
        if cfg.CAREFUL_MODE:
            incidence.verify_contracted_shis(out)
    return out
