"""Exception hierarchy for polyhedral-complex verification and repair.

Every error raised by :mod:`relucent.graph.incidence`, :mod:`relucent.verify.certify`,
:mod:`relucent.graph.meta_graph`, and :class:`~relucent.core.complex.Complex` when a complex fails an
invariant check lives here, so callers only need one import to catch domain
errors from the verification pipeline.
"""

from __future__ import annotations

__all__ = [
    "AmbiguousGeometryError",
    "ComplexNotCompleteError",
    "ComplexNotVerifiedError",
    "CubicalAmbiguityError",
    "CubicalConsistencyError",
    "DualGraphAsymmetricEdgeError",
    "IncompleteDualGraphError",
    "NonGenericArrangementError",
    "ShiFlipInvariantError",
    "ShiProofError",
    "SolverTimeLimitError",
]


class ComplexNotCompleteError(RuntimeError):
    """Topology routine requires a fully explored complex."""


class ComplexNotVerifiedError(RuntimeError):
    """Topology routine requires a complex that passed certification."""


class IncompleteDualGraphError(ValueError):
    """The dual graph has missing boundary neighbors (partially explored complex).

    :meth:`~relucent.core.complex.Complex.contract`, :meth:`~relucent.core.complex.Complex.chain_complex`,
    and :meth:`~relucent.core.complex.Complex.meta_graph` require a complete adjacency
    structure among top-dimensional cells. Explore the complex further (e.g. BFS/DFS)
    before building the chain complex or running topology routines.
    """


class DualGraphAsymmetricEdgeError(ValueError):
    """A dual-graph edge is not supported by both endpoints' SHI lists."""


class ShiFlipInvariantError(ValueError):
    """A cached SHI lacks a symmetric flip neighbor in the complex."""


class ShiProofError(ValueError):
    """SHI facet proof failed under strict certification."""


class CubicalConsistencyError(ValueError):
    """Dual-graph or incidence data violates the cubical face-star convention."""


class CubicalAmbiguityError(CubicalConsistencyError):
    """More than two top-dimensional cells share one codimension-one face tag."""


class NonGenericArrangementError(ValueError):
    """Geometric genericity / transversality is violated (degenerate endpoints or junctions)."""


class AmbiguousGeometryError(RuntimeError):
    """A geometric decision falls inside its float64 error bound, so it cannot be made correctly.

    Raised instead of guessing: which side of a hyperplane a point lies on, whether a row is a
    facet, whether a cell is empty, or whether two computed points coincide, when the computed
    value is within the rounding error of the network's exact rows. A ``RuntimeError`` rather than
    a ``ValueError`` so that handlers written for recoverable ``ValueError`` exceptions do not swallow it.
    """


class SolverTimeLimitError(AmbiguousGeometryError):
    """A Gurobi solve reached :data:`~relucent.config.GUROBI_TIME_LIMIT`, so the decision it was for is not made.

    A few badly conditioned LPs never finish, and the limit stops them. Where a cold re-solve or
    exact arithmetic can make the same decision (a SHI LP), relucent does that first and raises
    only if neither can. An :class:`AmbiguousGeometryError`, so every handler that refuses to guess
    on an undecidable geometric question also refuses to treat a timed-out solve as an answer.
    """
