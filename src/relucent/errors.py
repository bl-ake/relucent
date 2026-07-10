"""Exception hierarchy for polyhedral-complex verification and repair.

Every error raised by :mod:`relucent.incidence`, :mod:`relucent.certify`,
:mod:`relucent.meta_graph`, and :mod:`relucent.complex` when a complex fails an
invariant check lives here, so callers only need one import to catch domain
errors from the verification pipeline.
"""

from __future__ import annotations

__all__ = [
    "ComplexNotCompleteError",
    "ComplexNotVerifiedError",
    "CubicalAmbiguityError",
    "CubicalConsistencyError",
    "DualGraphAsymmetricEdgeError",
    "IncompleteDualGraphError",
    "NonGenericArrangementError",
    "ShiFlipInvariantError",
    "ShiProofError",
]


class ComplexNotCompleteError(RuntimeError):
    """Topology routine requires a fully explored complex."""


class ComplexNotVerifiedError(RuntimeError):
    """Topology routine requires a complex that passed certification."""


class IncompleteDualGraphError(ValueError):
    """The dual graph has missing boundary neighbors (partially explored complex).

    :meth:`~relucent.complex.Complex.contract`, :meth:`~relucent.complex.Complex.get_chain_complex`,
    and :meth:`~relucent.complex.Complex.get_meta_graph` require a complete adjacency
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
