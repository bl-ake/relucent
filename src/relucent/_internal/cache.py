"""The marker for a cache slot whose value has not been computed yet.

A :class:`~relucent.core.poly.Polyhedron` caches each computed property in one private slot.
Slots whose answer can itself be ``None`` (an empty cell's ``finite``, Qhull geometry that
could not be built) start at :data:`UNSET`, so "not computed" and "computed: ``None``" stay
distinct without a separate flag.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

__all__ = ["UNSET", "Unset"]


class Unset(Enum):
    """Type of :data:`UNSET`. An enum, so it pickles by name and type checkers narrow on it."""

    UNSET = "UNSET"


UNSET: Final = Unset.UNSET
