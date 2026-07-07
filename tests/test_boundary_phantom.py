"""Regression tests for combinatorially feasible but geometrically spurious boundary cells.

Full witness coverage lives in ``tests/integration/test_phantom_boundary.py`` (opt-in).
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")

pytestmark = pytest.mark.skip(
    reason="Moved to opt-in integration suite (RELUCENT_RUN_INTEGRATION=1).",
)


def test_phantom_moved_to_integration() -> None:
    """Placeholder so this module documents the migration."""
