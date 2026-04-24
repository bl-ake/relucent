"""Checks for importing relucent without the torch extra."""

from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="This test targets no-torch environments only.")
def test_submodules_are_importable_without_torch():
    import relucent

    assert relucent.__version__
    # These symbols should be importable even when torch is absent.
    assert relucent.Complex is not None
    assert relucent.Polyhedron is not None
    assert callable(relucent.convert)
    assert callable(relucent.get_colors)
    assert callable(relucent.plot_complex)
    assert callable(relucent.plot_polyhedron)
