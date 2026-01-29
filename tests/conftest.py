"""Shared pytest fixtures for relucent tests."""

import pytest

from relucent import get_mlp_model, set_seeds


@pytest.fixture
def seed():
    """Default RNG seed for reproducible tests."""
    return 0


@pytest.fixture
def seeded(seed):
    """Set all RNG seeds before test. Use as: def test_foo(seeded): ..."""
    set_seeds(seed)
    return seed


@pytest.fixture
def small_mlp(seeded):
    """Small MLP [4, 8] with ReLU on last layer, for fast complex/search tests."""
    return get_mlp_model(widths=[4, 8], add_last_relu=True)


@pytest.fixture
def tiny_mlp(seeded):
    """Tiny MLP [2, 4, 2] with last ReLU, for quick sanity checks."""
    return get_mlp_model(widths=[2, 4, 2], add_last_relu=True)


@pytest.fixture
def mlp_2d(seeded):
    """2D input MLP [2, 10, 5, 1] for plotting and 2D-specific tests."""
    return get_mlp_model(widths=[2, 10, 5, 1])
