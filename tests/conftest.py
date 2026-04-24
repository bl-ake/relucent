"""Shared pytest fixtures for relucent tests."""

import pytest

from relucent import mlp, set_seeds


@pytest.fixture
def seed():
    """Default RNG seed for reproducible tests."""
    return 0


@pytest.fixture
def seeded(seed):
    """Set all RNG seeds before test."""
    set_seeds(seed)
    return seed


@pytest.fixture
def small_mlp(seed):
    """Small MLP [4, 8] with ReLU on last layer, for fast complex/search tests."""
    set_seeds(seed)
    return mlp(widths=[4, 8], add_last_relu=True)


@pytest.fixture
def tiny_mlp(seed):
    """Tiny MLP [2, 4, 2] with last ReLU, for quick sanity checks."""
    set_seeds(seed)
    return mlp(widths=[2, 4, 2], add_last_relu=True)


@pytest.fixture
def mlp_2d(seed):
    """2D input MLP [2, 10, 5, 1] for plotting and 2D-specific tests."""
    set_seeds(seed)
    return mlp(widths=[2, 10, 5, 1])
