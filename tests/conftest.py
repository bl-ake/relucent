"""Shared pytest fixtures for relucent tests."""

import os

# Keep legacy config defaults for tolerance-sweep tests; package import bootstraps otherwise.
os.environ.setdefault("RELUCENT_SKIP_NUMERIC_BOOTSTRAP", "1")

import pytest

from relucent import Complex, mlp, set_seeds
from relucent.config import update_settings
from relucent.topology import C_BACKEND_AVAILABLE

_complex_init = Complex.__init__


def _complex_init_no_auto_tolerances(self, net, *args, **kwargs):
    kwargs.setdefault("auto_tolerances", False)
    return _complex_init(self, net, *args, **kwargs)


Complex.__init__ = _complex_init_no_auto_tolerances  # type: ignore[method-assign]


# All tests run with extra consistency checks (see :data:`relucent.config.CAREFUL_MODE`).
update_settings(CAREFUL_MODE=True)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip C GF(2) tests when the JIT backend is unavailable (typical on Windows CI).

    The dedicated ``gf2-backend`` workflow job sets ``RELUCENT_REQUIRE_C_GF2=1`` so those
    tests run and fail loudly if gcc cannot compile ``_gf2_rank.c``.
    """
    if os.environ.get("RELUCENT_REQUIRE_C_GF2") == "1":
        return
    if C_BACKEND_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="C GF(2) backend not available (gcc compile/load failed)")
    for item in items:
        if "requires_c_gf2" in item.keywords:
            item.add_marker(skip)


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
