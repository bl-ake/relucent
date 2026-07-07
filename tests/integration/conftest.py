"""Pytest configuration for opt-in integration tests."""

from __future__ import annotations

import os

import pytest

INTEGRATION_ENV = "RELUCENT_RUN_INTEGRATION"

os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if os.environ.get(INTEGRATION_ENV, "0") == "1":
        return
    skip = pytest.mark.skip(
        reason=f"Opt-in integration tests. Set {INTEGRATION_ENV}=1.",
    )
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)


@pytest.fixture
def integration_nworkers() -> int:
    """Worker count for BFS / boundary discovery (default 1; use 64 on SLURM)."""
    raw = os.environ.get("RELUCENT_INTEGRATION_NWORKERS", "1")
    return max(1, int(raw))


@pytest.fixture
def integration_outdir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Directory for failure artifacts (override with RELUCENT_INTEGRATION_OUTDIR)."""
    custom = os.environ.get("RELUCENT_INTEGRATION_OUTDIR")
    if custom:
        path = os.path.abspath(custom)
        os.makedirs(path, exist_ok=True)
        return path
    return str(tmp_path_factory.mktemp("integration_artifacts"))
