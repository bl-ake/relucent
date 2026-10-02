"""Tests for the verbose convention (relucent._internal.logging)."""

from __future__ import annotations

import logging

import numpy as np

import relucent.config as cfg
from relucent import Complex, set_seeds, torch_mlp
from relucent._internal.logging import logger, show_progress, verbosity, with_verbosity


@with_verbosity
def _probe(*, verbose: int | None = None) -> tuple[int | None, bool, int]:
    return verbose, show_progress(), logger.level


@with_verbosity
def _outer(*, verbose: int | None = None) -> tuple[int | None, bool, int]:
    del verbose
    return _probe()


def test_levels_and_restore() -> None:
    before = logger.level
    assert _probe(verbose=0) == (0, False, logging.WARNING)
    assert _probe(verbose=1) == (1, True, logging.INFO)
    assert _probe(verbose=2) == (2, True, logging.DEBUG)
    assert _probe(verbose=True)[0] == 1
    assert logger.level == before


def test_none_uses_config_at_call_time() -> None:
    saved = cfg.VERBOSE
    try:
        cfg.VERBOSE = 0  # plain assignment, not update_settings
        assert _probe() == (0, False, logging.WARNING)
        assert _probe(verbose=1)[1] is True  # an explicit argument wins over the setting
    finally:
        cfg.VERBOSE = saved


def test_nested_none_inherits_outer_level() -> None:
    assert _outer(verbose=0)[:2] == (0, False)
    assert _outer(verbose=2)[:2] == (2, True)
    with verbosity(0):
        assert _probe()[0] == 0


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[tuple[int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append((record.levelno, record.getMessage()))


def _bfs_messages(verbose: int) -> list[tuple[int, str]]:
    set_seeds(0)
    cplx = Complex(torch_mlp([2, 6, 1]))
    handler = _Records()
    logger.addHandler(handler)
    try:
        cplx.bfs(start=np.zeros((1, 2)) + 0.123, verbose=verbose, nworkers=2)
        cplx.betti_numbers(verbose=verbose)
    finally:
        logger.removeHandler(handler)
    return handler.messages


def test_bfs_verbosity_levels() -> None:
    assert _bfs_messages(0) == []
    one = _bfs_messages(1)
    assert (logging.INFO, "searcher running on 2 workers") in one
    assert all(level == logging.INFO for level, _ in one)
    two = _bfs_messages(2)
    assert any(level == logging.DEBUG for level, _ in two)
    assert set(one) <= set(two)
