"""Multiprocessing worker state for search and geometry workers."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from relucent.utils import get_env

if TYPE_CHECKING:
    from gurobipy import Env

    from relucent.model.model import ReLUNetwork

__all__ = ["WorkerContext", "get_worker_context", "set_worker_context", "worker_context_scope"]

# Set by set_worker_context() when used as a pool initializer in worker processes.
_context: WorkerContext | None = None


@dataclass
class WorkerContext:
    """Picklable network and Gurobi state shared by multiprocessing workers."""

    net: ReLUNetwork
    env: Env
    dim: int
    get_vol_calc: bool = True


def set_worker_context(
    get_net: ReLUNetwork,
    get_volumes: bool = True,
    num_threads: int | None = None,
) -> None:
    """Initialize worker globals for multiprocessing pools.

    Intended as a pool initializer in worker processes. For main-process serial
    paths (e.g. ``nworkers=1`` in :func:`~relucent.search.hamming_astar`), prefer
    :func:`worker_context_scope` so globals are restored afterward.
    """
    global _context
    _context = WorkerContext(
        net=get_net,
        env=get_env(num_threads=num_threads),
        dim=int(np.prod(get_net.input_shape)),
        get_vol_calc=get_volumes,
    )


def get_worker_context() -> WorkerContext:
    """Return worker globals; requires :func:`set_worker_context` as pool initializer."""
    if _context is None:
        raise RuntimeError("set_worker_context must be used as pool initializer")
    return _context


@contextmanager
def worker_context_scope(
    net: ReLUNetwork,
    get_volumes: bool = True,
    num_threads: int | None = None,
) -> Generator[WorkerContext, None, None]:
    """Temporarily set worker context (main-process serial paths / tests)."""
    global _context
    prev = _context
    try:
        set_worker_context(net, get_volumes, num_threads)
        yield get_worker_context()
    finally:
        _context = prev
