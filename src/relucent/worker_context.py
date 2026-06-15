"""Multiprocessing worker state for search and geometry workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from relucent.utils import get_env

if TYPE_CHECKING:
    from gurobipy import Env

    from relucent.model import ReLUNetwork

__all__ = ["WorkerContext", "get_worker_context", "set_worker_context"]

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

    Use only as a pool initializer in worker processes, never from the main process.
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
