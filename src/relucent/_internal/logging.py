"""Package-level logger, verbosity scoping, and progress bars for relucent.

Progress and status messages go through the ``"relucent"`` logger, and progress bars are
shown only while that logger is enabled for ``INFO``. Its level follows a verbosity:

* ``0``: ``WARNING``, quiet.
* ``1`` (default): ``INFO``, progress bars and one-line summaries.
* ``2`` or more: ``DEBUG``, per-stage detail.

Every long-running public function takes ``verbose: int | None = None`` and runs inside
:func:`verbosity`, which sets the level for the duration of the call. ``None`` means
:data:`relucent.config.VERBOSE`, or the enclosing call's level when one relucent call
makes another.

The logger has one :class:`~logging.StreamHandler` writing plain messages to stderr.
Handlers you add to the ``"relucent"`` logger take precedence; set ``logger.propagate = True``
to also forward records to the root logger.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ParamSpec, TypeVar

from tqdm.auto import tqdm

__all__ = ["logger", "progress", "show_progress", "verbosity", "with_verbosity"]

P = ParamSpec("P")
R = TypeVar("R")

logger: logging.Logger = logging.getLogger("relucent")

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)  # mirrors VERBOSE=1 default
logger.propagate = False

# Verbosity of the innermost active verbosity() scope in this context, if any.
_active: ContextVar[int | None] = ContextVar("relucent_verbosity", default=None)


def _level(verbose: int) -> int:
    if verbose <= 0:
        return logging.WARNING
    return logging.INFO if verbose == 1 else logging.DEBUG


def _apply_verbose(verbose: int) -> None:
    """Set the relucent logger level for a verbosity (see :data:`relucent.config.VERBOSE`)."""
    logger.setLevel(_level(int(verbose)))


@contextmanager
def verbosity(verbose: int | None) -> Generator[int, None, None]:
    """Run a block at a verbosity: the ``verbose`` argument of a public relucent call.

    ``None`` inherits the enclosing scope's verbosity, or :data:`relucent.config.VERBOSE`
    outside any scope. ``True``/``False`` mean ``1``/``0``. Yields the resolved level.
    The logger level is process-wide, so concurrent calls from different threads with
    different verbosities affect each other's output.
    """
    if verbose is None:
        verbose = _active.get()
    if verbose is None:
        import relucent.config as cfg

        verbose = cfg.VERBOSE
    verbose = int(verbose)
    old_level = logger.level
    token = _active.set(verbose)
    _apply_verbose(verbose)
    try:
        yield verbose
    finally:
        _active.reset(token)
        logger.setLevel(old_level)


def with_verbosity(fn: Callable[P, R]) -> Callable[P, R]:
    """Run ``fn`` inside :func:`verbosity` of its ``verbose`` argument.

    ``fn`` sees ``verbose`` already resolved to an ``int``.
    """
    sig = inspect.signature(fn)
    if "verbose" not in sig.parameters:
        raise TypeError(f"{fn.__qualname__} has no 'verbose' parameter")

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        with verbosity(bound.arguments["verbose"]) as level:
            bound.arguments["verbose"] = level
            return fn(*bound.args, **bound.kwargs)

    return wrapper


def show_progress() -> bool:
    """Whether progress output is on (the relucent logger is enabled for ``INFO``)."""
    return logger.isEnabledFor(logging.INFO)


def progress(iterable: Any = None, **kwargs: Any) -> Any:
    """A :func:`tqdm.auto.tqdm` bar that is hidden unless :func:`show_progress`.

    Bars appear only after ``delay`` seconds (default 1), so quick calls print nothing.
    An explicit ``disable=True`` still hides the bar.
    """
    kwargs["disable"] = bool(kwargs.get("disable", False)) or not show_progress()
    kwargs.setdefault("delay", 1.0)
    return tqdm(iterable, **kwargs)
