"""Multiprocessing context, worker pools, CPU count, and the thread-safe queues the search uses."""

import itertools
import multiprocessing as mp
import os
import sys
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Iterator, Sized
from heapq import heappop, heappush
from multiprocessing.context import BaseContext
from multiprocessing.pool import IMapIterator
from threading import Condition
from types import TracebackType
from typing import Any, Generic, TypeVar
from weakref import WeakSet

import relucent.config as cfg
from relucent._internal.logging import logger

__all__ = [
    "get_mp_context",
    "worker_pool",
    "WorkerPool",
    "process_aware_cpu_count",
    "NonBlockingQueue",
    "BlockingQueue",
    "UpdatablePriorityQueue",
]


def get_mp_context() -> BaseContext:
    """Return the appropriate multiprocessing context for the current platform.

    On macOS, uses ``spawn`` to avoid fork-after-PyTorch/BLAS issues that can
    segfault worker processes. Elsewhere, prefers ``fork`` when available so
    workers inherit the parent's already-loaded model weights without
    serialisation overhead. Falls back to ``spawn`` where ``fork`` is
    unavailable (e.g. Windows). When using ``spawn``, the caller's main
    module must use the standard ``if __name__ == "__main__":`` guard to
    prevent worker processes from re-executing top-level code.

    The environment variable ``RELUCENT_MP_START_METHOD`` can be set to force
    a specific start method (e.g. ``spawn``) for debugging and CI parity
    testing across platforms.

    ``forkserver`` is intentionally avoided: it uses OS semaphores for
    inter-process coordination that are not always released before Python's
    resource tracker runs at shutdown, producing spurious leaked-semaphore
    warnings (CPython issue #91435).

    Returns:
        A multiprocessing context object whose ``.Pool(...)`` method can be
        used to create a process pool.
    """
    forced = os.environ.get("RELUCENT_MP_START_METHOD")
    if forced:
        return mp.get_context(forced)
    available = mp.get_all_start_methods()
    if sys.platform == "darwin":
        return mp.get_context("spawn")
    return mp.get_context("fork" if "fork" in available else "spawn")


R = TypeVar("R")

# Seconds to wait for each in-flight result while a pool shuts down before giving up and
# terminating its workers (only reached if a worker died or a task hangs).
_DRAIN_TIMEOUT = 300.0

# Set in each worker of a WorkerPool by its initializer: the pool's shared stop flag.
_worker_stop_flag: Any = None


def _init_pool_worker(stop_flag: Any, initializer: Callable[..., object] | None, initargs: tuple[Any, ...]) -> None:
    global _worker_stop_flag
    _worker_stop_flag = stop_flag
    if initializer is not None:
        initializer(*initargs)


class _SkipAfterStop:
    """Run ``func`` unless the pool has started shutting down; then return ``None`` at once."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func

    def __call__(self, *args: Any) -> Any:
        if _worker_stop_flag is not None and _worker_stop_flag.value:
            return None
        return self.func(*args)


class _CallOnChunk:
    """Apply ``func`` to every item of a chunk (the pool's own ``mapstar``, but picklable here)."""

    def __init__(self, func: Callable[[Any], Any]) -> None:
        self.func = func

    def __call__(self, chunk: list[Any]) -> list[Any]:
        return [self.func(x) for x in chunk]


def _until_stopped(iterable: Iterable[Any], stop_flag: Any) -> Iterator[Any]:
    # Runs in the pool's task-handler thread: stop handing out tasks once shutdown starts.
    for item in iterable:
        if stop_flag.value:
            return
        yield item


def _chunks(items: Iterator[Any], size: int) -> Iterator[list[Any]]:
    while chunk := list(itertools.islice(items, size)):
        yield chunk


def _flatten(chunks: Iterator[list[Any] | None]) -> Iterator[Any]:
    for chunk in chunks:
        if chunk is not None:  # None marks a chunk skipped after shutdown began
            yield from chunk


class WorkerPool:
    """A process pool that shuts down without killing workers mid-write.

    ``multiprocessing.Pool.terminate()``, which ``with Pool() as pool`` also calls on exit,
    kills workers with SIGTERM. A worker killed while it holds the result queue's lock (while
    it sends a result, including the moment after the parent has read that result) never
    releases it. The pool's task-handler thread then blocks on that lock forever, and so does
    ``terminate()``.

    On exit this pool instead raises a shared stop flag, stops handing out tasks, lets tasks
    already started finish (tasks that start afterwards return ``None`` without running),
    drains every iterator it returned, and then closes and joins the pool. Iterables passed
    to :meth:`imap` and :meth:`imap_unordered` must not block forever: close a
    :class:`BlockingQueue` before leaving the ``with`` block. On ``KeyboardInterrupt`` (or any
    other non-``Exception``) the workers are terminated at once instead.
    """

    def __init__(
        self,
        processes: int | None = None,
        initializer: Callable[..., object] | None = None,
        initargs: Iterable[Any] = (),
    ) -> None:
        ctx = get_mp_context()
        self._stop_flag: Any = ctx.RawValue("b", 0)
        self._pool = ctx.Pool(
            processes, initializer=_init_pool_worker, initargs=(self._stop_flag, initializer, tuple(initargs))
        )
        # Unfinished iterators stay alive in the pool's own cache, so a WeakSet keeps every one
        # that still needs draining while letting finished ones go.
        self._iterators: WeakSet[IMapIterator[Any]] = WeakSet()
        self._closed = False

    def map(self, func: Callable[[Any], R], iterable: Iterable[Any], chunksize: int | None = None) -> list[R]:
        """Like :meth:`multiprocessing.pool.Pool.map`."""
        return self._pool.map(_SkipAfterStop(func), iterable, chunksize)

    def starmap(self, func: Callable[..., R], iterable: Iterable[Iterable[Any]], chunksize: int | None = None) -> list[R]:
        """Like :meth:`multiprocessing.pool.Pool.starmap`."""
        return self._pool.starmap(_SkipAfterStop(func), iterable, chunksize)

    def imap(self, func: Callable[[Any], R], iterable: Iterable[Any], chunksize: int = 1) -> Iterator[R]:
        """Like :meth:`multiprocessing.pool.Pool.imap`."""
        return self._lazy_map(self._pool.imap, func, iterable, chunksize)

    def imap_unordered(self, func: Callable[[Any], R], iterable: Iterable[Any], chunksize: int = 1) -> Iterator[R]:
        """Like :meth:`multiprocessing.pool.Pool.imap_unordered`."""
        return self._lazy_map(self._pool.imap_unordered, func, iterable, chunksize)

    def _lazy_map(
        self,
        method: Callable[..., "IMapIterator[Any]"],
        func: Callable[[Any], Any],
        iterable: Iterable[Any],
        chunksize: int,
    ) -> Iterator[Any]:
        # Chunk here rather than in the pool: with chunksize > 1, Pool.imap* returns a plain
        # generator, and only its IMapIterator can be drained with a timeout at shutdown.
        items = _until_stopped(iterable, self._stop_flag)
        if chunksize <= 1:
            it = method(_SkipAfterStop(func), items)
            self._iterators.add(it)
            return it
        it = method(_SkipAfterStop(_CallOnChunk(func)), _chunks(items, chunksize))
        self._iterators.add(it)
        return _flatten(it)

    def shutdown(self) -> None:
        """Stop handing out tasks, finish or skip those in flight, then close and join the pool."""
        if self._closed:
            return
        self._closed = True
        self._stop_flag.value = 1
        for it in list(self._iterators):
            if not _drain(it):
                logger.warning("A pool worker did not return within %.0f s while shutting down; terminating.", _DRAIN_TIMEOUT)
                self.terminate()
                return
        self._iterators.clear()
        self._pool.close()
        self._pool.join()

    def terminate(self) -> None:
        """Kill the workers at once (``multiprocessing.pool.Pool.terminate``); see the class docstring for the risk."""
        self._closed = True
        self._stop_flag.value = 1
        self._iterators.clear()
        self._pool.terminate()
        self._pool.join()

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None and not issubclass(exc_type, Exception):
            self.terminate()
        else:
            self.shutdown()


def _drain(it: "IMapIterator[Any]") -> bool:
    """Consume what is left of ``it``; ``False`` if a result takes longer than ``_DRAIN_TIMEOUT``."""
    while True:
        try:
            it.next(timeout=_DRAIN_TIMEOUT)
        except StopIteration:
            return True
        except mp.TimeoutError:
            return False
        except Exception:
            continue  # a task that raised; its error is not needed during shutdown


def worker_pool(
    processes: int | None = None,
    initializer: Callable[..., object] | None = None,
    initargs: Iterable[Any] = (),
) -> WorkerPool:
    """Create a :class:`WorkerPool` with the context from :func:`get_mp_context`."""
    return WorkerPool(processes, initializer=initializer, initargs=initargs)


def process_aware_cpu_count() -> int | None:
    """Return process CPU count when available, else system CPU count."""
    fn = getattr(os, "process_cpu_count", None)
    if callable(fn):
        count = fn()
        return count if isinstance(count, int) else None
    return os.cpu_count()


T = TypeVar("T")


Q = TypeVar("Q", bound=Sized)


def _deque_pop(q: deque[object]) -> object:
    return q.pop()


def _deque_append(q: deque[object], x: object) -> None:
    q.append(x)


def _new_deque() -> deque[object]:
    return deque()


class NonBlockingQueue(Generic[T, Q]):
    """Plain queue; ``pop`` raises ``IndexError``/``KeyError`` when empty, which ends iteration."""

    def __init__(
        self,
        queue_class: Callable[[], Q] = _new_deque,
        *,
        pop: Callable[[Q], T] = _deque_pop,
        push: Callable[[Q, T], None] = _deque_append,
        push_with_priority: Callable[[Q, T, float], None] | None = None,
    ) -> None:
        """Initialize a non-blocking queue.

        Args:
            queue_class: The underlying container class (e.g., deque, list).
                Defaults to deque.
            pop: Function to pop an element from the queue. Defaults to deque.pop().
            push: Function to push an element to the queue. Defaults to deque.append().
        """
        self.deque: Q = queue_class()
        self._pop_element: Callable[[Q], T] = pop
        self._push_element: Callable[[Q, T], None] = push
        self._push_with_priority: Callable[[Q, T, float], None] | None = push_with_priority

        self.closed: bool = False

    def __iter__(self) -> Iterator[T]:
        while True:
            try:
                task = self.pop()
            except (IndexError, KeyError):
                # Some queue backends (e.g. list/deque) raise IndexError when empty,
                # while others (e.g. UpdatablePriorityQueue) raise KeyError.
                return
            yield task

    def pop(self) -> T:
        """Remove and return the next element."""
        return self._pop_element(self.deque)

    def push(self, element: T, priority: float | None = None) -> None:
        """Add an element; ``priority`` is used only if the queue was built with ``push_with_priority``."""
        if priority is None or self._push_with_priority is None:
            self._push_element(self.deque, element)
        else:
            self._push_with_priority(self.deque, element, priority)

    def close(self) -> None:
        """Mark the queue as closed."""
        self.closed = True

    def __len__(self) -> int:
        return len(self.deque)


class BlockingQueue(Generic[T, Q]):
    """Thread-safe queue whose ``pop`` waits for new elements while empty, until the queue is closed."""

    def __init__(
        self,
        queue_class: Callable[[], Q] = _new_deque,
        *,
        pop: Callable[[Q], T] = _deque_pop,
        push: Callable[[Q, T], None] = _deque_append,
        push_with_priority: Callable[[Q, T, float], None] | None = None,
    ) -> None:
        """Create a blocking queue.

        Args:
            queue_class: The underlying container class (e.g., deque, list).
                Defaults to deque.
            pop: Function to pop an element from the queue. Defaults to deque.pop().
            push: Function to push an element to the queue. Defaults to deque.append().

        Note:
            pop and push can both be functions with kwargs; the corresponding
            methods in this class will pass their arguments along.
        """
        self.deque: Q = queue_class()
        self._pop_element: Callable[[Q], T] = pop
        self._push_element: Callable[[Q, T], None] = push
        self._push_with_priority: Callable[[Q, T, float], None] | None = push_with_priority

        self.lock: Condition = Condition()
        self.closed: bool = False

    def __iter__(self) -> Iterator[T]:
        while True:
            try:
                task = self.pop()
            except (IndexError, KeyError):
                return
            yield task

    def pop(self) -> T:
        """Remove and return the next element, waiting if empty; raises ``IndexError`` once closed and drained."""
        with self.lock:
            while len(self.deque) == 0 and not self.closed:
                self.lock.wait(timeout=cfg.advanced.BLOCKING_QUEUE_WAIT_TIMEOUT)
            if self.closed and len(self.deque) == 0:
                raise IndexError("Queue closed")
            return self._pop_element(self.deque)

    def push(self, element: T, priority: float | None = None) -> None:
        """Add an element; ``priority`` is used only if the queue was built with ``push_with_priority``."""
        with self.lock:
            if priority is None or self._push_with_priority is None:
                self._push_element(self.deque, element)
            else:
                self._push_with_priority(self.deque, element, priority)
            self.lock.notify()

    def close(self) -> None:
        """Mark the queue as closed and wake all waiting consumers."""
        with self.lock:
            self.closed = True
            self.lock.notify_all()

    def __len__(self) -> int:
        with self.lock:
            return len(self.deque)


class UpdatablePriorityQueue:
    """Priority queue that supports updating task priorities and removing tasks.

    Tasks are hashable objects. The full task object is used as the identity
    key for updates: pushing a task that is equal to an existing task replaces
    the previous entry.
    Lower priority value means higher priority.

    Based on the heapq implementation from Python docs.
    Reference: https://docs.python.org/3/library/heapq.html
    """

    REMOVED: Hashable = "<removed>"  # placeholder for a removed task (must be hashable for heap entry typing)
    _EntryItem = float | int | Hashable
    _Entry = list[_EntryItem]

    def __init__(self) -> None:
        self.pq: list[UpdatablePriorityQueue._Entry] = []  # list of entries arranged in a heap
        self.entry_finder: dict[Hashable, UpdatablePriorityQueue._Entry] = {}  # mapping of task -> entry
        self.counter: int = 0

    def push(self, task: Hashable, priority: float = 0) -> None:
        """Add a new task or update the priority of an existing task.

        Args:
            task: A hashable task object. Equal tasks are considered the same
                for updates.
            priority: The priority value (lower = higher priority). Defaults to 0.
        """
        if task in self.entry_finder:
            self.remove_task(task)
        entry = [priority, self.counter, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)
        self.counter += 1

    def remove_task(self, task: Hashable) -> None:
        """Mark an existing task as REMOVED. Raise KeyError if not found.

        Args:
            task: The full task object to remove.
        """
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop(self) -> Hashable:
        """Remove and return the lowest-priority task.

        Returns:
            The full task object.

        Raises:
            KeyError: If the queue is empty.
        """
        while self.pq:
            _, _, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError("pop from an empty priority queue")

    def __len__(self) -> int:
        return len(self.entry_finder)
