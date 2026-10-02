"""Multiprocessing context, CPU count, and the thread-safe queues the search uses."""

import multiprocessing as mp
import os
import sys
from collections import deque
from collections.abc import Callable, Hashable, Iterator, Sized
from heapq import heappop, heappush
from multiprocessing.context import BaseContext
from threading import Condition
from typing import Generic, TypeVar

import relucent.config as cfg

__all__ = ["get_mp_context", "process_aware_cpu_count", "NonBlockingQueue", "BlockingQueue", "UpdatablePriorityQueue"]


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
