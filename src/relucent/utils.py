"""Utility functions and data structures for relucent.

Provides RNG seeding, sign-sequence encoding, a cached Gurobi environment,
thread-safe queue implementations, and network manipulation helpers used
throughout the package.
"""

import multiprocessing as mp
import os
import random
import sys
from collections import OrderedDict, deque
from collections.abc import Callable, Hashable, Iterable, Iterator, Sized
from heapq import heappop, heappush
from multiprocessing.context import BaseContext
from threading import Condition
from typing import Generic, TypeVar

import numpy as np
import torch
import torch.nn as nn
from gurobipy import Env, disposeDefaultEnv

import relucent.config as cfg
from relucent.model import NN

__all__ = [
    "BlockingQueue",
    "NonBlockingQueue",
    "UpdatablePriorityQueue",
    "close_env",
    "encode_ss",
    "get_env",
    "get_mp_context",
    "mlp",
    "normalize_weights",
    "process_aware_cpu_count",
    "set_seeds",
    "split_sequential",
]

_env: Env | None = None
_default_env_disposed: bool = False


def mlp(widths: Iterable[int], add_last_relu: bool = False) -> NN:
    """Create a standard fully connected ReLU MLP wrapped as :class:`NN`.

    Args:
        widths: Layer widths including input and output widths.
            For example, ``[2, 10, 5, 1]``.
        add_last_relu: If ``True``, append a ReLU after the final Linear layer.

    Returns:
        An :class:`NN` instance with named Linear/ReLU layers.
    """
    widths = list(widths)
    layers: list[tuple[str, nn.Module]] = []
    for i in range(len(widths) - 1):
        layers.append((f"fc{i}", nn.Linear(widths[i], widths[i + 1])))
        if i < len(widths) - 2 or add_last_relu:
            layers.append((f"relu{i}", nn.ReLU()))
    net = NN(layers=OrderedDict(layers))
    object.__setattr__(net, "widths", widths)
    return net


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


def set_seeds(seed: int) -> None:
    """Set all RNG seeds to a given value.

    Args:
        seed: Integer seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def process_aware_cpu_count() -> int | None:
    """Return process CPU count when available, else system CPU count."""
    fn = getattr(os, "process_cpu_count", None)
    if callable(fn):
        count = fn()
        return count if isinstance(count, int) else None
    return os.cpu_count()


def encode_ss(ss: np.ndarray | torch.Tensor) -> bytes:
    """Create a hashable representation of a sign sequence.

    Converts a sign sequence array into a bytes object that can be used as a
    dictionary key or for hashing.

    The sign sequence is always encoded using an integer dtype so that float
    and integer representations with the same logical values (in {-1, 0, 1})
    produce identical tags.

    Args:
        ss: A sign sequence as np.ndarray or torch.Tensor with values in {-1, 0, 1}.

    Returns:
        bytes: A hashable bytes representation of the flattened sign sequence.
    """
    ss = ss.detach().cpu().numpy() if isinstance(ss, torch.Tensor) else np.asarray(ss)

    ss = ss.astype(np.int8, copy=False)
    return ss.ravel().tobytes()


def get_env(num_threads: int | None = None) -> Env:
    """Get a cached Gurobi environment.

    Creates and caches a Gurobi environment with logging disabled. This avoids
    the overhead of creating multiple environments. For more control over the
    environment, create and pass one directly to functions that need it.

    If ``num_threads`` is provided, the environment's ``Threads`` parameter is
    set accordingly on first creation. Subsequent calls ignore ``num_threads``
    and return the cached environment.

    Args:
        num_threads: Optional limit on the number of threads Gurobi may use.
            If None, Gurobi uses its default threading behavior.

    Returns:
        gurobipy.Env: A Gurobi environment with logging disabled.
    """
    global _env, _default_env_disposed
    if _env is not None:
        return _env
    if not _default_env_disposed:
        disposeDefaultEnv()
        _default_env_disposed = True
    _env = Env(logfilename="", empty=True)
    if num_threads is not None:
        _env.setParam("Threads", num_threads)
    _env.setParam("OutputFlag", 0)
    _env.setParam("LogToConsole", 0)
    _env.start()
    return _env


def close_env() -> None:
    """Close the cached Gurobi environment."""
    global _env
    if _env is None:
        return
    _env.close()
    _env = None


T = TypeVar("T")
Q = TypeVar("Q", bound=Sized)


def _deque_pop(q: deque[object]) -> object:
    return q.pop()


def _deque_append(q: deque[object], x: object) -> None:
    q.append(x)


def _new_deque() -> deque[object]:
    return deque()


class NonBlockingQueue(Generic[T, Q]):
    """Just a normal queue"""

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
        return self._pop_element(self.deque)

    def push(self, element: T, priority: float | None = None) -> None:
        if priority is None or self._push_with_priority is None:
            self._push_element(self.deque, element)
        else:
            self._push_with_priority(self.deque, element, priority)

    def close(self) -> None:
        self.closed = True

    def __len__(self) -> int:
        return len(self.deque)


class BlockingQueue(Generic[T, Q]):
    """Queue that patiently waits for new elements if you pop() while it's empty"""

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
        with self.lock:
            while len(self.deque) == 0 and not self.closed:
                self.lock.wait(timeout=cfg.BLOCKING_QUEUE_WAIT_TIMEOUT)
            if self.closed and len(self.deque) == 0:
                raise IndexError("Queue closed")
            return self._pop_element(self.deque)

    def push(self, element: T, priority: float | None = None) -> None:
        with self.lock:
            if priority is None or self._push_with_priority is None:
                self._push_element(self.deque, element)
            else:
                self._push_with_priority(self.deque, element, priority)
            self.lock.notify()

    def close(self) -> None:
        self.closed = True
        with self.lock:
            self.lock.notify()

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


def split_sequential(model: NN, split_layer: str) -> tuple[NN, NN]:
    """Split a neural network into two sequential parts.

    Creates two separate NN objects by splitting the model at a specified layer.
    The first network contains layers up to and including split_layer, and the
    second contains all subsequent layers.

    Args:
        model: The NN object to split.
        split_layer: Name of the layer at which to split (this layer goes to
            the first network).

    Returns:
        tuple: (nn1, nn2) where nn1 contains layers up to split_layer and
            nn2 contains the remaining layers.
    """
    layers1, layers2 = OrderedDict(), OrderedDict()
    current_layers = layers1
    for name, layer in model.layers.items():
        current_layers[name] = layer
        if name == split_layer:
            current_layers = layers2
    nn1 = NN(layers1, input_shape=model.input_shape, device=model.device, dtype=model.dtype)
    nn2 = NN(
        layers2,
        input_shape=nn1(torch.zeros((1,) + model.input_shape, device=model.device, dtype=model.dtype)).squeeze().shape,
        device=model.device,
        dtype=model.dtype,
    )
    return nn1, nn2


def normalize_weights(model: NN) -> NN:
    """Normalize hidden neuron weights to unit norm without changing the network function.

    The incoming weights (and biases) of each Linear layer except the last one are rescaled so that each
    neuron's weight vector has unit ℓ2 norm.

    Args:
        model: The NN object whose weights should be normalized in-place.

    Returns:
        The same NN object with normalized hidden-layer weights.

    Raises:
        ValueError: If the network contains layers other than Linear or ReLU.
    """
    layers = list(model.layers.values())

    # Ensure only supported layer types are present.
    for layer in layers:
        if not isinstance(layer, (torch.nn.Linear, torch.nn.ReLU)):
            raise ValueError(f"Unsupported layer type: {type(layer)}")

    # Indices of all Linear layers in order.
    linear_indices = [i for i, layer in enumerate(layers) if isinstance(layer, torch.nn.Linear)]

    for idx, lin_idx in enumerate(linear_indices):
        layer = layers[lin_idx]

        # Do not modify the final Linear layer
        is_last_linear = idx == len(linear_indices) - 1
        if is_last_linear:
            continue

        assert isinstance(layer.weight.data, torch.Tensor)
        assert isinstance(layer.bias.data, torch.Tensor)

        w = layer.weight.data
        norms = w.norm(dim=1, keepdim=True)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))

        layer.weight.data = w / safe_norms
        if layer.bias is not None:
            layer.bias.data = layer.bias.data / safe_norms.squeeze(1)

        next_linear = layers[linear_indices[idx + 1]]
        assert isinstance(next_linear.weight.data, torch.Tensor)

        next_w = next_linear.weight.data  # shape (out_next, out_current)
        scale = safe_norms.squeeze(1)  # shape (out_current,)
        next_linear.weight.data = next_w * scale

    return model
