import os
import random
from collections import OrderedDict, deque
from heapq import heappop, heappush
from threading import Condition
from typing import Any, Callable, Iterator

import numpy as np
import torch
from gurobipy import Env, disposeDefaultEnv

from relucent.config import BLOCKING_QUEUE_WAIT_TIMEOUT
from relucent.model import NN

disposeDefaultEnv()


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
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if callable(process_cpu_count):
        return process_cpu_count()
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
    if isinstance(ss, torch.Tensor):
        ss = ss.detach().cpu().numpy()
    else:
        ss = np.asarray(ss)

    if not np.issubdtype(ss.dtype, np.integer):
        ss = ss.astype(np.int8, copy=False)
    else:
        ss = ss.astype(np.int8, copy=False)

    return ss.ravel().tobytes()


_env: Env | None = None


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
    global _env
    if _env is not None:
        return _env
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


class NonBlockingQueue:
    """Just a normal queue"""

    stopFlag = "<stop>"

    def __init__(
        self,
        queue_class: Callable[[], Any] = deque,
        pop: Callable[..., Any] = lambda q: q.pop(),
        push: Callable[..., Any] = lambda q, x: q.append(x),
    ) -> None:
        """Initialize a non-blocking queue.

        Args:
            queue_class: The underlying container class (e.g., deque, list).
                Defaults to deque.
            pop: Function to pop an element from the queue. Defaults to deque.pop().
            push: Function to push an element to the queue. Defaults to deque.append().
        """
        self.deque: Any = queue_class()
        self.pop_element: Callable[..., Any] = pop
        self.push_element: Callable[..., Any] = push

        self.closed: bool = False

    def __iter__(self) -> Iterator[Any]:
        while True:
            task = self.pop()
            if task == self.stopFlag:
                return
            yield task

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        return self.pop_element(self.deque, *args, **kwargs)

    def push(self, element: Any, *args: Any, **kwargs: Any) -> None:
        self.push_element(self.deque, element, *args, **kwargs)

    def close(self) -> None:
        self.closed = True
        self.pop = lambda q: self.stopFlag

    def __len__(self) -> int:
        return len(self.deque)


class BlockingQueue:
    """Queue that patiently waits for new elements if you pop() while it's empty"""

    stopFlag = "<stop>"

    def __init__(
        self,
        queue_class: Callable[[], Any] = deque,
        pop: Callable[..., Any] = lambda q: q.pop(),
        push: Callable[..., Any] = lambda q, x: q.append(x),
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
        self.deque: Any = queue_class()
        self.pop_element: Callable[..., Any] = pop
        self.push_element: Callable[..., Any] = push

        self.lock: Condition = Condition()
        self.closed: bool = False

    def __iter__(self) -> Iterator[Any]:
        while True:
            task = self.pop()
            if task == self.stopFlag:
                return
            yield task

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            while len(self.deque) == 0 and not self.closed:
                self.lock.wait(timeout=BLOCKING_QUEUE_WAIT_TIMEOUT)
            return self.pop_element(self.deque, *args, **kwargs)

    def push(self, element: Any, *args: Any, **kwargs: Any) -> None:
        with self.lock:
            self.push_element(self.deque, element, *args, **kwargs)
            self.lock.notify()

    def close(self) -> None:
        self.closed = True
        with self.lock:
            self.pop = lambda q: self.stopFlag
            self.lock.notify()

    def __len__(self) -> int:
        with self.lock:
            return len(self.deque)


class UpdatablePriorityQueue:
    """Priority queue that supports updating task priorities and removing tasks.

    Tasks are tuples (head, *tail). The tail is used as the identity key for
    updates: pushing a task with the same tail replaces the previous entry.
    Lower priority value means higher priority.

    Based on the heapq implementation from Python docs.
    Reference: https://docs.python.org/3/library/heapq.html
    """

    REMOVED = "<removed-task>"  # placeholder for a removed task

    def __init__(self) -> None:
        self.pq: list[list[Any]] = []  # list of entries arranged in a heap
        self.entry_finder: dict[tuple[Any, ...], list[Any]] = {}  # mapping of tail -> entry
        self.counter: int = 0

    def push(self, task: tuple[Any, ...], priority: float = 0) -> None:
        """Add a new task or update the priority of an existing task.

        Args:
            task: A tuple (head, *tail). Tasks with the same tail are
                considered the same for updates.
            priority: The priority value (lower = higher priority). Defaults to 0.
        """
        head, *tail = task
        tail = tuple(tail)
        if tail in self.entry_finder:
            self.remove_task(tail)
        entry = [priority, self.counter, head, tail]
        self.entry_finder[tail] = entry
        heappush(self.pq, entry)
        self.counter += 1

    def remove_task(self, task_tail: tuple[Any, ...]) -> None:
        """Mark an existing task as REMOVED. Raise KeyError if not found.

        Args:
            task_tail: The tail of the task to remove (i.e. task[1:]).
        """
        entry = self.entry_finder.pop(task_tail)
        entry[-1] = self.REMOVED

    def pop(self) -> tuple[Any, ...]:
        """Remove and return the lowest-priority task.

        Returns:
            tuple: The full task (head, *tail).

        Raises:
            KeyError: If the queue is empty.
        """
        while self.pq:
            _, _, head, tail = heappop(self.pq)
            if tail is not self.REMOVED:
                del self.entry_finder[tail]
                return (head, *tail)
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
