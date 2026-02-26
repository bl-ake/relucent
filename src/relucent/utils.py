import random
from collections import OrderedDict, deque
from heapq import heappop, heappush
from threading import Condition
from typing import Any, Callable, Iterator, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from gurobipy import Env, disposeDefaultEnv
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from tqdm.auto import tqdm

from relucent.config import (
    BLOCKING_QUEUE_WAIT_TIMEOUT,
    DEFAULT_PYVIS_SAVE_FILE,
    MAX_IMAGES_PYVIS,
    MAX_NUM_EXAMPLES_PYVIS,
    PIE_LABEL_DISTANCE,
)
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


def encode_ss(ss: np.ndarray | torch.Tensor) -> bytes:
    """Create a hashable representation of a sign sequence.

    Converts a sign sequence array into a bytes object that can be used as a
    dictionary key or for hashing.

    Args:
        ss: A sign sequence as np.ndarray or torch.Tensor with values in {-1, 0, 1}.

    Returns:
        bytes: A hashable bytes representation of the flattened sign sequence.
    """
    if isinstance(ss, torch.Tensor):
        ss = ss.detach().cpu().numpy()
    return ss.flatten().tobytes()


_env: Env | None = None


def get_env() -> Env:
    """Get a cached Gurobi environment.

    Creates and caches a Gurobi environment with logging disabled. This avoids
    the overhead of creating multiple environments. For more control over the
    environment, create and pass one directly to functions that need it.

    Returns:
        gurobipy.Env: A Gurobi environment with logging disabled.
    """
    global _env
    if _env is not None:
        return _env
    _env = Env(logfilename="", empty=True)
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


def get_colors(data: Sequence[float], cmap: str = "viridis", **kwargs: Any) -> list[str]:
    """Map numeric values to hex color strings via a colormap.

    Values are normalized to [0, 1] over the input range, then mapped through
    the given colormap.

    Args:
        data: Sequence of numeric values.
        cmap: Matplotlib colormap name. Defaults to "viridis".
        **kwargs: Passed to the colormap callable.

    Returns:
        list[str]: Hex color strings (e.g. "#rrggbb") in the same order as data.
    """
    if not data:
        return []
    a = np.asarray(data)
    a = a - np.min(a)
    am = np.max(a)
    a = a / (am if am > 0 else 1)
    a = colormaps[cmap](a)
    a = (a * 255).astype(int)
    return [f"#{x[0]:02x}{x[1]:02x}{x[2]:02x}" for x in a]


def data_graph(
    node_df: Any,
    edge_df: Any,
    dataset: Any | None = None,
    draw_function: Callable[..., Any] = lambda x, **__: x,
    class_labels: bool | None = True,
    node_title_formatter: Callable[[int, Mapping[str, Any]], str] = lambda i, row: (
        row["title"] if "title" in row else str(row)
    ),
    node_label_formatter: Callable[[int, Mapping[str, Any]], str] = lambda i, row: (
        row["label"] if "label" in row else str(i)
    ),
    node_size_formatter: Callable[[Mapping[str, Any]], int] = lambda row: row["size"] if "size" in row else 10,
    edge_title_formatter: Callable[[Mapping[str, Any]], str] = lambda row: row["title"] if "title" in row else "",
    edge_label_formatter: Callable[[Mapping[str, Any]], str] = lambda row: row["label"] if "label" in row else "",
    edge_value_formatter: Callable[[Mapping[str, Any]], float | int] = lambda row: (
        row["value"] if "value" in row else 1
    ),
    max_images: int = MAX_IMAGES_PYVIS,
    max_num_examples: int = MAX_NUM_EXAMPLES_PYVIS,
    save_file: str = DEFAULT_PYVIS_SAVE_FILE,
) -> None:
    """Create an interactive pyvis graph from dataframes of nodes and edges.

    Creates a visual graph representation where nodes can contain images of
    data examples. Useful for visualizing relationships in datasets or
    polyhedral complexes.

    Args:
        node_df: DataFrame with node information. Each row should have 'data'
            (list of examples) and optionally 'title', 'label', 'size', etc.
        edge_df: DataFrame with edge information. Index should be (node1, node2)
            tuples, and rows can have 'title', 'label', 'value', etc.
        dataset: Optional dataset object for extracting class labels. Defaults to None.
        draw_function: Function to draw individual data examples. Should accept
            ``data`` and ``ax``. Defaults to identity (pass-through).
        class_labels: If True and dataset is provided, shows class proportions
            as pie charts. Defaults to True.
        node_title_formatter: Function to format node titles. Defaults to using
            'title' column or string representation.
        node_label_formatter: Function to format node labels. Defaults to using
            'label' column or index.
        node_size_formatter: Function to determine node sizes. Defaults to using
            'size' column or 10.
        edge_title_formatter: Function to format edge titles. Defaults to using
            'title' column or empty string.
        edge_label_formatter: Function to format edge labels. Defaults to using
            'label' column or empty string.
        edge_value_formatter: Function to determine edge values/weights.
            Defaults to using 'value' column or 1.
        max_images: Maximum number of node images to generate. Defaults to 3000.
        max_num_examples: Maximum number of data examples to show per node.
            Defaults to 3.
        save_file: Path to save the HTML graph file. Defaults to "./graph.html".
    """
    from pyvis.network import Network

    if class_labels is True and dataset is not None:
        class_labels_list = torch.unique(torch.tensor([dataset[i][1] for i in range(len(dataset))])).tolist()

    G = nx.Graph()
    bar = tqdm(node_df.iterrows(), total=len(node_df), desc="Adding Nodes")
    for i, row in bar:
        if i < max_images:
            num_examples = min(len(row["data"]), max_num_examples) + (class_labels is not False)
            num_rows = np.ceil(np.sqrt(num_examples)).astype(int)
            num_cols = num_examples // num_rows
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
            axs = axs.flatten() if isinstance(axs, np.ndarray) and num_rows > 1 else [axs]
            for j, ax in enumerate(axs[:-1]):
                ax.axis("equal")
                ax.set_axis_off()
                if j <= num_examples:
                    data = row["data"][j]
                    draw_function(data=data, ax=ax)

            if class_labels and "class_proportions" in row:
                axs[-1].pie(
                    row["class_proportions"],
                    labeldistance=PIE_LABEL_DISTANCE,
                    labels=class_labels_list,
                )
            axs[-1].axis("equal")
            axs[-1].set_axis_off()

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            img = Image.frombytes("RGBA", canvas.get_width_height(), bytes(canvas.buffer_rgba()))
            plt.close(fig)
            img.convert("RGB").save(f"images/{i}.png")

        G.add_node(
            i,
            title=node_title_formatter(i, row),
            label=node_label_formatter(i, row),
            image=f"images/{i}.png",
            shape="image",
            size=node_size_formatter(row),  # 10 * (np.log(row["count"]) + 3)
            **{k: str(v) for k, v in row.items() if k not in ["label", "title", "size", "image", "data"]},
        )
    pbar = tqdm(edge_df.iterrows(), total=len(edge_df), desc="Adding Edges")
    for (A, B), row in pbar:
        G.add_edge(
            A,
            B,
            title=edge_title_formatter(row),
            label=edge_label_formatter(row),
            value=edge_value_formatter(row),
        )
        bar.set_postfix({"Nodes": G.number_of_nodes(), "Edges": G.number_of_edges()})
    print(f"Number of Nodes: {G.number_of_nodes()}\nNumber of Edges: {G.number_of_edges()}")

    nt = Network(height="1000px", width="100%")
    nt.from_nx(G)
    nt.show_buttons()
    # layout = nx.spring_layout(G)
    # nt.repulsion(node_distance=300, central_gravity=0.2, spring_length=200, spring_strength=0.05)
    nt.toggle_physics(False)
    nt.save_graph(save_file)
