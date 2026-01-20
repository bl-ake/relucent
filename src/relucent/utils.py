from collections import OrderedDict, deque
from functools import cache
from threading import Condition

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from gurobipy import Env, disposeDefaultEnv
from matplotlib import colormaps
from PIL import Image
from tqdm.auto import tqdm

from relucent.model import NN
import random

disposeDefaultEnv()


def set_seeds(seed):
    """Set all RNG seeds to a given value.

    Args:
        seed: Integer seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@cache
def get_env():
    """Get a cached Gurobi environment.

    Creates and caches a Gurobi environment with logging disabled. This avoids
    the overhead of creating multiple environments. For more control over the
    environment, create and pass one directly to functions that need it.

    Returns:
        gurobipy.Env: A Gurobi environment with logging disabled.
    """
    env = Env(logfilename="", empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("LogToConsole", 0)
    env.start()
    return env


class NonBlockingQueue:
    """Just a normal queue"""

    stopFlag = "<stop>"

    def __init__(self, queue_class=deque, pop=lambda q: q.pop(), push=lambda q, x: q.append(x)):
        """Initialize a non-blocking queue.

        Args:
            queue_class: The underlying container class (e.g., deque, list).
                Defaults to deque.
            pop: Function to pop an element from the queue. Defaults to deque.pop().
            push: Function to push an element to the queue. Defaults to deque.append().
        """
        self.deque = queue_class()
        self.pop_element = pop
        self.push_element = push

        self.closed = False

    def __iter__(self):
        while True:
            task = self.pop()
            if task == self.stopFlag:
                return
            yield task

    def pop(self, *args, **kwargs):
        return self.pop_element(self.deque, *args, **kwargs)

    def push(self, element, *args, **kwargs):
        self.push_element(self.deque, element, *args, **kwargs)

    def close(self):
        self.closed = True
        self.pop = lambda q: self.stopFlag

    def __len__(self):
        return -1 if self.closed else len(self.deque)


class BlockingQueue:
    """Queue that patiently waits for new elements if you pop() while it's empty"""

    stopFlag = "<stop>"

    def __init__(self, queue_class=deque, pop=lambda q: q.pop(), push=lambda q, x: q.append(x)):
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
        self.deque = queue_class()
        self.pop_element = pop
        self.push_element = push

        self.lock = Condition()
        self.closed = False

    def __iter__(self):
        while True:
            task = self.pop()
            if task == self.stopFlag:
                return
            yield task

    def pop(self, *args, **kwargs):
        with self.lock:
            while len(self.deque) == 0 and not self.closed:
                self.lock.wait(timeout=0.5)
            return self.pop_element(self.deque, *args, **kwargs)

    def push(self, element, *args, **kwargs):
        with self.lock:
            self.push_element(self.deque, element, *args, **kwargs)
            self.lock.notify()

    def close(self):
        self.closed = True
        with self.lock:
            self.pop = lambda q: self.stopFlag
            self.lock.notify()

    def __len__(self):
        with self.lock:
            return -1 if self.closed else len(self.deque)


def split_sequential(model, split_layer):
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


def get_colors(data, cmap="viridis", **kwargs):
    """Map some numbers to some colors"""
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
    node_df,
    edge_df,
    dataset=None,
    draw_function=None,
    class_labels=True,
    node_title_formatter=lambda i, row: row["title"] if "title" in row else str(row),
    node_label_formatter=lambda i, row: row["label"] if "label" in row else str(i),
    node_size_formatter=lambda row: row["size"] if "size" in row else 10,
    edge_title_formatter=lambda row: row["title"] if "title" in row else "",
    edge_label_formatter=lambda row: row["label"] if "label" in row else "",
    edge_value_formatter=lambda row: row["value"] if "value" in row else 1,
    max_images=3000,
    max_num_examples=3,
    save_file="./graph.html",
):
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
            'data' and 'ax' parameters. Defaults to None.
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
        class_labels = torch.unique(torch.tensor([dataset[i][1] for i in range(len(dataset))])).tolist()

    G = nx.Graph()
    bar = tqdm(node_df.iterrows(), total=len(node_df), desc="Adding Nodes")
    for i, row in bar:
        if i < max_images:
            num_examples = min(len(row["data"]), max_num_examples) + (class_labels is not False)
            num_rows = np.ceil(np.sqrt(num_examples)).astype(int)
            num_cols = num_examples // num_rows
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
            axs = axs.flatten() if num_rows > 1 else [axs]
            for j, ax in enumerate(axs[:-1]):
                ax.axis("equal")
                ax.set_axis_off()
                if j <= num_examples:
                    data = row["data"][j]
                    draw_function(data=data, ax=ax)

            if class_labels and "class_proportions" in row:
                axs[-1].pie(row["class_proportions"], labeldistance=0.6, labels=class_labels)
            axs[-1].axis("equal")
            axs[-1].set_axis_off()

            fig.canvas.draw()
            img = Image.frombytes("RGBa", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
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
