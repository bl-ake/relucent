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
from pyvis.network import Network
from tqdm.auto import tqdm

from .model import NN
import random

disposeDefaultEnv()


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@cache
def get_env():
    env = Env(logfilename="", empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("LogToConsole", 0)
    env.start()
    return env


class NonBlockingQueue:
    stopFlag = "<stop>"

    def __init__(self, queue_class=deque, pop=lambda q: q.pop(), push=lambda q, x: q.append(x)):
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
    ## Queue that patiently waits for new elements before popping
    stopFlag = "<stop>"

    def __init__(self, queue_class=deque, pop=lambda q: q.pop(), push=lambda q, x: q.append(x)):
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
    if not data:
        return []
    a = np.asarray(data)
    a = a - np.min(a)
    am = np.max(a)
    a = a / (am if am > 0 else 1)
    a = colormaps[cmap](a)
    a = (a * 255).astype(int)
    return [f"#{x[0]:02x}{x[1]:02x}{x[2]:02x}" for x in a]


def draw_image(data, ax):
    img = data.view(28, 28).cpu().numpy()
    ax.imshow(img, cmap="gray")


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

            # fig, ax = draw_function(data=dataset.data[row["indices"][0][0]])

            # fig, ax = plt.subplots()
            # plt.margins(0,0)
            # ax.pie(row["class_proportions"], labeldistance=.6, labels = list(range(dataset.num_classes)))
            # ax.set_box_aspect(1)
            # ax.set_axis_off()
            # fig.tight_layout()
            # plt.tight_layout(pad=0)

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
        # G.add_edge(mask_tuple, other_node, weight=bits_different)
        bar.set_postfix({"Nodes": G.number_of_nodes(), "Edges": G.number_of_edges()})
    # G = nx.relabel_nodes(G, {node: str(node) for node in G.nodes}, copy=False)
    print(f"Number of Nodes: {G.number_of_nodes()}\nNumber of Edges: {G.number_of_edges()}")

    nt = Network(height="1000px", width="100%")
    nt.from_nx(G)
    # nt.from_nx(G.subgraph(choices(list(G.nodes), k=300)))
    nt.show_buttons()
    # layout = nx.spring_layout(G)
    # for node in nt.nodes:
    #     node_id = node["id"]
    #     if node_id in layout:
    #         node["x"], node["y"] = layout[node_id][0]*1000, layout[node_id][1]*1000
    # nt.repulsion(node_distance=300, central_gravity=0.2, spring_length=200, spring_strength=0.05)
    nt.toggle_physics(False)
    nt.save_graph(save_file)
