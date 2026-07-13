"""Dual-graph contraction and network surgery helpers for :class:`~relucent.core.complex.Complex`."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

import networkx as nx
import numpy as np

from relucent._internal.torch_compat import TORCH_AVAILABLE, torch
from relucent.model.model import Layer, LinearLayer, ReLULayer, ReLUNetwork

__all__ = [
    "contract_dual_graph_for_shi",
    "delete_ss_columns",
    "net_remove_ss_layer_and_following_relu",
    "net_without_last_ss_layer_neuron",
]


def delete_ss_columns(ss: np.ndarray | torch.Tensor, deleted_shis: Iterable[int]) -> np.ndarray | torch.Tensor:
    axis = int(ss.ndim - 1)
    for shi in sorted(set(int(s) for s in deleted_shis), reverse=True):
        if isinstance(ss, np.ndarray):
            ss = np.delete(ss, shi, axis=axis)
        elif TORCH_AVAILABLE and isinstance(ss, torch.Tensor):
            keep = [i for i in range(ss.shape[axis]) if i != shi]
            ss = ss.index_select(axis, torch.tensor(keep, device=ss.device))
        else:
            raise TypeError(f"Unsupported ss type: {type(ss)}")
    return ss


def contract_dual_graph_for_shi(
    graph: nx.Graph[int],
    deleted_shi: int,
) -> tuple[nx.Graph[int], dict[int, int]]:
    """Quotient a relabeled dual graph by edges with ``shi == deleted_shi``.

    Returns the contracted graph (nodes ``0 .. n-1``) and a map from each new
    node to a representative old node id.
    """
    parent = {node: node for node in graph.nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for u, v, data in graph.edges(data=True):
        if data.get("shi") == deleted_shi:
            union(u, v)

    roots = sorted({find(node) for node in graph.nodes})
    root_to_new = {root: i for i, root in enumerate(roots)}
    old_representative = {root_to_new[root]: root for root in roots}
    node_map = {node: root_to_new[find(node)] for node in graph.nodes}

    contracted = nx.Graph()
    contracted.add_nodes_from(range(len(roots)))
    for u, v, data in graph.edges(data=True):
        shi = data.get("shi")
        if shi == deleted_shi:
            continue
        a, b = node_map[u], node_map[v]
        if a == b:
            continue
        if shi is None:
            continue
        new_shi = shi - 1 if shi > deleted_shi else shi
        if contracted.has_edge(a, b):
            assert contracted.edges[a, b]["shi"] == new_shi
        else:
            contracted.add_edge(a, b, shi=new_shi)
    return contracted, old_representative


def net_remove_ss_layer_and_following_relu(net: ReLUNetwork, ss_layer_idx: int) -> ReLUNetwork:
    """Remove a width-1 linear layer and the ReLU immediately after it."""
    items = list(net.layers.items())
    relu_idx = ss_layer_idx + 1
    if relu_idx >= len(items) or not isinstance(items[relu_idx][1], ReLULayer):
        raise ValueError(f"Layer index {ss_layer_idx} is not immediately followed by a ReLU.")
    removed_linear = items[ss_layer_idx][1]
    if not isinstance(removed_linear, LinearLayer):
        raise ValueError(f"Layer index {ss_layer_idx} is not a LinearLayer.")
    n_removed_outputs = int(removed_linear.weight.shape[0])

    new_items: list[tuple[str, Layer]] = []
    skip_next_relu = False
    for i, (name, layer) in enumerate(items):
        if i == ss_layer_idx:
            skip_next_relu = True
            continue
        if skip_next_relu and i == relu_idx:
            skip_next_relu = False
            continue
        if skip_next_relu:
            raise RuntimeError("Expected ReLU immediately after removed linear layer.")
        if i == relu_idx + 1 and isinstance(layer, LinearLayer):
            weight = np.delete(layer.weight, np.arange(n_removed_outputs), axis=1)
            new_items.append((name, LinearLayer(weight=weight, bias=layer.bias)))
        else:
            new_items.append((name, layer))
    return ReLUNetwork(OrderedDict(new_items), input_shape=net.input_shape)


def net_without_last_ss_layer_neuron(
    net: ReLUNetwork,
    ss_layer_idx: int,
    neuron_idx: int,
) -> ReLUNetwork:
    """Return a copy of ``net`` with one neuron removed from the given ReLU hidden layer."""
    layer = list(net.layers.values())[ss_layer_idx]
    if not isinstance(layer, LinearLayer):
        raise ValueError(f"Layer index {ss_layer_idx} is not a LinearLayer.")
    if int(layer.weight.shape[0]) == 1:
        return net_remove_ss_layer_and_following_relu(net, ss_layer_idx)

    items = list(net.layers.items())
    new_items: list[tuple[str, Layer]] = []
    delete_column_from_next_linear = False
    for i, (name, layer) in enumerate(items):
        if i == ss_layer_idx and isinstance(layer, LinearLayer):
            weight = np.delete(layer.weight, neuron_idx, axis=0)
            bias = np.delete(layer.bias, neuron_idx, axis=1)
            new_items.append((name, LinearLayer(weight=weight, bias=bias)))
            delete_column_from_next_linear = True
        elif delete_column_from_next_linear and isinstance(layer, LinearLayer):
            weight = np.delete(layer.weight, neuron_idx, axis=1)
            new_items.append((name, LinearLayer(weight=weight, bias=layer.bias)))
            delete_column_from_next_linear = False
        else:
            new_items.append((name, layer))
    return ReLUNetwork(OrderedDict(new_items), input_shape=net.input_shape)
