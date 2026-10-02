"""Rebuilding a complex from its dual graph, and deleting a neuron from an explored complex.

:func:`recover_from_dual_graph` refills a complex from a stored dual graph (top cells plus
``shi`` edge labels). :func:`without_last_layer_neuron` contracts dual-graph edges on a
removed neuron's hyperplane, relabels sign sequences, shrinks the network, and rebuilds the
cells with :func:`recover_from_dual_graph`.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

import relucent.config as cfg
import relucent.verify.certify as certify
from relucent._internal.logging import progress
from relucent._internal.torch_compat import TORCH_AVAILABLE, torch
from relucent.model.model import Layer, LinearLayer, ReLULayer, ReLUNetwork
from relucent.utils import flip_ss_at_shi
from relucent.verify.certify import CertifyLevel

if TYPE_CHECKING:
    from relucent.core.complex import Complex

__all__ = [
    "contract_dual_graph_for_shi",
    "delete_ss_columns",
    "net_without_last_ss_layer_neuron",
    "recover_from_dual_graph",
    "without_last_layer_neuron",
]


def delete_ss_columns(ss: np.ndarray | torch.Tensor, deleted_shis: Iterable[int]) -> np.ndarray | torch.Tensor:
    """Drop sign-sequence columns for deleted supporting-hyperplane indices.

    Called from :meth:`~relucent.core.complex.Complex.without_last_layer_neuron` when seeding
    :meth:`~relucent.core.complex.Complex.recover_from_dual_graph` with the representative cell
    of each contracted dual-graph component.
    """
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

    Used by :meth:`~relucent.core.complex.Complex.without_last_layer_neuron` after
    :meth:`~relucent.core.complex.Complex.get_dual_graph` to merge top cells that shared the
    removed neuron's facet. Returns the contracted graph (nodes ``0 .. n-1``) and a map from
    each new node to a representative old node id.
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
    """Remove a width-1 linear layer and the ReLU immediately after it.

    Called from :func:`net_without_last_ss_layer_neuron` when the target hidden layer has a
    single neuron (the linear layer and its ReLU are removed entirely rather than one column).
    """
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
    """Return a copy of ``net`` with one neuron removed from the given ReLU hidden layer.

    Called from :meth:`~relucent.core.complex.Complex.without_last_layer_neuron` to build the
    smaller :class:`~relucent.core.complex.Complex` before dual-graph recovery.
    """
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


def recover_from_dual_graph(
    cplx: Complex,
    graph: nx.Graph[int],
    initial_ss: np.ndarray | torch.Tensor,
    source: int,
    copy: bool = False,
) -> None:
    """Add to ``cplx`` the top cells of a complex stored as its dual graph.

    Reconstructs polyhedra in the complex by traversing the adjacency graph
    of top-dimensional cells, using the supporting hyperplane indices stored
    on edges to determine how to flip sign sequence elements. This is useful
    for storing large complexes efficiently, as you only need to store the
    graph structure and SHI indices on edges rather than full polyhedron data.

    Args:
        cplx: The (empty) complex to fill, over the same network.
        graph: A networkx.Graph representing the dual graph. Edges must have
            a "shi" attribute indicating the supporting hyperplane index.
        initial_ss: The sign sequence of the starting polyhedron as
            torch.Tensor or np.ndarray.
        source: The node key in ``graph`` for the polyhedron with sign sequence initial_ss.
        copy: If True, operate on a copy of ``graph``; otherwise modify it in place.
            Defaults to False.

    Notes:
        Runs combinatorial certification (:class:`~relucent.verify.certify.CertifyLevel.COMBINATORIAL`)
        on the reconstructed complex and sets exploration state accordingly. Only
        recover graphs that were built from a complete, verified ambient search.
    """
    if copy:
        graph = graph.copy()
    initial_p = cplx.add_ss(initial_ss)
    graph.nodes[source]["poly"] = initial_p
    # ``nx.bfs_edges`` yields only the N-1 tree edges, so the progress bar
    # total must be in terms of nodes, not the total number of dual edges.
    for edge in progress(
        nx.bfs_edges(graph, source=source),
        desc="Recovering Polyhedra",
        total=graph.number_of_nodes() - 1,
    ):
        poly1, shi = graph.nodes[edge[0]]["poly"], graph.edges[edge]["shi"]
        if cfg.CAREFUL_MODE:
            assert poly1.ss_np.ravel()[shi] != 0
        poly2 = cplx.add_ss(flip_ss_at_shi(poly1.ss_np, shi), check_exists=False)
        graph.nodes[edge[1]]["poly"] = poly2

    # Populate each polyhedron's ``_shis`` from the full dual graph in a
    # single pass: iterating ``graph.edges(node)`` per-node would visit each
    # edge twice and also force a redundant ``cplx[...]`` SSManager lookup.
    shis_per_node: dict[Any, list[int]] = {n: [] for n in graph}
    for u, v, data in graph.edges(data=True):
        shi = data["shi"]
        shis_per_node[u].append(shi)
        shis_per_node[v].append(shi)
    for node, shis in shis_per_node.items():
        graph.nodes[node]["poly"]._shis = shis
        # Caches are written only for complete+verified complexes; trust SHIs on reload.
        graph.nodes[node]["poly"]._shis_strict = True
    # Dual-graph recovery reconstructs a previously explored complex, but still
    # certifies combinatorially (rebuilding + repairing the dual graph) rather
    # than blindly trusting the reconstruction.
    cplx.set_exploration_state(complete=True, verified=False)
    certify.certify_complex(cplx, level=CertifyLevel.COMBINATORIAL, repair=True, record_state=True)


def _deleted_shi_for_last_layer_neuron(cplx: Complex, neuron_idx: int) -> int:
    last_ss_layer = max(cplx.ss_layers)
    for shi, (layer_idx, (_, idx)) in enumerate(cplx.ssi2maski):
        if layer_idx == last_ss_layer and idx == neuron_idx:
            return shi
    raise RuntimeError(f"neuron_idx {neuron_idx} is not in the last ReLU hidden layer.")


def without_last_layer_neuron(cplx: Complex, neuron_idx: int) -> Complex:
    """Return the complex obtained by deleting a neuron from the last ReLU layer.

    The last ReLU layer is the final :class:`~relucent.model.model.LinearLayer` that is
    immediately followed by a ReLU in the canonical network (the same layer used
    by output-neuron boundary analysis).  Top-dimensional cells that shared a
    facet on the removed neuron are merged; recovered SHIs are the union of the
    two sides' facet indices, minus the removed neuron.

    If that layer has only one neuron, the linear layer and its following ReLU are
    removed from the network entirely.

    Implementation: contract dual-graph edges for the removed SHI, then rebuild
    cells with :func:`recover_from_dual_graph` on the smaller network.

    Args:
        neuron_idx: Index of the neuron within that last hidden linear layer
            (not the global supporting-hyperplane index).  Must be ``0`` when the
            layer has width ``1``.

    Returns:
        A new :class:`Complex` over the smaller network.  The dual graph is not
        copied.

    Raises:
        ValueError: If there is no ReLU hidden layer or ``neuron_idx`` is out of
            range for the last one.
    """
    if not cplx.ss_layers:
        raise ValueError("Network has no ReLU layers; cannot delete a neuron.")
    last_ss_layer = max(cplx.ss_layers)
    layer = list(cplx._net.layers.values())[last_ss_layer]
    if not isinstance(layer, LinearLayer):
        raise ValueError("Last sign-sequence layer is not a LinearLayer.")
    n_neurons = int(layer.weight.shape[0])
    if not (0 <= neuron_idx < n_neurons):
        raise ValueError(f"neuron_idx must be in [0, {n_neurons}), got {neuron_idx} for the last ReLU layer.")

    deleted_shi = _deleted_shi_for_last_layer_neuron(cplx, neuron_idx)
    new_net = net_without_last_ss_layer_neuron(cplx._net, last_ss_layer, neuron_idx)
    out = type(cplx)(new_net)

    dual = cplx.get_dual_graph(relabel=True)
    if dual.number_of_nodes() == 0:
        return out

    contracted, old_rep = contract_dual_graph_for_shi(dual, deleted_shi)
    for component in nx.connected_components(contracted):
        sub = contracted.subgraph(component).copy()
        source = min(component)
        initial_ss = delete_ss_columns(cplx.index2poly[old_rep[source]].ss_np, [deleted_shi])
        recover_from_dual_graph(out, sub, initial_ss, source=source, copy=True)

    return out
