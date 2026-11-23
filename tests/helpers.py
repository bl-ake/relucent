"""Shared test helpers."""

import numpy as np
import torch


def ss_to_numpy(ss):
    """Convert sign sequence to numpy for comparison (handles torch.Tensor or ndarray)."""
    if isinstance(ss, torch.Tensor):
        return ss.detach().cpu().numpy()
    return np.asarray(ss)


def cover_by_components(ref, model, shi, **kwargs):
    """Cover the full-BFS boundary ``ref`` with :meth:`Complex.boundary_component` calls.

    Starts each call from a cell of ``ref`` that no earlier component holds, and checks that every
    component contains its start, stays inside ``ref``, and shares no cell with another. Returns the
    components; together they hold exactly ``ref``'s cells.
    """
    import networkx as nx

    from relucent import Complex

    remaining = {p.tag: p for p in ref}
    covered: set[bytes] = set()
    components = []
    while remaining:
        tag = min(remaining)
        comp = Complex(model).boundary_component(shi, remaining[tag].ss_np, verbose=0, **kwargs)
        tags = {p.tag for p in comp}
        assert tag in tags, "a component must contain its start cell"
        assert tags <= {p.tag for p in ref}, "a component found cells the full boundary does not have"
        assert not tags & covered, "two components share a cell"
        assert nx.is_connected(comp.dual_graph(require_complete=False)), "a component must be connected"
        covered |= tags
        for t in tags:
            remaining.pop(t)
        components.append(comp)
    return components


def summed_betti(complexes, **kwargs):
    """Betti numbers of a disjoint union: the sum over its parts, as ``{dim: count}`` without zeros."""
    total: dict[int, int] = {}
    for c in complexes:
        for k, v in c.betti_numbers(**kwargs).items():
            total[int(k)] = total.get(int(k), 0) + int(v)
    return {k: v for k, v in total.items() if v}
