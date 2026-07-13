"""Trie-based compression of boundary MIP sign-pattern exclusions."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from gurobipy import Model

__all__ = [
    "CompileStats",
    "ForbiddenPatternTrie",
    "NogoodPathSpec",
]

NogoodPathSpec = tuple[tuple[int, int], ...]


@dataclass
class CompileStats:
    n_tags: int
    n_constraints: int
    n_saturated_nodes: int
    compile_seconds: float
    fully_saturated: bool = False

    @property
    def compression_ratio(self) -> float:
        if self.n_constraints == 0:
            return float(self.n_tags) if self.n_tags > 0 else 1.0
        return self.n_tags / self.n_constraints


@dataclass
class _TrieNode:
    count: int = 0
    saturated: bool = False
    children: dict[int, _TrieNode] = field(default_factory=dict)


class ForbiddenPatternTrie:
    """Binary trie of forbidden sign patterns over free ReLU indicators."""

    def __init__(self, n: int, boundary_shi: int) -> None:
        if not (0 <= boundary_shi < n):
            raise ValueError(f"boundary_shi must be in [0, {n}), got {boundary_shi}")
        self.n = n
        self.boundary_shi = boundary_shi
        self.free_indices = [j for j in range(n) if j != boundary_shi]
        self.n_free = len(self.free_indices)
        self.root = _TrieNode()
        self.n_tags = 0
        self.n_saturated_constraints = 0
        self.fully_saturated = False

    @classmethod
    def from_tags(
        cls,
        tags: Iterable[bytes],
        n: int,
        boundary_shi: int,
        *,
        verbose: bool = False,
    ) -> ForbiddenPatternTrie:
        trie = cls(n, boundary_shi)
        tag_list = list(tags)
        if not tag_list:
            return trie

        rows = np.frombuffer(b"".join(tag_list), dtype=np.int8).reshape(len(tag_list), n)
        iterator: Sequence[np.ndarray] | Iterable[np.ndarray] = rows
        if verbose:
            iterator = tqdm(rows, desc="Building exclusion trie", unit="tag")
        for signs in iterator:
            trie._insert_signs(signs)
        return trie

    def _y_bit(self, signs: np.ndarray, j: int) -> int:
        return 0 if int(signs[j]) > 0 else 1

    def _mark_saturated(self, node: _TrieNode, depth: int) -> None:
        if node.saturated or not self.is_saturated(node, depth):
            return
        node.saturated = True
        if depth == 0:
            self.fully_saturated = True
        else:
            self.n_saturated_constraints += 1

    def _insert_signs(self, signs: np.ndarray) -> None:
        node = self.root
        if node.saturated:
            self.n_tags += 1
            return
        node.count += 1
        self._mark_saturated(node, 0)
        for depth, j in enumerate(self.free_indices):
            y_bit = self._y_bit(signs, j)
            child = node.children.get(y_bit)
            if child is None:
                child = _TrieNode()
                node.children[y_bit] = child
            node = child
            if node.saturated:
                self.n_tags += 1
                return
            node.count += 1
            self._mark_saturated(node, depth + 1)
        self.n_tags += 1

    def insert(self, tag: bytes) -> None:
        signs = np.frombuffer(tag, dtype=np.int8, count=self.n)
        self._insert_signs(signs)

    def _subtree_capacity(self, depth: int) -> int:
        remaining = self.n_free - depth
        return 0 if remaining < 0 else 1 << remaining

    def is_saturated(self, node: _TrieNode, depth: int) -> bool:
        if depth < 0 or depth >= self.n_free:
            return False
        capacity = self._subtree_capacity(depth)
        if capacity <= 0:
            return False
        if depth == 0:
            return node.count == capacity
        return capacity > 1 and node.count == capacity

    def count_saturated_constraints(self, *, include_leaves: bool = False) -> int:
        """Count saturated-subtree constraints without traversing the trie."""
        del include_leaves  # leaf nogoods are never counted here
        return self.n_saturated_constraints

    def _collect_specs_node(
        self,
        node: _TrieNode,
        depth: int,
        path: list[tuple[int, int]],
        specs: list[NogoodPathSpec],
        *,
        include_leaves: bool,
    ) -> None:
        if node.count == 0:
            return

        if self.is_saturated(node, depth):
            if depth == 0:
                return
            if path:
                specs.append(tuple(path))
            return

        if depth == self.n_free:
            if include_leaves and node.count > 0 and path:
                specs.append(tuple(path))
            return

        j = self.free_indices[depth]
        for y_bit in sorted(node.children):
            child = node.children[y_bit]
            sign = 1 if y_bit == 0 else -1
            path.append((j, sign))
            self._collect_specs_node(child, depth + 1, path, specs, include_leaves=include_leaves)
            path.pop()

    def collect_saturated_specs(self, *, include_leaves: bool = False) -> list[NogoodPathSpec]:
        specs: list[NogoodPathSpec] = []
        self._collect_specs_node(self.root, 0, [], specs, include_leaves=include_leaves)
        return specs

    def compile_to_model(
        self,
        model: Model,
        y_vars: list[Any],
        *,
        start_idx: int = 0,
        include_leaves: bool = False,
    ) -> CompileStats:
        """Emit compressed no-good constraints; return compile statistics.

        By default only saturated subtrees are compiled. Unsaturated leaves are
        left to static or lazy exclusion paths when compression is poor.
        """
        from relucent.search.boundary_mip import _batch_add_nogood_constraints

        t0 = time.perf_counter()
        stats = CompileStats(
            n_tags=self.n_tags,
            n_constraints=0,
            n_saturated_nodes=0,
            compile_seconds=0.0,
        )
        specs = self.collect_saturated_specs(include_leaves=include_leaves)
        if self.is_saturated(self.root, 0):
            stats.fully_saturated = True
        if specs:
            next_idx = _batch_add_nogood_constraints(
                model,
                specs,
                y_vars,
                name_prefix="exclude_comp_",
                start_idx=start_idx,
            )
            stats.n_constraints = next_idx - start_idx
            stats.n_saturated_nodes = len(specs)
        stats.compile_seconds = time.perf_counter() - t0
        return stats
