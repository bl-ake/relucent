"""Tests for trie-based boundary exclusion compilation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gurobipy import GRB, Model

from relucent import Complex, set_seeds
from relucent.config import update_settings
from relucent.search.boundary_exclusion_trie import ForbiddenPatternTrie
from relucent.search.boundary_mip import price_boundary_witness
from relucent.search.exploration import explore_for_topology
from relucent.utils import add_output_relu, encode_ss, get_env, mlp

update_settings(VERBOSE=0)


def _make_tag(n: int, boundary_shi: int, signs: dict[int, int]) -> bytes:
    row = np.zeros(n, dtype=np.int8)
    row[boundary_shi] = 0
    for j, sign in signs.items():
        row[j] = int(sign)
    return encode_ss(row.reshape(1, -1))


def _compile_tags(
    tags: set[bytes],
    *,
    n: int,
    boundary_shi: int,
    include_leaves: bool = False,
) -> tuple[int, bool]:
    env = get_env()
    model = Model("trie_test", env)
    y_vars = [model.addVar(vtype=GRB.BINARY, name=f"y_{j}") for j in range(n)]
    trie = ForbiddenPatternTrie.from_tags(tags, n, boundary_shi, verbose=False)
    stats = trie.compile_to_model(model, y_vars, include_leaves=include_leaves)
    model.close()
    return stats.n_constraints, stats.fully_saturated


def test_full_subcube_compiles_to_one_constraint():
    n = 4
    boundary_shi = 0
    tags = {_make_tag(n, boundary_shi, {1: 1, 2: s2, 3: s3}) for s2 in (-1, 1) for s3 in (-1, 1)}
    assert len(tags) == 4
    n_constraints, fully_saturated = _compile_tags(tags, n=n, boundary_shi=boundary_shi)
    assert not fully_saturated
    assert n_constraints == 1


def test_distant_patterns_emit_separate_constraints():
    n = 3
    boundary_shi = 0
    tags = {
        _make_tag(n, boundary_shi, {1: 1, 2: 1}),
        _make_tag(n, boundary_shi, {1: -1, 2: -1}),
    }
    n_constraints, fully_saturated = _compile_tags(
        tags,
        n=n,
        boundary_shi=boundary_shi,
        include_leaves=True,
    )
    assert not fully_saturated
    assert n_constraints == 2


def test_unstructured_tags_skip_leaf_compile_by_default():
    n = 5
    boundary_shi = 0
    tags = {
        _make_tag(n, boundary_shi, {1: 1, 2: 1, 3: 1, 4: 1}),
        _make_tag(n, boundary_shi, {1: -1, 2: -1, 3: -1, 4: -1}),
    }
    n_constraints, fully_saturated = _compile_tags(tags, n=n, boundary_shi=boundary_shi)
    assert not fully_saturated
    assert n_constraints == 0


def test_saturated_root_detects_proven_infeasible():
    n = 2
    boundary_shi = 0
    tags = {
        _make_tag(n, boundary_shi, {1: 1}),
        _make_tag(n, boundary_shi, {1: -1}),
    }
    n_constraints, fully_saturated = _compile_tags(tags, n=n, boundary_shi=boundary_shi)
    assert fully_saturated
    assert n_constraints == 0


def test_price_boundary_witness_with_compiled_exclusions(seeded: int):
    """MIP pricing with trie-compiled exclusions still finds an unvisited witness."""
    update_settings(BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS=0)
    set_seeds(seeded)
    model = add_output_relu(mlp(widths=[2, 20, 1]))
    cplx = Complex(model)
    shi = cplx.n - 1

    excluded: set[bytes] = set()
    found: list[bytes] = []
    for _ in range(5):
        witness = price_boundary_witness(cplx._net, shi, excluded)
        if witness is None:
            break
        found.append(witness.tag)
        excluded.add(witness.tag)
    assert len(found) >= 2

    target = found[-1]
    witness = price_boundary_witness(cplx._net, shi, set(found[:-1]))
    assert witness is not None
    assert witness.tag == target


def test_price_boundary_witness_proven_none_via_saturated_trie(seeded: int):
    update_settings(BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS=0)
    set_seeds(seeded)
    fc = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    model = nn.Sequential(fc, nn.ReLU())
    cplx = Complex(model)
    xs = np.linspace(-2.0, 2.0, 25)
    ys = np.linspace(-2.0, 2.0, 25)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    eps = 1e-2
    left = grid.copy()
    left[:, 0] = -eps
    right = grid.copy()
    right[:, 0] = eps
    for x in np.vstack([left, right, np.random.randn(200, 2)]):
        ss = cplx.point2ss(x.reshape(1, -1))
        if (np.asarray(ss) == 0).any():
            continue
        cplx.add_point(x.reshape(1, -1), check_exists=True)

    explore_for_topology(cplx, np.array([0.5, 0.0]))
    shi = cplx.n - 1
    ref = cplx.get_boundary_complex(shi, verbose=False)
    tags = {p.tag for p in ref}
    witness = price_boundary_witness(cplx._net, shi, tags)
    assert witness is None
