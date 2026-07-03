"""Unit tests for boundary MIP pricing correctness."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from relucent import Complex, set_seeds
from relucent.boundary_mip import (
    BoundaryPricingIncompleteError,
    _batch_add_nogood_constraints,
    _nogood_flip_indices,
    _nogood_flip_terms,
    _parallel_build_nogood_specs,
    _tags_requiring_cuts,
    _unique_sign_patterns,
    price_boundary_witness,
)
from relucent.config import update_settings
from relucent.utils import encode_ss

update_settings(VERBOSE=0)


def _line_boundary_model() -> nn.Sequential:
    fc = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    return nn.Sequential(fc, nn.ReLU())


def _populate_line(cplx: Complex) -> None:
    xs = np.linspace(-2.0, 2.0, 25)
    ys = np.linspace(-2.0, 2.0, 25)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    eps = 1e-2
    left = grid.copy()
    left[:, 0] = -eps
    right = grid.copy()
    right[:, 0] = eps
    pts = np.vstack([left, right, np.random.randn(200, 2)])
    for x in pts:
        ss = cplx.point2ss(x.reshape(1, -1))
        if (np.asarray(ss) == 0).any():
            continue
        cplx.add_point(x.reshape(1, -1), check_exists=True)


def test_nogood_flip_terms_empty_for_single_relu_boundary():
    """Single-ReLU networks have no free indicators to flip at the boundary pin."""
    tag = b"\x00"
    y_vars = [object()]  # placeholder; only length matters for boundary_shi check
    assert _nogood_flip_terms(tag, y_vars, boundary_shi=0, n=1) is None


def test_unique_sign_patterns_dedupes():
    ss_a = np.array([[0, 1, -1]], dtype=np.int8)
    ss_b = np.array([[0, -1, 1]], dtype=np.int8)
    unique = _unique_sign_patterns([ss_a, ss_a, ss_b])
    assert len(unique) == 2
    assert {encode_ss(ss) for ss in unique} == {encode_ss(ss_a), encode_ss(ss_b)}


def test_tags_requiring_cuts_marks_excluded_pattern(seeded: int):
    set_seeds(seeded)
    model = _line_boundary_model()
    cplx = Complex(model)
    _populate_line(cplx)
    shi = cplx.n - 1
    ref = cplx.get_boundary_complex(shi, verbose=False)
    assert len(ref) >= 1
    known = next(iter(ref))
    ss = np.asarray(known.ss_np, dtype=np.int8).reshape(1, -1)

    to_cut = _tags_requiring_cuts(
        cplx._net,
        [ss],
        exclude_tags={known.tag},
        boundary_shi=shi,
        rejected=set(),
    )
    assert known.tag in to_cut


def test_tags_requiring_cuts_empty_when_already_rejected(seeded: int):
    set_seeds(seeded)
    model = _line_boundary_model()
    cplx = Complex(model)
    _populate_line(cplx)
    shi = cplx.n - 1
    ref = cplx.get_boundary_complex(shi, verbose=False)
    known = next(iter(ref))
    ss = np.asarray(known.ss_np, dtype=np.int8).reshape(1, -1)
    tag = encode_ss(ss)

    to_cut = _tags_requiring_cuts(
        cplx._net,
        [ss],
        exclude_tags=set(),
        boundary_shi=shi,
        rejected={tag},
    )
    assert to_cut == set()


def test_price_boundary_witness_finds_unvisited_cell(seeded: int):
    set_seeds(seeded)
    model = _line_boundary_model()
    cplx = Complex(model)
    _populate_line(cplx)
    shi = cplx.n - 1
    ref = cplx.get_boundary_complex(shi, verbose=False)
    tags = {p.tag for p in ref}
    assert len(tags) >= 1
    target = next(iter(tags))
    witness = price_boundary_witness(cplx._net, shi, tags - {target})
    assert witness is not None
    assert witness.tag == target


def test_price_boundary_witness_proven_none_when_all_visited(seeded: int):
    set_seeds(seeded)
    model = _line_boundary_model()
    cplx = Complex(model)
    _populate_line(cplx)
    shi = cplx.n - 1
    ref = cplx.get_boundary_complex(shi, verbose=False)
    tags = {p.tag for p in ref}
    witness = price_boundary_witness(cplx._net, shi, tags)
    assert witness is None


def test_diamond_discover_finds_all_components(seeded: int):
    """Regression: discovery must not stop early when a second component exists."""
    import networkx as nx

    set_seeds(seeded)
    fc0 = nn.Linear(2, 6, bias=False, dtype=torch.float64)
    base = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    fc0.weight.data[:] = base + 1e-3 * torch.randn_like(base)
    fc1 = nn.Linear(6, 2, bias=False, dtype=torch.float64)
    fc1.weight.data[:] = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    fc2 = nn.Linear(2, 1, bias=True, dtype=torch.float64)
    fc2.weight.data[:] = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    fc2.bias.data[:] = torch.tensor([-1.0], dtype=torch.float64)
    model = nn.Sequential(fc0, nn.ReLU(), fc1, fc2, nn.ReLU())

    cplx = Complex(model)
    thetas = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    pts = np.vstack([0.9 * dirs, 1.1 * dirs, np.random.randn(200, 2)])
    for x in pts:
        ss = cplx.point2ss(x.reshape(1, -1))
        if (np.asarray(ss) == 0).any():
            continue
        cplx.add_point(x.reshape(1, -1), check_exists=True)

    shi = cplx.n - 1
    ref = cplx.get_boundary_complex(shi, verbose=False)
    ref_components = nx.number_connected_components(ref.get_dual_graph(verbose=False, require_complete=False))
    new, stats = Complex(model).discover_boundary_complex(
        shi,
        verbose=False,
        return_stats=True,
        nworkers=1,
    )
    assert stats["n_components"] == ref_components
    assert {p.tag for p in ref} == {p.tag for p in new}


def test_boundary_pricing_incomplete_error_is_runtime_error():
    assert issubclass(BoundaryPricingIncompleteError, RuntimeError)


def test_nogood_flip_indices_matches_flip_terms(seeded: int):
    del seeded
    tag = encode_ss(np.array([[0, 1, -1]], dtype=np.int8))
    indices = _nogood_flip_indices(tag, boundary_shi=0, n=3)
    assert indices == ((1, 1), (2, -1))


def test_parallel_nogood_specs_match_serial(seeded: int):
    del seeded
    tags = {
        encode_ss(np.array([[0, 1, -1]], dtype=np.int8)),
        encode_ss(np.array([[0, -1, 1]], dtype=np.int8)),
        encode_ss(np.array([[0, 1, 1]], dtype=np.int8)),
    }
    serial = _parallel_build_nogood_specs(tags, boundary_shi=0, n=3, nworkers=1)
    parallel = _parallel_build_nogood_specs(tags, boundary_shi=0, n=3, nworkers=2)
    assert len(serial) == len(parallel) == 3


def test_batch_add_nogood_constraints_emits_expected_count():
    from gurobipy import GRB, Model

    from relucent.utils import get_env

    update_settings(BOUNDARY_MIP_BULK_NOGOOD_EMIT="on")
    env = get_env()
    model = Model("batch_nogood_test", env)
    y_mvar = model.addMVar(3, vtype=GRB.BINARY, name="y")
    y_vars = [y_mvar[j] for j in range(3)]
    model.update()
    tag_a = encode_ss(np.array([[0, 1, -1]], dtype=np.int8))
    tag_b = encode_ss(np.array([[0, -1, 1]], dtype=np.int8))
    specs = [
        idx
        for idx in (
            _nogood_flip_indices(tag_a, boundary_shi=0, n=3),
            _nogood_flip_indices(tag_b, boundary_shi=0, n=3),
        )
        if idx is not None
    ]
    next_idx = _batch_add_nogood_constraints(model, specs, y_vars, name_prefix="exclude_test_", y_mvar=y_mvar)
    model.update()
    assert next_idx == 2
    assert model.NumConstrs == 2
    model.close()


def test_cut_ordering_preserves_witness_tag(seeded: int):
    update_settings(
        BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS=0,
        BOUNDARY_MIP_STATIC_EXCLUSION_MIN_TAGS=1,
        BOUNDARY_MIP_BULK_NOGOOD_EMIT="on",
    )
    set_seeds(seeded)
    model = _line_boundary_model()
    cplx = Complex(model)
    _populate_line(cplx)
    shi = cplx.n - 1
    ref = cplx.get_boundary_complex(shi, verbose=False)
    tags = {p.tag for p in ref}
    target = next(iter(tags))
    excluded = tags - {target}

    witnesses: list[bytes] = []
    for order in ("as_is", "tag_lex", "random"):
        update_settings(BOUNDARY_MIP_CUT_ORDER=order)
        witness = price_boundary_witness(cplx._net, shi, excluded)
        assert witness is not None
        witnesses.append(witness.tag)
    assert len(set(witnesses)) == 1
    assert witnesses[0] == target


def test_price_boundary_witness_static_exclusions_match_lazy(seeded: int):
    """Static batched exclusions find the same witness as trie-only on small models."""
    update_settings(
        BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS=0,
        BOUNDARY_MIP_STATIC_EXCLUSION_MIN_TAGS=1,
        BOUNDARY_MIP_EXCLUSION_WORKERS=1,
    )
    set_seeds(seeded)
    model = _line_boundary_model()
    cplx = Complex(model)
    _populate_line(cplx)
    shi = cplx.n - 1
    ref = cplx.get_boundary_complex(shi, verbose=False)
    tags = {p.tag for p in ref}
    assert len(tags) >= 1
    target = next(iter(tags))
    witness = price_boundary_witness(cplx._net, shi, tags - {target})
    assert witness is not None
    assert witness.tag == target


def test_lazy_only_mode_skips_compile_and_finds_witness(seeded: int, monkeypatch):
    """Large exclude sets use lazy-only pricing (no trie/static precompile)."""
    import secrets

    from relucent import boundary_mip

    compile_called = False

    def _fake_compile(*_args, **_kwargs):
        nonlocal compile_called
        compile_called = True
        raise AssertionError("_compile_exclude_tags should not run in lazy-only mode")

    monkeypatch.setattr(boundary_mip, "_compile_exclude_tags", _fake_compile)
    update_settings(
        BOUNDARY_MIP_LAZY_ONLY_MIN_TAGS=1,
        BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS=0,
    )
    set_seeds(seeded)
    model = _line_boundary_model()
    cplx = Complex(model)
    _populate_line(cplx)
    shi = cplx.n - 1
    ref = cplx.get_boundary_complex(shi, verbose=False)
    tags = {p.tag for p in ref}
    assert len(tags) >= 1
    target = next(iter(tags))
    exclude = set(tags) - {target}
    tag_bytes = (cplx.n + 7) // 8
    while len(exclude) < 2:
        exclude.add(secrets.token_bytes(tag_bytes))
    witness = price_boundary_witness(cplx._net, shi, exclude)
    assert not compile_called
    assert witness is not None
    assert witness.tag == target
