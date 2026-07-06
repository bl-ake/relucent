"""MIP pricing for undiscovered bent-hyperplane boundary cells."""

from __future__ import annotations

import itertools
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from gurobipy import GRB, Env, Model, quicksum
from tqdm.auto import tqdm

import relucent.config as cfg
from relucent._network_scale import count_relu_units, estimate_input_bound, relu_linear_blocks
from relucent.model import ReLUNetwork
from relucent.poly import Polyhedron
from relucent.utils import encode_ss, get_env, get_mp_context, process_aware_cpu_count

if TYPE_CHECKING:
    pass

__all__ = [
    "BoundaryPricingIncompleteError",
    "decode_ss_tag",
    "price_boundary_witness",
]


class BoundaryPricingIncompleteError(RuntimeError):
    """Boundary MIP pricing stopped without a witness or proven infeasibility."""


NogoodFlipIndices = tuple[tuple[int, int], ...]


@dataclass
class _ExclusionCompileResult:
    n_trie_constraints: int = 0
    n_static_constraints: int = 0
    n_tags: int = 0
    fully_saturated: bool = False
    static_precompiled: bool = False
    pending_static_specs: list[NogoodFlipIndices] = field(default_factory=list)
    trie_build_s: float = 0.0
    trie_emit_s: float = 0.0
    static_emit_s: float = 0.0


@dataclass
class _PricingCallbackState:
    net: ReLUNetwork
    boundary_shi: int
    exclude_tags: set[bytes]
    n: int
    x: Any
    y_vars: list[Any]
    rejected: set[bytes] = field(default_factory=set)
    n_cuts: int = 0
    witness: Polyhedron | None = None
    proven_infeasible: bool = False
    stall: bool = False
    precompiled_exclusions: bool = False
    verbose: bool = False
    excl_pbar: tqdm | None = None


def _pricing_log(msg: str, *, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


def _configure_pricing_mip_logging(model: Model, *, log_path: Path | None) -> None:
    """Configure Gurobi solver logging for a pricing MIP without mutating the shared env.

    Controlled by :data:`~relucent.config.BOUNDARY_MIP_GUROBI_LOG`, not by the
    Relucent ``verbose`` flag. Model parameters override the cached
    :func:`~relucent.utils.get_env` defaults.
    """
    if cfg.BOUNDARY_MIP_GUROBI_LOG:
        model.Params.OutputFlag = 1
        model.Params.LogToConsole = 1
        if log_path is not None:
            model.Params.LogFile = str(log_path)
    else:
        model.Params.OutputFlag = 0


def decode_ss_tag(tag: bytes, n: int) -> np.ndarray:
    """Decode a sign-sequence tag produced by :func:`~relucent.utils.encode_ss`."""
    return _decode_tag_signs(tag, n).reshape(1, -1)


def _decode_tag_signs(tag: bytes, n: int) -> np.ndarray:
    return np.frombuffer(tag, dtype=np.int8, count=n)


def _nogood_flip_indices(tag: bytes, boundary_shi: int, n: int) -> NogoodFlipIndices | None:
    """Return ``(relu_index, sign)`` pairs for a no-good over free indicators."""
    known = _decode_tag_signs(tag, n)
    indices: list[tuple[int, int]] = []
    for j in range(n):
        if j == boundary_shi:
            continue
        if int(known[j]) > 0:
            indices.append((j, 1))
        elif int(known[j]) < 0:
            indices.append((j, -1))
    return tuple(indices) if indices else None


def _nogood_linexpr_from_indices(indices: NogoodFlipIndices, y_vars: list[Any]) -> Any:
    terms: list[Any] = []
    for j, sign in indices:
        if sign > 0:
            terms.append(1 - y_vars[j])
        else:
            terms.append(y_vars[j])
    return quicksum(terms)


def _relu_layer_boundaries(net: ReLUNetwork) -> list[tuple[int, int]]:
    boundaries: list[tuple[int, int]] = []
    shi = 0
    for block in relu_linear_blocks(net):
        n_out = int(block.weight.shape[0])
        boundaries.append((shi, shi + n_out))
        shi += n_out
    return boundaries


def _layer_index_for_neuron(j: int, boundaries: list[tuple[int, int]]) -> int:
    for layer_idx, (start, end) in enumerate(boundaries):
        if start <= j < end:
            return layer_idx
    return 0


def _order_tag_spec_pairs(
    pairs: list[tuple[bytes, NogoodFlipIndices]],
    *,
    order: str,
    n: int,
    boundary_shi: int,
    net: ReLUNetwork | None = None,
    seed: int = 0,
) -> list[tuple[bytes, NogoodFlipIndices]]:
    if not pairs or order == "as_is":
        return pairs

    boundaries = _relu_layer_boundaries(net) if net is not None else []

    if order == "literal_count_asc":
        return sorted(pairs, key=lambda item: len(item[1]))
    if order == "trie_depth_desc":
        return sorted(pairs, key=lambda item: len(item[1]), reverse=True)
    if order == "tag_lex":
        return sorted(pairs, key=lambda item: item[0])

    if order == "layer_major" and boundaries:

        def _layer_key(item: tuple[bytes, NogoodFlipIndices]) -> tuple[int, bytes]:
            spec = item[1]
            if not spec:
                return (0, item[0])
            layer = min(_layer_index_for_neuron(j, boundaries) for j, _ in spec)
            return (layer, item[0])

        return sorted(pairs, key=_layer_key)

    if order == "hamming_median":
        rows = np.frombuffer(b"".join(tag for tag, _ in pairs), dtype=np.int8).reshape(len(pairs), n)
        free_cols = [j for j in range(n) if j != boundary_shi]
        median_sign = np.sign(np.median(rows[:, free_cols], axis=0)).astype(np.int8)

        def _hamming_key(item: tuple[bytes, NogoodFlipIndices]) -> tuple[int, bytes]:
            signs = _decode_tag_signs(item[0], n)
            dist = int(np.count_nonzero(signs[free_cols] != median_sign))
            return (dist, item[0])

        return sorted(pairs, key=_hamming_key)

    if order == "random":
        shuffled = list(pairs)
        rng = random.Random(seed)
        rng.shuffle(shuffled)
        return shuffled

    return pairs


def _order_tags(tags: Iterable[bytes], *, n: int, boundary_shi: int, net: ReLUNetwork | None) -> list[bytes]:
    tag_list = list(tags)
    pairs: list[tuple[bytes, NogoodFlipIndices]] = []
    for tag in tag_list:
        indices = _nogood_flip_indices(tag, boundary_shi, n)
        if indices is not None:
            pairs.append((tag, indices))
    ordered = _order_tag_spec_pairs(
        pairs,
        order=str(cfg.BOUNDARY_MIP_CUT_ORDER),
        n=n,
        boundary_shi=boundary_shi,
        net=net,
    )
    return [tag for tag, _ in ordered]


def _use_bulk_nogood_emit(n_specs: int) -> bool:
    mode = str(cfg.BOUNDARY_MIP_BULK_NOGOOD_EMIT).strip().lower()
    if mode == "on":
        return True
    if mode == "off":
        return False
    return n_specs >= 500


def _bulk_add_nogood_constraints(
    model: Model,
    specs: Sequence[NogoodFlipIndices],
    y_vars: list[Any],
    *,
    name_prefix: str,
    start_idx: int = 0,
    trie_depths: Sequence[int] | None = None,
    y_mvar: Any | None = None,
) -> int:
    """Add nogoods via one sparse ``addMConstr`` matrix."""
    n_y = len(y_vars)
    n_c = len(specs)
    if n_c == 0:
        return start_idx

    model.update()

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.empty(n_c, dtype=np.float64)
    for row_i, spec in enumerate(specs):
        n_pos = 0
        for j, sign in spec:
            rows.append(row_i)
            cols.append(j)
            if sign > 0:
                data.append(-1.0)
                n_pos += 1
            else:
                data.append(1.0)
        rhs[row_i] = 1.0 - float(n_pos)

    matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_c, n_y))
    y_target = y_mvar if y_mvar is not None else y_vars
    constrs = model.addMConstr(matrix.toarray(), y_target, ">=", rhs)
    constr_list: Any = constrs
    for i in range(n_c):
        constr_list[i].ConstrName = f"{name_prefix}{start_idx + i}"
    if cfg.BOUNDARY_MIP_CUT_PRIORITY_ENABLED and trie_depths is not None:
        for i, depth in enumerate(trie_depths):
            constr_list[i].Priority = int(depth)

    model.update()
    return start_idx + n_c


def _batch_add_nogood_constraints(
    model: Model,
    specs: Sequence[NogoodFlipIndices],
    y_vars: list[Any],
    *,
    name_prefix: str,
    start_idx: int = 0,
    chunk_size: int | None = None,
    defer_update: bool = True,
    trie_depths: Sequence[int] | None = None,
    y_mvar: Any | None = None,
) -> int:
    """Add no-good constraints in batches."""
    if _use_bulk_nogood_emit(len(specs)):
        return _bulk_add_nogood_constraints(
            model,
            specs,
            y_vars,
            name_prefix=name_prefix,
            start_idx=start_idx,
            trie_depths=trie_depths,
            y_mvar=y_mvar,
        )

    chunk_size = int(chunk_size or cfg.BOUNDARY_MIP_EXCLUSION_BATCH_SIZE)
    next_idx = start_idx
    batch_exprs: list[Any] = []
    batch_depths: list[int] = []

    def _flush() -> None:
        nonlocal next_idx
        if not batch_exprs:
            return
        for i, expr in enumerate(batch_exprs):
            constr = model.addConstr(expr >= 1, name=f"{name_prefix}{next_idx + i}")
            if cfg.BOUNDARY_MIP_CUT_PRIORITY_ENABLED and trie_depths is not None:
                constr.Priority = int(batch_depths[i])
        next_idx += len(batch_exprs)
        batch_exprs.clear()
        batch_depths.clear()
        if not defer_update:
            model.update()

    for spec_i, spec in enumerate(specs):
        if not spec:
            continue
        batch_exprs.append(_nogood_linexpr_from_indices(spec, y_vars))
        if trie_depths is not None:
            batch_depths.append(int(trie_depths[spec_i]))
        if len(batch_exprs) >= chunk_size:
            _flush()
    _flush()
    if defer_update:
        model.update()
    return next_idx


def _exclusion_worker_count() -> int:
    configured = int(cfg.BOUNDARY_MIP_EXCLUSION_WORKERS)
    if configured == 1:
        return 1
    if configured > 1:
        return configured
    return process_aware_cpu_count() or 1


def _build_nogood_specs_chunk(args: tuple[list[bytes], int, int]) -> list[NogoodFlipIndices]:
    tags, boundary_shi, n = args
    specs: list[NogoodFlipIndices] = []
    for tag in tags:
        indices = _nogood_flip_indices(tag, boundary_shi, n)
        if indices is not None:
            specs.append(indices)
    return specs


def _parallel_build_nogood_specs(
    tags: Iterable[bytes],
    *,
    boundary_shi: int,
    n: int,
    nworkers: int | None = None,
    net: ReLUNetwork | None = None,
) -> list[NogoodFlipIndices]:
    ordered_tags = _order_tags(tags, n=n, boundary_shi=boundary_shi, net=net)
    if not ordered_tags:
        return []
    workers = nworkers or _exclusion_worker_count()
    if workers <= 1 or len(ordered_tags) < workers * 4:
        specs = _build_nogood_specs_chunk((ordered_tags, boundary_shi, n))
    else:
        chunk_size = max(1, (len(ordered_tags) + workers - 1) // workers)
        chunks = [ordered_tags[i : i + chunk_size] for i in range(0, len(ordered_tags), chunk_size)]
        with get_mp_context().Pool(workers) as pool:
            parts = pool.map(_build_nogood_specs_chunk, [(chunk, boundary_shi, n) for chunk in chunks])
        specs = [spec for part in parts for spec in part]
    return specs


def _build_ordered_static_pairs(
    exclude_tags: Iterable[bytes],
    *,
    boundary_shi: int,
    n: int,
    net: ReLUNetwork | None,
) -> list[tuple[bytes, NogoodFlipIndices]]:
    pairs: list[tuple[bytes, NogoodFlipIndices]] = []
    for tag in exclude_tags:
        indices = _nogood_flip_indices(tag, boundary_shi, n)
        if indices is not None:
            pairs.append((tag, indices))
    ordered = _order_tag_spec_pairs(
        pairs,
        order=str(cfg.BOUNDARY_MIP_CUT_ORDER),
        n=n,
        boundary_shi=boundary_shi,
        net=net,
    )
    return ordered


def _static_add_exclude_tags(
    model: Model,
    y_vars: list[Any],
    exclude_tags: Iterable[bytes],
    *,
    boundary_shi: int,
    n: int,
    net: ReLUNetwork | None = None,
    y_mvar: Any | None = None,
    name_prefix: str = "exclude_static_",
    start_idx: int = 0,
    verbose: bool = False,
) -> int:
    tag_list = list(exclude_tags)
    if not tag_list:
        return start_idx
    t0 = time.perf_counter()
    pairs = _build_ordered_static_pairs(tag_list, boundary_shi=boundary_shi, n=n, net=net)
    specs = [spec for _, spec in pairs]
    if verbose:
        _pricing_log(
            "boundary pricing MIP: built "
            + f"{len(specs)} static nogood specs from {len(tag_list)} tags "
            + f"(order={cfg.BOUNDARY_MIP_CUT_ORDER})",
            verbose=True,
        )
    next_idx = _batch_add_nogood_constraints(
        model,
        specs,
        y_vars,
        name_prefix=name_prefix,
        start_idx=start_idx,
        y_mvar=y_mvar,
    )
    static_emit_s = time.perf_counter() - t0
    if verbose:
        _pricing_log(
            "boundary pricing MIP: static exclusions emitted in "
            + f"{static_emit_s:.3f}s ({next_idx - start_idx} constraints)",
            verbose=True,
        )
    return next_idx


def _compile_exclude_tags(
    model: Model,
    y_vars: list[Any],
    exclude_tags: set[bytes],
    *,
    n: int,
    boundary_shi: int,
    net: ReLUNetwork | None = None,
    y_mvar: Any | None = None,
    verbose: bool = False,
) -> _ExclusionCompileResult:
    """Tiered exclusion compiler: trie compression, then batched static nogoods."""
    result = _ExclusionCompileResult(n_tags=len(exclude_tags))
    if not exclude_tags:
        return result

    n_free = max(0, n - 1)
    full_capacity = 1 << n_free if n_free < 30 else 0
    if full_capacity and len(exclude_tags) >= full_capacity:
        result.fully_saturated = True
        return result

    wave_size = int(cfg.BOUNDARY_MIP_STATIC_WAVE_SIZE)
    skip_trie = len(exclude_tags) >= cfg.BOUNDARY_MIP_STATIC_EXCLUSION_MIN_TAGS
    trie = None
    trie_stats = None

    if not skip_trie and len(exclude_tags) >= cfg.BOUNDARY_MIP_COMPILE_EXCLUSIONS_MIN_TAGS:
        from relucent.boundary_exclusion_trie import ForbiddenPatternTrie

        _pricing_log(
            "boundary pricing MIP: compiling " + f"{len(exclude_tags)} excluded tags into exclusion trie ...",
            verbose=verbose,
        )
        t_compile = time.perf_counter()
        trie = ForbiddenPatternTrie.from_tags(exclude_tags, n, boundary_shi, verbose=verbose)
        result.trie_build_s = time.perf_counter() - t_compile
        _pricing_log(
            "boundary pricing MIP: exclusion trie built in " + f"{result.trie_build_s:.3f}s",
            verbose=verbose,
        )
        if trie.fully_saturated:
            result.fully_saturated = True
            return result

    static_all = skip_trie
    if trie is not None:
        n_saturated = trie.count_saturated_constraints(include_leaves=False)
        if n_saturated > 0:
            t_emit = time.perf_counter()
            trie_stats = trie.compile_to_model(model, y_vars, include_leaves=False)
            result.trie_emit_s = time.perf_counter() - t_emit
            result.n_trie_constraints = trie_stats.n_constraints
            _pricing_log(
                "boundary pricing MIP: trie constraints emitted in "
                + f"{result.trie_emit_s:.3f}s "
                + f"({trie_stats.n_constraints} constraints, "
                + f"ratio={trie_stats.compression_ratio:.1f}x)",
                verbose=verbose,
            )
        compression_ratio = trie_stats.compression_ratio if trie_stats is not None else float(len(exclude_tags))
        static_all = n_saturated == 0 or compression_ratio < cfg.BOUNDARY_MIP_STATIC_EXCLUSION_MIN_RATIO

    if static_all:
        if wave_size > 0:
            pairs = _build_ordered_static_pairs(exclude_tags, boundary_shi=boundary_shi, n=n, net=net)
            result.pending_static_specs = [spec for _, spec in pairs]
            result.static_precompiled = False
        else:
            start_idx = result.n_trie_constraints
            next_idx = _static_add_exclude_tags(
                model,
                y_vars,
                exclude_tags,
                boundary_shi=boundary_shi,
                n=n,
                net=net,
                y_mvar=y_mvar,
                name_prefix="exclude_static_",
                start_idx=start_idx,
                verbose=verbose,
            )
            result.n_static_constraints = next_idx - start_idx
            result.static_precompiled = result.n_trie_constraints > 0 or result.n_static_constraints > 0

    elif trie is not None and trie_stats is not None and trie_stats.n_constraints > 0:
        result.static_precompiled = True

    return result


def _is_top_boundary_ss(ss: np.ndarray, boundary_shi: int) -> bool:
    row = np.asarray(ss, dtype=np.int8).ravel()
    if row[int(boundary_shi)] != 0:
        return False
    return int(np.count_nonzero(row == 0)) == 1


def _witness_from_ss(
    net: ReLUNetwork,
    ss: np.ndarray,
    *,
    exclude_tags: set[bytes],
    boundary_shi: int,
) -> Polyhedron | None:
    """Return a feasible boundary polyhedron for ``ss`` if it is new and top-dimensional."""
    tag = encode_ss(ss)
    if tag in exclude_tags:
        return None
    if not _is_top_boundary_ss(ss, boundary_shi):
        return None
    poly = Polyhedron(net, ss)
    if not poly.feasible:
        return None
    from relucent.boundary_search import _both_ambient_cofaces_feasible

    if not _both_ambient_cofaces_feasible(poly, boundary_shi):
        return None
    return poly


def _brute_force_boundary_witness(
    net: ReLUNetwork,
    boundary_shi: int,
    exclude_tags: set[bytes],
) -> Polyhedron | None:
    n = count_relu_units(net)
    if n == 0 or boundary_shi >= n or n > cfg.BOUNDARY_PRICING_BRUTE_FORCE_MAX_N:
        return None
    indices = [j for j in range(n) if j != boundary_shi]
    for signs in itertools.product((-1, 1), repeat=len(indices)):
        ss = np.zeros((1, n), dtype=np.int8)
        ss[0, boundary_shi] = 0
        for idx, sign in zip(indices, signs, strict=True):
            ss[0, idx] = int(sign)
        witness = _witness_from_ss(net, ss, exclude_tags=exclude_tags, boundary_shi=boundary_shi)
        if witness is not None:
            return witness
    return None


def _nogood_flip_terms(
    tag: bytes,
    y_vars: list[Any],
    boundary_shi: int,
    n: int,
) -> list[Any] | None:
    """Return no-good literals forcing the solution to differ from ``tag``."""
    indices = _nogood_flip_indices(tag, boundary_shi, n)
    if indices is None:
        return None
    terms: list[Any] = []
    for j, sign in indices:
        if sign > 0:
            terms.append(1 - y_vars[j])
        else:
            terms.append(y_vars[j])
    return terms


def _cb_lazy_nogood_from_indices(
    model: Model,
    indices: NogoodFlipIndices,
    y_vars: list[Any],
) -> bool:
    if not indices:
        return False
    model.cbLazy(_nogood_linexpr_from_indices(indices, y_vars) >= 1)
    return True


def _cb_lazy_nogood(
    model: Model,
    tag: bytes,
    y_vars: list[Any],
    boundary_shi: int,
    n: int,
) -> bool:
    """Add a lazy no-good cut excluding ``tag``; return False if the cut would be empty."""
    indices = _nogood_flip_indices(tag, boundary_shi, n)
    if indices is None:
        return False
    return _cb_lazy_nogood_from_indices(model, indices, y_vars)


def _add_pattern_exclusion(
    model: Model,
    y_vars: list[Any],
    boundary_shi: int,
    exclude_tags: Iterable[bytes],
    n: int,
    *,
    verbose: bool = False,
    start_idx: int = 0,
) -> int:
    """Add no-good cuts requiring the solution pattern to differ from each excluded tag.

    Returns the next constraint index (for stable ``exclude_{i}`` names).
    """
    tag_list = list(exclude_tags)
    if not tag_list:
        return start_idx
    specs = _parallel_build_nogood_specs(tag_list, boundary_shi=boundary_shi, n=n, nworkers=1)
    if verbose and len(specs) < len(tag_list):
        _pricing_log(
            "boundary pricing MIP: skipped "
            + f"{len(tag_list) - len(specs)} empty no-goods (no free ReLU indicators to flip)",
            verbose=True,
        )
    return _batch_add_nogood_constraints(
        model,
        specs,
        y_vars,
        name_prefix="exclude_",
        start_idx=start_idx,
    )


def _ss_from_y_values(
    y_vals: list[float],
    *,
    boundary_shi: int,
) -> np.ndarray:
    ss_parts: list[int] = []
    for j, y_val in enumerate(y_vals):
        if j == boundary_shi:
            ss_parts.append(0)
        else:
            ss_parts.append(1 if y_val >= 0.5 else -1)
    return np.asarray(ss_parts, dtype=np.int8).reshape(1, -1)


def _ss_from_mip_solution(
    y_vars: list[Any],
    z_by_shi: list[Any],
    *,
    boundary_shi: int,
    eps: float,
) -> np.ndarray:
    """Build a sign sequence from solved MIP ReLU indicators ``y``."""
    del z_by_shi, eps  # sign bits are enforced via ``y``; ``z`` may be loose at large big-M.
    y_vals = [float(np.asarray(yj.X).item()) for yj in y_vars]
    return _ss_from_y_values(y_vals, boundary_shi=boundary_shi)


def _is_mip_proven_infeasible(status: int) -> bool:
    return status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD)


def _mip_has_solution(model: Model, status: int) -> bool:
    if _is_mip_proven_infeasible(status):
        return False
    if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return int(model.SolCount) > 0
    if status == GRB.TIME_LIMIT:
        return int(model.SolCount) > 0
    if status == GRB.INTERRUPTED:
        return int(model.SolCount) > 0
    return False


def _unique_sign_patterns(patterns: Iterable[np.ndarray]) -> list[np.ndarray]:
    seen: set[bytes] = set()
    unique: list[np.ndarray] = []
    for ss in patterns:
        tag = encode_ss(ss)
        if tag in seen:
            continue
        seen.add(tag)
        unique.append(np.asarray(ss, dtype=np.int8).reshape(1, -1))
    return unique


def _patterns_from_final_solution(
    x: Any,
    y_vars: list[Any],
    net: ReLUNetwork,
    *,
    boundary_shi: int,
) -> list[np.ndarray]:
    """Sign patterns from the post-optimize incumbent (safety fallback)."""
    from relucent.complex import Complex

    x_val = np.asarray(x.X, dtype=np.float64).reshape(1, -1)
    y_vals = [float(np.asarray(yj.X).item()) for yj in y_vars]
    ss = _ss_from_y_values(y_vals, boundary_shi=boundary_shi)
    cx = Complex(net)
    ss_fwd = np.sign(np.asarray(cx.point2ss(x_val), dtype=np.int8)).reshape(1, -1)
    return _unique_sign_patterns([ss, ss_fwd])


def _find_witness_in_patterns(
    net: ReLUNetwork,
    patterns: Iterable[np.ndarray],
    *,
    exclude_tags: set[bytes],
    boundary_shi: int,
) -> Polyhedron | None:
    for ss in patterns:
        witness = _witness_from_ss(net, ss, exclude_tags=exclude_tags, boundary_shi=boundary_shi)
        if witness is not None:
            return witness
    return None


def _tags_requiring_cuts(
    net: ReLUNetwork,
    patterns: Iterable[np.ndarray],
    *,
    exclude_tags: set[bytes],
    boundary_shi: int,
    rejected: set[bytes],
    precompiled_exclusions: bool = False,
) -> set[bytes]:
    """Tags whose sign patterns failed witness checks and are not yet cut."""
    to_cut: set[bytes] = set()
    for ss in patterns:
        if _witness_from_ss(net, ss, exclude_tags=exclude_tags, boundary_shi=boundary_shi) is not None:
            continue
        tag = encode_ss(ss)
        if tag not in rejected:
            if precompiled_exclusions and tag in exclude_tags:
                continue
            to_cut.add(tag)
    return to_cut


def _collect_callback_patterns(
    model: Model,
    state: _PricingCallbackState,
    *,
    y_vals: list[float] | None = None,
) -> list[np.ndarray]:
    """Sign patterns from a MIPSOL callback incumbent."""
    from relucent.complex import Complex

    if y_vals is None:
        y_vals = [float(model.cbGetSolution(yj)) for yj in state.y_vars]
    ss = _ss_from_y_values(y_vals, boundary_shi=state.boundary_shi)
    x_val = np.asarray(model.cbGetSolution(state.x), dtype=np.float64).reshape(1, -1)
    cx = Complex(state.net)
    ss_fwd = np.sign(np.asarray(cx.point2ss(x_val), dtype=np.int8)).reshape(1, -1)
    return _unique_sign_patterns([ss, ss_fwd])


def _add_lazy_cuts_for_tags(
    model: Model,
    state: _PricingCallbackState,
    tags: Iterable[bytes],
) -> bool:
    """Add lazy nogood cuts for ``tags``. Returns False if a cut would be empty."""
    ordered_tags = _order_tags(tags, n=state.n, boundary_shi=state.boundary_shi, net=state.net)
    for tag in ordered_tags:
        if tag in state.rejected:
            continue
        indices = _nogood_flip_indices(tag, state.boundary_shi, state.n)
        if indices is None:
            state.proven_infeasible = True
            model.terminate()
            return False
        if not _cb_lazy_nogood_from_indices(model, indices, state.y_vars):
            state.proven_infeasible = True
            model.terminate()
            return False
        state.n_cuts += 1
        state.rejected.add(tag)
        if state.excl_pbar is not None:
            state.excl_pbar.update(1)
    return True


def _mip_pricing_callback(model: Model, where: int, state: _PricingCallbackState) -> None:
    if where != GRB.Callback.MIPSOL:
        return
    if state.witness is not None or state.proven_infeasible or state.stall:
        return

    y_vals = [float(model.cbGetSolution(yj)) for yj in state.y_vars]
    tag_y = encode_ss(_ss_from_y_values(y_vals, boundary_shi=state.boundary_shi))
    if tag_y in state.exclude_tags:
        if not _add_lazy_cuts_for_tags(model, state, [tag_y]):
            return
        return
    if tag_y in state.rejected:
        return

    patterns = _collect_callback_patterns(model, state, y_vals=y_vals)
    witness = _find_witness_in_patterns(
        state.net,
        patterns,
        exclude_tags=state.exclude_tags,
        boundary_shi=state.boundary_shi,
    )
    if witness is not None:
        state.witness = witness
        model.terminate()
        return

    tags_to_cut = _tags_requiring_cuts(
        state.net,
        patterns,
        exclude_tags=state.exclude_tags,
        boundary_shi=state.boundary_shi,
        rejected=state.rejected,
        precompiled_exclusions=state.precompiled_exclusions,
    )
    if not tags_to_cut:
        pattern_tags = {encode_ss(ss) for ss in patterns}
        if pattern_tags.issubset(state.rejected):
            state.stall = True
            model.terminate()
        return

    if not _add_lazy_cuts_for_tags(model, state, tags_to_cut):
        return


def _run_pricing_optimize(
    model: Model,
    state: _PricingCallbackState,
    *,
    verbose: bool,
    log_path: Path | None,
) -> float:
    time_limit = float(cfg.BOUNDARY_MIP_TIME_LIMIT)
    _pricing_log(
        "boundary pricing MIP: optimize "
        + f"(time_limit={'none' if time_limit <= 0 else f'{time_limit:.1f}s'}"
        + (f", log_file={log_path.resolve()})" if log_path is not None else ")"),
        verbose=verbose,
    )
    t_opt = time.perf_counter()
    model.setObjective(0.0, GRB.MINIMIZE)
    model.optimize(lambda m, w: _mip_pricing_callback(m, w, state))
    mip_optimize_s = time.perf_counter() - t_opt
    _pricing_log(
        "boundary pricing MIP: optimize finished in "
        + f"{mip_optimize_s:.3f}s, status={model.Status}, "
        + f"cuts={state.n_cuts}",
        verbose=verbose,
    )
    return mip_optimize_s


def _mip_boundary_witness(
    net: ReLUNetwork,
    boundary_shi: int,
    exclude_tags: set[bytes],
    *,
    bound: float,
    env: Env,
    eps: float,
    verbose: bool = False,
    pricing_call: int | None = None,
) -> Polyhedron | None:
    blocks = relu_linear_blocks(net)
    n = count_relu_units(net)
    if not (0 <= boundary_shi < n):
        raise ValueError(f"boundary_shi must be in [0, {n}), got {boundary_shi}")

    input_dim = int(np.prod(net.input_shape))
    big_m = float(bound)

    log_path: Path | None = None
    if verbose:
        call_suffix = f", pricing_call={pricing_call}" if pricing_call is not None else ""
        _pricing_log(
            "boundary pricing MIP: building model "
            + f"(boundary_shi={boundary_shi}, excluded_tags={len(exclude_tags)}, "
            + f"lazy_callback=True{call_suffix}, "
            + f"log_file={log_path.resolve() if log_path is not None else None})",
            verbose=True,
        )

    model = Model("boundary_pricing", env)
    _configure_pricing_mip_logging(model, log_path=log_path)
    if cfg.BOUNDARY_MIP_TIME_LIMIT > 0:
        model.setParam(GRB.Param.TimeLimit, float(cfg.BOUNDARY_MIP_TIME_LIMIT))
    model.Params.LazyConstraints = 1

    x = model.addMVar((input_dim, 1), lb=-bound, ub=bound, vtype=GRB.CONTINUOUS, name="x")
    y_mvar = model.addMVar(n, vtype=GRB.BINARY, name="y")
    y_vars: list[Any] = [y_mvar[j] for j in range(n)]
    z_by_shi: list[Any] = []

    prev: Any = x
    shi = 0
    for block in blocks:
        w = np.asarray(block.weight, dtype=np.float64)
        b = np.asarray(block.bias, dtype=np.float64).reshape(-1, 1)
        n_out = int(w.shape[0])
        z = model.addMVar((n_out, 1), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")
        a = model.addMVar((n_out, 1), lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a")
        model.addConstr(z == w @ prev + b, name="affine")
        for j in range(n_out):
            global_j = shi + j
            model.addConstr(a[j, 0] >= z[j, 0], name=f"relu_low_{global_j}")
            model.addConstr(a[j, 0] <= z[j, 0] + big_m * (1 - y_mvar[global_j]), name=f"relu_up_on_{global_j}")
            model.addConstr(a[j, 0] <= big_m * y_mvar[global_j], name=f"relu_up_off_{global_j}")
            z_by_shi.append(z[j, 0])
        prev = a
        shi += n_out

    model.addConstr(z_by_shi[boundary_shi] == 0.0, name="boundary_zero")
    for j, z_j in enumerate(z_by_shi):
        if j == boundary_shi:
            continue
        model.addConstr(z_j >= eps - big_m * (1 - y_vars[j]), name=f"sign_pos_{j}")
        model.addConstr(z_j <= -eps + big_m * y_vars[j], name=f"sign_neg_{j}")

    excl_pbar = tqdm(desc="Pattern exclusions", disable=not verbose, unit=" cuts")
    state = _PricingCallbackState(
        net=net,
        boundary_shi=boundary_shi,
        exclude_tags=exclude_tags,
        n=n,
        x=x,
        y_vars=y_vars,
        verbose=verbose,
        excl_pbar=excl_pbar,
    )

    compile_result: _ExclusionCompileResult | None = None
    mip_optimize_s = 0.0

    try:
        lazy_only = len(exclude_tags) >= int(cfg.BOUNDARY_MIP_LAZY_ONLY_MIN_TAGS)
        if exclude_tags and not lazy_only:
            compile_result = _compile_exclude_tags(
                model,
                y_vars,
                exclude_tags,
                n=n,
                boundary_shi=boundary_shi,
                net=net,
                y_mvar=y_mvar,
                verbose=verbose,
            )
            if compile_result.fully_saturated:
                _pricing_log(
                    "boundary pricing MIP: compiled exclusions cover all sign patterns " + "(proven infeasible)",
                    verbose=verbose,
                )
                return None

            wave_size = int(cfg.BOUNDARY_MIP_STATIC_WAVE_SIZE)
            if compile_result.pending_static_specs:
                state.precompiled_exclusions = False
                start_idx = compile_result.n_trie_constraints
                for wave_start in range(0, len(compile_result.pending_static_specs), wave_size):
                    wave = compile_result.pending_static_specs[wave_start : wave_start + wave_size]
                    next_idx = _batch_add_nogood_constraints(
                        model,
                        wave,
                        y_vars,
                        name_prefix="exclude_static_",
                        start_idx=start_idx,
                        y_mvar=y_mvar,
                    )
                    compile_result.n_static_constraints += next_idx - start_idx
                    start_idx = next_idx
                    state.precompiled_exclusions = True
                    mip_optimize_s += _run_pricing_optimize(model, state, verbose=verbose, log_path=log_path)
                    if state.witness is not None:
                        return state.witness
                    if state.proven_infeasible or state.stall:
                        break
                    status = int(model.Status)
                    if _is_mip_proven_infeasible(status):
                        return None
            else:
                state.precompiled_exclusions = compile_result.static_precompiled
                if compile_result.static_precompiled:
                    _pricing_log(
                        "boundary pricing MIP: precompiled exclusions "
                        + f"(trie={compile_result.n_trie_constraints}, "
                        + f"static={compile_result.n_static_constraints}, "
                        + f"tags={compile_result.n_tags}, order={cfg.BOUNDARY_MIP_CUT_ORDER})",
                        verbose=verbose,
                    )
                else:
                    _pricing_log(
                        "boundary pricing MIP: lazy callback mode "
                        + f"({len(exclude_tags)} visited tags; cuts added on integer incumbents)",
                        verbose=verbose,
                    )
                mip_optimize_s = _run_pricing_optimize(model, state, verbose=verbose, log_path=log_path)
        elif exclude_tags and lazy_only:
            _pricing_log(
                "boundary pricing MIP: lazy-only mode " + f"({len(exclude_tags)} visited tags; no static/trie precompilation)",
                verbose=verbose,
            )
            state.precompiled_exclusions = False
            mip_optimize_s = _run_pricing_optimize(model, state, verbose=verbose, log_path=log_path)
        else:
            mip_optimize_s = _run_pricing_optimize(model, state, verbose=verbose, log_path=log_path)

        if state.witness is not None:
            _pricing_log(
                "boundary pricing MIP: witness found via lazy callback " + f"({state.n_cuts} cut(s))",
                verbose=verbose,
            )
            return state.witness

        if state.stall:
            raise BoundaryPricingIncompleteError(
                "boundary pricing MIP stalled: callback received only already-cut sign patterns"
            )

        status = int(model.Status)
        if state.proven_infeasible or _is_mip_proven_infeasible(status):
            _pricing_log(
                "boundary pricing MIP: proven infeasible (no new witness exists)",
                verbose=verbose,
            )
            return None

        if status == GRB.TIME_LIMIT and int(model.SolCount) == 0:
            raise BoundaryPricingIncompleteError(
                "boundary pricing MIP hit the time limit without producing a feasible solution"
            )

        if not _mip_has_solution(model, status):
            raise BoundaryPricingIncompleteError(
                f"boundary pricing MIP stopped with unsupported status={status} " + f"and sol_count={int(model.SolCount)}"
            )

        patterns = _patterns_from_final_solution(x, y_vars, net, boundary_shi=boundary_shi)
        witness = _find_witness_in_patterns(
            net,
            patterns,
            exclude_tags=exclude_tags,
            boundary_shi=boundary_shi,
        )
        if witness is not None:
            _pricing_log(
                "boundary pricing MIP: witness found from final incumbent (callback fallback)",
                verbose=verbose,
            )
            return witness

        raise BoundaryPricingIncompleteError("boundary pricing MIP finished with a feasible incumbent but no valid witness")
    finally:
        excl_pbar.close()
        model.close()


def price_boundary_witness(
    net: ReLUNetwork,
    boundary_shi: int,
    exclude_tags: set[bytes] | None = None,
    *,
    bound: float | None = None,
    env: Env | None = None,
    eps: float | None = None,
    verbose: bool = False,
    pricing_call: int | None = None,
) -> Polyhedron | None:
    """Find a new top-dimensional cell on ``boundary_shi`` not in ``exclude_tags``.

    Tries brute-force sign scan on tiny networks, then a Gurobi MIP pricing model.

    Returns:
        A new witness polyhedron, or ``None`` when the MIP is **proven infeasible**
        (no uncut feasible sign pattern remains).

    Raises:
        BoundaryPricingIncompleteError: If pricing stops without a witness or proven
            infeasibility (e.g. time limit without solution, or solver stall).
    """
    exclude_tags = exclude_tags or set()
    bound = estimate_input_bound(net, margin=float(cfg.BOUNDARY_MIP_BOUND_MARGIN)) if bound is None else float(bound)
    env = env or get_env()
    eps = float(cfg.BOUNDARY_MIP_EPS if eps is None else eps)

    witness = _brute_force_boundary_witness(net, boundary_shi, exclude_tags)
    if witness is not None:
        _pricing_log(
            "boundary pricing: brute-force witness found " + f"(excluded_tags={len(exclude_tags)})",
            verbose=verbose,
        )
        return witness

    n = count_relu_units(net)
    if n <= cfg.BOUNDARY_PRICING_BRUTE_FORCE_MAX_N:
        _pricing_log(
            "boundary pricing: brute-force found no witness "
            + f"(all {2 ** max(0, n - 1)} sign patterns excluded or infeasible)",
            verbose=verbose,
        )
        return None

    return _mip_boundary_witness(
        net,
        boundary_shi,
        exclude_tags,
        bound=bound,
        env=env,
        eps=eps,
        verbose=verbose,
        pricing_call=pricing_call,
    )
