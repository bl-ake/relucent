"""Search completeness when a flip-neighbor fails: nothing is excused as a "phantom".

Neighbors are queued only across certified facets, so the region across one is nonempty. A
neighbor that comes back empty therefore signals an upstream error and must leave BFS
incomplete; an undecidable one (``AmbiguousGeometryError``) is raised outright.
"""

from __future__ import annotations

import numpy as np
import pytest

from relucent import AmbiguousGeometryError, Complex, torch_mlp
from relucent.search.engine import blocking_bad_shi_computations, true_phantom_neighbor_error


def test_no_error_is_a_phantom() -> None:
    assert not true_phantom_neighbor_error("Polyhedron is infeasible (empty).")
    assert not true_phantom_neighbor_error("Inradius -1.0000e-12")
    assert not true_phantom_neighbor_error("Model status: 5")


def test_every_failed_neighbor_blocks_completeness() -> None:
    empty = (object(), 1, 1, "Polyhedron is infeasible (empty).")
    fault = (object(), 2, 1, "Model status: 5")
    assert blocking_bad_shi_computations([empty, fault]) == [empty, fault]


class _SyncPool:
    """In-process ``WorkerPool`` stand-in so monkeypatches survive on macOS (spawn workers)."""

    def __init__(self, _nworkers: int, *, initializer=None, initargs=()) -> None:
        if initializer is not None:
            initializer(*initargs)

    def imap_unordered(self, func, iterable, chunksize=1):
        _ = chunksize
        for item in iterable:
            yield func(item)

    def shutdown(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        pass


def _patch_first_worker(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    import relucent.search.engine as search_mod

    original = search_mod._worker_prepare_poly
    calls = {"n": 0}

    def _fake_worker(p, props, *, env, shis_kwargs=None, need_interior=False, shuffle_shis=False):
        calls["n"] += 1
        if calls["n"] == 1:
            return error
        return original(
            p,
            props,
            env=env,
            shis_kwargs=shis_kwargs,
            need_interior=need_interior,
            shuffle_shis=shuffle_shis,
        )

    monkeypatch.setattr(search_mod, "_worker_prepare_poly", _fake_worker)
    monkeypatch.setattr(search_mod, "worker_pool", _SyncPool)


def test_bfs_incomplete_when_a_neighbor_comes_back_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_first_worker(monkeypatch, ValueError("Polyhedron is infeasible (empty)."))
    cplx = Complex(torch_mlp(widths=[2, 6, 1], add_last_relu=True))
    stats = cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, nworkers=1, verify=False)
    assert len(stats.bad_shi_computations) >= 1
    assert cplx.complete is not True


def test_bfs_raises_on_ambiguous_neighbor(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_first_worker(monkeypatch, AmbiguousGeometryError("cannot decide"))
    cplx = Complex(torch_mlp(widths=[2, 6, 1], add_last_relu=True))
    with pytest.raises(AmbiguousGeometryError):
        cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, nworkers=1)
