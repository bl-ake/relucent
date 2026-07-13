"""Search completeness when flip-neighbors are geometrically empty (true phantoms)."""

from __future__ import annotations

import numpy as np
import pytest

import relucent.config as cfg
from relucent import Complex, mlp
from relucent.search import blocking_bad_shi_computations, true_phantom_neighbor_error


def test_true_phantom_neighbor_error_empty_polyhedron() -> None:
    assert true_phantom_neighbor_error("Polyhedron is infeasible (empty).")


def test_true_phantom_neighbor_error_near_zero_negative_inradius() -> None:
    tol = float(cfg.TOL_INTERIOR_VERIFY)
    assert true_phantom_neighbor_error(f"Inradius {-tol / 2:.4e}")
    assert not true_phantom_neighbor_error(f"Inradius {-2 * tol:.4e}")


def test_true_phantom_neighbor_error_rejects_worker_faults() -> None:
    assert not true_phantom_neighbor_error("Model status: 5")
    assert not true_phantom_neighbor_error("Polyhedron inradius 1.0000e-12 is below MIN_SEARCH_INRADIUS")


def test_blocking_bad_shi_computations_filters_phantoms() -> None:
    phantom = (object(), 1, 1, "Polyhedron is infeasible (empty).")
    real = (object(), 2, 1, "Model status: 5")
    assert blocking_bad_shi_computations([phantom, real]) == [real]


class _SyncPool:
    """In-process Pool stand-in so monkeypatches survive on macOS (spawn workers)."""

    def __init__(self, _nworkers: int, *, initializer=None, initargs=()) -> None:
        if initializer is not None:
            initializer(*initargs)

    def imap_unordered(self, func, iterable, chunksize=1):
        _ = chunksize
        for item in iterable:
            yield func(item)

    def terminate(self) -> None:
        pass

    def join(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        pass


class _SyncMpContext:
    def Pool(self, nworkers: int, initializer=None, initargs=()):
        return _SyncPool(nworkers, initializer=initializer, initargs=initargs)


def test_bfs_complete_when_only_phantom_mistakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phantom-only failures must not leave BFS incomplete."""
    model = mlp(widths=[2, 6, 1], add_last_relu=True)
    cplx = Complex(model)
    real_calls = {"n": 0}

    import relucent.search.engine as search_mod

    original = search_mod._worker_prepare_poly

    def _fake_worker(p, props, *, env, shis_kwargs=None, need_interior=False, shuffle_shis=False):
        real_calls["n"] += 1
        if real_calls["n"] == 1:
            return ValueError("Polyhedron is infeasible (empty).")
        return original(
            p,
            props,
            env=env,
            shis_kwargs=shis_kwargs,
            need_interior=need_interior,
            shuffle_shis=shuffle_shis,
        )

    monkeypatch.setattr(search_mod, "_worker_prepare_poly", _fake_worker)
    monkeypatch.setattr(search_mod, "get_mp_context", lambda: _SyncMpContext())

    stats = cplx.bfs(start=np.zeros((1, 2), dtype=np.float64), verbose=False, nworkers=1)
    assert len(stats["Bad SHI Computations"]) >= 1
    assert cplx.complete is True
