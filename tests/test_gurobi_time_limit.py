"""GUROBI_TIME_LIMIT: Gurobi's TimeLimit on relucent's environment, and what a timed-out solve does."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from gurobipy import GRB, Model

import relucent.config as cfg
from relucent import AmbiguousGeometryError, Complex, Polyhedron, SolverTimeLimitError, set_seeds, torch_mlp
from relucent._internal.gurobi import get_env, time_limit_param
from relucent._internal.network_scale import default_polyhedron_bound
from relucent.geometry import calculations
from relucent.search import boundary_mip
from relucent.search.exploration import explore_for_topology

# A limit no solve can meet: every solve stops with status TIME_LIMIT.
NO_TIME = 1e-9

# Unit square, rows a . x + b <= 0: its Chebyshev radius is 1.
SQUARE = np.array([[1.0, 0.0, -1.0], [-1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0]])


@pytest.fixture
def gurobi_time_limit() -> Iterator[Callable[[float], None]]:
    """Set GUROBI_TIME_LIMIT for one test; the cached environment is put back afterwards."""
    saved = cfg.GUROBI_TIME_LIMIT

    def set_limit(seconds: float) -> None:
        cfg.update_settings(GUROBI_TIME_LIMIT=seconds)
        get_env()  # write it to the cached environment now, for code holding the environment

    yield set_limit
    set_limit(saved)


def _model_time_limit() -> float:
    model = Model("probe", get_env())
    try:
        return float(model.Params.TimeLimit)
    finally:
        model.close()


def test_time_limit_param() -> None:
    assert time_limit_param(3) == 3.0
    for no_limit in (None, 0, 0.0, -1.0):
        assert time_limit_param(no_limit) == GRB.INFINITY


def test_models_in_relucent_env_follow_the_setting(gurobi_time_limit: Callable[[float], None]) -> None:
    """Models built in relucent's environment get the setting, re-read for the models built next."""
    assert _model_time_limit() == time_limit_param(cfg.GUROBI_TIME_LIMIT)
    gurobi_time_limit(2.5)
    assert _model_time_limit() == 2.5
    gurobi_time_limit(0.0)
    assert math.isinf(_model_time_limit())


def test_timed_out_chebyshev_lp_raises(gurobi_time_limit: Callable[[float], None]) -> None:
    """A Chebyshev LP that stops at the limit is never read as an empty or unbounded cell."""
    _center, radius = calculations.solve_radius(get_env(), SQUARE, max_radius=10.0)
    assert radius == pytest.approx(1.0)
    gurobi_time_limit(NO_TIME)
    with pytest.raises(SolverTimeLimitError, match="GUROBI_TIME_LIMIT"):
        calculations.solve_radius(get_env(), SQUARE, max_radius=10.0)


@pytest.mark.usefixtures("seeded")
def test_timed_out_search_raises_instead_of_running_on(gurobi_time_limit: Callable[[float], None]) -> None:
    """A search whose solves all time out raises SolverTimeLimitError, never a ValueError.

    Some handlers read a ValueError from the Chebyshev LP as "this cell is empty", so a timed-out
    solve must not be one: it is an AmbiguousGeometryError, a RuntimeError.
    """
    assert issubclass(SolverTimeLimitError, AmbiguousGeometryError)
    assert not issubclass(SolverTimeLimitError, ValueError)
    cplx = Complex(torch_mlp(widths=[2, 4, 1]))
    gurobi_time_limit(NO_TIME)
    with pytest.raises(SolverTimeLimitError):
        cplx.bfs(start=np.zeros((1, 2)), nworkers=1, verbose=False)


def test_timed_out_shi_lps_are_decided_exactly(seeded: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """When every SHI LP after a cell's first solve times out, exact arithmetic gives the same facets."""
    set_seeds(seeded)
    cplx = Complex(torch_mlp(widths=[2, 5, 5, 1]))
    cplx.bfs(start=np.zeros((1, 2)), nworkers=1, verbose=False, max_polys=12)
    net = cplx._net
    bound = default_polyhedron_bound(net)
    cells = [p.ss_np for p in cplx][:8]
    assert len(cells) >= 4

    def facets() -> list[list[int]]:
        return [sorted(int(i) for i in calculations.shis(Polyhedron(net, ss, bound=bound), bound=bound)) for ss in cells]

    expected = facets()
    timed_out: list[int] = []

    class TimesOutAfterFirstSolve(Model):
        """A SHI LP model whose first solve (the cell's feasibility check) has no limit, and every later one none to spare."""

        _solved_once: bool = False

        def optimize(self, *args: Any, **kwargs: Any) -> None:
            if self.ModelName == "SHIS" and self._solved_once:
                self.Params.TimeLimit = NO_TIME
            self._solved_once = True
            super().optimize(*args, **kwargs)
            if self.Status == GRB.TIME_LIMIT:
                timed_out.append(1)

    monkeypatch.setattr(calculations, "Model", TimesOutAfterFirstSolve)
    assert facets() == expected
    assert timed_out, "no SHI LP timed out, so the exact path was not exercised"


@pytest.mark.usefixtures("seeded")
def test_boundary_pricing_mip_keeps_its_own_time_limit(
    monkeypatch: pytest.MonkeyPatch, gurobi_time_limit: Callable[[float], None]
) -> None:
    """The per-LP GUROBI_TIME_LIMIT never cuts a pricing MIP short: it uses BOUNDARY_MIP_TIME_LIMIT."""
    fc = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    fc.weight.data[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    cplx = Complex(nn.Sequential(fc, nn.ReLU()))
    explore_for_topology(cplx, np.array([0.5, 0.0]))
    shi = cplx.n - 1
    limits: list[float] = []

    class RecordingModel(Model):
        def optimize(self, *args: Any, **kwargs: Any) -> None:
            if self.ModelName == "boundary_pricing":
                limits.append(float(self.Params.TimeLimit))
            super().optimize(*args, **kwargs)

    monkeypatch.setattr(boundary_mip, "Model", RecordingModel)
    monkeypatch.setattr(cfg.advanced, "BOUNDARY_PRICING_BRUTE_FORCE_MAX_N", 0)
    gurobi_time_limit(1.5)
    for mip_limit, expected in ((0.0, math.inf), (7.0, 7.0)):
        monkeypatch.setattr(cfg, "BOUNDARY_MIP_TIME_LIMIT", mip_limit)
        limits.clear()
        boundary_mip.price_boundary_witness(cplx._net, shi, set())
        assert limits, "the pricing MIP was not solved"
        assert all(limit == expected for limit in limits)
