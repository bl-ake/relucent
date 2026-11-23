"""Sublevel filtrations of a network output on unbounded complexes, against exact answers.

Each network below is a sum of axis-parallel ReLU ramps in the plane, so its sublevel sets
``{g <= t}`` are known exactly at every threshold. The units on the other axis (output
weight zero) only add vertices, so every cell is pointed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from relucent import Complex
from relucent.search.exploration import explore_for_topology
from relucent.topology.filtration import AffineOutputFiltration, LogitSublevelFiltration, TrainingDistanceFiltration
from relucent.topology.persistence import betti_curve, compute_persistent_homology

# Units: ReLU(x1 - 1), ReLU(-x1 - 1), ReLU(x2 - 1), ReLU(-x2 - 1).
_W0 = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
_B0 = [-1.0, -1.0, -1.0, -1.0]


def _ramp_net(out_weights: list[float], out_bias: float = 0.0, *, trailing_relu: bool = False) -> nn.Sequential:
    fc0 = nn.Linear(2, 4, dtype=torch.float64)
    fc1 = nn.Linear(4, 1, dtype=torch.float64)
    with torch.no_grad():
        fc0.weight.copy_(torch.tensor(_W0, dtype=torch.float64))
        fc0.bias.copy_(torch.tensor(_B0, dtype=torch.float64))
        fc1.weight.copy_(torch.tensor([out_weights], dtype=torch.float64))
        fc1.bias.fill_(out_bias)
    layers: list[nn.Module] = [fc0, nn.ReLU(), fc1]
    if trailing_relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _diagram(net: nn.Sequential, fil=None):
    cplx = Complex(net)
    explore_for_topology(cplx, np.array([0.1, 0.2]))
    return compute_persistent_homology(cplx, fil or LogitSublevelFiltration())


def _betti_at(diagram, ts: list[float], dim: int) -> list[int]:
    curve = betti_curve(diagram, ts, max_dim=max(dim, 1))
    return [int(b) for b in curve[dim]]


THRESHOLDS = [-100.0, -2.0, -1.5, -0.5, 0.0, 0.5, 100.0, math.inf]


def test_bowl_is_empty_below_its_minimum_then_a_disk():
    """g = sum of all four ramps - 1: minimum -1 on the unit square, +inf in every direction."""
    d = _diagram(_ramp_net([1.0, 1.0, 1.0, 1.0], -1.0))
    assert _betti_at(d, THRESHOLDS, 0) == [0, 0, 0, 1, 1, 1, 1, 1]
    assert _betti_at(d, THRESHOLDS, 1) == [0] * len(THRESHOLDS)
    (h0,) = [p for p in d.pairs if p.dimension == 0 and p.death_cell is None]
    assert h0.birth == pytest.approx(-1.0)
    # The circle of caps at infinity must not enter at a finite value.
    assert not [p for p in d.pairs if p.dimension == 1 and math.isfinite(p.birth) and p.birth < p.death]


def test_trailing_relu_is_ignored_by_the_logit():
    """With an output ReLU the complex holds g = 0, but the logit filtration still reads g."""
    d = _diagram(_ramp_net([1.0, 1.0, 1.0, 1.0], -1.0, trailing_relu=True))
    assert _betti_at(d, THRESHOLDS, 0) == [0, 0, 0, 1, 1, 1, 1, 1]
    (h0,) = [p for p in d.pairs if p.dimension == 0 and p.death_cell is None]
    assert h0.birth == pytest.approx(-1.0)

    # The network output itself is ReLU(g): 0 on the square, so its sublevel sets start at 0.
    out = _diagram(_ramp_net([1.0, 1.0, 1.0, 1.0], -1.0, trailing_relu=True), AffineOutputFiltration())
    (h0_out,) = [p for p in out.pairs if p.dimension == 0 and p.death_cell is None]
    assert h0_out.birth == pytest.approx(0.0)


def test_monotone_ramp_is_one_component_born_at_minus_infinity():
    """g = ReLU(x1 - 1) - ReLU(-x1 - 1): nondecreasing in x1, so {g <= t} is a nonempty half-plane-like set."""
    d = _diagram(_ramp_net([1.0, -1.0, 0.0, 0.0]))
    assert _betti_at(d, THRESHOLDS, 0) == [1] * len(THRESHOLDS)
    assert _betti_at(d, THRESHOLDS, 1) == [0] * len(THRESHOLDS)
    (h0,) = [p for p in d.pairs if p.dimension == 0 and p.death_cell is None]
    assert h0.birth == -math.inf


def test_two_valleys_at_infinity_merge_at_zero():
    """g = -ReLU(x1 - 1) - ReLU(-x1 - 1): two half-planes for t < 0, the whole plane from t = 0."""
    d = _diagram(_ramp_net([-1.0, -1.0, 0.0, 0.0]))
    assert _betti_at(d, THRESHOLDS, 0) == [2, 2, 2, 2, 1, 1, 1, 1]
    assert _betti_at(d, THRESHOLDS, 1) == [0] * len(THRESHOLDS)
    finite_h0 = [p for p in d.pairs if p.dimension == 0 and p.birth < p.death and math.isfinite(p.death)]
    assert [(p.birth, p.death) for p in finite_h0] == [(-math.inf, pytest.approx(0.0))]


def test_lower_star_off_is_rejected():
    cplx = Complex(_ramp_net([1.0, 1.0, 1.0, 1.0], -1.0))
    explore_for_topology(cplx, np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="not monotone"):
        compute_persistent_homology(cplx, LogitSublevelFiltration(), lower_star=False)


def test_training_distance_sends_unbounded_cells_and_caps_to_infinity():
    cplx = Complex(_ramp_net([1.0, 1.0, 1.0, 1.0], -1.0))
    explore_for_topology(cplx, np.array([0.1, 0.2]))
    d = compute_persistent_homology(cplx, TrainingDistanceFiltration(np.zeros((1, 2))))
    meta_unbounded = [v for v in d.cell_filtration.values() if not math.isfinite(v)]
    assert meta_unbounded and all(v == math.inf for v in meta_unbounded)
    # The bounded cells (the central square and its faces) form a disk.
    assert _betti_at(d, [100.0], 0) == [1]
    assert _betti_at(d, [100.0], 1) == [0]


@pytest.mark.parametrize("d", [2, 3])
def test_cells_containing_a_line_are_rejected(d: int):
    """Two parallel hyperplanes: every cell contains a line, so no vertex anchors the values.

    At d = 3 the complex has no 1-cells at all (slabs and planes), so this needs the
    pointedness check, not just a check for 1-cells without endpoints.
    """
    fc0 = nn.Linear(d, 2, dtype=torch.float64)
    fc1 = nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        normal = [1.0] + [0.0] * (d - 1)
        fc0.weight.copy_(torch.tensor([normal, normal], dtype=torch.float64))
        fc0.bias.copy_(torch.tensor([-1.0, 1.0], dtype=torch.float64))
        fc1.weight.copy_(torch.tensor([[1.0, 1.0]], dtype=torch.float64))
        fc1.bias.fill_(0.0)
    cplx = Complex(nn.Sequential(fc0, nn.ReLU(), fc1))
    explore_for_topology(cplx, np.full(d, 0.1))
    with pytest.raises(ValueError, match="no vertex"):
        compute_persistent_homology(cplx, LogitSublevelFiltration())


def test_binary_needs_two_logits():
    cplx = Complex(_ramp_net([1.0, 1.0, 1.0, 1.0], -1.0))
    explore_for_topology(cplx, np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="two outputs"):
        compute_persistent_homology(cplx, LogitSublevelFiltration(binary=True))


def _random_net(d: int, widths: list[int], seed: int) -> nn.Sequential:
    gen = torch.Generator().manual_seed(seed)
    dims = [d, *widths, 1]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        fc = nn.Linear(dims[i], dims[i + 1], dtype=torch.float64)
        with torch.no_grad():
            fc.weight.copy_(torch.randn(dims[i + 1], dims[i], generator=gen, dtype=torch.float64))
            fc.bias.copy_(torch.randn(dims[i + 1], generator=gen, dtype=torch.float64))
        layers.append(fc)
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _region_betti(net: nn.Sequential, t: float, d: int) -> list[int]:
    """Homology of the closed region {g <= t}, computed without any filtration values.

    Shift the output bias by -t and append a ReLU, so the level set g = t is in the complex,
    truncate, and rank the closed subcomplex of cells where the output unit is not active.
    """
    import copy

    from relucent import add_output_relu
    from relucent.graph.meta_graph import truncate_meta_graph
    from relucent.topology import betti_numbers

    shifted = copy.deepcopy(net)
    last = shifted[-1]
    assert isinstance(last, nn.Linear)
    with torch.no_grad():
        last.bias.sub_(t)
    cplx = Complex(add_output_relu(shifted))
    explore_for_topology(cplx, np.zeros(d) + 0.01)
    meta = cplx.meta_graph()
    truncate_meta_graph(meta)
    keep = [n for n, a in meta.nodes(data=True) if int(np.asarray(a["ss"]).reshape(-1)[cplx.n - 1]) <= 0]
    if not keep:
        return [0] * d
    b = betti_numbers(meta.subgraph(keep).copy())
    return [int(b.get(k, 0)) for k in range(d)]


@pytest.mark.parametrize(("d", "widths", "seed"), [(2, [6, 6], 0), (2, [5, 5, 4], 3), (3, [5, 5], 1)])
def test_logit_sublevel_homology_matches_the_region(d: int, widths: list[int], seed: int):
    net = _random_net(d, widths, seed)
    cplx = Complex(net)
    explore_for_topology(cplx, np.zeros(d) + 0.01)
    diagram = compute_persistent_homology(cplx, LogitSublevelFiltration())
    values = sorted(v for v in diagram.cell_filtration.values() if math.isfinite(v))
    ts = list(np.random.default_rng(seed).uniform(values[0] - 1.0, values[-1] + 1.0, 3))
    curve = betti_curve(diagram, ts, max_dim=d - 1)
    for i, t in enumerate(ts):
        assert [int(curve[k][i]) for k in range(d)] == _region_betti(net, t, d), f"t = {t}"
