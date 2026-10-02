"""SHI LPs that Gurobi fails on under the configured and the automatic scaling.

Each fixture is a decision-boundary network (scalar output with a final ReLU, float64) from a
training checkpoint, plus ``region_ss``: the sign sequence of one cell whose SHI LP for some row
returned NUMERIC, UNBOUNDED, INFEASIBLE or INF_OR_UNBD although it relaxes a feasible LP. Expected
SHIs come from the Sep 30 2026 code, which recovered every case, and agree with an independent
HiGHS LP.

Retrying with no scaling (ScaleFlag 0) recovers the first five. In the last three, some row fails
under every scaling tried and is decided in exact arithmetic without the LP. Whether Gurobi fails at
all is platform- and version-dependent, so each case checks only the final SHIs. See
docs/search_shi_and_graphs.rst, "LP solver failures".
"""

from __future__ import annotations

import pytest
import torch

from relucent import Complex
from tests.integration.helpers import FIXTURES_DIR, _build_torch_mlp_from_bundle

pytestmark = [pytest.mark.integration]

# (fixture, cell repr, expected SHIs)
_CASES = [
    pytest.param("shi_lp_v2_23471.pt", "111635cf", [4, 27, 40, 43, 45, 48], id="v2_23471"),
    pytest.param("shi_lp_v2_deep_4341.pt", "87451879", [0, 2, 35, 40], id="v2_deep_4341"),
    pytest.param("shi_lp_v2_deep_11256.pt", "4668d0eb", [3, 12, 38, 40], id="v2_deep_11256"),
    pytest.param("shi_lp_v2_deep_22979.pt", "7649826a", [8, 12, 33, 40], id="v2_deep_22979"),
    pytest.param("shi_lp_v2_deep_linear_11192.pt", "dcd7bae0", [5, 32, 39, 47], id="v2_deep_linear_11192"),
    pytest.param("shi_lp_real_70871.pt", "a2e75b34", [6, 19, 32], id="real_70871"),
    pytest.param("shi_lp_v2_deep_4995.pt", "c3044d73", [8, 12, 20, 22, 25, 35, 36], id="v2_deep_4995"),
    pytest.param("shi_lp_v2_deep_linear_22747.pt", "82d93c87", [9, 15, 46, 48], id="v2_deep_linear_22747"),
]


@pytest.mark.parametrize(("fixture", "cell", "expected"), _CASES)
def test_shi_lp_solver_failure_is_recovered(fixture: str, cell: str, expected: list[int]) -> None:
    bundle = torch.load(FIXTURES_DIR / fixture, map_location="cpu", weights_only=True)
    model = _build_torch_mlp_from_bundle(bundle["state_dict"], [int(w) for w in bundle["widths"]])
    poly = Complex(model).add_ss(bundle["region_ss"].numpy(), check_exists=False)
    assert repr(poly) == cell
    assert sorted(int(s) for s in poly.shis) == expected
