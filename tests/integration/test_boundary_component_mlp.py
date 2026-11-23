"""Heavy MLP boundary-component parity test (opt-in via RELUCENT_RUN_INTEGRATION)."""

from __future__ import annotations

import pytest

from relucent import Complex, add_output_relu, set_seeds, torch_mlp
from tests.helpers import cover_by_components, summed_betti

pytestmark = [pytest.mark.integration]


def test_boundary_components_cover_mlp_medium(seeded: int, integration_nworkers: int) -> None:
    set_seeds(seeded)
    model = add_output_relu(torch_mlp(widths=[4, 12, 12, 12, 12, 1]))
    cplx = Complex(model)
    cplx.bfs(verbose=0, nworkers=integration_nworkers)
    shi = cplx.n - 1
    ref = cplx.boundary_complex(shi, verbose=False)
    components = cover_by_components(ref, model, shi, nworkers=integration_nworkers)
    assert summed_betti(components) == {int(k): int(v) for k, v in ref.betti_numbers().items() if v}
