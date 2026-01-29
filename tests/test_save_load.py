"""Tests for Complex save/load roundtrip."""

import numpy as np
import torch

from relucent import Complex, get_mlp_model

from tests.helpers import ss_to_numpy


def test_complex_save_load_roundtrip_with_ssm(tmp_path, seeded):
    net = get_mlp_model(widths=[4, 7, 3], add_last_relu=True)
    cplx = Complex(net)
    points = [torch.rand((1, 4), device=net.device, dtype=net.dtype) for _ in range(3)]
    for pt in points:
        cplx.add_point(pt)

    path_with_ssm = tmp_path / "cplx_with_ssm.pkl"
    cplx.save(path_with_ssm, save_ssm=True)
    loaded_with_ssm = Complex.load(path_with_ssm)

    assert len(loaded_with_ssm) == len(cplx)
    assert all(p.net is loaded_with_ssm.net for p in loaded_with_ssm)

    for p in cplx:
        assert p.ss in loaded_with_ssm
        assert np.array_equal(ss_to_numpy(loaded_with_ssm[p.ss].ss), ss_to_numpy(p.ss))

    x = torch.rand((2, 4), device=net.device, dtype=net.dtype)
    assert torch.allclose(loaded_with_ssm.net(x), net(x))


def test_complex_save_load_roundtrip_no_ssm(tmp_path, seeded):
    net = get_mlp_model(widths=[4, 7, 3], add_last_relu=True)
    cplx = Complex(net)
    points = [torch.rand((1, 4), device=net.device, dtype=net.dtype) for _ in range(3)]
    for pt in points:
        cplx.add_point(pt)

    path_no_ssm = tmp_path / "cplx_no_ssm.pkl"
    cplx.save(path_no_ssm, save_ssm=False)
    loaded_no_ssm = Complex.load(path_no_ssm)

    assert len(loaded_no_ssm) == len(cplx)
    for p in cplx:
        assert p.ss in loaded_no_ssm
        assert np.array_equal(ss_to_numpy(loaded_no_ssm[p.ss].ss), ss_to_numpy(p.ss))
