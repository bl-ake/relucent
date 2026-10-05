"""Tests for Complex save/load roundtrip."""

import numpy as np
import torch

from relucent import Complex, set_seeds, torch_mlp
from relucent.model.builders import TorchMLP
from tests.helpers import ss_to_numpy


def test_complex_save_load_roundtrip_with_ssm(tmp_path, seed: int):
    set_seeds(seed)
    net = torch_mlp(widths=[4, 7, 3], add_last_relu=True)
    assert isinstance(net, TorchMLP)
    cplx = Complex(net)
    points = [torch.rand((1, 4), device=net.device, dtype=net.dtype) for _ in range(3)]
    for pt in points:
        cplx.add_point(pt)

    path_with_ssm = tmp_path / "cplx_with_ssm.pkl"
    cplx.save(path_with_ssm, save_ssm=True)
    loaded_with_ssm = Complex.load(path_with_ssm)

    assert len(loaded_with_ssm) == len(cplx)
    assert all(p._net is loaded_with_ssm._net for p in loaded_with_ssm)

    for p in cplx:
        assert p.ss in loaded_with_ssm
        assert np.array_equal(ss_to_numpy(loaded_with_ssm[p.ss].ss), ss_to_numpy(p.ss))

    x = torch.rand((2, 4), device=net.device, dtype=net.dtype)
    assert torch.allclose(torch.as_tensor(loaded_with_ssm._net(x), dtype=x.dtype), torch.as_tensor(net(x), dtype=x.dtype))


def test_complex_save_load_roundtrip_no_ssm(tmp_path, seed: int):
    set_seeds(seed)
    net = torch_mlp(widths=[4, 7, 3], add_last_relu=True)
    assert isinstance(net, TorchMLP)
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


def _explored(seed: int) -> Complex:
    import relucent

    relucent.set_seeds(seed)
    cplx = Complex(relucent.mlp(widths=[2, 5, 1]))
    cplx.bfs(verbose=0)
    return cplx


def test_load_keeps_exploration_state(tmp_path, seed: int):
    """A saved complete, verified complex runs topology after load without set_exploration_state."""
    cplx = _explored(seed)
    assert cplx.complete is True and cplx.verified is True
    expected = cplx.betti_numbers()
    path = tmp_path / "cplx.pkl"
    cplx.save(path)
    loaded = Complex.load(path)
    assert loaded.complete is True and loaded.verified is True
    loaded._betti_cache.clear()  # recompute, which requires the complex to be complete and verified
    assert loaded.betti_numbers() == expected


def test_load_refuses_a_newer_format(tmp_path, seed: int):
    import pickle

    import pytest

    path = tmp_path / "cplx.pkl"
    _explored(seed).save(path)
    with open(path, "rb") as f:
        state = pickle.load(f)
    state["format_version"] = Complex.SAVE_FORMAT_VERSION + 1
    with open(path, "wb") as f:
        pickle.dump(state, f)
    with pytest.raises(ValueError, match="Upgrade relucent"):
        Complex.load(path)


def test_load_accepts_a_file_without_a_format_version(tmp_path, seed: int):
    """Files saved before relucent 1.0 load, without their exploration state."""
    import pickle

    cplx = _explored(seed)
    state = cplx.__getstate__()
    for key in ("_complete", "_verified"):
        del state[key]
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as f:
        pickle.dump(state, f)
    loaded = Complex.load(path)
    assert len(loaded) == len(cplx)
    assert loaded.complete is None and loaded.verified is None
