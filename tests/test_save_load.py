import pickle

import numpy as np
import torch

from relucent import Complex, Polyhedron, get_mlp_model, set_seeds


def _bv_to_numpy(bv):
    if isinstance(bv, torch.Tensor):
        return bv.detach().cpu().numpy()
    return bv


def test_polyhedron_pickle_roundtrip():
    set_seeds(0)
    net = get_mlp_model(widths=[3, 6, 2], add_last_relu=True)

    cplx = Complex(net)
    x = torch.rand((1, 3)).to(net.dtype)
    bv = cplx.point2bv(x)

    p = Polyhedron(net, bv)
    # Ensure we can compute the affine map before pickling
    y1 = x @ p.W + p.b
    assert torch.allclose(y1, net(x))

    blob = pickle.dumps(p)
    p2 = pickle.loads(blob)

    # Polyhedron intentionally does NOT pickle the net
    assert p2.net is None
    # BV is saved as numpy in Polyhedron.__reduce__ (and remains numpy on load)
    assert isinstance(p2.bv, (np.ndarray, torch.Tensor))
    assert np.array_equal(_bv_to_numpy(p2.bv), _bv_to_numpy(p.bv))
    assert p2.tag == p.tag

    # After reattaching the net, the polyhedron should behave normally
    p2.net = net
    # When BV is numpy, Polyhedron uses the numpy halfspace path, which produces
    # float64 arrays (e.g. via np.eye). Convert explicitly to net.dtype tensors
    # to avoid Float/Double mismatches in comparisons.
    W2 = torch.as_tensor(p2.W, device=net.device, dtype=net.dtype)
    b2 = torch.as_tensor(p2.b, device=net.device, dtype=net.dtype)
    y2 = x @ W2 + b2
    assert torch.allclose(y2, net(x))
    assert p2.halfspaces.shape == p.halfspaces.shape


def test_complex_save_load_roundtrip(tmp_path):
    set_seeds(0)
    net = get_mlp_model(widths=[4, 7, 3], add_last_relu=True)

    cplx = Complex(net)
    points = [torch.rand((1, 4)).to(net.dtype) for _ in range(3)]
    for pt in points:
        cplx.add_point(pt)

    path_with_bvm = tmp_path / "cplx_with_bvm.pkl"
    cplx.save(path_with_bvm, save_bvm=True)
    loaded_with_bvm = Complex.load(path_with_bvm)

    assert len(loaded_with_bvm) == len(cplx)
    assert all(p.net is loaded_with_bvm.net for p in loaded_with_bvm)

    # Membership / indexing should work after load
    for p in cplx:
        assert p.bv in loaded_with_bvm
        assert np.array_equal(_bv_to_numpy(loaded_with_bvm[p.bv].bv), _bv_to_numpy(p.bv))

    # The network is part of the saved state; it should round-trip too.
    x = torch.rand((2, 4)).to(net.dtype)
    assert torch.allclose(loaded_with_bvm.net(x), net(x))

    # Also exercise the "rebuild bvm on load" path.
    path_no_bvm = tmp_path / "cplx_no_bvm.pkl"
    cplx.save(path_no_bvm, save_bvm=False)
    loaded_no_bvm = Complex.load(path_no_bvm)
    assert len(loaded_no_bvm) == len(cplx)
    for p in cplx:
        assert p.bv in loaded_no_bvm
        assert np.array_equal(_bv_to_numpy(loaded_no_bvm[p.bv].bv), _bv_to_numpy(p.bv))
