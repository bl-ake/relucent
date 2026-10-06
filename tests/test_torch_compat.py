"""The tensor check and gradient decorator in ``relucent._internal.torch_compat``."""

import numpy as np
import pytest

from relucent._internal.torch_compat import is_torch_tensor, no_grad


def test_is_torch_tensor():
    assert not is_torch_tensor(np.zeros(3))
    assert not is_torch_tensor([0.0, 1.0])
    torch = pytest.importorskip("torch")
    assert is_torch_tensor(torch.zeros(3))
    assert not is_torch_tensor(np.zeros(3))


def test_no_grad_disables_gradients_in_functions_and_generators():
    torch = pytest.importorskip("torch")

    @no_grad
    def grad_enabled() -> bool:
        return torch.is_grad_enabled()

    @no_grad
    def grad_enabled_while_iterating():
        yield torch.is_grad_enabled()

    assert torch.is_grad_enabled()
    assert grad_enabled() is False
    assert list(grad_enabled_while_iterating()) == [False]
    assert torch.is_grad_enabled()
