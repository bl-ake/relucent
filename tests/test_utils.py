"""Tests for relucent.utils."""

import copy

import numpy as np
import pytest
import torch

from relucent import mlp
from relucent.model import LinearLayer, ReLULayer, ReLUNetwork
from relucent.utils import (
    BlockingQueue,
    NonBlockingQueue,
    TorchMLP,
    UpdatablePriorityQueue,
    add_output_relu,
    encode_ss,
    flip_ss_at_shi,
    flip_ss_at_shi_inplace,
    get_env,
    normalize_weights,
    set_seeds,
    split_sequential,
)
from relucent.vis import get_colors


class TestSetSeeds:
    def test_determinism(self):
        set_seeds(42)
        a = np.random.rand(3).tolist()
        set_seeds(42)
        b = np.random.rand(3).tolist()
        assert a == b


class TestEncodeSs:
    def test_numpy(self):
        ss = np.array([[1, -1, 0, 1]])
        out = encode_ss(ss)
        assert isinstance(out, bytes)
        assert encode_ss(ss) == encode_ss(ss.copy())

    def test_torch(self):
        ss = torch.tensor([[1.0, -1.0, 0.0]])
        out = encode_ss(ss)
        assert isinstance(out, bytes)

    def test_same_content_same_tag(self):
        a = np.array([[1, -1, 0]])
        b = np.array([[1, -1, 0]])
        assert encode_ss(a) == encode_ss(b)


class TestFlipSsAtShi:
    def test_flips_coordinate(self):
        ss = np.array([[1, -1, 0, 1]], dtype=np.int8)
        flipped = flip_ss_at_shi(ss, 1)
        assert flipped.shape == ss.shape
        assert flipped[0, 1] == 1
        assert encode_ss(flipped) == encode_ss(np.array([[1, 1, 0, 1]], dtype=np.int8))

    def test_does_not_mutate_input(self):
        ss = np.array([[1, -1]], dtype=np.int8)
        original = ss.copy()
        flip_ss_at_shi(ss, 0)
        assert np.array_equal(ss, original)

    def test_torch(self):
        ss = torch.tensor([[1.0, -1.0, 0.0]])
        flipped = flip_ss_at_shi(ss, 1)
        assert isinstance(flipped, np.ndarray)
        assert flipped[0, 1] == 1

    def test_inplace_toggle_restores(self):
        ss = np.array([[1, -1, 0]], dtype=np.int8)
        original = ss.copy()
        flip_ss_at_shi_inplace(ss, 1)
        assert ss[0, 1] == 1
        flip_ss_at_shi_inplace(ss, 1)
        assert np.array_equal(ss, original)


class TestGetEnv:
    def test_returns_env(self):
        env = get_env()
        assert env is not None

    def test_cached(self):
        e1 = get_env()
        e2 = get_env()
        assert e1 is e2


class TestBlockingQueue:
    def test_default_pop_order(self):
        """Default pop is deque.pop() (right end), so LIFO order."""
        q = BlockingQueue()
        q.push(1)
        q.push(2)
        q.push(3)
        assert q.pop() == 3
        assert q.pop() == 2
        assert q.pop() == 1

    def test_lifo_pop(self):
        q = BlockingQueue(pop=lambda d: d.popleft(), push=lambda d, x: d.append(x))
        q.push(1)
        q.push(2)
        q.push(3)
        assert q.pop() == 1
        assert q.pop() == 2
        assert q.pop() == 3


class TestNonBlockingQueue:
    def test_push_pop(self):
        q = NonBlockingQueue()
        q.push(10)
        q.push(20)
        assert q.pop() == 20
        assert q.pop() == 10

    def test_len(self):
        q = NonBlockingQueue()
        assert len(q) == 0
        q.push(1)
        assert len(q) == 1


class TestUpdatablePriorityQueue:
    def test_push_pop_order(self):
        pq = UpdatablePriorityQueue()
        pq.push(("a", 1), priority=2)
        pq.push(("b", 2), priority=1)
        pq.push(("c", 3), priority=3)
        assert pq.pop() == ("b", 2)
        assert pq.pop() == ("a", 1)
        assert pq.pop() == ("c", 3)

    def test_update_priority(self):
        pq = UpdatablePriorityQueue()
        pq.push(("x", 1), priority=10)
        pq.push(("y", 2), priority=5)
        pq.push(("x", 1), priority=0)  # same task -> update
        assert pq.pop() == ("x", 1)
        assert pq.pop() == ("y", 2)

    def test_remove_task(self):
        pq = UpdatablePriorityQueue()
        pq.push(("a", 1), priority=1)
        pq.push(("b", 2), priority=2)
        pq.remove_task(("b", 2))
        assert pq.pop() == ("a", 1)
        with pytest.raises(KeyError):
            pq.pop()

    def test_len(self):
        pq = UpdatablePriorityQueue()
        assert len(pq) == 0
        pq.push(("a", 1), priority=0)
        assert len(pq) == 1
        pq.push(("a", 1), priority=1)
        assert len(pq) == 1

    def test_hashable_non_tuple_tasks(self):
        pq = UpdatablePriorityQueue()
        pq.push("alpha", priority=2)
        pq.push("beta", priority=1)
        pq.push("alpha", priority=0)  # update existing task
        assert pq.pop() == "alpha"
        assert pq.pop() == "beta"


class TestGetColors:
    def test_empty(self):
        assert get_colors([]) == []

    def test_single(self):
        out = get_colors([0.5])
        assert len(out) == 1
        assert out[0].startswith("#") and len(out[0]) == 7

    def test_range(self):
        out = get_colors([0, 0.5, 1.0])
        assert len(out) == 3
        assert out[0] != out[-1]


class TestSplitSequential:
    def test_split(self, seeded):
        assert seeded is not None
        net = mlp(widths=[4, 8, 6, 2])
        assert isinstance(net, TorchMLP)
        nn1, nn2 = split_sequential(net, "relu0")
        x = torch.zeros((1, 4), device=net.device, dtype=net.dtype)
        y1 = nn1(x)
        y_full = net(x)
        y2 = nn2(y1)
        assert torch.allclose(torch.as_tensor(y_full), torch.as_tensor(y2), atol=1e-5)

    def test_split_layer_in_first(self, seeded):
        assert seeded is not None
        net = mlp(widths=[2, 4, 2])
        assert isinstance(net, TorchMLP)
        nn1, nn2 = split_sequential(net, "fc0")
        assert "fc0" in nn1.layers
        assert "fc1" in nn2.layers or "relu0" in nn2.layers


class TestAddOutputRelu:
    def test_torch_mlp_appends_relu(self, seeded):
        assert seeded is not None
        net = mlp(widths=[2, 4, 1], add_last_relu=False)
        assert isinstance(net, TorchMLP)
        topo = add_output_relu(net)
        assert isinstance(topo, TorchMLP)
        assert topo is not net
        assert list(topo.children())[-1].__class__.__name__ == "ReLU"
        assert topo.widths == net.widths

    def test_preserves_linear_outputs_before_relu(self, seeded):
        assert seeded is not None
        net = mlp(widths=[3, 5, 1], add_last_relu=False)
        assert isinstance(net, TorchMLP)
        topo = add_output_relu(net)
        x = torch.randn(4, 3, device=net.device, dtype=net.dtype)
        pre_relu = net(x)
        post_relu = topo(x)
        assert torch.allclose(post_relu, torch.relu(pre_relu))

    def test_already_has_output_relu_raises(self, seeded):
        assert seeded is not None
        net = mlp(widths=[2, 4, 1], add_last_relu=True)
        assert isinstance(net, TorchMLP)
        with pytest.raises(ValueError, match="already ends with a ReLU"):
            add_output_relu(net)

    def test_relu_network(self, seeded):
        assert seeded is not None
        layers = {
            "fc0": LinearLayer(
                weight=np.random.randn(4, 2).astype(np.float64),
                bias=np.random.randn(1, 4).astype(np.float64),
            ),
            "relu0": ReLULayer(),
            "fc1": LinearLayer(
                weight=np.random.randn(1, 4).astype(np.float64),
                bias=np.random.randn(1, 1).astype(np.float64),
            ),
        }
        net = ReLUNetwork(layers, input_shape=(2,))
        topo = add_output_relu(net)
        assert isinstance(topo, ReLUNetwork)
        assert isinstance(list(topo.layers.values())[-1], ReLULayer)
        assert topo.num_relus == net.num_relus + 1


class TestNormalizeWeights:
    @pytest.mark.parametrize("widths", ([2, 4, 2], [4, 8, 6, 2]))
    def test_function_invariant(self, seeded, widths):
        assert seeded is not None
        net = mlp(widths=widths)
        assert isinstance(net, TorchMLP)
        original = copy.deepcopy(net)

        x = torch.randn(32, widths[0], device=net.device, dtype=net.dtype)
        y_before = original(x)

        normalize_weights(net)
        y_after = net(x)

        assert torch.allclose(torch.as_tensor(y_before), torch.as_tensor(y_after), atol=1e-5, rtol=1e-5)
