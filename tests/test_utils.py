"""Tests for relucent.utils."""

import numpy as np
import pytest
import torch

from relucent.utils import (
    BlockingQueue,
    NonBlockingQueue,
    UpdatablePriorityQueue,
    encode_ss,
    get_colors,
    get_env,
    set_seeds,
    split_sequential,
)

from relucent import get_mlp_model


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
        pq.push(("x", 1), priority=0)  # same tail (1,) -> update
        assert pq.pop() == ("x", 1)
        assert pq.pop() == ("y", 2)

    def test_remove_task(self):
        pq = UpdatablePriorityQueue()
        pq.push(("a", 1), priority=1)
        pq.push(("b", 2), priority=2)
        pq.remove_task((2,))
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
        net = get_mlp_model(widths=[4, 8, 6, 2])
        nn1, nn2 = split_sequential(net, "relu0")
        x = torch.zeros((1, 4), device=net.device, dtype=net.dtype)
        y1 = nn1(x)
        y_full = net(x)
        y2 = nn2(y1)
        assert torch.allclose(y_full, y2, atol=1e-5)

    def test_split_layer_in_first(self, seeded):
        net = get_mlp_model(widths=[2, 4, 2])
        nn1, nn2 = split_sequential(net, "fc0")
        assert "fc0" in nn1.layers
        assert "fc1" in nn2.layers or "relu0" in nn2.layers
