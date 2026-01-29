"""Tests for relucent.ss (SSManager)."""

import numpy as np
import pytest
import torch

from relucent.ss import SSManager


class TestSSManager:
    def test_add_and_contains(self):
        m = SSManager()
        ss = np.array([[1, -1, 0, 1]])
        m.add(ss)
        assert ss in m
        assert np.array([[1, -1, 0, 1]]) in m

    def test_getitem(self):
        m = SSManager()
        a = np.array([[1, 0, -1]])
        b = np.array([[0, 1, 0]])
        m.add(a)
        m.add(b)
        assert m[a] == 0
        assert m[b] == 1

    def test_tensor(self):
        m = SSManager()
        ss = torch.tensor([[1.0, -1.0, 0.0]])
        m.add(ss)
        assert ss in m
        assert m[ss] == 0

    def test_delitem(self):
        m = SSManager()
        a = np.array([[1, 0]])
        m.add(a)
        assert len(m) == 1
        del m[a]
        assert len(m) == 0
        assert a not in m

    def test_getitem_keyerror(self):
        m = SSManager()
        m.add(np.array([[1, 0]]))
        with pytest.raises(KeyError):
            m.__getitem__(np.array([[1, 1]]))  # never added

    def test_no_duplicate_add(self):
        m = SSManager()
        a = np.array([[1, -1, 0]])
        m.add(a)
        m.add(a.copy())
        assert len(m) == 1

    def test_iter(self):
        m = SSManager()
        m.add(np.array([[1, 0]]))
        m.add(np.array([[0, 1]]))
        out = list(m)
        assert len(out) == 2

    def test_len(self):
        m = SSManager()
        assert len(m) == 0
        m.add(np.array([[1]]))
        assert len(m) == 1
        m.add(np.array([[-1]]))
        assert len(m) == 2
