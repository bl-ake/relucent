from typing import Iterator

import numpy as np
import torch

from relucent.utils import encode_ss


class SSManager:
    """Manages storage and lookup of sign sequences.

    This class provides a dictionary-like interface for storing and retrieving
    sign sequences (arrays with values in {-1, 0, 1}). It maintains an index
    mapping and allows efficient membership testing and retrieval.

    Sign sequences are encoded as hashable tags for efficient storage and lookup.
    """

    def __init__(self) -> None:
        """Initialize an empty sign-sequence manager."""
        self.index2ss: list[np.ndarray | torch.Tensor | None] = []
        # Tags are just hashable versions of sign sequences, should be unique
        self.tag2index: dict[bytes, int] = {}
        self._len: int = 0

    def _get_tag(self, ss: np.ndarray | torch.Tensor) -> bytes:
        return encode_ss(ss)

    def add(self, ss: np.ndarray | torch.Tensor) -> None:
        """Add a sign sequence to the manager.

        Args:
            ss: A sign sequence as torch.Tensor or np.ndarray.
        """
        tag: bytes = self._get_tag(ss)
        if tag not in self.tag2index:
            self.tag2index[tag] = len(self.index2ss)
            self.index2ss.append(ss)
            self._len += 1

    def __getitem__(self, ss: np.ndarray | torch.Tensor) -> int:
        tag = self._get_tag(ss)
        index: int = self.tag2index[tag]
        if self.index2ss[index] is None:
            raise KeyError
        return index

    def __contains__(self, ss: np.ndarray | torch.Tensor) -> bool:
        tag = self._get_tag(ss)
        if tag not in self.tag2index:
            return False
        return self.index2ss[self.tag2index[tag]] is not None

    def __delitem__(self, ss: np.ndarray | torch.Tensor) -> None:
        tag = self._get_tag(ss)
        index: int = self.tag2index[tag]
        self.index2ss[index] = None
        self._len -= 1

    def __iter__(self) -> Iterator[np.ndarray | torch.Tensor]:
        return iter(ss for ss in self.index2ss if ss is not None)

    def __len__(self) -> int:
        return self._len
