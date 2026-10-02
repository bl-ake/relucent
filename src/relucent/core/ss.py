"""Sign-sequence storage and lookup for the polyhedral complex.

Provides :class:`SSManager`, a dict-like container that maps sign sequences
(arrays of {-1, 0, 1}) to contiguous integer indices, enabling O(1) membership
testing and insertion via a bytes-encoded hash key.
"""

from collections.abc import Iterator

import numpy as np

from relucent._internal.torch_compat import torch

__all__ = ["SSManager", "encode_ss", "flip_ss_at_shi", "flip_ss_at_shi_inplace"]


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


def encode_ss(ss: np.ndarray | torch.Tensor) -> bytes:
    """Create a hashable representation of a sign sequence.

    Converts a sign sequence array into a bytes object that can be used as a
    dictionary key or for hashing.

    The sign sequence is always encoded using an integer dtype so that float
    and integer representations with the same logical values (in {-1, 0, 1})
    produce identical tags.

    Args:
        ss: A sign sequence as np.ndarray or torch.Tensor with values in {-1, 0, 1}.

    Returns:
        bytes: A hashable bytes representation of the flattened sign sequence.
    """
    # Hot path (millions of calls from cubical incidence / chain-complex assembly): a
    # sign sequence that is already a C-contiguous int8 ndarray needs no coercion —
    # ``tobytes()`` flattens in C order, exactly what ``.ravel().tobytes()`` produced.
    if type(ss) is np.ndarray and ss.dtype == np.int8 and ss.flags["C_CONTIGUOUS"]:
        return ss.tobytes()

    ss = ss.detach().cpu().numpy() if isinstance(ss, torch.Tensor) else np.asarray(ss)

    ss = ss.astype(np.int8, copy=False)
    return ss.ravel().tobytes()


def flip_ss_at_shi_inplace(ss: np.ndarray, shi: int) -> None:
    """Negate coordinate ``shi`` of ``ss`` in place.

    Call twice on the same ``shi`` to restore the original sign sequence.  Useful
    when iterating many SHIs over one working copy (e.g. dual-graph construction).
    """
    ss.ravel()[int(shi)] *= -1


def flip_ss_at_shi(ss: np.ndarray | torch.Tensor, shi: int) -> np.ndarray:
    """Return a copy of ``ss`` with coordinate ``shi`` negated.

    Codimension-1 neighbors across supporting hyperplane ``shi`` differ by this flip.
    """
    if isinstance(ss, torch.Tensor):
        ss = ss.detach().cpu().numpy()
    flipped = np.asarray(ss, dtype=np.int8).copy()
    flip_ss_at_shi_inplace(flipped, shi)
    return flipped
