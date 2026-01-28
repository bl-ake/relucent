from heapq import heappop, heappush

from torch import Tensor

from relucent.poly import encode_ss


class SSManager:
    """Manages storage and lookup of sign sequences.

    This class provides a dictionary-like interface for storing and retrieving
    sign sequences (arrays with values in {-1, 0, 1}). It maintains an index
    mapping and allows efficient membership testing and retrieval.

    Sign sequences are encoded as hashable tags for efficient storage and lookup.
    """

    def __init__(self):
        self.index2ss = list()
        self.tag2index = dict()  # Tags are just hashable versions of sign sequences, should be unique
        self._len = 0

    def _get_tag(self, ss):
        if isinstance(ss, Tensor):
            ss = ss.detach().cpu().numpy()
        return encode_ss(ss)

    def add(self, ss):
        """Add a sign sequence to the manager.

        Args:
            ss: A sign sequence as torch.Tensor or np.ndarray.
        """
        tag = self._get_tag(ss)
        if tag not in self.tag2index:
            self.tag2index[tag] = len(self.index2ss)
            self.index2ss.append(ss)
            self._len += 1

    def __getitem__(self, ss):
        tag = self._get_tag(ss)
        index = self.tag2index[tag]
        if self.index2ss[index] is None:
            raise KeyError
        return index

    def __contains__(self, ss):
        tag = self._get_tag(ss)
        if tag not in self.tag2index:
            return False
        return self.index2ss[self.tag2index[tag]] is not None

    def __delitem__(self, ss):
        tag = self._get_tag(ss)
        index = self.tag2index[tag]
        self.index2ss[index] = None
        self._len -= 1

    def __iter__(self):
        return iter((ss for ss in self.index2ss if ss is not None))

    def __len__(self):
        return self._len


# TODO: Move to utils as general priority queue
class SSPriorityQueue:
    """Priority queue for tasks with sign sequences.

    A priority queue implementation that supports updating task priorities and
    removing tasks. Tasks are tuples starting with a sign sequence followed
    by additional data. Based on the heapq implementation from Python docs.

    Reference: https://docs.python.org/3/library/heapq.html
    """

    REMOVED = "<removed-task>"  # placeholder for a removed task

    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.counter = 0  # unique sequence count

    def push(self, task, priority=0):
        """Add a new task or update the priority of an existing task.

        Args:
            task: A tuple starting with a sign sequence followed by
                additional task data.
            priority: The priority value (lower = higher priority). Defaults to 0.
        """
        ss, *task = task
        task = tuple(task)
        if task in self.entry_finder:
            self.remove_task(task)
        entry = [priority, self.counter, ss, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)
        self.counter += 1

    def remove_task(self, task):
        "Mark an existing task as REMOVED.  Raise KeyError if not found."
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop(self):
        """Remove and return the lowest priority task.

        Returns:
            tuple: A tuple starting with the sign sequence followed by
                the task data.

        Raises:
            KeyError: If the queue is empty.
        """
        while self.pq:
            _, _, ss, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return ss, *task
        raise KeyError("pop from an empty priority queue")

    def __len__(self):
        return len(self.entry_finder)
