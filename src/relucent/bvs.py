from heapq import heappop, heappush

from torch import Tensor

from relucent.poly import encode_bv


class BVManager:
    """Manages storage and lookup of sign sequences.

    This class provides a dictionary-like interface for storing and retrieving
    sign sequences (arrays with values in {-1, 0, 1}). It maintains an index
    mapping and allows efficient membership testing and retrieval.

    Sign sequences are encoded as hashable tags for efficient storage and lookup.
    """

    def __init__(self):
        self.index2bv = list()
        self.tag2index = dict()  ## Tags are just hashable versions of bvs, should be unique
        self._len = 0

    def _get_tag(self, bv):
        if isinstance(bv, Tensor):
            bv = bv.detach().cpu().numpy()
        return encode_bv(bv)

    def add(self, bv):
        """Add a sign sequence to the manager.

        Args:
            bv: A sign sequence as torch.Tensor or np.ndarray.
        """
        tag = self._get_tag(bv)
        if tag not in self.tag2index:
            self.tag2index[tag] = len(self.index2bv)
            self.index2bv.append(bv)
            self._len += 1

    def __getitem__(self, bv):
        tag = self._get_tag(bv)
        index = self.tag2index[tag]
        if self.index2bv[index] is None:
            raise KeyError
        return index

    def __contains__(self, bv):
        tag = self._get_tag(bv)
        if tag not in self.tag2index:
            return False
        return self.index2bv[self.tag2index[tag]] is not None

    def __delitem__(self, bv):
        tag = self._get_tag(bv)
        index = self.tag2index[tag]
        self.index2bv[index] = None
        self._len -= 1

    def __iter__(self):
        return iter((bv for bv in self.index2bv if bv is not None))

    def __len__(self):
        return self._len


# TODO: Move to utils as general priority queue
class BVPriorityQueue:
    """Priority queue for tasks with sign sequences.

    A priority queue implementation that supports updating task priorities and
    removing tasks. Tasks are tuples starting with a sign sequence (BV) followed
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
        bv, *task = task
        task = tuple(task)
        if task in self.entry_finder:
            self.remove_task(task)
        entry = [priority, self.counter, bv, task]
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
            tuple: A tuple starting with the sign sequence (BV) followed by
                the task data.

        Raises:
            KeyError: If the queue is empty.
        """
        while self.pq:
            _, _, bv, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return bv, *task
        raise KeyError("pop from an empty priority queue")

    def __len__(self):
        return len(self.entry_finder)


# class BVPriorityQueue:
#     """Simpler, less efficient version of the one above for debugging"""
#     def __init__(self):
#         self.pq = []  # list of entries arranged in a heap

#     def push(self, task, priority=0):
#         "Add a new task or update the priority of an existing task"
#         # bv, *task = task
#         self.pq.append((priority, task))
#         self.pq.sort(reverse=True, key=lambda x: x[0])

#     def remove_task(self, task):
#         "Mark an existing task as REMOVED.  Raise KeyError if not found."
#         self.pq = [(p, t) for p, t in self.pq if t != task]

#     def pop(self):
#         "Remove and return the lowest priority task. Raise KeyError if empty."
#         return self.pq.pop(-1)[1]

#     def __len__(self):
#         return len(self.pq)


# class BVNode:
#     def __init__(self, key):
#         self.key = key  ## Key of bv being set
#         self.left = None  ## for all nodes in subtree, bv[0, key] = -1
#         self.middle = None  ## ...bv[0, key] = 0
#         self.right = None  ## ...bv[0, key] = 1

#     def get_child(self, bv):
#         # return either BVNode or int
#         if bv[0, self.key] == -1:
#             next_node = self.left
#         elif bv[0, self.key] == 0:
#             next_node = self.middle
#         elif bv[0, self.key] == 1:
#             next_node = self.right
#         return next_node

#     def set_child(self, bv, node):
#         if bv[0, self.key] == -1:
#             self.left = node
#         elif bv[0, self.key] == 0:
#             self.middle = node
#         elif bv[0, self.key] == 1:
#             self.right = node

#     def print(self, level=0):
#         print(" " * level + str(self.key) + ":")
#         for name, k in zip(("L", "M", "R"), (self.left, self.middle, self.right)):
#             if isinstance(k, BVNode):
#                 print(" " * (level + 2) + name)
#                 k.print(level=level + 4)
#             elif isinstance(k, int):
#                 print(" " * (level + 2) + name + ": leaf " + str(k))

# # Trie
# # Each edge in the tree sets a dimension to a value
# # Leaf nodes are just indices of bvs in index2bv
# class BVManager:
#     def __init__(self):
#         self.root = BVNode(0)
#         self.index2bv = list()

#     def add(self, bv):
#         assert bv.ndim == 2
#         node = self.root
#         child = self.root.get_child(bv)
#         while isinstance(child, BVNode):
#             node = child
#             child = node.get_child(bv)
#         if child is None:
#             node.set_child(bv, len(self.index2bv))
#             self.index2bv.append(bv)
#         elif isinstance(child, int):  ## TODO: This check should be redundant
#             child_bv = self.index2bv[child]
#             if not isinstance(child_bv, type(bv)):
#                 if isinstance(bv, Tensor):
#                     bv = bv.detach().cpu().numpy()
#                 else:
#                     child_bv = child_bv.detach().cpu().numpy()
#             if not (child_bv == bv).all():
#                 if child_bv[0, node.key] == bv[0, node.key]:  ## TODO: This check should be redundant
#                     for i in range(bv.shape[1]):
#                         if child_bv[0, i] != bv[0, i]:
#                             break

#                     ## Replace the existing child of the node with a new node
#                     new_bvnode = BVNode(i)
#                     node.set_child(bv, new_bvnode)

#                     ## Set the new node's children
#                     new_bvnode.set_child(child_bv, child)
#                     new_bvnode.set_child(bv, len(self.index2bv))
#                     self.index2bv.append(bv)
#                 else:
#                     raise ValueError("Something went wrong")

#     def __getitem__(self, bv):
#         assert bv.ndim == 2
#         node = self.root
#         while isinstance(node, BVNode):
#             node = node.get_child(bv)
#         if isinstance(node, int):
#             found_bv = self.index2bv[node]
#             if not isinstance(bv, type(found_bv)):
#                 if isinstance(bv, Tensor):
#                     bv = bv.detach().cpu().numpy()
#                 else:
#                     found_bv = found_bv.detach().cpu().numpy()
#             if (found_bv == bv).all():
#                 return node
#         raise KeyError

#     def __contains__(self, bv):
#         try:
#             self[bv]
#             return True
#         except KeyError:
#             return False

#     def __iter__(self):
#         return iter(self.index2bv)

#     def __len__(self):
#         return len(self.index2bv)

#     def print(self):
#         print("Number of BVs:", len(self))
#         self.root.print()
