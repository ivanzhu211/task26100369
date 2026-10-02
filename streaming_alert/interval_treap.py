"""Randomized treap storing disjoint half-open intervals [start, end).

Intervals are keyed by ``start``; because stored intervals are disjoint and
non-touching, the in-order sequence is sorted by both ``start`` and ``end``.
Only structural primitives live here; all alert logic lives in tracker.py.

All operations are O(log n) expected.  Priorities are a deterministic hash of
the (unique) start key, so no RNG state is needed.
"""

_MASK = 0xFFFFFFFFFFFFFFFF


class IntervalNode:
    __slots__ = ("start", "end", "prio", "left", "right")

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end
        self.prio = _priority(start)
        self.left = None
        self.right = None


def _priority(key: int) -> int:
    """Deterministic splitmix64 hash used as treap priority."""
    z = (key + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def lower_bound(root, start):
    """Return the node with the smallest start >= ``start``, or None."""
    best = None
    while root is not None:
        if root.start >= start:
            best = root
            root = root.left
        else:
            root = root.right
    return best


def first_end_greater(root, end):
    """Return the in-order-first node whose end > ``end``, or None.

    Valid because ``end`` is monotonically increasing along the in-order
    traversal (disjoint intervals sorted by start are sorted by end).
    """
    best = None
    while root is not None:
        if root.end > end:
            best = root
            root = root.left
        else:
            root = root.right
    return best


def treap_min(root):
    """Return the node with the smallest start, or None."""
    if root is None:
        return None
    while root.left is not None:
        root = root.left
    return root


def _split(root, start):
    """Split into (nodes with start < ``start``, nodes with start >= ``start``)."""
    if root is None:
        return None, None
    if root.start < start:
        left, right = _split(root.right, start)
        root.right = left
        return root, right
    left, right = _split(root.left, start)
    root.left = right
    return left, root


def _merge(left, right):
    """Merge two treaps where all keys in ``left`` are smaller than in ``right``."""
    if left is None:
        return right
    if right is None:
        return left
    if left.prio < right.prio:
        left.right = _merge(left.right, right)
        return left
    right.left = _merge(left, right.left)
    return right


def insert(root, node):
    """Insert ``node``; caller guarantees no overlap with existing nodes."""
    if root is None:
        return node
    if node.prio < root.prio:
        left, right = _split(root, node.start)
        node.left = left
        node.right = right
        return node
    if node.start < root.start:
        root.left = insert(root.left, node)
    else:
        root.right = insert(root.right, node)
    return root


def delete(root, start):
    """Remove the node with the given start key."""
    if root is None:
        return None
    if start < root.start:
        root.left = delete(root.left, start)
    elif start > root.start:
        root.right = delete(root.right, start)
    else:
        root = _merge(root.left, root.right)
    return root


def delete_min(root):
    """Remove and return ``(new_root, min_node)``."""
    if root.left is None:
        return root.right, root
    new_left, node = delete_min(root.left)
    root.left = new_left
    return root, node
