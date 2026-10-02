"""Randomized treap storing a disjoint union of [start, end) intervals.

Intervals kept here are always disjoint and non-touching, so both starts and
ends are strictly increasing in in-order.  Augmentations per node:

- total:   sum of (end - start) over the subtree (covered milliseconds)
- max_end: max end over the subtree (used to prune range scans)

All operations are O(log n) expected; endpoint scans are O(log n + k) where k
is the number of reported endpoints.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple


class _Node:
    __slots__ = ("s", "e", "prio", "left", "right", "total", "max_end")

    def __init__(self, s: int, e: int, prio: int) -> None:
        self.s = s
        self.e = e
        self.prio = prio
        self.left: Optional[_Node] = None
        self.right: Optional[_Node] = None
        self.total = e - s
        self.max_end = e


def _update(node: _Node) -> None:
    total = node.e - node.s
    max_end = node.e
    if node.left is not None:
        total += node.left.total
        if node.left.max_end > max_end:
            max_end = node.left.max_end
    if node.right is not None:
        total += node.right.total
        if node.right.max_end > max_end:
            max_end = node.right.max_end
    node.total = total
    node.max_end = max_end


def _split(node: Optional[_Node], key: int) -> Tuple[Optional[_Node], Optional[_Node]]:
    """Split into (< key, >= key) by start."""
    if node is None:
        return None, None
    if node.s < key:
        left, right = _split(node.right, key)
        node.right = left
        _update(node)
        return node, right
    left, right = _split(node.left, key)
    node.left = right
    _update(node)
    return left, node


def _merge(a: Optional[_Node], b: Optional[_Node]) -> Optional[_Node]:
    if a is None:
        return b
    if b is None:
        return a
    if a.prio <= b.prio:
        a.right = _merge(a.right, b)
        _update(a)
        return a
    b.left = _merge(a, b.left)
    _update(b)
    return b


def _insert_node(node: Optional[_Node], new: _Node) -> _Node:
    if node is None:
        return new
    if new.prio < node.prio:
        if new.s < node.s:
            node.left = _insert_node(node.left, new)
        else:
            node.right = _insert_node(node.right, new)
        _update(node)
        return node
    left, right = _split(node, new.s)
    new.left = left
    new.right = right
    _update(new)
    return new


def _delete(node: Optional[_Node], key: int) -> Optional[_Node]:
    if node is None:
        return None
    if node.s == key:
        return _merge(node.left, node.right)
    if key < node.s:
        node.left = _delete(node.left, key)
    else:
        node.right = _delete(node.right, key)
    _update(node)
    return node


def union_insert(root: Optional[_Node], s: int, e: int, prio: int) -> Tuple[Optional[_Node], int]:
    """Insert [s, e), merging every overlapping or touching interval.

    Returns (new_root, number_of_removed_intervals).
    """
    # Collect stored intervals [a, b) with a <= e and b >= s (touching merges).
    victims: list[_Node] = []
    _collect_overlaps(root, s, e, victims)
    lo, hi = s, e
    for v in victims:
        if v.s < lo:
            lo = v.s
        if v.e > hi:
            hi = v.e
        root = _delete(root, v.s)
    return _insert_node(root, _Node(lo, hi, prio)), len(victims)


def _collect_overlaps(node: Optional[_Node], s: int, e: int, out: list[_Node]) -> None:
    if node is None or node.max_end < s:
        return
    _collect_overlaps(node.left, s, e, out)
    if node.s > e:
        return  # starts only grow in-order: nothing further can overlap
    if node.e >= s:
        out.append(node)
    _collect_overlaps(node.right, s, e, out)


def prune_ended(root: Optional[_Node], cutoff: int) -> Tuple[Optional[_Node], int]:
    """Drop all intervals with end <= cutoff (they form an in-order prefix).

    Returns (new_root, number_of_removed_intervals).
    """
    removed = 0
    while root is not None:
        leftmost = root
        while leftmost.left is not None:
            leftmost = leftmost.left
        if leftmost.e > cutoff:
            break
        root = _delete(root, leftmost.s)
        removed += 1
    return root, removed


def covered_prefix(root: Optional[_Node], x: int) -> int:
    """Total covered milliseconds in [0, x) (0 for x <= 0)."""
    if x <= 0:
        return 0
    total = 0
    node = root
    while node is not None:
        if x <= node.s:
            node = node.left
        elif x >= node.e:
            total += (node.left.total if node.left else 0) + (node.e - node.s)
            node = node.right
        else:
            total += (node.left.total if node.left else 0) + (x - node.s)
            break
    return total


def is_covered(root: Optional[_Node], x: int) -> bool:
    """Whether instant x is covered (right-continuous)."""
    node = root
    while node is not None:
        if x < node.s:
            node = node.left
        elif x >= node.e:
            node = node.right
        else:
            return True
    return False


class _ScanDone(Exception):
    pass


def iter_endpoints(root: Optional[_Node], lo: int, hi: int) -> Iterator[Tuple[int, int]]:
    """Yield (time, delta) for endpoints in (lo, hi], in increasing time.

    delta is +1 at an interval start, -1 at an interval end.
    """
    try:
        yield from _iter_endpoints(root, lo, hi)
    except _ScanDone:
        return


def _iter_endpoints(node: Optional[_Node], lo: int, hi: int) -> Iterator[Tuple[int, int]]:
    if node is None:
        return
    if node.left is not None and node.left.max_end > lo:
        yield from _iter_endpoints(node.left, lo, hi)
    if node.s > hi:
        raise _ScanDone  # starts only grow in-order
    if lo < node.s <= hi:
        yield node.s, 1
    if lo < node.e <= hi:
        yield node.e, -1
    if node.right is not None and node.right.max_end > lo:
        yield from _iter_endpoints(node.right, lo, hi)
