"""Streaming alert computation over non-productive activity intervals.

Model
-----
For integer time t (milliseconds, t >= 0), let D(t) be the total milliseconds
of the half-open window [t - window_ms, t) covered by at least one activity
interval.  The tracker is in alert at t iff D(t) > threshold_ms.  Alert
segments are maximal half-open integer ranges [start, end): start is the
first alert moment, end is the first subsequent non-alert moment.

Method
------
The union of all ingested intervals is maintained in a treap of disjoint
intervals.  Because callers promise that new intervals start no earlier than
the watermark, coverage below the watermark is final.  Each union interval
[a, b) contributes a trapezoid to D, fully described by four slope events:

    +1 at a        (window's right edge enters the interval)
    -1 at b        (window's right edge leaves)
    -1 at a + W    (window's left edge enters)
    +1 at b + W    (window's left edge leaves)

Events for an interval are emitted lazily and exactly once: the left pair
once its start is below the watermark, the right pair once its end is at or
below the watermark (at that point the interval can no longer be merged with
future input, so its events are final).  Events are consumed in time order
from a heap; between consecutive events D is linear with slope in {-1, 0, 1},
so threshold crossings are computed analytically with exact integer
arithmetic.  Intervals whose last event (end + W) is at or below the
watermark are pruned, so memory stays proportional to recent activity plus
not-yet-reached input.
"""

from heapq import heappop, heappush

from .interval_treap import (
    IntervalNode,
    delete,
    delete_min,
    first_end_greater,
    insert,
    lower_bound,
    treap_min,
)

WINDOW_MS = 7_200_000      # 2 hours
THRESHOLD_MS = 2_700_000   # 45 minutes


class AlertTracker:
    """Tracks one employee's non-productive activity and emits alert segments."""

    def __init__(self, window_ms: int = WINDOW_MS, threshold_ms: int = THRESHOLD_MS) -> None:
        if not (0 < threshold_ms < window_ms):
            raise ValueError("require 0 < threshold_ms < window_ms")
        self._window = window_ms
        self._threshold = threshold_ms
        self._root = None            # treap of disjoint union intervals
        self._size = 0               # number of treap nodes
        self._heap = []              # pending slope events: (time, delta)
        self._watermark = 0
        self._frontier = 0           # alert status decided for integers <= frontier
        self._d = 0                  # D(frontier)
        self._slope = 0              # slope of D just after frontier
        self._alert_start = None     # start of the currently open alert, if any
        self._scan_start = 0         # left events emitted for all starts < this
        self._right_scan = 0         # right events emitted for all ends <= this

    # ------------------------------------------------------------------ add

    def add(self, intervals) -> None:
        """Ingest a batch of (start, end) intervals.

        Every interval must satisfy 0 <= start < end and start >= current
        watermark.  Intervals may be unordered, overlapping or duplicated;
        overlapping parts are counted only once.
        """
        for interval in intervals:
            start, end = int(interval[0]), int(interval[1])
            if not (0 <= start < end):
                raise ValueError(f"invalid interval ({start}, {end})")
            if start < self._watermark:
                raise ValueError(
                    f"interval start {start} is below watermark {self._watermark}"
                )
            self._insert_union(start, end)

    def _insert_union(self, start: int, end: int) -> None:
        root = self._root
        # Absorb every stored interval overlapping [start, end).  Overlapping
        # nodes form a contiguous in-order range; the leftmost candidate is
        # the first node whose end exceeds ``start``.
        node = first_end_greater(root, start)
        while node is not None and node.start < end:
            start = min(start, node.start)
            end = max(end, node.end)
            root = self._delete_node(root, node.start)
            node = first_end_greater(root, start)
        self._root = insert(root, IntervalNode(start, end))
        self._size += 1

    def _delete_node(self, root, start: int):
        self._size -= 1
        return delete(root, start)

    # -------------------------------------------------------------- advance

    def advance(self, t: int):
        """Advance the watermark to ``t`` (must be >= current watermark).

        Completes all alert decisions for integer moments <= t and returns
        the alert segments that closed during this call, ordered by start.
        """
        t = int(t)
        if t < self._watermark:
            raise ValueError(f"cannot move watermark backwards: {t} < {self._watermark}")
        if t == self._watermark:
            return []
        window = self._window

        # Emit right-edge events for union intervals whose end is now final.
        node = first_end_greater(self._root, self._right_scan)
        while node is not None and node.end <= t:
            heappush(self._heap, (node.end, -1))
            heappush(self._heap, (node.end + window, 1))
            self._right_scan = node.end
            node = first_end_greater(self._root, self._right_scan)

        # Emit left-edge events for union intervals reached by the watermark.
        node = lower_bound(self._root, self._scan_start)
        while node is not None and node.start < t:
            heappush(self._heap, (node.start, 1))
            heappush(self._heap, (node.start + window, -1))
            self._scan_start = node.start + 1
            node = lower_bound(self._root, self._scan_start)

        # Consume slope events in time order, resolving threshold crossings.
        closed = []
        heap = self._heap
        while heap and heap[0][0] <= t:
            tau = heap[0][0]
            delta = 0
            while heap and heap[0][0] == tau:
                delta += heappop(heap)[1]
            self._process_piece(tau, closed)
            self._slope += delta
        self._process_piece(t, closed)

        # Prune intervals that can no longer influence any future D value.
        node = treap_min(self._root)
        while node is not None and node.end + window <= t:
            self._root, node = delete_min(self._root)
            self._size -= 1
            node = treap_min(self._root)

        self._watermark = t
        return closed

    def _process_piece(self, to_time: int, closed: list) -> None:
        """Decide alert status for integer moments in (frontier, to_time].

        D is linear on this range with the current slope in {-1, 0, 1} and
        takes integer values at integer times, so crossings are exact.
        """
        left, right = self._frontier, to_time
        if right <= left:
            return
        slope = self._slope
        d = self._d
        threshold = self._threshold
        if slope > 0:
            if self._alert_start is None:
                # First integer x with D(x) >= threshold + 1.
                x0 = left + (threshold + 1 - d)
                if x0 <= left:
                    x0 = left  # defensive; invariant says d <= threshold here
                if x0 <= right:
                    self._alert_start = x0
        elif slope < 0:
            if self._alert_start is not None:
                # First integer x with D(x) <= threshold.
                x1 = left + (d - threshold)
                if x1 <= right:
                    closed.append((self._alert_start, x1))
                    self._alert_start = None
        self._d = d + slope * (right - left)
        self._frontier = right

    # ---------------------------------------------------------------- state

    def state(self):
        """Return ``(watermark, D(watermark), open_alert_start_or_None)``."""
        return self._watermark, self._d, self._alert_start
