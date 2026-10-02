"""Streaming alert tracker for one employee's non-productive activity.

Definitions (all times are non-negative integer milliseconds):
- D(t) = milliseconds covered by at least one activity interval inside the
  half-open window [t - window, t).  No activity exists before time 0.
- Alerting at integer instant t iff D(t) > threshold.
- An alert segment is a maximal run of alerting instants, reported half-open:
  [first alerting integer, first non-alerting integer afterwards).

D(t) = C(t) - C(t - window) where C(x) is the covered length of [0, x) under
the union of all activity intervals.  C is piecewise linear with slope 0/1, so
D is continuous and piecewise linear with slope in {-1, 0, 1}; its breakpoints
are the union-interval endpoints and those endpoints shifted by +window.  The
tracker keeps the union in a treap and, per advance, sweeps only the endpoints
inside the newly covered time ranges, so all arithmetic stays exact to 1 ms.
"""

from __future__ import annotations

import random
from typing import Iterable, Iterator, List, NamedTuple, Optional, Tuple

from . import _treap

WINDOW_MS = 7_200_000  # 2 hours
THRESHOLD_MS = 2_700_000  # 45 minutes


class AlertState(NamedTuple):
    watermark: int
    covered: int  # D(watermark)
    open_alert_start: Optional[int]  # None when no unclosed alert


class AlertTracker:
    """Tracks alerts for a single employee.  Calls are serial; nothing is persisted."""

    def __init__(self, window_ms: int = WINDOW_MS, threshold_ms: int = THRESHOLD_MS) -> None:
        if not (0 < threshold_ms < window_ms):
            raise ValueError("require 0 < threshold_ms < window_ms")
        self.window = int(window_ms)
        self.threshold = int(threshold_ms)
        self._root: Optional[_treap._Node] = None
        self._rng = random.Random(0x5EED)
        self._mark = 0
        self._covered = 0  # D(self._mark)
        self._open_x: Optional[int] = None  # crossing point where D rose past threshold
        self._intervals = 0  # number of stored union intervals (for diagnostics)

    # ------------------------------------------------------------------ add

    def add(self, intervals: Iterable[Tuple[int, int]]) -> None:
        mark = self._mark
        for iv in intervals:
            s, e = int(iv[0]), int(iv[1])
            if s < 0 or e <= s:
                raise ValueError(f"invalid interval [{s}, {e})")
            if s < mark:
                raise ValueError(f"interval start {s} before watermark {mark}")
            self._root, removed = _treap.union_insert(self._root, s, e, self._rng.getrandbits(62))
            self._intervals += 1 - removed

    # -------------------------------------------------------------- advance

    def advance(self, t: int) -> List[Tuple[int, int]]:
        t = int(t)
        if t < self._mark:
            raise ValueError(f"advance({t}) before watermark {self._mark}")
        if t == self._mark:
            return []
        t0 = self._mark
        window = self.window
        threshold = self.threshold
        root = self._root

        covered = _treap.covered_prefix(root, t0) - _treap.covered_prefix(root, t0 - window)
        slope = int(_treap.is_covered(root, t0)) - int(_treap.is_covered(root, t0 - window))

        emitted: List[Tuple[int, int]] = []
        open_x = self._open_x
        cur_t = t0
        cur_d = covered

        for tm, delta in self._events_between(t0, t):
            seg = tm - cur_t
            if slope == 1 and open_x is None and cur_d + seg > threshold:
                open_x = cur_t + (threshold - cur_d)
            elif slope == -1 and open_x is not None and cur_d - seg <= threshold:
                emitted.append((open_x + 1, cur_t + (cur_d - threshold)))
                open_x = None
            cur_d += slope * seg
            cur_t = tm
            slope += delta

        seg = t - cur_t
        if slope == 1 and open_x is None and cur_d + seg > threshold:
            open_x = cur_t + (threshold - cur_d)
        elif slope == -1 and open_x is not None and cur_d - seg <= threshold:
            emitted.append((open_x + 1, cur_t + (cur_d - threshold)))
            open_x = None
        cur_d += slope * seg

        self._root, removed = _treap.prune_ended(root, t - window)
        self._intervals -= removed
        self._mark = t
        self._covered = cur_d
        self._open_x = open_x
        return emitted

    def _events_between(self, t0: int, t1: int) -> Iterator[Tuple[int, int]]:
        """Yield (time, net_slope_delta) with t0 < time <= t1, grouped by time.

        Direct events: union endpoints in (t0, t1] (start +1 / end -1).
        Mirror events: union endpoints in (t0 - W, t1 - W], replayed at +W
        with negated delta (they leave the sliding window).
        """
        window = self.window
        gen_a = _treap.iter_endpoints(self._root, t0 - window, t1 - window)
        gen_b = _treap.iter_endpoints(self._root, t0, t1)
        pa = next(gen_a, None)
        pb = next(gen_b, None)
        while pa is not None or pb is not None:
            ta = pa[0] + window if pa is not None else None
            tb = pb[0] if pb is not None else None
            if tb is None or (ta is not None and ta < tb):
                tm, da = ta, -pa[1]
                pa = next(gen_a, None)
            elif ta is None or tb < ta:
                tm, da = tb, pb[1]
                pb = next(gen_b, None)
            else:
                tm, da = ta, pb[1] - pa[1]
                pa = next(gen_a, None)
                pb = next(gen_b, None)
            # merge further events at the same instant
            while pa is not None and pa[0] + window == tm:
                da -= pa[1]
                pa = next(gen_a, None)
            while pb is not None and pb[0] == tm:
                da += pb[1]
                pb = next(gen_b, None)
            if da != 0:
                yield tm, da

    # --------------------------------------------------------------- state

    def state(self) -> AlertState:
        open_start = None if self._open_x is None else self._open_x + 1
        return AlertState(self._mark, self._covered, open_start)

    @property
    def interval_count(self) -> int:
        """Number of stored union intervals (diagnostics / memory checks)."""
        return self._intervals

