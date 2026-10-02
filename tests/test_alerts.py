"""Functional tests for streaming_alert.AlertTracker."""

import random

import pytest

from streaming_alert import AlertTracker

W = 7_200_000
TH = 2_700_000


def test_duplicate_45min_never_alerts():
    t = AlertTracker()
    seg = (0, TH)  # exactly 45 minutes
    for _ in range(3):
        t.add([seg])
    t.add([(1_000_000, 2_000_000)])  # contained in seg
    assert t.advance(3 * W) == []
    st = t.state()
    assert st.watermark == 3 * W
    assert st.covered == 0
    assert st.open_alert_start is None


def test_duplicate_45min_during_window():
    # D peaks at exactly TH (not strictly greater) -> no alert at any time.
    t = AlertTracker()
    t.add([(0, TH), (0, TH)])
    for mark in range(900_000, 3 * W + 1, 900_000):
        assert t.advance(mark) == []
        assert t.state().covered <= TH
        assert t.state().open_alert_start is None


def test_single_hour_alert_segment():
    t = AlertTracker()
    t.add([(0, 3_600_000)])
    assert t.advance(10_800_000) == [(2_700_001, 8_100_000)]
    st = t.state()
    assert st.covered == 0
    assert st.open_alert_start is None


def test_advance_is_idempotent_and_keeps_state():
    t = AlertTracker()
    t.add([(0, 3_600_000)])
    t.advance(3_000_000)
    st = t.state()
    assert st.covered == 3_000_000
    assert st.open_alert_start == 2_700_001  # alert opened, not yet closed
    assert t.advance(3_000_000) == []  # same watermark: no-op
    assert t.advance(5_000_000) == []  # still open, must not be chopped
    assert t.state().open_alert_start == 2_700_001
    assert t.advance(10_800_000) == [(2_700_001, 8_100_000)]
    assert t.state().open_alert_start is None


def test_alert_closes_after_activity_stops():
    # No further input; advancing alone must close the alert exactly once.
    t = AlertTracker()
    t.add([(0, 3_600_000)])
    assert t.advance(5_000_000) == []
    assert t.advance(20_000_000) == [(2_700_001, 8_100_000)]
    assert t.advance(50_000_000) == []
    assert t.state().covered == 0


def make_overlap_set():
    # overlapping, touching, same-start and same-end intervals; union [0, 9_000_000)
    return [
        (0, 3_000_000),
        (2_000_000, 5_000_000),      # overlaps previous
        (5_000_000, 6_000_000),      # touches previous
        (6_000_000, 7_500_000),      # same start as next
        (6_000_000, 8_000_000),
        (8_500_000, 9_000_000),      # same end as next
        (8_000_000, 9_000_000),
    ]


def test_overlap_big_jump_vs_interleaved():
    intervals = make_overlap_set()
    expected_alerts = [(2_700_001, 13_500_000)]  # union [0, 9e6), window 7.2e6

    # (a) single batch, single big advance
    big = AlertTracker()
    big.add(intervals)
    assert big.advance(20_000_000) == expected_alerts

    # (b) interleaved adds and advances (starts always >= watermark)
    it = AlertTracker()
    it.add(intervals[:2])
    assert it.advance(1_000_000) == []
    it.add(intervals[2:5])
    assert it.advance(2_000_000) == []
    # alert opens at 2_700_001 and closes at 13_500_000: fully inside this gap
    it.add(intervals[5:])
    assert it.advance(20_000_000) == expected_alerts

    assert it.state() == big.state()
    assert it.state().covered == 0


def test_alert_spanning_multiple_watermarks_single_segment():
    t = AlertTracker()
    t.add(make_overlap_set())
    out = []
    for mark in (5_000_000, 10_000_000, 12_000_000, 20_000_000):
        out.extend(t.advance(mark))
        if mark < 13_500_000:
            assert t.state().open_alert_start == 2_700_001
    assert out == [(2_700_001, 13_500_000)]  # one segment, emitted once
    assert t.state().open_alert_start is None


def test_alert_fully_between_two_advances():
    t = AlertTracker()
    t.add(make_overlap_set())
    assert t.advance(2_000_000) == []
    assert t.advance(20_000_000) == [(2_700_001, 13_500_000)]


def test_validation():
    t = AlertTracker()
    with pytest.raises(ValueError):
        t.add([(5, 5)])
    with pytest.raises(ValueError):
        t.add([(7, 3)])
    with pytest.raises(ValueError):
        t.add([(-1, 3)])
    t.add([(10, 20)])
    t.advance(15)
    with pytest.raises(ValueError):
        t.add([(14, 30)])  # start before watermark
    with pytest.raises(ValueError):
        t.advance(14)  # backwards watermark


def brute_force_segments(intervals, upto, window, threshold):
    """Exact per-millisecond reference: alert segments closed by `upto`."""
    cov = [0] * (upto + 1)
    for s, e in intervals:
        for x in range(max(s, 0), min(e, upto)):
            cov[x] += 1
    pref = [0] * (upto + 1)
    for x in range(upto):
        pref[x + 1] = pref[x] + (1 if cov[x] else 0)

    def covered_d(t):
        lo = max(0, t - window)
        return pref[t] - pref[lo]

    segs = []
    open_start = None
    for t in range(0, upto + 1):
        alerting = covered_d(t) > threshold
        if alerting and open_start is None:
            open_start = t
        elif not alerting and open_start is not None:
            segs.append((open_start, t))
            open_start = None
    return segs, open_start, covered_d(upto)


@pytest.mark.parametrize("seed", range(40))
def test_randomized_vs_brute_force(seed):
    rng = random.Random(seed)
    window, threshold = 120, 45
    horizon = 2_000
    intervals = []
    for _ in range(rng.randint(5, 40)):
        s = rng.randint(0, horizon - 1)
        intervals.append((s, s + rng.randint(1, 300)))

    # reference over the whole horizon
    ref_segs, ref_open, ref_d = brute_force_segments(intervals, horizon, window, threshold)

    # streamed: random legal interleaving of adds and advances
    t = AlertTracker(window_ms=window, threshold_ms=threshold)
    pending = sorted(intervals)  # add in start order to respect the watermark
    got = []
    mark = 0
    idx = 0
    while mark < horizon:
        step = rng.randint(1, 400)
            # add every pending interval whose start is >= mark, chosen so that
            # starts never fall behind the watermark
        while idx < len(pending) and pending[idx][0] < mark + step:
            idx += 1
        batch = [iv for iv in intervals if mark <= iv[0] < mark + step]
        if batch:
            t.add(batch)
        mark = min(horizon, mark + step)
        got.extend(t.advance(mark))
    assert got == ref_segs
    st = t.state()
    assert st.watermark == horizon
    assert st.covered == ref_d
    assert st.open_alert_start == ref_open


def test_initial_state_and_empty_add():
    t = AlertTracker()
    st = t.state()
    assert st.watermark == 0 and st.covered == 0 and st.open_alert_start is None
    t.add([])
    assert t.advance(0) == []
    assert t.state().covered == 0


def test_add_at_exact_watermark_after_advance():
    # interval starting exactly at the current watermark must still count
    t = AlertTracker(window_ms=100, threshold_ms=40)
    t.advance(50)
    t.add([(50, 150)])  # 100ms of activity starting at the watermark
    got = []
    got += t.advance(100)
    got += t.advance(500)
    # D(t) = t-50 for 50<=t<=150 -> exceeds 40 at t=91; window slides out by 250
    assert got == [(91, 210)]
