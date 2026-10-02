"""Correctness tests for streaming_alert.AlertTracker."""

import random
import unittest

from streaming_alert import AlertTracker, THRESHOLD_MS, WINDOW_MS


def brute_force(intervals, t_final, window, threshold):
    """Independent reference: walks every integer moment up to t_final.

    Uses D(t+1) = D(t) + c(t) - c(t-window) where c(x) is 1 iff slot
    [x, x+1) is covered by the union of intervals.  Returns
    (closed_alert_segments, D(t_final), open_alert_start).
    """
    union = []
    for s, e in sorted((max(0, s), e) for s, e in intervals):
        if union and s <= union[-1][1]:
            union[-1][1] = max(union[-1][1], e)
        else:
            union.append([s, e])

    def make_cursor():
        return {"idx": 0}

    def coverage(x, cursor):
        # coverage of slot [x, x+1); x >= 0, non-decreasing queries
        if x < 0:
            return 0
        i = cursor["idx"]
        while i < len(union) and union[i][1] <= x:
            i += 1
        cursor["idx"] = i
        if i < len(union) and union[i][0] <= x < union[i][1]:
            return 1
        return 0

    cur_new = make_cursor()
    cur_old = make_cursor()
    d = 0
    closed = []
    open_start = None
    for t in range(0, t_final + 1):
        if t > 0:
            d += coverage(t - 1, cur_new) - coverage(t - 1 - window, cur_old)
        if d > threshold and open_start is None:
            open_start = t
        elif d <= threshold and open_start is not None:
            closed.append((open_start, t))
            open_start = None
    return closed, d, open_start


def random_plan(rng, intervals_sorted, t_final):
    """Random legal interleaving of adds and advances.

    Returns a plan: list of (add_count, advance_to).  Never advances the
    watermark past the start of an interval that has not been added yet.
    """
    n = len(intervals_sorted)
    idx = 0
    wm = 0
    plan = []
    while idx < n or wm < t_final:
        limit = intervals_sorted[idx][0] if idx < n else t_final
        choices = []
        if idx < n:
            choices.append("add")
        if wm < limit:
            choices.append("advance")
        action = rng.choice(choices) if choices else "add"
        if action == "add":
            k = rng.randrange(1, min(5, n - idx) + 1)
            plan.append((k, None))
            idx += k
        else:
            wm = rng.randrange(wm + 1, limit + 1)
            plan.append((0, wm))
    return plan


def run_plan(intervals_sorted, plan, window, threshold):
    tracker = AlertTracker(window_ms=window, threshold_ms=threshold)
    closed = []
    idx = 0
    for add_n, adv_to in plan:
        if add_n:
            tracker.add(intervals_sorted[idx:idx + add_n])
            idx += add_n
        if adv_to is not None:
            closed.extend(tracker.advance(adv_to))
    return closed, tracker.state()


class SpecCasesTest(unittest.TestCase):
    def test_exactly_45_minutes_never_alerts(self):
        tracker = AlertTracker()
        # Same 45-minute activity submitted repeatedly, plus contained ones.
        tracker.add([(0, 2_700_000)])
        tracker.add([(0, 2_700_000), (100, 2_000_000), (1_000, 2_699_999)])
        tracker.add([(0, 2_700_000)])
        self.assertEqual(tracker.advance(10_000_000), [])
        self.assertEqual(tracker.advance(20_000_000), [])
        self.assertEqual(tracker.state(), (20_000_000, 0, None))

    def test_spec_example_single_interval(self):
        tracker = AlertTracker()
        tracker.add([(0, 3_600_000)])
        # The whole alert happens between the two calls.
        self.assertEqual(tracker.advance(10_800_000), [(2_700_001, 8_100_000)])
        self.assertEqual(tracker.state(), (10_800_000, 0, None))

    def test_alert_spanning_multiple_watermarks_reported_once(self):
        tracker = AlertTracker()
        tracker.add([(0, 3_600_000)])
        self.assertEqual(tracker.advance(2_700_001), [])
        self.assertEqual(tracker.state(), (2_700_001, 2_700_001, 2_700_001))
        self.assertEqual(tracker.advance(5_000_000), [])
        self.assertEqual(tracker.state(), (5_000_000, 3_600_000, 2_700_001))
        self.assertEqual(tracker.advance(8_000_000), [])
        self.assertEqual(tracker.state(), (8_000_000, 2_800_000, 2_700_001))
        # One single segment, returned exactly once, at close time.
        self.assertEqual(tracker.advance(10_800_000), [(2_700_001, 8_100_000)])
        self.assertEqual(tracker.advance(20_000_000), [])
        self.assertEqual(tracker.state(), (20_000_000, 0, None))

    def test_alert_closes_without_further_input(self):
        tracker = AlertTracker()
        tracker.add([(0, 10_000_000)])
        self.assertEqual(tracker.advance(5_000_000), [])
        # D decays after t = 14_200_000... crosses threshold at 14_500_000.
        self.assertEqual(tracker.advance(20_000_000), [(2_700_001, 14_500_000)])
        self.assertEqual(tracker.state(), (20_000_000, 0, None))

    def test_advance_noop_and_validation(self):
        tracker = AlertTracker()
        tracker.add([(0, 1_000_000)])
        self.assertEqual(tracker.advance(1_000_000), [])
        self.assertEqual(tracker.advance(1_000_000), [])  # same watermark
        with self.assertRaises(ValueError):
            tracker.advance(999_999)
        with self.assertRaises(ValueError):
            tracker.add([(500_000, 600_000)])  # start below watermark
        with self.assertRaises(ValueError):
            tracker.add([(2_000_000, 1_000_000)])  # start >= end


class ConsistencyTest(unittest.TestCase):
    """Same input, different legal batching/advance rhythms -> same result."""

    INTERVALS = [
        (0, 3_000_000),
        (500_000, 2_000_000),        # overlaps previous
        (3_000_000, 4_000_000),      # touches the merged union
        (5_000_000, 6_500_000),
        (5_000_000, 5_500_000),      # same start as previous
        (7_000_000, 8_000_000),
        (7_500_000, 8_000_000),      # same end as previous
        (9_000_000, 9_000_001),      # 1 ms blip
        (10_000_000, 11_000_000),
        (10_500_000, 12_500_000),    # overlaps previous, extends end
    ]
    T_FINAL = 22_000_000

    def test_rhythms_agree_with_each_other_and_brute_force(self):
        intervals = sorted(self.INTERVALS)
        expected = brute_force(intervals, self.T_FINAL, WINDOW_MS, THRESHOLD_MS)

        # Rhythm 1: everything up front, one big advance.
        closed, state = run_plan(
            intervals, [(len(intervals), self.T_FINAL)], WINDOW_MS, THRESHOLD_MS
        )
        self.assertEqual((closed, state[1], state[2]), expected)
        self.assertEqual(state[0], self.T_FINAL)

        # Rhythm 2: many random legal interleavings.
        rng = random.Random(20261003)
        for _ in range(30):
            plan = random_plan(rng, intervals, self.T_FINAL)
            closed, state = run_plan(intervals, plan, WINDOW_MS, THRESHOLD_MS)
            self.assertEqual((closed, state[1], state[2]), expected)
            self.assertEqual(state[0], self.T_FINAL)

    def test_alert_fully_between_advances_and_across_watermarks(self):
        intervals = sorted(self.INTERVALS)
        # One giant advance: every alert opens and closes inside one call.
        closed_big, _ = run_plan(
            intervals, [(len(intervals), self.T_FINAL)], WINDOW_MS, THRESHOLD_MS
        )
        self.assertTrue(len(closed_big) >= 1)
        # Fine-grained advances: identical segments, each returned once.
        plan = [(len(intervals), None)] + [
            (0, t) for t in range(500_000, self.T_FINAL + 1, 500_000)
        ]
        closed_fine, state_fine = run_plan(intervals, plan, WINDOW_MS, THRESHOLD_MS)
        self.assertEqual(closed_fine, closed_big)
        self.assertEqual(len(closed_fine), len(set(closed_fine)))
        self.assertEqual(state_fine[2], None)


class PropertyTest(unittest.TestCase):
    """Random small-scale interleavings checked against the brute force."""

    def test_random_interleavings(self):
        window, threshold = 100, 45
        rng = random.Random(12345)
        for seed in range(200):
            t_final = 3_000
            intervals = sorted(
                (rng.randrange(0, 2_500), 0) for _ in range(40)
            )
            intervals = [
                (s, s + rng.randrange(1, 200)) for s, _ in intervals
            ]
            plan = random_plan(rng, intervals, t_final)
            closed, state = run_plan(intervals, plan, window, threshold)
            expected = brute_force(intervals, t_final, window, threshold)
            self.assertEqual(
                (closed, state[1], state[2]),
                expected,
                f"seed={seed} intervals={intervals} plan={plan}",
            )
            self.assertEqual(state[0], t_final)


if __name__ == "__main__":
    unittest.main()
