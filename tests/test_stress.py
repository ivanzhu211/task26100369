"""Streaming stress test: 300k intervals, 100k advances, 20-year span."""

import resource
import time
import unittest

from streaming_alert import AlertTracker, WINDOW_MS

N_INTERVALS = 300_000
N_ADVANCES = 100_000
SPAN_MS = 20 * 365 * 86_400_000  # 20 years in milliseconds
STEP = SPAN_MS // N_ADVANCES     # ~6.3e6 ms (~1.75 h) per advance


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def make_batches(seed=99):
    """N_ADVANCES batches; batch i has starts in [i*STEP, (i+1)*STEP)."""
    import random

    rng = random.Random(seed)
    batches = []
    total = 0
    for i in range(N_ADVANCES):
        if total >= N_INTERVALS:
            break
        base = i * STEP
        batch = []
        for _ in range(min(3, N_INTERVALS - total)):
            start = base + rng.randrange(0, STEP)
            # Mix of short blips and long stretches (some exceed 45 min).
            duration = rng.choice(
                (rng.randrange(1, 60_000), rng.randrange(60_000, 3_600_000),
                 rng.randrange(3_600_000, 10_800_000))
            )
            batch.append((start, start + duration))
            total += 1
        batches.append(batch)
    return batches


class StressTest(unittest.TestCase):
    def test_streaming_stress(self):
        batches = make_batches()
        n_intervals = sum(len(b) for b in batches)
        self.assertEqual(n_intervals, N_INTERVALS)
        t_final = SPAN_MS + WINDOW_MS + 11_000_000

        # Run A: interleaved add/advance, batch i added before step i+1.
        tracker_a = AlertTracker()
        closed_a = []
        max_nodes = max_heap = 0
        t0 = time.perf_counter()
        wm = 0
        for batch in batches:
            tracker_a.add(batch)
            wm += STEP
            closed_a.extend(tracker_a.advance(wm))
            if len(tracker_a._heap) > max_heap:
                max_heap = len(tracker_a._heap)
            if tracker_a._size > max_nodes:
                max_nodes = tracker_a._size
        closed_a.extend(tracker_a.advance(t_final))
        elapsed_a = time.perf_counter() - t0
        state_a = tracker_a.state()

        # Old history must be fully released once it is behind the window.
        self.assertEqual(tracker_a._size, 0)
        self.assertIsNone(tracker_a._root)
        self.assertEqual(len(tracker_a._heap), 0)
        self.assertEqual(state_a, (t_final, 0, None))

        # Run B: identical interval set, all added up front (legal: every
        # start >= 0), same advance schedule.  Results must be identical.
        tracker_b = AlertTracker()
        tracker_b.add(iv for batch in batches for iv in batch)
        closed_b = []
        t0 = time.perf_counter()
        wm = 0
        for _ in batches:
            wm += STEP
            closed_b.extend(tracker_b.advance(wm))
        closed_b.extend(tracker_b.advance(t_final))
        elapsed_b = time.perf_counter() - t0

        self.assertEqual(closed_a, closed_b)
        self.assertEqual(state_a, tracker_b.state())
        self.assertEqual(tracker_b._size, 0)
        self.assertGreater(len(closed_a), 0)  # alerts actually happened

        peak = peak_rss_mb()
        print(
            f"\nstress: {n_intervals} intervals, {len(batches)} advances, "
            f"{SPAN_MS / 86_400_000:.0f} days span\n"
            f"  alerts closed: {len(closed_a)}\n"
            f"  run A (interleaved): {elapsed_a:.2f}s, "
            f"max treap nodes {max_nodes}, max heap {max_heap}\n"
            f"  run B (upfront):     {elapsed_b:.2f}s\n"
            f"  peak RSS: {peak:.0f} MiB"
        )
        # Generous guards against accidental complexity blow-ups.
        self.assertLess(elapsed_a, 120)
        self.assertLess(elapsed_b, 120)
        self.assertLess(peak, 2048)

    def test_empty_time_jumps_are_constant_time(self):
        tracker = AlertTracker()
        t0 = time.perf_counter()
        wm = 0
        for _ in range(N_ADVANCES):
            wm += 1_000_000_000  # ~11.5 days per jump, no activity at all
            self.assertEqual(tracker.advance(wm), [])
        elapsed = time.perf_counter() - t0
        self.assertEqual(tracker.state(), (wm, 0, None))
        print(f"\nempty jumps: {N_ADVANCES} advances in {elapsed:.3f}s")
        self.assertLess(elapsed, 5)


if __name__ == "__main__":
    unittest.main()
