"""Streaming stress test: 300k intervals, 100k advances, 20-year span.

Verifies:
- total work stays near O((N + Q) log N + A): blank jumps cost nothing;
- state memory tracks only live intervals: old history is pruned;
- results stay exact (segments well-formed, sorted, non-overlapping).
"""

import random
import time
import tracemalloc

from streaming_alert import AlertTracker

TWENTY_YEARS_MS = 20 * 365 * 86_400_000
ADVANCES = 100_000
INTERVALS = 300_000
ACTIVE_SPAN_MS = 2 * 365 * 86_400_000  # all activity inside the first 2 years


def test_stress_stream():
    rng = random.Random(20261003)
    step = TWENTY_YEARS_MS // ADVANCES  # ~6.3e6 ms (~105 min) per advance
    per_advance = INTERVALS // (ACTIVE_SPAN_MS // step)  # intervals per active advance

    tracker = AlertTracker()
    alerts = []
    max_live = 0
    live_at_checks = []

    tracemalloc.start()
    t_start = time.perf_counter()

    mark = 0
    for i in range(ADVANCES):
        new_mark = (i + 1) * step
        if new_mark <= ACTIVE_SPAN_MS + step:
            batch = []
            if i % 10 == 0:
                # dense cluster: 1-4 h activities packed into ~26 min -> alert
                for _ in range(per_advance):
                    s = mark + rng.randrange(0, step // 4)
                    dur = rng.randrange(3_600_000, 14_400_000)
                    batch.append((s, s + dur))
            else:
                # sparse short activities: disjoint, no alert, must be pruned
                # (~17 min of activity per 105 min step -> D stays ~20 min)
                for _ in range(per_advance):
                    s = mark + rng.randrange(0, step)
                    dur = rng.randrange(10_000, 60_000)  # 10-60 s
                    batch.append((s, s + dur))
            tracker.add(batch)
        alerts.extend(tracker.advance(new_mark))
        mark = new_mark
        if tracker.interval_count > max_live:
            max_live = tracker.interval_count
        if i in (ADVANCES // 4, ADVANCES // 2, ADVANCES - 1):
            live_at_checks.append((mark, tracker.interval_count))

    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nstress: {INTERVALS} intervals, {ADVANCES} advances, "
          f"{TWENTY_YEARS_MS / 86_400_000 / 365:.0f} years")
    print(f"stress: elapsed {elapsed:.2f}s, peak python mem {peak / 1e6:.1f} MB, "
          f"closed alerts {len(alerts)}")
    print(f"stress: max live union intervals {max_live}, checks {live_at_checks}")

    # --- correctness of emitted stream
    assert len(alerts) > 100, f"expected many cluster alerts, got {len(alerts)}"
    assert all(0 <= a < b <= TWENTY_YEARS_MS for a, b in alerts)
    assert all(alerts[i][1] <= alerts[i + 1][0] for i in range(len(alerts) - 1))

    # --- blank-time jumps cost nothing per moment: 18 empty years must be fast
    assert elapsed < 180, f"too slow: {elapsed:.1f}s (would indicate per-moment work)"

    # --- old history does not accumulate: after passing all activity + window,
    # every stored interval is pruned; memory tracks only live boundaries.
    assert tracker.interval_count == 0
    assert live_at_checks[-1][1] == 0
    # live state stayed far below the total input size while streaming,
    # and pruning really had work to do (live set was non-trivial mid-stream)
    assert 10 < max_live < INTERVALS // 10

    st = tracker.state()
    assert st.watermark == TWENTY_YEARS_MS
    assert st.covered == 0
    assert st.open_alert_start is None


def test_stress_deep_live_set():
    """300k simultaneously-live disjoint intervals: treap at full scale.

    5 ms activities every 24 ms across 2 hours -> total coverage per 2 h
    window is 1.5e6 ms < threshold, so no alert may fire, while the union
    structure holds up to 300k intervals before pruning catches up.
    """
    count = INTERVALS
    spacing = 24
    duration = 5

    tracker = AlertTracker()
    tracemalloc.start()
    t_start = time.perf_counter()

    # one batch: all starts are >= watermark 0
    tracker.add((i * spacing, i * spacing + duration) for i in range(count))
    assert tracker.interval_count == count

    horizon = 2 * 7_200_000
    steps = 40
    max_live = 0
    for i in range(steps):
        assert tracker.advance((i + 1) * horizon // steps) == []
        max_live = max(max_live, tracker.interval_count)

    elapsed = time.perf_counter() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\ndeep-set: {count} live intervals, elapsed {elapsed:.2f}s, "
          f"peak python mem {peak / 1e6:.1f} MB")
    assert max_live == count  # full 2h window was resident before pruning
    assert tracker.interval_count == 0  # everything pruned after window slid past
    assert elapsed < 60
    st = tracker.state()
    assert st.covered == 0 and st.open_alert_start is None
