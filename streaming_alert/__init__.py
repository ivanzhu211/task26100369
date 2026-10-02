"""Streaming alert computation for employee non-productive activity."""

from .tracker import AlertTracker, THRESHOLD_MS, WINDOW_MS

__all__ = ["AlertTracker", "WINDOW_MS", "THRESHOLD_MS"]
