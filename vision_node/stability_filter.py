"""
stability_filter.py

Determines when a weight reading is stable enough to be confirmed.

Strategy:
- Majority vote across N readings to handle SSOCR flicker / decimal instability.
- Variance check across the buffer to confirm the value has settled.
- Both conditions must hold for STABLE_DURATION_SECONDS to confirm.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import Optional


# ─── Tunable Parameters ───────────────────────────────────────────────────────

# Number of readings kept in the rolling buffer.
BUFFER_SIZE = 10

# Weight must not vary by more than this (kg) across the buffer to be stable.
VARIANCE_THRESHOLD_KG = 0.5

# Majority vote window — most common reading across last N readings wins.
# Must be <= BUFFER_SIZE.
VOTE_WINDOW = 5

# Once a stable reading is detected, it must hold for this many seconds
# before being confirmed.
STABLE_DURATION_SECONDS = 5.0


class StabilityFilter:
    """
    Tracks incoming weight readings and determines when the weight is stable.

    Usage:
        sf = StabilityFilter()
        sf.add(75.5)
        ...
        result = sf.get_stable_weight()   # returns float once stable, else None
    """

    def __init__(self) -> None:
        self._buffer: deque[float] = deque(maxlen=BUFFER_SIZE)
        self._stable_since: Optional[float] = None   # timestamp
        self._last_stable_value: Optional[float] = None

    def add(self, weight: Optional[float]) -> None:
        """
        Add a new weight reading.
        Pass None if the detector failed on this frame (treated as a gap).
        """
        if weight is None:
            return
        self._buffer.append(weight)

    def get_stable_weight(self) -> Optional[float]:
        """
        Return the confirmed stable weight if stability criteria are met,
        otherwise return None.

        A weight is considered stable when:
        1. The buffer has enough readings.
        2. The dominant vote reading is consistent across VOTE_WINDOW readings.
        3. The variance of the buffer is below VARIANCE_THRESHOLD_KG.
        4. These conditions have held for STABLE_DURATION_SECONDS.
        """
        if len(self._buffer) < VOTE_WINDOW:
            self._reset_stable_timer()
            return None

        voted = self._majority_vote()
        if voted is None:
            self._reset_stable_timer()
            return None

        if not self._variance_ok():
            self._reset_stable_timer()
            return None

        # Criteria met — start or continue the stability timer
        now = time.monotonic()

        if self._stable_since is None or voted != self._last_stable_value:
            # New stable candidate — restart timer
            self._stable_since = now
            self._last_stable_value = voted
            return None

        if (now - self._stable_since) >= STABLE_DURATION_SECONDS:
            return voted

        return None

    def reset(self) -> None:
        """Clear all state. Call when session resets to IDLE."""
        self._buffer.clear()
        self._reset_stable_timer()

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _majority_vote(self) -> Optional[float]:
        """
        Return the most common reading across the last VOTE_WINDOW entries.
        Readings are rounded to 1 decimal place before voting to group
        near-identical values (e.g., 75.4 and 75.5 remain distinct).
        """
        recent = list(self._buffer)[-VOTE_WINDOW:]
        rounded = [round(v, 1) for v in recent]
        counts = Counter(rounded)
        most_common_value, most_common_count = counts.most_common(1)[0]

        # Require a clear majority (more than half the vote window)
        if most_common_count > VOTE_WINDOW // 2:
            return most_common_value
        return None

    def _variance_ok(self) -> bool:
        """Return True if weight variation across the buffer is within threshold."""
        values = list(self._buffer)
        return (max(values) - min(values)) <= VARIANCE_THRESHOLD_KG

    def _reset_stable_timer(self) -> None:
        self._stable_since = None
        self._last_stable_value = None
