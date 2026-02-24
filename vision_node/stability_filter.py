"""
stability_filter.py

Determines when a weight reading is stable enough to confirm.

Algorithm (per master plan §7):
    - Rolling buffer of last BUFFER_SIZE readings.
    - Stability condition:
        np.var(buffer) < VARIANCE_THRESHOLD   — buffer not bouncing
        np.mean(buffer) >= MIN_WEIGHT_KG      — not empty-scale noise
    - Both conditions must hold continuously for STABLE_DURATION_SEC.

Why variance over range check?
    - Variance is sensitive to individual outliers (single misread spikes variance).
    - Natural scale flicker (±0.1 kg) stays well within 0.05 kg².
    - np.var == 0.05 kg²  →  σ ≈ 0.22 kg  →  tight but realistic for a stable load.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np

from config import (
    STABILITY_BUFFER_SIZE,
    STABILITY_DURATION_SEC,
    STABILITY_VARIANCE_THRESH,
    STABILITY_MIN_WEIGHT_KG,
)


class StabilityFilter:
    """
    Rolling variance-based stability filter.

    Usage:
        sf = StabilityFilter()
        sf.add(482.5)
        ...
        weight = sf.get_stable_weight()   # float once stable, else None
    """

    def __init__(self) -> None:
        self._buffer: deque[float] = deque(maxlen=STABILITY_BUFFER_SIZE)
        self._stable_since: Optional[float] = None

    def add(self, weight: float) -> None:
        """Append a valid weight reading to the rolling buffer."""
        self._buffer.append(weight)

    def get_stable_weight(self) -> Optional[float]:
        """
        Return the smoothed confirmed weight when all criteria are met,
        otherwise return None.

        Criteria:
            1. Buffer has at least half its capacity filled.
            2. np.var(buffer) < STABILITY_VARIANCE_THRESH
            3. np.mean(buffer) >= STABILITY_MIN_WEIGHT_KG
            4. All of the above held for >= STABILITY_DURATION_SEC continuously.
        """
        min_samples = STABILITY_BUFFER_SIZE // 2
        if len(self._buffer) < min_samples:
            self._reset_timer()
            return None

        values   = np.array(self._buffer, dtype=np.float64)
        mean     = float(np.mean(values))
        variance = float(np.var(values))

        stable = (
            variance < STABILITY_VARIANCE_THRESH
            and mean >= STABILITY_MIN_WEIGHT_KG
        )

        if not stable:
            self._reset_timer()
            return None

        now = time.monotonic()
        if self._stable_since is None:
            self._stable_since = now
            return None

        if (now - self._stable_since) >= STABILITY_DURATION_SEC:
            return round(mean, 1)

        return None

    def reset(self) -> None:
        """Clear all state. Call when a session resets."""
        self._buffer.clear()
        self._reset_timer()

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _reset_timer(self) -> None:
        self._stable_since = None
