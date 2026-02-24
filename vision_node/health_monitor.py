"""
health_monitor.py

Monitors camera frame quality every HEALTH_PUBLISH_INTERVAL seconds.
Reports issues to the HMI via MQTT.

Checks (per master plan §10):
    1. Blur          — Laplacian variance. Low = dirty lens or out of focus.
    2. Brightness    — Mean pixel intensity. Low = too dark, high = overexposed.
    3. Watchdog      — If in an active session but no valid weight read for
                       OBSTRUCTION_TIMEOUT_SECONDS, the display is likely blocked.

All thresholds are read from config.yaml so they can be tuned without code changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config import (
    BLUR_THRESHOLD,
    BRIGHTNESS_MIN,
    BRIGHTNESS_MAX,
    OBSTRUCTION_TIMEOUT_SECONDS,
)


@dataclass
class HealthStatus:
    ok: bool          = True
    blurry: bool      = False
    too_dark: bool    = False
    overexposed: bool = False
    obstructed: bool  = False
    blur_score: float = 0.0
    brightness: float = 0.0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok":          self.ok,
            "blurry":      self.blurry,
            "too_dark":    self.too_dark,
            "overexposed": self.overexposed,
            "obstructed":  self.obstructed,
            "blur_score":  round(self.blur_score, 2),
            "brightness":  round(self.brightness, 2),
            "issues":      self.issues,
        }


class HealthMonitor:
    """
    Per-frame camera health evaluator.

    Usage:
        monitor = HealthMonitor()
        status  = monitor.check(frame, session_active=True, weight_detected=False)
        if not status.ok:
            mqtt.publish_health(status.to_dict())
    """

    def __init__(self) -> None:
        self._last_weight_time: Optional[float] = None

    def check(
        self,
        frame: np.ndarray,
        session_active: bool,
        weight_detected: bool,
    ) -> HealthStatus:
        """
        Evaluate frame health.

        Args:
            frame:           BGR numpy array from the camera.
            session_active:  True when state is STABILIZING or CONFIRMED.
            weight_detected: True if weight_detector returned a value this frame.

        Returns:
            HealthStatus with all flags populated.
        """
        status = HealthStatus()
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Blur detection (Laplacian variance)
        blur_score       = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        status.blur_score = blur_score
        if blur_score < BLUR_THRESHOLD:
            status.blurry = True
            status.issues.append(
                f"Camera blurry (score={blur_score:.1f}, threshold={BLUR_THRESHOLD})"
            )

        # 2. Brightness
        brightness       = float(gray.mean())
        status.brightness = brightness
        if brightness < BRIGHTNESS_MIN:
            status.too_dark = True
            status.issues.append(
                f"Too dark (brightness={brightness:.1f}, min={BRIGHTNESS_MIN})"
            )
        elif brightness > BRIGHTNESS_MAX:
            status.overexposed = True
            status.issues.append(
                f"Overexposed (brightness={brightness:.1f}, max={BRIGHTNESS_MAX})"
            )

        # 3. Watchdog — display obstruction / unreadable display
        now = time.monotonic()
        if weight_detected:
            self._last_weight_time = now

        if session_active:
            if self._last_weight_time is None:
                self._last_weight_time = now
            elapsed = now - self._last_weight_time
            if elapsed >= OBSTRUCTION_TIMEOUT_SECONDS:
                status.obstructed = True
                status.issues.append(
                    f"Display obstructed — no digit read for {elapsed:.1f}s "
                    f"(threshold={OBSTRUCTION_TIMEOUT_SECONDS}s)"
                )

        status.ok = len(status.issues) == 0
        return status

    def reset(self) -> None:
        """Reset watchdog timer. Call when session returns to IDLE."""
        self._last_weight_time = None
