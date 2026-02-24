"""
health_monitor.py

Monitors camera frame quality and reports issues to the HMI via MQTT.

Checks performed every frame:
    1. Blur detection     — Laplacian variance. Low = camera dirty or out of focus.
    2. Brightness check   — Mean pixel intensity. Too dark or too bright = bad lighting.
    3. Obstruction check  — If session is active but no digits detected for too long,
                            the camera view is likely blocked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# ─── Thresholds ───────────────────────────────────────────────────────────────

# Laplacian variance below this = blurry frame.
# Tune on actual hardware — typical clear frames score 100+, blurry < 20.
BLUR_THRESHOLD = 30.0

# Mean pixel brightness (0-255). Outside this range = lighting problem.
BRIGHTNESS_MIN = 20    # too dark
BRIGHTNESS_MAX = 240   # overexposed

# If in an active session (weight expected) but no digit read for this long,
# flag as obstructed.
OBSTRUCTION_TIMEOUT_SECONDS = 4.0


@dataclass
class HealthStatus:
    ok: bool               = True
    blurry: bool           = False
    too_dark: bool         = False
    overexposed: bool      = False
    obstructed: bool       = False
    blur_score: float      = 0.0
    brightness: float      = 0.0
    issues: list[str]      = field(default_factory=list)

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
    Evaluates camera frame quality on each update call.

    Usage:
        monitor = HealthMonitor()

        # In main loop:
        status = monitor.check(frame, session_active=True, weight_detected=False)
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
            frame:           BGR numpy array from camera.
            session_active:  True if state machine is in STABILIZING or CONFIRMED.
            weight_detected: True if weight_detector returned a value this frame.

        Returns:
            HealthStatus with all flags populated.
        """
        status = HealthStatus()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Blur detection
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        status.blur_score = blur_score
        if blur_score < BLUR_THRESHOLD:
            status.blurry = True
            status.issues.append(f"Camera blurry (score={blur_score:.1f}, min={BLUR_THRESHOLD})")

        # 2. Brightness check
        brightness = float(gray.mean())
        status.brightness = brightness
        if brightness < BRIGHTNESS_MIN:
            status.too_dark = True
            status.issues.append(f"Too dark (brightness={brightness:.1f}, min={BRIGHTNESS_MIN})")
        elif brightness > BRIGHTNESS_MAX:
            status.overexposed = True
            status.issues.append(f"Overexposed (brightness={brightness:.1f}, max={BRIGHTNESS_MAX})")

        # 3. Obstruction check
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
                    f"Display obstructed or unreadable "
                    f"({elapsed:.1f}s without digit read during active session)"
                )

        status.ok = not status.issues
        return status

    def reset(self) -> None:
        """Reset obstruction timer. Call when session returns to IDLE."""
        self._last_weight_time = None
