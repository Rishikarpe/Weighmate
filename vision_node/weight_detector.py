"""
weight_detector.py

Extracts weight reading from the scale display image using SSOCR.
Target display: Eagle brand, green 7-segment LED on dark background.

Install SSOCR on Raspberry Pi:
    sudo apt install ssocr
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional

import cv2
import numpy as np

# ─── ROI Configuration ────────────────────────────────────────────────────────
# Crop region for the digit area within the camera frame.
# Tune these values once the camera is physically mounted above the display.
# Use the debug mode (save_debug_image=True) to visually verify the crop.
#
#   +──────────────── frame width ─────────────────+
#   |                                               |
#   |   (ROI_X, ROI_Y) ┌──── ROI_W ────┐           |
#   |                  │               │  ROI_H     |
#   |                  └───────────────┘           |
#   +───────────────────────────────────────────────+
#
ROI_X = 0      # pixels from left edge of frame   ← TUNE
ROI_Y = 0      # pixels from top edge of frame    ← TUNE
ROI_W = 640    # width of crop region              ← TUNE
ROI_H = 480    # height of crop region             ← TUNE

# ─── SSOCR Configuration ──────────────────────────────────────────────────────
# Luminance threshold for green channel isolation.
# 20 is a good starting point for LED displays.
# Increase if getting false reads, decrease if missing digits.
SSOCR_THRESHOLD = 20     # ← TUNE on actual hardware

SSOCR_MIN_DIGITS = 3     # 50.0  → 3 digits (5, 0, 0)
SSOCR_MAX_DIGITS = 4     # 500.0 → 4 digits (5, 0, 0, 0)

# ─── Weight Validation ────────────────────────────────────────────────────────
WEIGHT_MIN_KG = 50.0
WEIGHT_MAX_KG = 500.0


def extract_weight(
    frame: np.ndarray,
    save_debug_image: Optional[str] = None,
) -> Optional[float]:
    """
    Extract weight reading from a camera frame.

    Args:
        frame: BGR numpy array from OpenCV (full camera frame).
        save_debug_image: If provided, saves SSOCR debug image to this path.
                          Useful during camera setup to verify ROI and tuning.

    Returns:
        Weight as float (e.g., 75.5) or None if extraction failed.
    """
    cropped = _crop_roi(frame)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cv2.imwrite(tmp_path, cropped)
        raw = _run_ssocr(tmp_path, save_debug_image=save_debug_image)
        if raw is None:
            return None
        return _parse_weight(raw)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _crop_roi(frame: np.ndarray) -> np.ndarray:
    """Crop the digit region from the full camera frame."""
    h, w = frame.shape[:2]
    x1 = max(0, ROI_X)
    y1 = max(0, ROI_Y)
    x2 = min(w, ROI_X + ROI_W)
    y2 = min(h, ROI_Y + ROI_H)
    return frame[y1:y2, x1:x2]


def _run_ssocr(
    image_path: str,
    save_debug_image: Optional[str] = None,
) -> Optional[str]:
    """
    Run SSOCR on the image and return raw string output.

    Uses green channel isolation (g_threshold) for the green LED display.
    Returns None on any failure — caller retries next frame.
    """
    cmd = [
        "ssocr",
        "-d", f"{SSOCR_MIN_DIGITS}-{SSOCR_MAX_DIGITS}",
        "--background", "black",
        "-c", "digits",
        "-a",                            # use absolute threshold (not auto)
        f"--threshold={SSOCR_THRESHOLD}",
        "g_threshold",                   # isolate green channel
        "erosion",                       # fill segment gaps
    ]

    if save_debug_image:
        cmd += [f"--debug-image={save_debug_image}"]

    cmd.append(image_path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2.0,   # fail fast — never block the main loop
        )

        output = result.stdout.strip()

        if not output or result.returncode != 0:
            return None

        return output

    except subprocess.TimeoutExpired:
        return None

    except FileNotFoundError:
        raise RuntimeError(
            "ssocr executable not found.\n"
            "Install with: sudo apt install ssocr"
        )


def _parse_weight(raw: str) -> Optional[float]:
    """
    Parse SSOCR output string to a validated float weight.

    Args:
        raw: string from SSOCR e.g. "75.5" or "500.0"

    Returns:
        Float weight or None if unparseable or out of valid range.
    """
    try:
        value = float(raw)
    except ValueError:
        return None

    if not (WEIGHT_MIN_KG <= value <= WEIGHT_MAX_KG):
        return None

    return value
