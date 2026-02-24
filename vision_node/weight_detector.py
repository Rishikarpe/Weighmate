"""
weight_detector.py

Extracts the weight reading from the scale display image using SSOCR.
Target display: Eagle brand, green 7-segment LED on a dark background.

Preprocessing pipeline (per master plan §6):
    1. Crop fixed ROI                — eliminate everything outside the display
    2. Convert to grayscale          — remove colour noise
    3. convertScaleAbs(α=2.2, β=-60) — boost segment contrast, darken background
    4. GaussianBlur(3×3)             — reduce sensor noise
    5. adaptiveThreshold             — handle uneven LED brightness across digits
    6. morphologyEx OPEN             — remove small noise specks
    → Pass preprocessed binary image to SSOCR

SSOCR note:
    The input to SSOCR is already a clean binary image (segments = white,
    background = black) so we do NOT use the g_threshold filter here — that
    flag isolates the green channel from raw BGR, which we have already handled
    in step 2-6 above.

Install SSOCR:
    sudo apt install ssocr          # Debian / Raspberry Pi OS
    # or build from source:
    #   cd ssocr/ && make && sudo make install
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Optional

import cv2
import numpy as np

from config import (
    ROI_X, ROI_Y, ROI_W, ROI_H,
    SSOCR_THRESHOLD, SSOCR_MIN_DIGITS, SSOCR_MAX_DIGITS,
    SSOCR_BACKGROUND,
    WEIGHT_MIN_KG, WEIGHT_MAX_KG,
)

logger = logging.getLogger(__name__)


def extract_weight(
    frame: np.ndarray,
    debug_image_path: Optional[str] = None,
) -> Optional[float]:
    """
    Extract weight reading from a full camera frame.

    Args:
        frame:            BGR numpy array (full camera frame).
        debug_image_path: If set, saves the preprocessed ROI image here.
                          Use during hardware setup to verify the pipeline.

    Returns:
        Weight as float (e.g., 482.5) or None if extraction failed.
    """
    roi       = _crop_roi(frame)
    processed = _preprocess(roi)

    if debug_image_path:
        try:
            cv2.imwrite(debug_image_path, processed)
        except Exception as exc:
            logger.debug("Debug image write failed: %s", exc)

    raw = _run_ssocr(processed)
    if raw is None:
        return None
    return _parse_weight(raw)


# ─── Preprocessing pipeline ───────────────────────────────────────────────────

def _crop_roi(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, ROI_X);      y1 = max(0, ROI_Y)
    x2 = min(w, ROI_X + ROI_W); y2 = min(h, ROI_Y + ROI_H)
    return frame[y1:y2, x1:x2]


def _preprocess(roi: np.ndarray) -> np.ndarray:
    """
    Produce a clean binary image (bright segments on black background)
    suitable for SSOCR.
    """
    # Step 1: Grayscale — collapse colour channels
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Step 2: Contrast boost — amplify LED segments, push dark background to black
    boosted = cv2.convertScaleAbs(gray, alpha=2.2, beta=-60)

    # Step 3: Denoise — remove high-frequency sensor noise before thresholding
    blurred = cv2.GaussianBlur(boosted, (3, 3), 0)

    # Step 4: Adaptive threshold — handles uneven brightness across individual digits
    #   blockSize=15: neighbourhood for local threshold (tune if digits are large/small)
    #   C=-5: bias toward brighter pixels (keeps dim segments while dropping background)
    binary = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=15,
        C=-5,
    )

    # Step 5: Morphological opening — remove isolated noise specks smaller than kernel
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned


# ─── SSOCR interface ──────────────────────────────────────────────────────────

def _run_ssocr(image: np.ndarray) -> Optional[str]:
    """
    Write *image* to a temp PNG, invoke SSOCR, return the raw text output.
    Returns None on any failure so the caller retries next frame.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, image)
        return _invoke_ssocr(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _invoke_ssocr(image_path: str) -> Optional[str]:
    cmd = [
        "ssocr",
        "--background", SSOCR_BACKGROUND,
        "-c", "digits",
        "-d", f"{SSOCR_MIN_DIGITS}-{SSOCR_MAX_DIGITS}",
        "-a",                               # absolute threshold mode
        f"--threshold={SSOCR_THRESHOLD}",
        "erosion",                          # close any remaining segment gaps
        image_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2.0,   # fail fast — never stall the main loop
        )
        output = result.stdout.strip()
        if output and result.returncode == 0:
            return output
        if result.returncode != 0:
            logger.debug("SSOCR exit %d: %s", result.returncode, result.stderr.strip())
        return None

    except subprocess.TimeoutExpired:
        logger.debug("SSOCR timed out")
        return None

    except FileNotFoundError:
        raise RuntimeError(
            "ssocr executable not found.\n"
            "Install:   sudo apt install ssocr\n"
            "Or build:  cd ssocr/ && make && sudo make install"
        )


# ─── Weight parsing ───────────────────────────────────────────────────────────

def _parse_weight(raw: str) -> Optional[float]:
    """
    Parse SSOCR output to a validated float.

    Args:
        raw: SSOCR stdout, e.g. "482.5" or "500"

    Returns:
        Float weight or None if unparseable / outside the valid range.
    """
    try:
        value = float(raw.replace(",", "."))   # handle locale decimal separator
    except ValueError:
        logger.debug("Unparseable SSOCR output: %r", raw)
        return None

    if not (WEIGHT_MIN_KG <= value <= WEIGHT_MAX_KG):
        logger.debug(
            "Weight %.1f kg out of range [%.0f, %.0f]",
            value, WEIGHT_MIN_KG, WEIGHT_MAX_KG,
        )
        return None

    return value
