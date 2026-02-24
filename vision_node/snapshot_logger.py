"""
snapshot_logger.py

Saves an annotated JPEG snapshot and appends a CSV row for every
confirmed weighing event.

Output:
    SNAPSHOT_DIR/<scale_id>_<timestamp>_<weight>kg.jpg
    SNAPSHOT_CSV  — one row per confirmation, appended on each save

Called by main.py's _on_confirmed callback with the live camera frame.
"""

from __future__ import annotations

import csv
import datetime
import logging
import os
from typing import Optional

import cv2
import numpy as np

from config import SNAPSHOT_DIR, SNAPSHOT_CSV, SCALE_ID

logger = logging.getLogger(__name__)


def save(frame: Optional[np.ndarray], weight: float) -> Optional[str]:
    """
    Write a snapshot image + CSV row for a confirmed weight.

    Args:
        frame:  BGR numpy array (the camera frame at the moment of confirmation).
                If None, a black placeholder is saved so the CSV row still exists.
        weight: Confirmed stable weight in kg.

    Returns:
        Absolute path to the saved JPEG, or None on failure.
    """
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    _ensure_csv_header()

    ts     = datetime.datetime.now()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    fname  = f"{SCALE_ID}_{ts_str}_{weight:.1f}kg.jpg"
    path   = os.path.join(SNAPSHOT_DIR, fname)

    try:
        img = _annotate(frame, weight, ts)
        cv2.imwrite(path, img)
    except Exception as exc:
        logger.error("Snapshot write failed: %s", exc)
        return None

    try:
        _append_csv(ts, weight, path)
    except Exception as exc:
        logger.warning("CSV append failed: %s", exc)

    logger.info("Snapshot saved: %s (%.1f kg)", path, weight)
    return path


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _annotate(
    frame: Optional[np.ndarray],
    weight: float,
    ts: datetime.datetime,
) -> np.ndarray:
    """Draw weight + timestamp overlay on the frame."""
    img = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    lines = [
        (f"{weight:.1f} kg — CONFIRMED", img.shape[0] - 70),
        (ts.strftime("%Y-%m-%d  %H:%M:%S"),  img.shape[0] - 45),
        (f"Scale: {SCALE_ID}",               img.shape[0] - 20),
    ]

    for text, y in lines:
        # Black outline for legibility on any background
        cv2.putText(img, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        # Green fill
        cv2.putText(img, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return img


def _ensure_csv_header() -> None:
    if not os.path.exists(SNAPSHOT_CSV):
        csv_dir = os.path.dirname(SNAPSHOT_CSV)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(SNAPSHOT_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["timestamp", "scale_id", "weight_kg", "snapshot_path"]
            )


def _append_csv(ts: datetime.datetime, weight: float, path: str) -> None:
    with open(SNAPSHOT_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts.isoformat(), SCALE_ID, weight, path])
