"""
tools/collect_training_data.py

Captures scale display images from the live camera feed and auto-labels them
using SSOCR. Produces a dataset ready for TFLite fine-tuning (plan step 3).

Output layout:
    ~/weighmate/training_data/
        raw/          — ROI crops, named  <weight>kg_<timestamp>[_discarded].jpg
        labeled.csv   — timestamp, image_path, ssocr_reading, confirmed

Usage:
    python tools/collect_training_data.py [--target N]

    --target N   Number of confirmed images to collect (default: 500)

Controls:
    SPACE   — accept current SSOCR reading, save image as confirmed
    D       — discard frame (saved as unconfirmed for negative examples)
    Q / Esc — quit
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, CAPTURE_FPS,
    ROI_X, ROI_Y, ROI_W, ROI_H,
    SSOCR_FRAME_SKIP,
)
from weight_detector import extract_weight

_OUTPUT_DIR  = os.path.expanduser("~/weighmate/training_data")
_RAW_DIR     = os.path.join(_OUTPUT_DIR, "raw")
_LABELED_CSV = os.path.join(_OUTPUT_DIR, "labeled.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled scale display images.")
    parser.add_argument("--target", type=int, default=500,
                        help="Number of confirmed images to collect (default: 500)")
    args = parser.parse_args()

    os.makedirs(_RAW_DIR, exist_ok=True)
    _ensure_csv()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera (index={CAMERA_INDEX}). Check config.yaml.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAPTURE_FPS)

    print(__doc__)
    print(f"Target: {args.target} confirmed images → {_OUTPUT_DIR}\n")

    count       = 0
    frame_idx   = 0
    last_weight = None

    while count < args.target:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_idx += 1
        if frame_idx % SSOCR_FRAME_SKIP == 0:
            last_weight = extract_weight(frame)

        roi    = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
        weight = last_weight

        # ── Build display ──────────────────────────────────────────────────────
        display = frame.copy()
        cv2.rectangle(display,
                      (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H),
                      (0, 255, 0), 2)

        wlabel = f"{weight:.1f} kg" if weight is not None else "NO READ"
        color  = (0, 255, 0) if weight is not None else (0, 0, 255)
        cv2.putText(display, wlabel, (ROI_X, max(ROI_Y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        cv2.putText(display, f"Confirmed: {count}/{args.target}",
                    (10, display.shape[0] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(display, "SPACE=save  D=discard  Q=quit",
                    (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Collect Training Data", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(" ") and weight is not None:
            path = _save_image(roi, weight)
            _append_csv(path, weight, confirmed=True)
            count += 1
            print(f"[{count}/{args.target}] Saved {weight:.1f} kg → {os.path.basename(path)}")

        elif key == ord("d"):
            if weight is not None:
                path = _save_image(roi, weight, suffix="_discarded")
                _append_csv(path, weight, confirmed=False)
            print("Discarded.")

        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {count} confirmed images saved to {_OUTPUT_DIR}")
    print(f"CSV:  {_LABELED_CSV}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _save_image(roi, weight: float, suffix: str = "") -> str:
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{weight:.1f}kg_{ts}{suffix}.jpg"
    path  = os.path.join(_RAW_DIR, fname)
    cv2.imwrite(path, roi)
    return path


def _ensure_csv() -> None:
    if not os.path.exists(_LABELED_CSV):
        with open(_LABELED_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["timestamp", "image_path", "ssocr_reading", "confirmed"]
            )


def _append_csv(path: str, weight: float, confirmed: bool) -> None:
    with open(_LABELED_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().isoformat(),
            path,
            weight,
            confirmed,
        ])


if __name__ == "__main__":
    main()
