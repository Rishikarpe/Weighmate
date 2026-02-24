"""
tools/calibrate_roi.py

Interactive ROI calibration tool.

Run this once after physically mounting the camera above the scale display.
Draw a rectangle around the digit region — coordinates are saved to config.yaml
and used by weight_detector.py on every subsequent run.

Usage:
    python tools/calibrate_roi.py

Controls:
    Drag     — draw rectangle over the scale display digits
    S        — save ROI to config.yaml and exit
    R        — reset selection
    Q / Esc  — quit without saving
"""

from __future__ import annotations

import os
import sys

import cv2
import yaml

# Allow running from the tools/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, CAPTURE_FPS

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)

# ─── Mouse state (module-level cells avoid global keyword noise) ───────────────
_start    = [(0, 0)]
_end      = [(0, 0)]
_dragging = [False]
_roi: list = [None]   # confirmed (x, y, w, h) or None


def _on_mouse(event, x: int, y: int, flags, param) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        _start[0]    = (x, y)
        _end[0]      = (x, y)
        _dragging[0] = True
        _roi[0]      = None

    elif event == cv2.EVENT_MOUSEMOVE and _dragging[0]:
        _end[0] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and _dragging[0]:
        _end[0]      = (x, y)
        _dragging[0] = False
        x0 = min(_start[0][0], _end[0][0])
        y0 = min(_start[0][1], _end[0][1])
        w  = abs(_end[0][0] - _start[0][0])
        h  = abs(_end[0][1] - _start[0][1])
        if w > 10 and h > 10:
            _roi[0] = (x0, y0, w, h)
            print(f"ROI selected → x={x0}, y={y0}, w={w}, h={h}")
        else:
            print("Selection too small — try again.")


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera (index={CAMERA_INDEX}). Check config.yaml.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAPTURE_FPS)

    win = "Calibrate ROI — drag to select  |  S=save  R=reset  Q=quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_mouse)

    print(__doc__)
    print(f"Camera index={CAMERA_INDEX}  resolution={FRAME_WIDTH}×{FRAME_HEIGHT}")
    print("Draw a rectangle around the scale display digits.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed — retrying...")
            continue

        display = frame.copy()
        roi     = _roi[0]

        # Live drag outline
        if _dragging[0]:
            cv2.rectangle(display, _start[0], _end[0], (0, 200, 255), 2)

        # Confirmed ROI
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                display, f"ROI  x={x} y={y} w={w} h={h}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
            )

            # Inset preview of the cropped region (top-right corner)
            crop = frame[y:y + h, x:x + w]
            if crop.size > 0:
                ph = 120
                pw = int(crop.shape[1] * ph / max(crop.shape[0], 1))
                preview = cv2.resize(crop, (pw, ph))
                py1 = 8;             py2 = 8 + ph
                px1 = display.shape[1] - pw - 8; px2 = display.shape[1] - 8
                display[py1:py2, px1:px2] = preview
                cv2.rectangle(display, (px1, py1), (px2, py2), (0, 255, 0), 1)

        cv2.putText(
            display, "S = save to config.yaml   R = reset   Q = quit",
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
        )

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if roi is None:
                print("Nothing to save — draw a rectangle first.")
            else:
                _save_roi(roi)
                break

        elif key == ord("r"):
            _roi[0] = None
            print("Selection reset.")

        elif key in (ord("q"), 27):   # Q or Esc
            print("Exiting without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()


def _save_roi(roi: tuple) -> None:
    x, y, w, h = roi
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["roi"] = {"x": x, "y": y, "w": w, "h": h}
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Saved to config.yaml — ROI: x={x}, y={y}, w={w}, h={h}")
    print("Restart the vision node to apply the new ROI.")


if __name__ == "__main__":
    main()
