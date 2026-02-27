"""
tools/test_weight_image.py

Test the weight detector against uploaded / saved images — no webcam needed.

Usage (run from the vision_node/ directory):
    python tools/test_weight_image.py path/to/image.jpg [more images ...]

    # Override ROI without touching config.yaml:
    python tools/test_weight_image.py scale.jpg --roi 100 50 300 80

    # Batch / headless (no window):
    python tools/test_weight_image.py *.jpg --no-display

    # Save the 200×31 preprocessed ROI crop used by the model:
    python tools/test_weight_image.py scale.jpg --debug

Controls (interactive mode):
    SPACE / N   — next image
    P           — previous image
    Q / Esc     — quit
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import sys

import cv2
import numpy as np

# ── Path setup so `from config import …` works ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weight_detector as _wd
from weight_detector import extract_weight


# ─── Auto ROI ─────────────────────────────────────────────────────────────────

def auto_detect_roi(frame: np.ndarray) -> tuple[int, int, int, int]:
    """
    Auto-detect the LCD/7-segment display ROI using edge density.
    Returns (x, y, w, h). Falls back to full frame if nothing is found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_img, w_img = frame.shape[:2]

    edges = cv2.Canny(gray, 20, 80)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    dilated = cv2.dilate(edges, kernel_h)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
    dilated = cv2.dilate(dilated, kernel_v)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0
        area = w * h
        if (2.0 < aspect < 16.0
                and 0.004 * w_img * h_img < area < 0.5 * w_img * h_img
                and x > 2 and y > 2
                and x + w < w_img - 2 and y + h < h_img - 2):
            roi_edges = edges[y:y + h, x:x + w]
            density = np.count_nonzero(roi_edges) / float(area)
            candidates.append((x, y, w, h, density))

    if candidates:
        x, y, w, h, _ = max(candidates, key=lambda b: b[4])
        # Add 8% padding so clipped display edges are included
        px, py = max(1, int(w * 0.08)), max(1, int(h * 0.08))
        x = max(0, x - px);       y = max(0, y - py)
        w = min(w_img - x, w + 2 * px)
        h = min(h_img - y, h + 2 * py)
        return (x, y, w, h)

    return (0, 0, w_img, h_img)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test weight detector on uploaded images (no webcam).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("images", nargs="+",
                   help="Image file path(s).  Glob patterns are expanded automatically.")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                   default=None,
                   help="Override ROI from config.yaml (pixels: x y w h).")
    p.add_argument("--no-display", action="store_true",
                   help="Print results only — skip the OpenCV window.")
    p.add_argument("--debug", action="store_true",
                   help="Save the 200×31 preprocessed ROI crop next to each image.")
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pat in patterns:
        expanded = _glob.glob(pat)
        if expanded:
            paths.extend(expanded)
        elif os.path.isfile(pat):
            paths.append(pat)
        else:
            print(f"[WARN] No file matched: {pat}")
    return sorted(set(paths))


def _patch_roi(rx: int, ry: int, rw: int, rh: int) -> None:
    """Overwrite the module-level ROI names that _crop_roi reads at call time."""
    _wd.ROI_X = rx
    _wd.ROI_Y = ry
    _wd.ROI_W = rw
    _wd.ROI_H = rh


def run_detector(
    frame: np.ndarray,
    rx: int, ry: int, rw: int, rh: int,
    debug_path: str | None = None,
) -> float | None:
    _patch_roi(rx, ry, rw, rh)
    return extract_weight(frame, debug_image_path=debug_path)


def debug_path_for(image_path: str) -> str:
    base, _ = os.path.splitext(image_path)
    return base + "_debug_roi.png"


def annotate(
    frame: np.ndarray,
    weight: float | None,
    rx: int, ry: int, rw: int, rh: int,
    img_path: str,
    idx: int,
    total: int,
) -> np.ndarray:
    out = frame.copy()

    # ROI box
    cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    cv2.putText(out, "ROI", (rx + 4, ry + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Weight label
    label = f"{weight:.1f} kg" if weight is not None else "NO READ"
    color = (0, 255, 0) if weight is not None else (0, 0, 255)
    cv2.putText(out, label, (rx, max(ry - 12, 28)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

    # Footer
    footer = (f"[{idx}/{total}]  {os.path.basename(img_path)}"
              f"   |   SPACE/N=next  P=prev  Q=quit")
    cv2.putText(out, footer, (8, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def print_summary(results: list[tuple[str, float | None]]) -> None:
    total   = len(results)
    ok      = [r for r in results if r[1] is not None]
    no_read = total - len(ok)
    print("\n" + "─" * 52)
    print(f"  Summary: {len(ok)}/{total} images produced a reading")
    if no_read:
        print(f"           {no_read} returned NO READ (out-of-range or bad crop)")
    if ok:
        weights = [r[1] for r in ok]
        print(f"           Min {min(weights):.1f} kg  "
              f"Max {max(weights):.1f} kg  "
              f"Avg {sum(weights)/len(weights):.1f} kg")
    print("─" * 52 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def _get_roi(frame: np.ndarray, override: list[int] | None) -> tuple[int, int, int, int]:
    """Return ROI from CLI override, or auto-detect from the frame."""
    if override:
        return tuple(override)  # type: ignore[return-value]
    roi = auto_detect_roi(frame)
    print(f"  [ROI] auto-detected: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
    return roi


def main() -> None:
    args = parse_args()

    if args.roi:
        print(f"\n[ROI] CLI --roi: x={args.roi[0]} y={args.roi[1]} w={args.roi[2]} h={args.roi[3]}")
    else:
        print("\n[ROI] auto-detect mode — ROI determined per image")

    paths = expand_paths(args.images)
    if not paths:
        print("No images found. Exiting.")
        sys.exit(1)

    print(f"[INFO] {len(paths)} image(s) to test\n")

    results: list[tuple[str, float | None]] = []

    # ── Batch / headless ─────────────────────────────────────────────────────
    if args.no_display:
        for path in paths:
            frame = cv2.imread(path)
            if frame is None:
                print(f"  [SKIP]   {path}  (cannot load)")
                results.append((path, None))
                continue
            rx, ry, rw, rh = _get_roi(frame, args.roi)
            dp = debug_path_for(path) if args.debug else None
            w  = run_detector(frame, rx, ry, rw, rh, dp)
            tag = f"{w:.1f} kg" if w is not None else "NO READ"
            print(f"  {tag:<14}  {path}")
            if dp and args.debug and w is not None:
                print(f"             └─ debug crop → {dp}")
            results.append((path, w))
        print_summary(results)
        return

    # ── Interactive ───────────────────────────────────────────────────────────
    # cache maps index → (frame, weight, roi)
    cache: dict[int, tuple[np.ndarray, float | None, tuple[int, int, int, int]]] = {}

    def load(i: int) -> tuple[np.ndarray, float | None, tuple[int, int, int, int]]:
        if i not in cache:
            path  = paths[i]
            frame = cv2.imread(path)
            if frame is None:
                print(f"  [SKIP] Cannot load: {path}")
                cache[i] = (np.zeros((480, 640, 3), dtype=np.uint8), None, (0, 0, 640, 480))
            else:
                rx, ry, rw, rh = _get_roi(frame, args.roi)
                dp = debug_path_for(path) if args.debug else None
                w  = run_detector(frame, rx, ry, rw, rh, dp)
                cache[i] = (frame, w, (rx, ry, rw, rh))
                tag = f"{w:.1f} kg" if w is not None else "NO READ"
                print(f"  [{i+1}/{len(paths)}]  {tag:<14}  {path}")
        return cache[i]

    idx = 0
    while True:
        frame, weight, (rx, ry, rw, rh) = load(idx)
        display = annotate(frame, weight, rx, ry, rw, rh, paths[idx], idx + 1, len(paths))
        cv2.imshow("WeightMate — Image Test", display)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord(" "), ord("n")):
            idx = min(idx + 1, len(paths) - 1)
        elif key == ord("p"):
            idx = max(idx - 1, 0)

    cv2.destroyAllWindows()
    print_summary([(p, cache[i][1]) for i, p in enumerate(paths) if i in cache])


if __name__ == "__main__":
    main()
