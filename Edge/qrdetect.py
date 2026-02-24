"""
QR Detection — ArUco-based detector for robust angled QR reading on RPi.

Uses cv2.QRCodeDetectorAruco as primary with cv2.QRCodeDetector as fallback:
- ArUco finder-pattern detection + homography warp before decoding
- Handles perspective distortion (camera at angle) far better than pyzbar/basic OpenCV
- Unsharp-mask sharpening + CLAHE preprocessing for contrast at angles / bad lighting
- Multi-scale detection: tries native resolution then 2× upscale for distant QR codes
- Fallback to cv2.QRCodeDetector when ArUco fails (different failure modes)
- Polygon area + shortest side for size thresholds (accurate at any angle)

Thresholds come from config.py:
  SIZE_THRESHOLD_AREA_PX  → min polygon area in px²
  MIN_WIDTH_PX            → min shortest polygon side in px
"""

import math

import cv2
import numpy as np

from config import (
    SIZE_THRESHOLD_AREA_PX,
    MIN_WIDTH_PX,
)


class QRSizeDetector:
    """
    Robust QR detector using cv2.QRCodeDetectorAruco (primary) with
    cv2.QRCodeDetector fallback and multi-scale upscale for distant codes.

    detect_and_decode_qr() returns:
        (bbox, area, width, threshold_met, decoded_data)
    """

    def __init__(self):
        self.pixels_per_cm = None
        self.detection_count = 0
        self.below_threshold_count = 0
        self.successful_decodes = 0
        self.last_bbox = None

        # CLAHE for contrast normalisation — clipLimit=3 is stronger, helps at distance
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Primary: QRCodeDetectorAruco — best for angled/perspective shots
        self._detector = cv2.QRCodeDetectorAruco()
        # Fallback: standard OpenCV detector — different algorithm, catches what ArUco misses
        self._fallback = cv2.QRCodeDetector()
        print("[QR] Using QRCodeDetectorAruco (primary) + QRCodeDetector (fallback), multi-scale enabled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_decode_qr(self, frame):
        """
        Detect and decode QR codes in frame.

        Detection pipeline (stops at first success):
          1. Native resolution  — QRCodeDetectorAruco
          2. 2× upscale         — QRCodeDetectorAruco  (catches small/distant codes)
          3. 2× upscale         — QRCodeDetector fallback

        Returns: (bbox, area, width, threshold_met, decoded_data)
            bbox          – (x1, y1, x2, y2) axis-aligned bounding box
            area          – polygon area in px²
            width         – shortest polygon side in px
            threshold_met – bool
            decoded_data  – decoded string or None
        """
        gray = self._preprocess(frame)

        # --- Pass 1: native resolution with primary detector ---
        retval, decoded_list, points_list = self._run_detector(gray, self._detector)

        # --- Pass 2: 2× upscale with primary detector (helps at 1.5–2 m range) ---
        if not retval or not any(decoded_list):
            gray_2x = cv2.resize(gray, None, fx=2.0, fy=2.0,
                                 interpolation=cv2.INTER_CUBIC)
            retval2, dl2, pl2 = self._run_detector(gray_2x, self._detector)
            if retval2 and pl2 is not None and len(pl2) > 0:
                pl2 = [pts * 0.5 for pts in pl2]   # scale coords back to original
                retval, decoded_list, points_list = retval2, dl2, pl2

        # --- Pass 3: 2× upscale with fallback detector ---
        if not retval or not any(decoded_list):
            gray_2x = cv2.resize(gray, None, fx=2.0, fy=2.0,
                                 interpolation=cv2.INTER_CUBIC)
            retval3, dl3, pl3 = self._run_detector(gray_2x, self._fallback)
            if retval3 and pl3 is not None and len(pl3) > 0:
                pl3 = [pts * 0.5 for pts in pl3]
                retval, decoded_list, points_list = retval3, dl3, pl3

        if not retval or points_list is None or len(points_list) == 0:
            self.last_bbox = None
            return None, 0, 0, False, None

        # Pick largest QR by polygon area
        best_data, best_poly, best_area = None, None, 0
        for data, pts in zip(decoded_list, points_list):
            if pts is None or len(pts) == 0:
                continue
            poly = [(int(p[0]), int(p[1])) for p in pts]
            area = self._polygon_area(poly)
            if area > best_area:
                best_area, best_poly, best_data = area, poly, data

        if best_poly is None:
            self.last_bbox = None
            return None, 0, 0, False, None

        # Bounding box and measurements
        x1 = min(p[0] for p in best_poly)
        y1 = min(p[1] for p in best_poly)
        x2 = max(p[0] for p in best_poly)
        y2 = max(p[1] for p in best_poly)
        bbox = (x1, y1, x2, y2)
        min_side = self._min_side(best_poly)
        decoded_data = best_data if best_data else None

        self.detection_count += 1
        if decoded_data:
            self.successful_decodes += 1

        # Threshold check
        threshold_met = (best_area >= SIZE_THRESHOLD_AREA_PX and
                         min_side >= MIN_WIDTH_PX)
        if not threshold_met:
            self.below_threshold_count += 1

        self.last_bbox = bbox

        if self.detection_count % 100 == 0:
            self._log_stats()

        return bbox, best_area, min_side, threshold_met, decoded_data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_detector(self, gray, detector):
        """
        Run detector.detectAndDecodeMulti on gray image.
        Returns (retval, decoded_list, points_list) or (False, [], None) on failure.
        """
        try:
            retval, decoded_list, points_list, _ = detector.detectAndDecodeMulti(gray)
            if retval and points_list is not None and len(points_list) > 0:
                return retval, decoded_list, points_list
        except Exception:
            pass
        return False, [], None

    def _preprocess(self, frame):
        """
        Convert to grayscale, apply unsharp-mask sharpening, then CLAHE.

        Sharpening recovers finder-pattern edges lost to motion blur or
        camera focus at distance. CLAHE normalises contrast for angled shots.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Unsharp mask: subtract a blurred copy to boost edges
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        gray = self._clahe.apply(gray)
        return gray

    @staticmethod
    def _polygon_area(pts) -> float:
        """Shoelace formula for polygon area."""
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    @staticmethod
    def _min_side(pts) -> float:
        """Shortest edge length of the polygon."""
        n = len(pts)
        return min(
            math.hypot(pts[(i+1)%n][0] - pts[i][0],
                       pts[(i+1)%n][1] - pts[i][1])
            for i in range(n)
        )

    def _log_stats(self):
        if self.detection_count > 0:
            t_rate = (self.below_threshold_count / self.detection_count) * 100
            d_rate = (self.successful_decodes / self.detection_count) * 100
            print(f"[QR] Stats: {self.detection_count} detected, "
                  f"{self.successful_decodes} decoded ({d_rate:.1f}%), "
                  f"{self.below_threshold_count} below threshold ({t_rate:.1f}%)")

    # ------------------------------------------------------------------
    # Legacy / calibration
    # ------------------------------------------------------------------

    def detect_qr(self, frame):
        """Legacy method — kept for backward compatibility."""
        bbox, area, width, threshold_met, decoded_data = self.detect_and_decode_qr(frame)
        return bbox, area, width, threshold_met, (threshold_met and bbox is not None)

    def calibrate(self, frame, qr_size_cm):
        """Calibrate pixels-per-cm using a QR code of known physical size."""
        _, _, min_side, _, _ = self.detect_and_decode_qr(frame)
        if min_side > 0:
            self.pixels_per_cm = min_side / qr_size_cm
            print(f"[QR] Calibration successful: {self.pixels_per_cm:.2f} px/cm")
            return True
        print("[QR] Calibration failed: no QR detected")
        return False
