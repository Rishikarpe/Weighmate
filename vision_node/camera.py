"""
camera.py

USB camera interface for the Vision Node.
Uses OpenCV VideoCapture with the index configured in config.py.

To find your camera index on Linux:
    ls /dev/video*          # usually /dev/video0 or /dev/video2
    v4l2-ctl --list-devices # more detail

Set CAMERA_INDEX in vision_node/config.py accordingly.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Optional

import cv2
import numpy as np

from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, CAPTURE_FPS

logger = logging.getLogger(__name__)

RETRY_DELAY  = 3.0           # seconds between open retries
WARMUP_SECS  = 1.0           # seconds to let the camera stabilise after open
WARMUP_READS = 10            # discard this many frames during warm-up


class Camera:
    """
    Wraps OpenCV VideoCapture for USB camera frame capture.

    Usage:
        camera = Camera()
        camera.open()

        for frame in camera.frames():
            process(frame)   # BGR numpy array, or None on read failure

        camera.close()
    """

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the USB camera. Retries indefinitely on failure."""
        while True:
            try:
                self._try_open()
                logger.info(
                    "USB camera opened (index=%d, %dx%d @ %dfps)",
                    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, CAPTURE_FPS,
                )
                return
            except Exception as exc:
                logger.error(
                    "Camera failed to open: %s. Retrying in %ds...", exc, RETRY_DELAY
                )
                time.sleep(RETRY_DELAY)

    def close(self) -> None:
        """Release the camera."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def frames(self) -> Iterator[Optional[np.ndarray]]:
        """
        Yield BGR frames at ~CAPTURE_FPS.
        Yields None on a bad read so the main loop stays alive.
        Pacing ensures we never spin faster than the configured FPS,
        even when reads fail.
        """
        interval = 1.0 / CAPTURE_FPS
        while True:
            t0 = time.monotonic()
            yield self._read_frame()
            elapsed = time.monotonic() - t0
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _try_open(self) -> None:
        # If CAMERA_INDEX is set to a specific value (not AUTO_SCAN sentinel),
        # try that index first; otherwise scan 0-5 for the first working camera.
        indices_to_try = (
            list(range(6)) if CAMERA_INDEX < 0 else [CAMERA_INDEX]
        )

        for idx in indices_to_try:
            cap = self._open_index(idx)
            if cap is not None:
                self._cap = cap
                return

        raise RuntimeError(
            "No working USB camera found. "
            "Run 'v4l2-ctl --list-devices' and set CAMERA_INDEX in config.py."
        )

    def _open_index(self, idx: int) -> "Optional[cv2.VideoCapture]":
        """
        Try to open camera at *idx* and verify it actually delivers a frame.
        Tries FRAME_WIDTH×FRAME_HEIGHT first, falls back to 640×480.
        Returns a ready VideoCapture on success, None on failure.
        """
        for w, h in [(FRAME_WIDTH, FRAME_HEIGHT), (640, 480)]:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                return None

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS,          CAPTURE_FPS)

            time.sleep(WARMUP_SECS)
            for _ in range(WARMUP_READS):
                cap.grab()

            ok, frame = cap.read()
            if ok and frame is not None:
                logger.info(
                    "Camera found: index=%d, resolution=%dx%d", idx, w, h
                )
                return cap

            cap.release()
            logger.debug("index=%d %dx%d: no frames", idx, w, h)

        return None

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        try:
            ok, frame = self._cap.read()
            return frame if ok else None
        except Exception as exc:
            logger.warning("Frame capture error: %s", exc)
            return None
