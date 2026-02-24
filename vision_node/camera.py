"""
camera.py

CSI camera interface for Raspberry Pi.
Provides a simple frame iterator used by the main loop.

Uses Picamera2 (Raspberry Pi official library for CSI cameras).
Falls back to OpenCV VideoCapture for development on non-Pi hardware.

Install on Raspberry Pi:
    sudo apt install python3-picamera2
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

# Resolution of captured frames.
# Higher = better digit detail, but heavier SSOCR processing.
# 1280x720 is a good balance for this use case.
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# Target capture frame rate. 10fps is more than enough for weighing.
CAPTURE_FPS = 10

# Seconds to wait between retries if camera fails to open.
RETRY_DELAY = 3.0


class Camera:
    """
    Wraps Picamera2 (or OpenCV fallback) for frame capture.

    Usage:
        camera = Camera()
        camera.open()

        for frame in camera.frames():
            process(frame)   # BGR numpy array

        camera.close()
    """

    def __init__(self) -> None:
        self._cap = None
        self._use_picamera2 = False

    def open(self) -> None:
        """Open the camera. Retries indefinitely on failure."""
        while True:
            try:
                self._try_open()
                logger.info(
                    "Camera opened (%dx%d @ %dfps)",
                    FRAME_WIDTH, FRAME_HEIGHT, CAPTURE_FPS,
                )
                return
            except Exception as exc:
                logger.error("Camera failed to open: %s. Retrying in %ds...", exc, RETRY_DELAY)
                time.sleep(RETRY_DELAY)

    def close(self) -> None:
        """Release the camera."""
        if self._cap is None:
            return
        try:
            if self._use_picamera2:
                self._cap.stop()
            else:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    def frames(self) -> Iterator[np.ndarray]:
        """
        Yield BGR frames continuously.
        On capture failure, logs a warning and yields None to keep the main loop alive.
        """
        while True:
            frame = self._read_frame()
            yield frame

    def _try_open(self) -> None:
        """Attempt to open Picamera2, fall back to OpenCV if not on Pi."""
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            config = cam.create_video_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"},
                controls={"FrameRate": CAPTURE_FPS},
            )
            cam.configure(config)
            cam.start()
            self._cap = cam
            self._use_picamera2 = True
            logger.info("Using Picamera2 (CSI camera)")
        except ImportError:
            logger.warning("Picamera2 not available. Falling back to OpenCV VideoCapture.")
            self._open_opencv()

    def _open_opencv(self) -> None:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("OpenCV: no camera found at index 0")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        self._cap = cap
        self._use_picamera2 = False

    def _read_frame(self) -> Optional[np.ndarray]:
        try:
            if self._use_picamera2:
                return self._cap.capture_array("main")
            else:
                ok, frame = self._cap.read()
                return frame if ok else None
        except Exception as exc:
            logger.warning("Frame capture error: %s", exc)
            return None
