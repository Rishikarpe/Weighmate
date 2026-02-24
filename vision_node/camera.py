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
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"No USB camera found at index {CAMERA_INDEX}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAPTURE_FPS)

        # Warm-up: give the sensor time to stabilise and drain initial frames.
        logger.info("Camera warm-up (%ds)...", WARMUP_SECS)
        time.sleep(WARMUP_SECS)
        for _ in range(WARMUP_READS):
            cap.grab()

        # Test read — verify the camera actually delivers frames.
        # If it fails here, raise so open() retries with a different config.
        ok, test_frame = cap.read()
        if not ok or test_frame is None:
            cap.release()
            # Many cheap USB cameras don't support 1280x720 in raw (YUYV) mode.
            # Fall back to 640x480 which is universally supported.
            logger.warning(
                "Camera at index %d could not read %dx%d. "
                "Retrying at 640x480...",
                CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
            )
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                raise RuntimeError(f"Camera index {CAMERA_INDEX} failed to reopen")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS,          CAPTURE_FPS)
            time.sleep(WARMUP_SECS)
            for _ in range(WARMUP_READS):
                cap.grab()
            ok, test_frame = cap.read()
            if not ok or test_frame is None:
                cap.release()
                raise RuntimeError(
                    f"Camera index {CAMERA_INDEX} opened but cannot deliver frames. "
                    "Check 'v4l2-ctl --list-devices' and set CAMERA_INDEX correctly."
                )
            logger.info("Fallback resolution 640x480 is working.")

        self._cap = cap

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        try:
            ok, frame = self._cap.read()
            return frame if ok else None
        except Exception as exc:
            logger.warning("Frame capture error: %s", exc)
            return None
