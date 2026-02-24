"""
stream_server.py

MJPEG live stream server for the Vision Node.
Runs as a background thread on a configurable port.

The HMI RPi embeds this stream in the browser UI:
    <img src="http://<vision-rpi-ip>:8080/stream">

Frame rate is intentionally low (5 fps) to keep network usage minimal
on the factory WiFi while still giving the operator a usable live view.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

STREAM_PORT    = 8080
STREAM_FPS     = 5        # frames per second sent to HMI
STREAM_QUALITY = 70       # JPEG quality (0-100), lower = less bandwidth


class StreamServer:
    """
    MJPEG stream server.

    Usage:
        server = StreamServer()
        server.start()

        # In the main loop, push new frames:
        server.push_frame(frame)
    """

    def __init__(self) -> None:
        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._app = Flask(__name__)
        self._app.add_url_rule("/stream", "stream", self._stream_route)
        self._app.add_url_rule("/", "index", self._index_route)

    def start(self) -> None:
        """Start the Flask server in a background daemon thread."""
        thread = threading.Thread(
            target=self._run_flask,
            daemon=True,
            name="StreamServer",
        )
        thread.start()
        logger.info("Stream server started on port %d", STREAM_PORT)

    def push_frame(self, frame: np.ndarray, overlay: Optional[str] = None) -> None:
        """
        Update the frame being served.

        Args:
            frame:   BGR numpy array from OpenCV.
            overlay: Optional text to draw on the frame (e.g., "75.5 kg | STABILIZING").
                     Drawn in top-left corner for operator reference.
        """
        display = frame.copy()

        if overlay:
            cv2.putText(
                display,
                overlay,
                org=(10, 35),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY]
        ok, buffer = cv2.imencode(".jpg", display, encode_params)
        if not ok:
            return

        with self._lock:
            self._frame = buffer.tobytes()

    # ─── Flask routes ─────────────────────────────────────────────────────────

    def _stream_route(self) -> Response:
        return Response(
            self._generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _index_route(self) -> str:
        return (
            "<html><body style='background:#000;margin:0'>"
            f"<img src='/stream' style='width:100%;height:100vh;object-fit:contain'>"
            "</body></html>"
        )

    def _generate_frames(self):
        interval = 1.0 / STREAM_FPS
        while True:
            with self._lock:
                frame_bytes = self._frame

            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )

            time.sleep(interval)

    def _run_flask(self) -> None:
        # Silence Flask startup banner
        import os
        os.environ["WERKZEUG_RUN_MAIN"] = "true"
        self._app.run(
            host="0.0.0.0",
            port=STREAM_PORT,
            threaded=True,
            use_reloader=False,
        )
