"""
main.py — Vision Node entry point.

Architecture:
    camera → weight_detector → session_manager (via StabilityFilter)
           → mqtt_client + stream_server + health_monitor + snapshot_logger

Main loop:   ~10 fps  (CAPTURE_FPS)
TFLite:      every TFLITE_FRAME_SKIP frames  (~3 fps)
Health pub:  every HEALTH_PUBLISH_INTERVAL seconds

Run:
    python main.py

On Raspberry Pi, install as a systemd service:
    sudo cp deploy/weighmate-vision.service /etc/systemd/system/
    sudo systemctl enable weighmate-vision
    sudo systemctl start weighmate-vision
"""

from __future__ import annotations

import datetime
import logging
import os
import signal
import time
from typing import Optional

import cv2
import numpy as np

import snapshot_logger
from camera import Camera
from config import (
    SNAPSHOT_DIR,
    HEALTH_PUBLISH_INTERVAL,
    TFLITE_FRAME_SKIP,
    ROI_X, ROI_Y, ROI_W, ROI_H,
)
from health_monitor import HealthMonitor
from mqtt_client import VisionMQTTClient
from session_manager import SessionManager, State
from stream_server import StreamServer
from weight_detector import extract_weight

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def main() -> None:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # ── Components ────────────────────────────────────────────────────────────
    mqtt   = VisionMQTTClient()
    stream = StreamServer()
    health = HealthMonitor()
    camera = Camera()

    # Mutable cell so the on_confirmed closure always has the latest frame.
    last_frame: list[Optional[np.ndarray]] = [None]

    session = SessionManager(
        on_confirmed=lambda w: _on_confirmed(w, last_frame[0], mqtt),
        on_error=lambda reason: _on_error(reason, mqtt),
        on_state_change=lambda s: _on_state_change(s, mqtt, health),
    )
    # Wire rescan after session is constructed
    mqtt.set_rescan_callback(session.rescan)

    # ── Start services ────────────────────────────────────────────────────────
    mqtt.connect()
    stream.start()
    camera.open()
    logger.info("WeighMate Vision Node started")

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    running = True

    def _shutdown(sig, _frame):
        nonlocal running
        logger.info("Shutdown signal received (%s)", sig)
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_counter       = 0
    last_health_publish = 0.0
    last_debug_save     = 0.0
    cached_weight: Optional[float] = None   # last valid TFLite result

    for frame in camera.frames():
        if not running:
            break
        if frame is None:
            logger.warning("Dropped frame — camera read failed")
            continue

        frame_counter += 1
        last_frame[0] = frame

        # 1. TFLite on every Nth frame; reuse cache on skipped frames so the
        #    state machine and MQTT feed stay live at full loop rate.
        if frame_counter % TFLITE_FRAME_SKIP == 0:
            cached_weight = extract_weight(frame)
        weight          = cached_weight
        weight_detected = weight is not None

        # 2. Feed state machine
        session.update(weight)

        # 3. Publish live weight every frame
        mqtt.publish_live_weight(weight)

        # 4. Build stream frame: ROI box + overlay text
        display = frame.copy()
        cv2.rectangle(
            display,
            (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H),
            (0, 255, 0), 2,
        )
        cv2.putText(display, "ROI", (ROI_X + 4, ROI_Y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        weight_label = f"{weight:.1f} kg" if weight is not None else "---"
        stream.push_frame(display, overlay=f"{weight_label}  |  {session.state.name}")

        now = time.monotonic()

        # 5. Dump preprocessed ROI to /tmp every 30 s for hardware verification
        if now - last_debug_save >= 30.0:
            try:
                roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
                cv2.imwrite("/tmp/debug_roi.jpg", roi)
            except Exception:
                pass
            last_debug_save = now

        # 6. Periodic health publish
        if now - last_health_publish >= HEALTH_PUBLISH_INTERVAL:
            session_active = session.state in (State.STABILIZING, State.CONFIRMED)
            status = health.check(frame, session_active, weight_detected)
            mqtt.publish_health(status.to_dict())
            if not status.ok:
                logger.warning("Health issues: %s", status.issues)
            last_health_publish = now

    # ── Cleanup ───────────────────────────────────────────────────────────────
    camera.close()
    mqtt.disconnect()
    logger.info("WeighMate Vision Node stopped")


# ─── Callbacks ────────────────────────────────────────────────────────────────

def _on_confirmed(
    weight: float,
    frame: Optional[np.ndarray],
    mqtt: VisionMQTTClient,
) -> None:
    logger.info("Weight CONFIRMED: %.1f kg", weight)
    mqtt.publish_stable_weight(weight)

    path = snapshot_logger.save(frame, weight)
    if path:
        mqtt.publish_snapshot(
            path,
            timestamp=datetime.datetime.now().isoformat(),
            weight=weight,
        )


def _on_error(reason: str, mqtt: VisionMQTTClient) -> None:
    logger.warning("Session ERROR: %s", reason)
    mqtt.publish_state(f"ERROR:{reason}")


def _on_state_change(
    state: State,
    mqtt: VisionMQTTClient,
    health: HealthMonitor,
) -> None:
    logger.info("State → %s", state.name)
    mqtt.publish_state(state.name)
    if state == State.IDLE:
        health.reset()


if __name__ == "__main__":
    main()
