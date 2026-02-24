"""
config.py — Vision Node configuration loader.

All parameters live in config.yaml.
This module loads them and exposes module-level constants so every other
module can do  `from config import X`  as before.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def _load() -> dict[str, Any]:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


_cfg = _load()

# ── Camera ─────────────────────────────────────────────────────────────────────
CAMERA_INDEX = _cfg["camera"]["index"]
FRAME_WIDTH  = _cfg["camera"]["resolution"][0]
FRAME_HEIGHT = _cfg["camera"]["resolution"][1]
CAPTURE_FPS  = _cfg["camera"]["fps"]

# ── ROI ────────────────────────────────────────────────────────────────────────
ROI_X = _cfg["roi"]["x"]
ROI_Y = _cfg["roi"]["y"]
ROI_W = _cfg["roi"]["w"]
ROI_H = _cfg["roi"]["h"]

# ── SSOCR ──────────────────────────────────────────────────────────────────────
SSOCR_THRESHOLD  = _cfg["ssocr"]["threshold"]
SSOCR_MIN_DIGITS = _cfg["ssocr"]["min_digits"]
SSOCR_MAX_DIGITS = _cfg["ssocr"]["max_digits"]
SSOCR_FRAME_SKIP = _cfg["ssocr"]["frame_skip"]
SSOCR_BACKGROUND = _cfg["ssocr"]["background"]

# ── Weight validation ──────────────────────────────────────────────────────────
WEIGHT_MIN_KG = _cfg["weight"]["min_kg"]
WEIGHT_MAX_KG = _cfg["weight"]["max_kg"]

# ── Stability filter ───────────────────────────────────────────────────────────
STABILITY_BUFFER_SIZE     = _cfg["stability"]["buffer_size"]
STABILITY_DURATION_SEC    = _cfg["stability"]["stable_duration_sec"]
STABILITY_VARIANCE_THRESH = _cfg["stability"]["variance_threshold"]
STABILITY_MIN_WEIGHT_KG   = _cfg["stability"]["min_weight_kg"]

# ── Session state machine ──────────────────────────────────────────────────────
IDLE_THRESHOLD_KG   = _cfg["session"]["idle_threshold_kg"]
DETECT_THRESHOLD_KG = _cfg["session"]["detect_threshold_kg"]
DETECT_HOLD_SEC     = _cfg["session"]["detect_hold_sec"]

# ── Health monitor ─────────────────────────────────────────────────────────────
BLUR_THRESHOLD              = _cfg["health"]["blur_threshold"]
BRIGHTNESS_MIN              = _cfg["health"]["brightness_min"]
BRIGHTNESS_MAX              = _cfg["health"]["brightness_max"]
OBSTRUCTION_TIMEOUT_SECONDS = _cfg["health"]["watchdog_timeout_sec"]
HEALTH_PUBLISH_INTERVAL     = _cfg["health"]["publish_interval_sec"]

# ── MQTT ───────────────────────────────────────────────────────────────────────
MQTT_BROKER    = _cfg["mqtt"]["broker"]
MQTT_PORT      = _cfg["mqtt"]["port"]
MQTT_USERNAME  = _cfg["mqtt"]["username"]
MQTT_PASSWORD  = _cfg["mqtt"]["password"]
MQTT_CLIENT_ID = _cfg["mqtt"]["client_id"]
MQTT_KEEPALIVE = _cfg["mqtt"]["keepalive"]
MQTT_TLS       = _cfg["mqtt"]["tls"]
SCALE_ID       = _cfg["mqtt"]["scale_id"]

_prefix = f"factory/{SCALE_ID}"
MQTT_TOPIC_LIVE_WEIGHT   = f"{_prefix}/live_weight"
MQTT_TOPIC_SESSION_STATE = f"{_prefix}/session_state"
MQTT_TOPIC_STABLE_WEIGHT = f"{_prefix}/stable_weight"
MQTT_TOPIC_HEALTH        = f"{_prefix}/health"
MQTT_TOPIC_SNAPSHOT      = f"{_prefix}/snapshot"
MQTT_TOPIC_COMMAND       = f"{_prefix}/command"

# ── Stream server ──────────────────────────────────────────────────────────────
STREAM_PORT    = _cfg["stream"]["port"]
STREAM_FPS     = _cfg["stream"]["fps"]
STREAM_QUALITY = _cfg["stream"]["quality"]

# ── Snapshots ──────────────────────────────────────────────────────────────────
SNAPSHOT_DIR = os.path.expanduser(_cfg["snapshots"]["dir"])
SNAPSHOT_CSV = os.path.expanduser(_cfg["snapshots"]["csv_path"])
