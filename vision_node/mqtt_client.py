"""
mqtt_client.py — Vision Node

Publishes scale data to EMQX Cloud broker (same as Edge).
Subscribes to command topic for rescan/ack from Edge.

Topics published:
    factory/scale1/live_weight       float kg, every frame
    factory/scale1/session_state     string state name, on change
    factory/scale1/stable_weight     float kg, once per session
    factory/scale1/health            JSON health status, periodic
    factory/scale1/snapshot          JSON snapshot metadata

Topics subscribed:
    factory/scale1/command           "rescan" from Edge operator
"""

from __future__ import annotations

import json
import logging
import ssl
from typing import Callable, Optional

import paho.mqtt.client as mqtt

from config import (
    MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
    MQTT_CLIENT_ID, MQTT_KEEPALIVE,
    MQTT_TOPIC_LIVE_WEIGHT, MQTT_TOPIC_SESSION_STATE,
    MQTT_TOPIC_STABLE_WEIGHT, MQTT_TOPIC_HEALTH,
    MQTT_TOPIC_SNAPSHOT, MQTT_TOPIC_COMMAND,
)

logger = logging.getLogger(__name__)


class VisionMQTTClient:
    """
    MQTT client for the Vision Node.

    Usage:
        client = VisionMQTTClient(on_rescan=session_manager.rescan)
        client.connect()
        client.publish_live_weight(75.5)
    """

    def __init__(self, on_rescan: Optional[Callable[[], None]] = None) -> None:
        self._on_rescan: Optional[Callable[[], None]] = on_rescan
        self._client = mqtt.Client(
            client_id=MQTT_CLIENT_ID,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self._client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

        # TLS — same setup as Edge
        self._client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
        self._client.tls_insecure_set(False)

        self._client.on_connect    = self._on_connect
        self._client.on_message    = self._on_message
        self._client.on_disconnect = self._on_disconnect

        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

    def set_rescan_callback(self, callback: Callable[[], None]) -> None:
        """Wire the rescan callback after construction (avoids circular init)."""
        self._on_rescan = callback

    # ─── Connection ───────────────────────────────────────────────────────────

    def connect(self) -> None:
        logger.info("MQTT connecting to %s:%d", MQTT_BROKER, MQTT_PORT)
        try:
            self._client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
            self._client.loop_start()
        except Exception as exc:
            logger.error("MQTT connection failed: %s", exc)

    def disconnect(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    # ─── Publishers ───────────────────────────────────────────────────────────

    def publish_live_weight(self, weight: Optional[float]) -> None:
        payload = str(weight) if weight is not None else "null"
        self._publish(MQTT_TOPIC_LIVE_WEIGHT, payload, qos=0)

    def publish_state(self, state_name: str) -> None:
        self._publish(MQTT_TOPIC_SESSION_STATE, state_name, qos=1)

    def publish_stable_weight(self, weight: float) -> None:
        self._publish(MQTT_TOPIC_STABLE_WEIGHT, str(weight), qos=1, retain=True)
        logger.info("Stable weight published: %.1f kg", weight)

    def publish_health(self, status: dict) -> None:
        self._publish(MQTT_TOPIC_HEALTH, json.dumps(status), qos=0)

    def publish_snapshot(self, snapshot_path: str, timestamp: str, weight: float) -> None:
        payload = json.dumps({
            "path":      snapshot_path,
            "timestamp": timestamp,
            "weight_kg": weight,
        })
        self._publish(MQTT_TOPIC_SNAPSHOT, payload, qos=1)

    # ─── Internal callbacks ───────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code == 0:
            logger.info("MQTT connected to %s", MQTT_BROKER)
            client.subscribe(MQTT_TOPIC_COMMAND, qos=1)
        else:
            logger.error("MQTT connection failed, code=%s", reason_code)

    def _on_message(self, client, userdata, msg) -> None:
        payload = msg.payload.decode("utf-8").strip().lower()
        if payload == "rescan" and self._on_rescan:
            logger.info("Rescan command received from Edge")
            self._on_rescan()

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code != 0:
            logger.warning("MQTT disconnected (code=%s). Auto-reconnecting...", reason_code)

    def _publish(
        self,
        topic: str,
        payload: str,
        qos: int = 0,
        retain: bool = False,
    ) -> None:
        self._client.publish(topic, payload, qos=qos, retain=retain)
