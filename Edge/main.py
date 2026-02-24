#!/usr/bin/env python3
"""
Edge UWB Tag Tracking Server for Raspberry Pi
- Reads location data from ESP32 via Serial Port (/dev/ttyUSB0 at 115200 baud)
- Detects QR codes using USB webcam
- Sends calculated position and QR data to Central server database
"""

import sys
import os
import time
import threading

# Add parent directory (FINAL) to path for config and qrdetect imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from Edge import state
from Edge.web_server import app, socketio
from Edge.mqtt_handler import init_mqtt_client, mqtt_client
from Edge.serial_reader import serial_reader_thread
from Edge.logger import periodic_logger
from Edge.qr_detection import qr_detection_loop
from config import (
    DEFAULT_FORKLIFT, DEFAULT_OPERATOR,
    SERIAL_PORT, SERIAL_BAUDRATE,
    MQTT_BROKER, MQTT_PORT,
    CAMERA_INDEX, ANCHORS, MAP_LENGTH, MAP_BREADTH
)


# ============== MAIN ==============
def main():
    print("\n" + "="*60)
    print("  Edge UWB Tag Tracking + QR Detection (Raspberry Pi)")
    print("="*60)

    # Use default values from config
    forklift_id = DEFAULT_FORKLIFT
    operator_name = DEFAULT_OPERATOR

    print(f"\n✅ Forklift ID: {forklift_id}")
    print(f"✅ Operator: {operator_name}")
    print("="*60)
    print(f"\n[CONFIG] Serial Port: {SERIAL_PORT} @ {SERIAL_BAUDRATE} baud")
    print(f"[CONFIG] MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"[CONFIG] USB Camera: Index {CAMERA_INDEX}")
    print(f"[CONFIG] Web Dashboard: http://0.0.0.0:8080")
    print(f"[CONFIG] ANCHORS: {ANCHORS}")
    print(f"[CONFIG] MAP: {MAP_LENGTH}m x {MAP_BREADTH}m")
    print("\n" + "="*60)
    print("\nStarting Edge server...")
    print("- Serial reader will read ESP32 location data")
    print("- QR detection will run with USB webcam (instant updates)")
    print("- Combined data sent to MQTT broker every second")
    print("- Materials synced via MQTT subscription")
    print("- Real-time web dashboard with live position tracking")
    print("\nPress Ctrl+C to stop\n")

    state.serial_active = True
    state.qr_detection_active = True

    # Initialize MQTT connection
    if not init_mqtt_client():
        print("[MQTT] WARNING: MQTT not connected, will retry automatically")

    # Start serial reader thread
    serial_thread = threading.Thread(target=serial_reader_thread, args=(forklift_id,), daemon=True)
    serial_thread.start()

    # Start periodic logger thread (sends to MQTT broker every second)
    logger_thread = threading.Thread(target=periodic_logger, args=(forklift_id, operator_name), daemon=True)
    logger_thread.start()

    # Start QR detection in separate thread
    qr_thread = threading.Thread(target=qr_detection_loop, args=(forklift_id,), daemon=True)
    qr_thread.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Stopping all threads...")
        state.serial_active = False
        state.qr_detection_active = False
        from Edge.mqtt_handler import mqtt_client
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            print("[MQTT] Disconnected")
        time.sleep(2)
        print("[SHUTDOWN] Edge server stopped")


if __name__ == '__main__':
    # Load scanned reels from JSON file
    state.load_scanned_reels()

    # Start Flask web server in separate thread
    print("\n[WEB] Starting web dashboard on http://0.0.0.0:8080")
    print(f"[WEB] Access from browser: http://localhost:8080")

    def start_flask():
        try:
            socketio.run(app, host='0.0.0.0', port=8080, debug=False, use_reloader=False, allow_unsafe_werkzeug=True, log_output=False)
        except Exception as e:
            print(f"[WEB] Flask error: {e}")

    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    time.sleep(2)  # Give Flask time to start
    print("[WEB] ✓ Dashboard ready\n")

    main()
