"""Periodic data logger — sends position and material data via MQTT."""

import time
from datetime import datetime

from Edge import state
from Edge.mqtt_handler import send_to_mqtt
from Edge.web_server import socketio


# ============== PERIODIC DATA LOGGER ==============
def periodic_logger(forklift_id, operator_name):
    """Send position and material data via MQTT every second"""

    print(f"\n[LOGGER] Starting periodic logger for forklift {forklift_id}")
    print(f"[LOGGER] Sending data via MQTT to broker\n")

    log_counter = 0
    start_time = time.time()

    while state.serial_active:
        log_counter += 1
        elapsed = time.time() - start_time

        # Get current position and material
        x = state.current_tag_position['x'] if state.current_tag_position else 0.0
        y = state.current_tag_position['y'] if state.current_tag_position else 0.0
        material = state.current_material_id if state.current_material_id else None

        # Broadcast to web dashboard FIRST (don't wait for HTTP request)
        try:
            socketio.emit('update', {
                'forklift_id': forklift_id,
                'operator_name': operator_name,
                'location_x': x,
                'location_y': y,
                'material_id': material,
                'distances': state.current_distances,
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        except:
            pass  # Ignore if no clients connected

        # Send to MQTT broker (in background, don't block dashboard)
        success = send_to_mqtt(
            forklift_id=forklift_id,
            location_x=x,
            location_y=y,
            material_id=material
        )

        # Print status
        status_symbol = "✓" if success else "✗"
        status_msg = f"[LOG {log_counter}] {status_symbol} [{elapsed:.1f}s] Pos: ({x:.2f}, {y:.2f}) | Material: {material if material else 'NULL'}"
        print(status_msg)

        time.sleep(0.5)  # Wait 1 second before next update
