"""MQTT client connection, callbacks, and publish functions."""

import json
import time
import ssl
import paho.mqtt.client as mqtt
from datetime import datetime

from Edge import state
from Edge.web_server import socketio
from config import (
    MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
    MQTT_CLIENT_ID, MQTT_KEEPALIVE,
    MQTT_TOPIC_POSITION, MQTT_TOPIC_EVENT, MQTT_TOPIC_MATERIALS,
    MQTT_TOPIC_SCALE_LIVE, MQTT_TOPIC_SCALE_STATE, MQTT_TOPIC_SCALE_STABLE,
    MQTT_TOPIC_SCALE_HEALTH, MQTT_TOPIC_SCALE_COMMAND, MQTT_TOPIC_SCALE_CONFIRMED,
)

# ============== MQTT CLIENT ==============
mqtt_client = None
mqtt_connected = False


def on_mqtt_connect(client, userdata, flags, reason_code, properties):
    """Callback when MQTT client connects to broker"""
    global mqtt_connected
    if reason_code == 0:
        mqtt_connected = True
        print(f"[MQTT] ✓ Connected to broker {MQTT_BROKER}:{MQTT_PORT}")
        # Subscribe to materials sync topic
        client.subscribe(MQTT_TOPIC_MATERIALS, qos=1)
        print(f"[MQTT] Subscribed to: {MQTT_TOPIC_MATERIALS}")
        # Subscribe to WeighMate (Vision Node) topics
        client.subscribe([
            (MQTT_TOPIC_SCALE_STABLE, 1),
            (MQTT_TOPIC_SCALE_STATE,  1),
            (MQTT_TOPIC_SCALE_LIVE,   0),
            (MQTT_TOPIC_SCALE_HEALTH, 0),
        ])
        print(f"[MQTT] Subscribed to scale topics (scale1)")
    else:
        mqtt_connected = False
        print(f"[MQTT] ✗ Connection failed with code: {reason_code}")


def on_mqtt_disconnect(client, userdata, flags, reason_code, properties):
    """Callback when MQTT client disconnects"""
    global mqtt_connected
    mqtt_connected = False
    print(f"[MQTT] Disconnected from broker (code: {reason_code})")


def on_mqtt_message(client, userdata, msg):
    """Callback when MQTT message is received"""
    payload_str = msg.payload.decode('utf-8').strip()

    # Route scale topics to dedicated handler
    if msg.topic in (MQTT_TOPIC_SCALE_LIVE, MQTT_TOPIC_SCALE_STATE,
                     MQTT_TOPIC_SCALE_STABLE, MQTT_TOPIC_SCALE_HEALTH):
        _handle_scale_message(msg.topic, payload_str)
        return

    # Materials sync (existing logic below)
    try:
        data = json.loads(payload_str)
        materials = data.get('materials', [])

        with state.materials_lock:
            central_material_ids = set()

            for material in materials:
                material_id = material.get('material_id')
                if material_id:
                    central_material_ids.add(material_id)
                    state.synced_materials[material_id] = {
                        'x': material.get('location_x', 0.0),
                        'y': material.get('location_y', 0.0),
                        'timestamp': material.get('timestamp', time.time()),
                        'datetime': material.get('datetime', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    }

            # Remove materials deleted on server side
            edge_material_ids = set(state.synced_materials.keys())
            deleted_materials = edge_material_ids - central_material_ids

            if deleted_materials:
                print(f"[MQTT] Removing {len(deleted_materials)} deleted materials")
                for material_id in deleted_materials:
                    del state.synced_materials[material_id]
                try:
                    socketio.emit('materials_deleted', {'material_ids': list(deleted_materials)})
                except:
                    pass

            # Broadcast to web dashboard
            try:
                socketio.emit('materials_sync', {
                    'materials': [{**{'material_id': k}, **v} for k, v in state.synced_materials.items()]
                })
            except:
                pass

        print(f"[MQTT] Synced {len(materials)} materials")
    except Exception as e:
        print(f"[MQTT] Error processing materials message: {e}")


def init_mqtt_client():
    """Initialize and connect the MQTT client with TLS"""
    global mqtt_client

    mqtt_client = mqtt.Client(
        client_id=MQTT_CLIENT_ID,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    # Configure TLS/SSL for EMQX Cloud
    mqtt_client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
    mqtt_client.tls_insecure_set(False)

    # Set callbacks
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_disconnect = on_mqtt_disconnect
    mqtt_client.on_message = on_mqtt_message

    # Enable automatic reconnection
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)

    print(f"[MQTT] Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
        mqtt_client.loop_start()  # Start background network loop
        return True
    except Exception as e:
        print(f"[MQTT] ✗ Failed to connect: {e}")
        return False


def publish_mqtt(topic, payload):
    """Publish a JSON payload to an MQTT topic"""
    global mqtt_client, mqtt_connected
    if mqtt_client is None or not mqtt_connected:
        print(f"[MQTT] ✗ Not connected, skipping publish")
        return False
    try:
        result = mqtt_client.publish(topic, json.dumps(payload), qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            return True
        else:
            print(f"[MQTT] ✗ Publish failed (rc={result.rc})")
            return False
    except Exception as e:
        print(f"[MQTT] ✗ Publish error: {e}")
        return False


def send_to_mqtt(forklift_id, location_x, location_y, material_id):
    """Send position data to MQTT broker"""
    topic = MQTT_TOPIC_POSITION.format(forklift_id=forklift_id)
    payload = {
        'mname': material_id,
        'x': location_x,
        'y': location_y,
        'timestamp': time.time()
    }
    return publish_mqtt(topic, payload)


def send_event_to_mqtt(forklift_id, event_type, material_id, location_x, location_y):
    """Send pickup/drop event to MQTT broker"""
    topic = MQTT_TOPIC_EVENT.format(forklift_id=forklift_id)
    payload = {
        'event_type': event_type,
        'mname': material_id,
        'x': location_x,
        'y': location_y,
        'timestamp': time.time(),
    }
    return publish_mqtt(topic, payload)


# ============== WEIGHMATE SCALE HANDLERS ==============

def _handle_scale_message(topic, payload_str):
    """Route and handle incoming messages from the Vision Node."""
    try:
        if topic == MQTT_TOPIC_SCALE_LIVE:
            weight = None if payload_str == 'null' else float(payload_str)
            with state.weight_lock:
                state.live_weight = weight
            try:
                socketio.emit('scale_live', {'weight': weight})
            except Exception:
                pass

        elif topic == MQTT_TOPIC_SCALE_STATE:
            with state.weight_lock:
                state.weighing_state = payload_str
                if payload_str == 'IDLE':
                    state.stable_weight = None
                    state.live_weight = None
            print(f"[SCALE] State → {payload_str}")
            try:
                socketio.emit('scale_state', {'state': payload_str})
            except Exception:
                pass

        elif topic == MQTT_TOPIC_SCALE_STABLE:
            weight = float(payload_str)
            with state.weight_lock:
                state.stable_weight = weight
            print(f"[SCALE] Stable weight: {weight} kg — awaiting operator confirmation")
            try:
                socketio.emit('scale_stable', {'weight': weight})
            except Exception:
                pass

        elif topic == MQTT_TOPIC_SCALE_HEALTH:
            status = json.loads(payload_str)
            if not status.get('ok'):
                print(f"[SCALE] Health warning: {status.get('issues', [])}")
                try:
                    socketio.emit('scale_health', status)
                except Exception:
                    pass

    except Exception as e:
        print(f"[SCALE] Error handling topic {topic}: {e}")


def send_weight_confirmation(weight):
    """
    Operator confirmed the weight on the Edge dashboard.
    Publishes to confirmed_weight topic for the external server to pick up.
    """
    from datetime import datetime as dt
    payload = {
        'weight_kg':    weight,
        'scale_id':     'scale1',
        'confirmed_at': dt.now().isoformat(),
    }
    result = publish_mqtt(MQTT_TOPIC_SCALE_CONFIRMED, payload)
    if result:
        print(f"[SCALE] ✓ Confirmed weight sent: {weight} kg")
    return result


def send_rescan_command():
    """Operator pressed RESCAN — tell Vision Node to reset the session."""
    global mqtt_client
    if mqtt_client:
        mqtt_client.publish(MQTT_TOPIC_SCALE_COMMAND, 'rescan', qos=1)
        print("[SCALE] Rescan command sent to Vision Node")
