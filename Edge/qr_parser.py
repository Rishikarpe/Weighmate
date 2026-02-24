"""QR code data parsing and reel event publishing."""

from Edge.mqtt_handler import publish_mqtt
from config import MQTT_TOPIC_REEL_EVENT


# ============== QR DATA PARSER ==============
def parse_qr_data(raw_data):
    """Parse new QR code CSV format: ReelNumber,BF,GSM,Shade,Size,Weight

    Example: 'A00001,18,180,NATURAL,191,900'
    Returns dict with parsed fields, or None if format is invalid.
    """
    try:
        parts = raw_data.strip().split(',')
        if len(parts) != 6:
            print(f"[QR] Invalid QR format: expected 6 fields, got {len(parts)}")
            return None

        return {
            'reel_number': parts[0].strip(),
            'bf': int(parts[1].strip()),
            'gsm': int(parts[2].strip()),
            'shade': parts[3].strip(),
            'size': int(parts[4].strip()),
            'weight': int(parts[5].strip()),
        }
    except (ValueError, IndexError) as e:
        print(f"[QR] Error parsing QR data '{raw_data}': {e}")
        return None


def send_reel_event_to_mqtt(reel_data, location_x, location_y):
    """Send parsed reel data to MQTT topic event/{reelNumber}"""
    topic = MQTT_TOPIC_REEL_EVENT.format(reel_number=reel_data['reel_number'])
    payload = {
        'bf': reel_data['bf'],
        'gsm': reel_data['gsm'],
        'shade': reel_data['shade'],
        'size': reel_data['size'],
        'weight': reel_data['weight'],
        'x': location_x,
        'y': location_y,
    }
    print(f"[MQTT] Publishing reel event to '{topic}': {payload}")
    return publish_mqtt(topic, payload)
