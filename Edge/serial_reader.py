"""Serial port reader for ESP32 UWB distance data."""

import serial
import json
import time

from Edge import state
from Edge.geometry import trilaterate
from config import SERIAL_PORT, SERIAL_BAUDRATE, SERIAL_TIMEOUT, ANCHORS


# ============== SERIAL PORT READER ==============
def serial_reader_thread(forklift_id):
    """Read UWB distance data from ESP32 via serial port"""

    print(f"\n[SERIAL] Starting serial reader for {SERIAL_PORT} at {SERIAL_BAUDRATE} baud")

    while state.serial_active:
        try:
            # Open serial port
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)
            print(f"[SERIAL] ✓ Connected to {SERIAL_PORT}")

            while state.serial_active:
                try:
                    # Read line from serial port
                    line = ser.readline().decode('utf-8').strip()

                    if not line:
                        continue

                    # Strip JSON: or $JSON: prefix if present
                    if line.startswith('$JSON:'):
                        line = line[6:]  # Remove '$JSON:' prefix
                    elif line.startswith('JSON:'):
                        line = line[5:]  # Remove 'JSON:' prefix

                    # Parse JSON data from ESP32
                    try:
                        msg = json.loads(line)

                        # Extract distances
                        dist = msg.get('dist', {})

                        # Update global distances for web dashboard
                        state.current_distances = {
                            'A': dist.get('A'),
                            'B': dist.get('B'),
                            'C': dist.get('C'),
                            'D': dist.get('D'),
                            'E': dist.get('E'),
                            'F': dist.get('F'),
                            'G': dist.get('G')
                        }

                        print(f"\n[ESP32] Seq:{msg.get('seq', '?')}")

                        for anchor in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                            d = dist.get(anchor)
                            if d is not None:
                                print(f"  {anchor}: {d:.2f}m")
                            else:
                                print(f"  {anchor}: --")

                        # Calculate position (weighted LS + outlier rejection + NLOS)
                        raw_pos = trilaterate(dist)
                        if raw_pos:
                            # Kalman filter smoothing
                            filt_x, filt_y = state.position_filter.update(raw_pos['x'], raw_pos['y'])
                            pos = {'x': round(filt_x, 3), 'y': round(filt_y, 3)}
                            print(f"  => Raw: ({raw_pos['x']:.2f}, {raw_pos['y']:.2f}) | Filtered: ({pos['x']:.2f}, {pos['y']:.2f})")
                            state.current_tag_position = pos
                        else:
                            print(f"  => Position: Cannot calculate")

                    except json.JSONDecodeError:
                        print(f"[SERIAL] Invalid JSON: {line[:50]}")

                except serial.SerialException as e:
                    print(f"[SERIAL] Error reading: {e}")
                    break
                except Exception as e:
                    print(f"[SERIAL] Unexpected error: {e}")
                    break

            ser.close()

        except serial.SerialException as e:
            print(f"[SERIAL] ✗ Cannot open {SERIAL_PORT}: {e}")
            print(f"[SERIAL] Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"[SERIAL] ✗ Error: {e}")
            time.sleep(5)

    print("[SERIAL] Serial reader thread stopped")
