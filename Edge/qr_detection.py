"""QR code detection loop using USB webcam."""

import cv2
import time
from datetime import datetime

from Edge import state
from Edge.web_server import socketio
from Edge.qr_parser import parse_qr_data, send_reel_event_to_mqtt
from Edge.mqtt_handler import send_event_to_mqtt
from qrdetect import QRSizeDetector
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FRAME_SKIP,
    SIZE_THRESHOLD_AREA_PX, MIN_WIDTH_PX, MIN_HEIGHT_PX
)


# ============== QR DETECTION LOOP ==============
def qr_detection_loop(forklift_id):
    """Run QR detection in separate thread"""

    print(f"\n[QR] Starting QR detection thread for forklift {forklift_id}")
    print(f"[QR] Camera resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"[QR] QR threshold area: {SIZE_THRESHOLD_AREA_PX} px, min width: {MIN_WIDTH_PX} px, min height: {MIN_HEIGHT_PX} px")


    cap = cv2.VideoCapture(CAMERA_INDEX)
        # RPi-optimized resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    frame_counter = 0
    detector = QRSizeDetector()

    while state.qr_detection_active:
        try:
            display_frame = None
            ret, frame = cap.read()
            if not ret:
                print("[QR] Failed to read frame from camera")
            else:
                frame_counter += 1
                # Frame skipping for performance (process every Nth frame)
                if frame_counter % FRAME_SKIP != 0:
                    continue
                # QR detection and decoding
                try:
                    bbox, area, _, threshold_met, decoded_data = detector.detect_and_decode_qr(frame)
                except Exception as qr_exc:
                    print(f"[QR] Error in detect_and_decode_qr: {qr_exc}")
                    bbox, area, _, threshold_met, decoded_data = None, 0, 0, False, None
                # QR logic
                if decoded_data and threshold_met:
                    state.frames_without_qr = 0
                    if state.current_tag_position:
                        state.last_qr_seen_position = {'x': state.current_tag_position['x'], 'y': state.current_tag_position['y']}
                    else:
                        state.last_qr_seen_position = {'x': 0.0, 'y': 0.0}

                    # Parse new QR CSV format: ReelNumber,BF,GSM,Shade,Size,Weight
                    reel_data = parse_qr_data(decoded_data)
                    reel_id = reel_data['reel_number'] if reel_data else decoded_data

                    if reel_id != state.current_material_id:
                        print(f"\n[QR] 🎯 Material DETECTED: {decoded_data}")
                        print(f"[QR] Position saved: X={state.last_qr_seen_position['x']:.2f}m, Y={state.last_qr_seen_position['y']:.2f}m")
                        state.current_material_id = reel_id

                        # Send reel event to event/{reelNumber} topic
                        if reel_data:
                            print(f"[QR] Parsed reel: {reel_data}")
                            send_reel_event_to_mqtt(reel_data,
                                                    state.last_qr_seen_position['x'],
                                                    state.last_qr_seen_position['y'])

                        # Also send legacy pickup event
                        print(f"[QR] Sending pickup event via MQTT")
                        send_event_to_mqtt(forklift_id, 'pickup', reel_id,
                                           state.last_qr_seen_position['x'], state.last_qr_seen_position['y'])
                        try:
                            socketio.emit('qr_pickup', {
                                'material_id': reel_id,
                                'location_x': state.last_qr_seen_position['x'],
                                'location_y': state.last_qr_seen_position['y'],
                                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                        except Exception as emit_exc:
                            print(f"[QR] Error emitting qr_pickup: {emit_exc}")
                else:
                    if state.current_material_id:
                        state.frames_without_qr += 1
                        print(f"[QR] ⏳ QR not detected ({state.frames_without_qr}/{state.QR_DROP_THRESHOLD})...")
                        if state.frames_without_qr >= state.QR_DROP_THRESHOLD:
                            print(f"\n[QR] 📤 Material DROPPED: {state.current_material_id}")
                            print(f"[QR] Not detected for {state.frames_without_qr} consecutive processed frames")
                            if state.last_qr_seen_position:
                                print(f"[QR] Drop location (last seen): X={state.last_qr_seen_position['x']:.2f}m, Y={state.last_qr_seen_position['y']:.2f}m")
                            else:
                                print(f"[QR] Drop location: Not available (using default)")
                            print(f"[QR] Sending drop event via MQTT")
                            prev_material = state.current_material_id
                            drop_position = state.last_qr_seen_position if state.last_qr_seen_position else {'x': 0.0, 'y': 0.0}
                            state.current_material_id = None
                            state.frames_without_qr = 0
                            state.scanned_reels[prev_material] = {
                                'material_id': prev_material,
                                'x': drop_position['x'],
                                'y': drop_position['y'],
                                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'timestamp': time.time()
                            }
                            state.save_scanned_reels()
                            send_event_to_mqtt(forklift_id, 'drop', prev_material,
                                               drop_position['x'], drop_position['y'])
                            try:
                                socketio.emit('qr_drop', {
                                    'material_id': prev_material,
                                    'location_x': drop_position['x'],
                                    'location_y': drop_position['y'],
                                    'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                })
                            except Exception as emit_exc:
                                print(f"[QR] Error emitting qr_drop: {emit_exc}")
                    else:
                        if state.frames_without_qr != 0:
                            state.frames_without_qr = 0
                # Display frame with overlay
                display_frame = frame.copy()
                # Only draw bounding box if area is above threshold — this suppresses
                # tiny false-positive detections (noise / nearby high-contrast objects)
                if bbox and area >= SIZE_THRESHOLD_AREA_PX:
                    x1, y1, x2, y2 = bbox
                    color = (0, 255, 0) if (threshold_met and decoded_data) else (0, 0, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"Area: {area:.0f}px", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if decoded_data:
                        cv2.putText(display_frame, f"ID: {decoded_data[:15]}", (x1, y2 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if state.current_material_id:
                    material_text = f"Material: {state.current_material_id[:20]}"
                    cv2.putText(display_frame, material_text,
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if state.frames_without_qr > 0:
                        counter_text = f"Drop in: {state.QR_DROP_THRESHOLD - state.frames_without_qr}"
                        cv2.putText(display_frame, counter_text,
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "Material: None",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
            # If frame read failed, use a blank image for display_frame
            if display_frame is None:
                import numpy as np
                display_frame = 255 * np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(display_frame, "Camera Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            with state.camera_lock:
                state.latest_display_frame = display_frame
        except Exception as main_exc:
            print(f"[QR] Error in QR detection loop: {main_exc}")
            time.sleep(0.5)

    cap.release()
    print("[QR] QR detection thread stopped")
