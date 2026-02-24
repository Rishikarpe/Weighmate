"""Global state variables, locks, and reel persistence functions."""

import threading
import json
import os

from Edge.geometry import KalmanFilter2D
from config import KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE

# ============== GLOBAL STATE ==============
current_tag_position = None
qr_detection_active = False
serial_active = False
current_distances = {'A': None, 'B': None, 'C': None, 'D': None, 'E': None, 'F': None, 'G': None}

# QR detection state
current_material_id = None
last_qr_seen_position = None  # Position when QR was last detected
frames_without_qr = 0  # Counter for consecutive frames without QR
QR_DROP_THRESHOLD = 15  # Number of processed frames without QR before drop (3 seconds at 30fps with FRAME_SKIP=2)

# Kalman filter for position smoothing
position_filter = KalmanFilter2D(KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE)

# Camera state for video streaming
camera = None
camera_lock = threading.Lock()
latest_display_frame = None  # Store the latest processed frame with overlays

# Material synchronization state
synced_materials = {}  # Dictionary to store materials synced from central server {material_id: {x, y, timestamp}}
materials_lock = threading.Lock()  # Thread-safe access to synced_materials

# Reel storage (all scanned reels, not just dropped)
SCANNED_REELS_FILE = 'scanned_reels.json'
scanned_reels = {}  # {reel_id: {...}}
reels_lock = threading.Lock()

# ============== WEIGHING STATE (WeighMate Vision Node) ==============
weighing_state = 'IDLE'    # IDLE | STABILIZING | CONFIRMED | ERROR:<reason>
live_weight = None          # float kg from Vision Node, updated every frame
stable_weight = None        # float kg — set when CONFIRMED, cleared on IDLE/rescan
weight_lock = threading.Lock()


def save_scanned_reels():
    with reels_lock:
        try:
            with open(SCANNED_REELS_FILE, 'w', encoding='utf-8') as f:
                json.dump(scanned_reels, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[REELS] Error saving scanned reels: {e}")


def load_scanned_reels():
    global scanned_reels
    try:
        if os.path.exists(SCANNED_REELS_FILE):
            with open(SCANNED_REELS_FILE, 'r', encoding='utf-8') as f:
                scanned_reels = json.load(f)
        else:
            scanned_reels = {}
    except Exception as e:
        print(f"[REELS] Error loading scanned reels: {e}")
        scanned_reels = {}
