import os

# Get the directory where this config file is located
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# ============== SERIAL PORT SETTINGS ==============
# ESP32 connection via USB serial
SERIAL_PORT = '/dev/ttyUSB0'  # Change this if your ESP32 is on a different port
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0  # seconds

# ============== MQTT BROKER SETTINGS ==============
# EMQX Cloud MQTT Broker (TLS/SSL)
MQTT_BROKER = 'r7e2272f.ala.eu-central-1.emqxsl.com'
MQTT_PORT = 8883  # MQTT over TLS/SSL
MQTT_USERNAME = 'edge_autonex'
MQTT_PASSWORD = 'autonex@2050'
MQTT_CLIENT_ID = 'edge_autonex_rpi'  # Unique client ID for this edge device
MQTT_KEEPALIVE = 60  # seconds

# MQTT Topics
MQTT_TOPIC_POSITION = 'position/{forklift_id}'
MQTT_TOPIC_EVENT = 'autonex/forklift/{forklift_id}/event'
MQTT_TOPIC_REEL_EVENT = 'event/{reel_number}'  # New QR reel event topic
MQTT_TOPIC_MATERIALS = 'autonex/materials/sync'

# ============== CENTRAL SERVER SETTINGS (Legacy - replaced by MQTT) ==============
CENTRAL_SERVER_IP = '192.168.187.1'
CENTRAL_SERVER_PORT = 5001
CENTRAL_API_ENDPOINT = f'http://{CENTRAL_SERVER_IP}:{CENTRAL_SERVER_PORT}/api/log_data'

# ============== CAMERA SETTINGS ==============
# Camera type: 'usb' for USB webcam, 'picamera' for RPi Camera Module
CAMERA_TYPE = 'usb'  # Changed to USB for better compatibility
CAMERA_INDEX = 0  # Use index 0 for first USB camera (not '/dev/video0')
# RPi-optimized resolution: lower resolution = faster processing
FRAME_WIDTH = 1280  # Set to 1280 for 720p resolution
FRAME_HEIGHT = 720  # Set to 720 for 720p resolution
FRAME_SKIP = 2      # Process every Nth frame (1=all frames, 2=every other frame, 3=every third)

# ============== QR DETECTION THRESHOLDS ==============
# Using pyzbar for detection (much faster than YOLO or OpenCV on RPi!)
# pyzbar combines detection + decoding in one pass - massive performance boost
SIZE_THRESHOLD_AREA_PX = 200  # Increased for large QR code at 1.5m
MIN_WIDTH_PX = 2              # Increased for large QR code at 1.5m
MIN_HEIGHT_PX = 2             # Increased for large QR code at 1.5m

# ============== DECODING SETTINGS ==============
DECODE_TIMEOUT_MS = 100
MAX_DECODE_ATTEMPTS = 3

# ============== STABILITY SETTINGS ==============
PICKUP_FRAMES = 3   # Reduced from 5 for easier confirmation (with FRAME_SKIP=2, this is ~6 frames or 0.2s)
DROP_FRAMES = 6     # Reduced from 8 for faster drop detection

# ============== LOGGING SETTINGS ==============
VERBOSE_OUTPUT = False
LOG_DETECTION_STATS = False
LOG_INTERVAL_SECONDS = 1.0
FIXED_INTERVAL = True
os.path.join(_CONFIG_DIR, "edge_backup.db")
# ============== CALIBRATION SETTINGS ==============
CALIBRATION_QR_SIZE_CM = 10.0

# ============== DATABASE SETTINGS (Local backup only) ==============
DB_PATH = "edge_backup.db"  # Local backup database
SEPARATE_TABLES = True
TABLE_PREFIX = "Forklift_"
DEFAULT_FORKLIFT = "F001"
DEFAULT_OPERATOR = "Rishabh"  # Default forklift operator name
LOCATION_TRACKING = True

# ============== ANCHOR POSITIONS ==============
# Anchor positions in meters (must match your physical setup)
# Configured for 52.5m x 35m warehouse, origin at bottom-right
ANCHORS = {
    'A': {'x': 13.5, 'y': 6.5},
    'B': {'x': 13.5, 'y': 14.5},
    'C': {'x': 13.5, 'y': 20.0},
    'D': {'x': 15.0, 'y': 35.0},
    'E': {'x': 37.5, 'y': 35.0},
    'F': {'x': 37.5, 'y': 20.0},
    'G': {'x': 52.5, 'y': 30.0},
}

# ============== MAP DIMENSIONS ==============
# Map dimensions in meters (length = x-axis, breadth = y-axis)
# Must be >= anchor extents. Includes ~0.5m margin for visibility.
MAP_LENGTH = 55.0   # Width of the area in meters (x-axis, anchors span 0-52.5m)
MAP_BREADTH = 37.0  # Height of the area in meters (y-axis, anchors span 0-35m)

# ============== WAREHOUSE BOUNDARY (L-shaped polygon) ==============
# Vertices of the warehouse boundary in order (clockwise or counter-clockwise)
# Used for point-in-polygon validation during trilateration
WAREHOUSE_BOUNDARY = [
    (0.0, 0.0),
    (22.5, 0.0),
    (22.5, 20.0),
    (52.5, 20.0),
    (52.5, 30.0),
    (37.5, 30.0),
    (37.5, 35.0),
    (0.0, 35.0),
]

# ============== PROCESSING ZONE ==============
# Rectangular area where reels dropped are considered "gone for processing"
# Adjust these coordinates to match your physical processing area
PROCESSING_ZONE = [
    (42.5, 20.0),
    (52.5, 20.0),
    (52.5, 30.0),
    (42.5, 30.0),
]

# ============== TRILATERATION TUNING ==============
# NLOS (Non-Line-of-Sight) detection: residual above this = likely NLOS anchor
NLOS_RESIDUAL_THRESHOLD = 1.5  # meters
NLOS_MAX_ITERATIONS = 2        # max NLOS rejection passes

# Outlier rejection: triplet results further than this from median are discarded
OUTLIER_DISTANCE_THRESHOLD = 3.0  # meters

# Kalman filter tuning (constant-velocity model)
KALMAN_PROCESS_NOISE = 0.5       # how unpredictable forklift motion is
KALMAN_MEASUREMENT_NOISE = 1.5   # how noisy trilateration readings are

# ============== WEIGHMATE (VISION NODE) ==============
# IP of the Vision Node RPi (fixed above scale).  ← SET BEFORE DEPLOYMENT
VISION_NODE_IP = '192.168.1.100'
VISION_NODE_STREAM_URL = f'http://{VISION_NODE_IP}:8080/stream'

# Scale MQTT topics (Vision Node publishes, Edge subscribes)
SCALE_ID = 'scale1'
MQTT_TOPIC_SCALE_LIVE      = f'factory/{SCALE_ID}/live_weight'
MQTT_TOPIC_SCALE_STATE     = f'factory/{SCALE_ID}/session_state'
MQTT_TOPIC_SCALE_STABLE    = f'factory/{SCALE_ID}/stable_weight'
MQTT_TOPIC_SCALE_HEALTH    = f'factory/{SCALE_ID}/health'
MQTT_TOPIC_SCALE_COMMAND   = f'factory/{SCALE_ID}/command'     # Edge → Vision Node
MQTT_TOPIC_SCALE_CONFIRMED = f'factory/{SCALE_ID}/confirmed_weight'  # Edge → external server