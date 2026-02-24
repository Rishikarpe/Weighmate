"""
config.py — Vision Node configuration

All tunable settings in one place.
"""

# ============== MQTT BROKER (EMQX Cloud) ==============
# Same broker as Edge — both nodes publish/subscribe here.
MQTT_BROKER    = 'r7e2272f.ala.eu-central-1.emqxsl.com'
MQTT_PORT      = 8883          # TLS/SSL
MQTT_USERNAME  = 'edge_autonex'
MQTT_PASSWORD  = 'autonex@2050'
MQTT_CLIENT_ID = 'vision_node_scale1'
MQTT_KEEPALIVE = 60

# ============== SCALE IDENTITY ==============
SCALE_ID = 'scale1'

# ============== MQTT TOPICS ==============
MQTT_TOPIC_LIVE_WEIGHT   = f'factory/{SCALE_ID}/live_weight'
MQTT_TOPIC_SESSION_STATE = f'factory/{SCALE_ID}/session_state'
MQTT_TOPIC_STABLE_WEIGHT = f'factory/{SCALE_ID}/stable_weight'
MQTT_TOPIC_HEALTH        = f'factory/{SCALE_ID}/health'
MQTT_TOPIC_SNAPSHOT      = f'factory/{SCALE_ID}/snapshot'
MQTT_TOPIC_COMMAND       = f'factory/{SCALE_ID}/command'

# ============== CAMERA ==============
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
CAPTURE_FPS  = 10

# ============== STREAM SERVER ==============
STREAM_PORT    = 8080
STREAM_FPS     = 5
STREAM_QUALITY = 70

# ============== ROI (tune after physical mount) ==============
# Pixel coordinates of the scale display within the camera frame.
# Use save_debug_image=True in weight_detector.extract_weight() to verify.
ROI_X = 0      # ← TUNE
ROI_Y = 0      # ← TUNE
ROI_W = 640    # ← TUNE
ROI_H = 480    # ← TUNE

# ============== SSOCR ==============
SSOCR_THRESHOLD  = 20    # luminance threshold, tune on hardware
SSOCR_MIN_DIGITS = 3     # 50.0  → 3 digits
SSOCR_MAX_DIGITS = 4     # 500.0 → 4 digits

# ============== WEIGHT VALIDATION ==============
WEIGHT_MIN_KG = 50.0
WEIGHT_MAX_KG = 500.0

# ============== SESSION ==============
IDLE_THRESHOLD_KG = 30.0   # weight above this = reel is on scale

# ============== SNAPSHOTS ==============
SNAPSHOT_DIR = '/home/pi/weighmate/snapshots'

# ============== HEALTH MONITOR ==============
BLUR_THRESHOLD             = 30.0
BRIGHTNESS_MIN             = 20
BRIGHTNESS_MAX             = 240
OBSTRUCTION_TIMEOUT_SECONDS = 4.0
HEALTH_PUBLISH_INTERVAL    = 10.0   # seconds between health publishes
