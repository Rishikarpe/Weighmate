"""Flask web server, SocketIO, and HTTP routes."""

import json
import os
import cv2
import time
from flask import Flask, Response
from flask_socketio import SocketIO
from flask_cors import CORS

from Edge import state
from Edge.dashboard import HTML_PAGE
from config import ANCHORS, MAP_LENGTH, MAP_BREADTH, WAREHOUSE_BOUNDARY, VISION_NODE_STREAM_URL

# ============== FLASK WEB SERVER ==============
app = Flask(__name__)
app.config['SECRET_KEY'] = 'edge-tracker-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/')
def index():
    # Inject ANCHORS data, MAP dimensions, and warehouse boundary into the HTML page
    import json
    anchors_json = json.dumps(ANCHORS)
    map_config_json = json.dumps({
        'length': MAP_LENGTH,
        'breadth': MAP_BREADTH
    })
    boundary_json = json.dumps(WAREHOUSE_BOUNDARY)
    page = HTML_PAGE.replace('{{ANCHORS_DATA}}', anchors_json)
    page = page.replace('{{MAP_CONFIG}}', map_config_json)
    page = page.replace('{{BOUNDARY_DATA}}', boundary_json)
    page = page.replace('{{SCALE_STREAM_URL}}', VISION_NODE_STREAM_URL)
    return Response(page, mimetype='text/html')


@app.route('/Logo.png')
def logo():
    """Serve the logo image"""
    try:
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Logo.png')
        with open(logo_path, 'rb') as f:
            return Response(f.read(), mimetype='image/png')
    except FileNotFoundError:
        # Return a fallback SVG logo if Logo.png doesn't exist
        svg_logo = '''<svg width="120" height="40" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="30" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="#000">autonex</text>
        </svg>'''
        return Response(svg_logo, mimetype='image/svg+xml')


def generate_frames():
    """Generator function for video streaming"""
    while True:
        if state.latest_display_frame is not None:
            with state.camera_lock:
                try:
                    # Use the processed frame with QR overlays
                    frame = state.latest_display_frame.copy()
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ret:
                        continue
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except:
                    pass
        time.sleep(0.033)  # ~30 FPS


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/materials')
def get_materials():
    """API endpoint to get current synced materials"""
    with state.materials_lock:
        materials_list = [{**{'material_id': k}, **v} for k, v in state.synced_materials.items()]
    return {'materials': materials_list, 'count': len(materials_list)}


@app.route('/api/reels')
def get_reels():
    """API endpoint to get all dropped/scanned reels from JSON file"""
    with state.reels_lock:
        reels_list = list(state.scanned_reels.values())
    return {'reels': reels_list, 'count': len(reels_list)}


@app.route('/api/reels/<material_id>', methods=['DELETE'])
def delete_reel(material_id):
    """API endpoint to delete a scanned reel by material_id"""
    with state.reels_lock:
        if material_id in state.scanned_reels:
            del state.scanned_reels[material_id]
    state.save_scanned_reels()
    return {'success': True}


@app.route('/api/anchors')
def get_anchors():
    """API endpoint to get current raw distances from anchors"""
    distances = {
        anchor: state.current_distances.get(anchor)
        for anchor in ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    }
    return {'distances': distances}


# ============== WEIGHMATE API ==============

@app.route('/api/weight/status')
def weight_status():
    """Current weighing session state."""
    with state.weight_lock:
        return {
            'state':         state.weighing_state,
            'live_weight':   state.live_weight,
            'stable_weight': state.stable_weight,
        }


@app.route('/api/weight/confirm', methods=['POST'])
def confirm_weight():
    """Operator confirmed the stable weight — publish to external server."""
    with state.weight_lock:
        weight = state.stable_weight
    if weight is None:
        return {'success': False, 'error': 'No stable weight to confirm'}, 400
    from Edge.mqtt_handler import send_weight_confirmation
    send_weight_confirmation(weight)
    return {'success': True, 'weight_kg': weight}


@app.route('/api/weight/rescan', methods=['POST'])
def rescan_weight():
    """Operator requested rescan — reset Vision Node session."""
    from Edge.mqtt_handler import send_rescan_command
    send_rescan_command()
    with state.weight_lock:
        state.stable_weight = None
    return {'success': True}
