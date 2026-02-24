"""Web dashboard HTML template."""

# ============== WEB DASHBOARD HTML ==============
HTML_PAGE = '''<!DOCTYPE html>
<html>
<head>
    <title>AUTONEX - Warehouse Tracking</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: #1e293b;
            color: #fff;
            width: 1024px;
            height: 600px;
            overflow: hidden;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 12px;
            background: #1e293b;
            height: 36px;
        }
        .logo-container {
            background: white;
            padding: 2px 8px;
            border-radius: 4px;
            display: flex;
            align-items: center;
        }
        .logo {
            font-size: 16px;
            font-weight: 900;
            color: #000;
            letter-spacing: -1px;
        }
        .header-info {
            display: flex;
            gap: 12px;
            align-items: center;
            background: white;
            padding: 4px 12px;
            border-radius: 4px;
            color: #1e293b;
            font-weight: 600;
            font-size: 12px;
        }
        .main-container {
            display: flex;
            padding: 6px 12px;
            gap: 12px;
            height: calc(600px - 36px);
            align-items: flex-start;
        }
        .map-wrapper {
            flex: 1;
            display: flex;
            justify-content: flex-start;
            align-items: center;
            height: 100%;
        }
        .map-container {
            position: relative;
            width: 99%;
            max-height: 99%;
            aspect-ratio: 55 / 37;
            background: #ffffff;
            border-radius: 6px;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .right-panel {
            width: 280px;
            min-width: 280px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            height: 100%;
        }
        .camera-preview {
            background: #0f172a;
            border-radius: 6px;
            padding: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            width: 100%;
        }
        .camera-preview h3 {
            margin: 0 0 4px 0;
            font-size: 11px;
            font-weight: 600;
            color: #94a3b8;
        }
        .camera-feed {
            width: 100%;
            height: auto;
            border-radius: 4px;
            background: #000;
            display: block;
            cursor: pointer;
        }
        .camera-modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            z-index: 4000;
            align-items: center;
            justify-content: center;
        }
        .camera-modal-overlay.active {
            display: flex;
        }
        .camera-modal-content {
            position: relative;
            width: 90%;
            height: 90%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .camera-modal-content img {
            max-width: 100%;
            max-height: 100%;
            border-radius: 8px;
            object-fit: contain;
        }
        .camera-modal-close {
            position: absolute;
            top: -10px;
            right: -10px;
            background: #1e293b;
            border: 2px solid #475569;
            color: #fff;
            font-size: 24px;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
            z-index: 4001;
        }
        .camera-modal-close:hover {
            background: #ef4444;
            border-color: #ef4444;
        }
        .materials-panel {
            background: #0f172a;
            border-radius: 6px;
            padding: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            flex: 1;
            overflow-y: auto;
            width: 100%;
        }
        .materials-panel h3 {
            margin: 0 0 6px 0;
            font-size: 15px;
            font-weight: 600;
            color: #94a3b8;a
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .sync-indicator {
            font-size: 13px;
            color: #10b981;
            display: flex;
            align-items: center;
            gap: 3px;
        }
        .sync-dot {
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 2s infinite;
        }
        .material-item {
            background: #1e293b;
            padding: 7px 10px;
            border-radius: 4px;
            margin-bottom: 5px;
            border-left: 3px solid #f59e0b;
            cursor: pointer;
            transition: all 0.2s;
        }
        .material-item:hover {
            background: #334155;
            transform: translateX(2px);
        }
        .material-id {
            font-weight: 600;
            font-size: 13px;
            color: #fbbf24;
            margin-bottom: 2px;
        }
        .material-location {
            font-size: 12px;
            color: #94a3b8;
            display: flex;
            gap: 8px;
        }
        .material-time {
            font-size: 11px;
            color: #94a3b8;
            margin-top: 2px;
        }
        .material-marker {
            fill: #f59e0b;
            stroke: #fff;
            stroke-width: 2;
            filter: drop-shadow(0 2px 6px rgba(245, 158, 11, 0.6));
            cursor: pointer;
            transition: all 0.2s;
        }
        .material-marker:hover {
            fill: #fbbf24;
        }
        .material-label {
            fill: white;
            font-size: 10px;
            font-weight: 600;
            pointer-events: none;
        }
        .materials-panel::-webkit-scrollbar {
            width: 4px;
        }
        .materials-panel::-webkit-scrollbar-track {
            background: #1e293b;
            border-radius: 3px;
        }
        .materials-panel::-webkit-scrollbar-thumb {
            background: #475569;
            border-radius: 3px;
        }
        .anchor-letter {
            font-size: 9px;
            font-weight: 700;
            fill: #1e293b;
            pointer-events: none;
        }
        svg {
            width: 100%;
            height: 100%;
        }
        .warehouse-floor {
            fill: #334155;
            stroke: #475569;
            stroke-width: 1.5;
        }
        .grid-lines line {
            stroke: #475569;
            stroke-width: 0.3;
            stroke-dasharray: 4 4;
        }
        .trail {
            fill: none;
            stroke: #60a5fa;
            stroke-width: 2;
            opacity: 0.6;
        }
        .tag-circle {
            fill: #3b82f6;
            filter: drop-shadow(0 2px 6px rgba(59, 130, 246, 0.6));
            transition: all 0.2s ease-out;
        }
        .tag-number {
            fill: white;
            font-size: 14px;
            font-weight: 700;
        }
        .distance-banner {
            position: absolute;
            bottom: 12px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(15, 23, 42, 0.95);
            color: #fff;
            padding: 14px 28px;
            border-radius: 10px;
            display: none;
            z-index: 100;
            font-size: 20px;
            font-weight: 700;
            align-items: center;
            gap: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.5);
            white-space: nowrap;
            border: 2px solid #3b82f6;
        }
        .distance-banner.show { display: flex; }
        .distance-value {
            color: #60a5fa;
            font-size: 32px;
            font-weight: 900;
        }
        .distance-close {
            background: none;
            border: none;
            color: #94a3b8;
            font-size: 26px;
            cursor: pointer;
            padding: 0 4px;
            line-height: 1;
        }
        .distance-close:hover { color: #fff; }
        .distance-line {
            fill: none;
            stroke: #60a5fa;
            stroke-width: 1.5;
            stroke-dasharray: 6 4;
            opacity: 0.8;
        }
        .qr-notification {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            padding: 18px 24px;
            border-radius: 12px;
            box-shadow: 0 12px 40px rgba(16, 185, 129, 0.5);
            display: none;
            z-index: 2000;
            min-width: 300px;
            animation: scaleIn 0.3s ease-out;
        }
        @keyframes scaleIn {
            from { transform: translate(-50%, -50%) scale(0.8); opacity: 0; }
            to { transform: translate(-50%, -50%) scale(1); opacity: 1; }
        }
        .qr-notification.drop {
            background: linear-gradient(135deg, #f59e0b, #d97706);
            box-shadow: 0 12px 40px rgba(245, 158, 11, 0.5);
        }
        .qr-notification.show { display: block; }
        .qr-notification h3 {
            margin: 0 0 8px 0;
            font-size: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 700;
        }
        .qr-notification p {
            margin: 4px 0;
            font-size: 12px;
            opacity: 0.95;
        }
        .qr-notification .material {
            font-weight: 700;
            font-size: 14px;
            margin: 6px 0;
        }
        /* ── WeighMate Scale Panel ── */
        .weighing-panel {
            background: #0f172a;
            border-radius: 6px;
            padding: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            width: 100%;
        }
        .weighing-panel h3 {
            margin: 0 0 4px 0;
            font-size: 11px;
            font-weight: 600;
            color: #94a3b8;
        }
        .scale-feed {
            width: 100%;
            height: auto;
            max-height: 78px;
            border-radius: 4px;
            background: #000;
            display: block;
            object-fit: contain;
        }
        .weight-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 4px;
        }
        .weight-value {
            font-size: 20px;
            font-weight: 700;
            color: #00e676;
            font-family: monospace;
            letter-spacing: 1px;
        }
        .weight-value.idle { color: #475569; font-size: 13px; font-weight: 400; }
        .weight-badge {
            font-size: 10px;
            padding: 2px 7px;
            border-radius: 10px;
            background: #1e293b;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .weight-badge.stabilizing { background: #1a3300; color: #76ff03; }
        .weight-badge.confirmed   { background: #003a1a; color: #00e676; }
        .weight-badge.error       { background: #3a0000; color: #ff5252; }
        .weight-buttons {
            display: flex;
            gap: 5px;
            margin-top: 5px;
        }
        .weight-buttons button {
            flex: 1;
            padding: 6px 4px;
            border: none;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
        }
        .btn-wconfirm { background: #00c853; color: #000; }
        .btn-wrescan  { background: #263238; color: #fff; }
    </style>
</head>
<body>
    <div class="qr-notification" id="qrNotification">
        <h3><span id="qrIcon">📦</span> <span id="qrTitle">QR Code Scanned!</span></h3>
        <div class="material" id="qrMaterial">Material: SAMPLE_M_ID</div>
        <p id="qrPosition">Position: X=3.400m, Y=1.400m</p>
        <p id="qrTime">Time: 2026-02-06 08:45:11.740</p>
    </div>

    <div class="header">
        <div class="logo-container">
            <img src="Logo.png" alt="AUTONEX" style="height: 23px;">
        </div>
        <div class="header-info">
            <span>Forklift: <span id="forkliftId">F001</span></span>
            <span>Operator: <span id="operatorId">Rishabh</span></span>
        </div>
    </div>

    <div class="main-container">
        <div class="map-wrapper">
            <div class="map-container">
                <svg viewBox="0 0 550 370" preserveAspectRatio="xMidYMid meet">
                    <!-- Warehouse L-shaped floor -->
                    <polygon id="warehouseFloor" class="warehouse-floor"/>
                    <!-- Grid lines -->
                    <g id="gridLines" class="grid-lines"></g>
                    <!-- Anchor markers on SVG -->
                    <g id="anchorDots"></g>
                    <polyline id="trail" class="trail"/>
                    <g id="materialMarkers"></g>
                    <line id="distanceLine" class="distance-line" x1="0" y1="0" x2="0" y2="0" style="display:none;"/>
                    <g id="tag">
                        <circle class="tag-circle" r="12" cx="0" cy="0"/>
                        <text class="tag-number" x="0" y="0" text-anchor="middle" dominant-baseline="central">1</text>
                    </g>
                </svg>
                <div class="distance-banner" id="distanceBanner">
                    <span id="distanceReelId"></span>
                    <span class="distance-value" id="distanceValue">--</span>
                    <button class="distance-close" id="distanceClose" title="Stop tracking">&times;</button>
                </div>
            </div>
        </div>

        <div class="right-panel">
            <div class="camera-preview">
                <h3>📹 Camera Feed</h3>
                <img src="/video_feed" alt="Camera Feed" class="camera-feed">
            </div>
            <!-- WeighMate Scale Panel -->
            <div class="weighing-panel">
                <h3>⚖️ Scale</h3>
                <img src="{{SCALE_STREAM_URL}}" class="scale-feed" id="scaleFeed" alt="Scale Feed">
                <div class="weight-row">
                    <span class="weight-value idle" id="weightValue">No Signal</span>
                    <span class="weight-badge" id="weightBadge">IDLE</span>
                </div>
                <div class="weight-buttons" id="weightButtons" style="display:none">
                    <button class="btn-wconfirm" id="btnWConfirm" onclick="confirmWeight()">✓ CONFIRM</button>
                    <button class="btn-wrescan"  onclick="rescanWeight()">↺ RESCAN</button>
                </div>
            </div>
            <div class="materials-panel">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0; display: flex; align-items: center; gap: 8px;">
                        <span>📦 Warehouse Materials</span>
                        <span class="sync-indicator">
                            <span class="sync-dot"></span>
                            <span>Synced</span>
                        </span>
                    </h3>
                </div>
                <div style="margin-top: 6px; text-align: left; display: flex; gap: 8px;">
                    <button id="searchReelBtn" style="background: #475569; color: #fff; border: none; border-radius: 4px; padding: 7px 16px; font-weight: 600; font-size: 15px; cursor: pointer;">Search Reel</button>
                    <button id="showAnchorBtn" style="background: #475569; color: #fff; border: none; border-radius: 4px; padding: 7px 16px; font-weight: 600; font-size: 15px; cursor: pointer;">Anchor Data</button>
                </div>
                <div id="materialsList" style="margin-top: 6px;"></div>
            </div>
        </div>
    </div>

    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
            <!-- Reel Search Modal -->
            <div id="reelSearchModal" style="display:none; position:fixed; top:0; left:0; width:1024px; height:600px; background:rgba(30,41,59,0.90); z-index:3000; align-items:center; justify-content:center;">
                <div style="background:#fff; color:#1e293b; border-radius:8px; padding:32px 36px; min-width:648px; max-width:960px; max-height:580px; overflow-y:auto; box-shadow:0 4px 20px rgba(0,0,0,0.15); position:relative;">
                    <h2 style="margin-top:0; font-size:24px; font-weight:700; letter-spacing:-0.3px; color:#1e293b;">Search Reel</h2>
                    <button id="closeReelModal" style="position:absolute; top:20px; right:20px; background:none; border:none; font-size:28px; color:#94a3b8; cursor:pointer; line-height:1;">&times;</button>
                    <p style="color:#64748b; margin-bottom:16px; margin-top:4px; font-size:13px;">Search and manage scanned material reels</p>
                    <input id="reelSearchInput" type="text" placeholder="Type material ID to search..." style="width:100%; padding:10px 14px; border:1px solid #cbd5e1; border-radius:6px; font-size:14px; margin:8px 0; outline:none; color:#1e293b;" oninput="filterReels()">
                    <div id="reelListContainer">
                        <div style="color:#64748b; text-align:center; padding:18px; font-size:18px;">Loading scanned reels...</div>
                    </div>
                </div>
            </div>
            <!-- Camera Feed Modal -->
            <div id="cameraModal" class="camera-modal-overlay">
                <div class="camera-modal-content">
                    <button class="camera-modal-close" id="closeCameraModal">&times;</button>
                    <img src="/video_feed" alt="Camera Feed">
                </div>
            </div>
            <!-- Anchor Distances Modal -->
            <div id="anchorModal" style="display:none; position:fixed; top:0; left:0; width:1024px; height:600px; background:rgba(30,41,59,0.90); z-index:3000; align-items:center; justify-content:center;">
                <div style="background:#fff; color:#1e293b; border-radius:8px; padding:32px 36px; min-width:520px; max-width:720px; box-shadow:0 4px 20px rgba(0,0,0,0.15); position:relative;">
                    <h2 style="margin-top:0; font-size:24px; font-weight:700; letter-spacing:-0.3px; color:#1e293b;">Anchor Distances</h2>
                    <button id="closeAnchorModal" style="position:absolute; top:20px; right:20px; background:none; border:none; font-size:28px; color:#94a3b8; cursor:pointer; line-height:1;">&times;</button>
                    <p style="color:#64748b; margin-bottom:24px; margin-top:4px; font-size:13px;">Raw UWB ranging measurements</p>
                    <div id="anchorDistancesContainer" style="display:grid; grid-template-columns:repeat(2, 1fr); gap:12px; margin-top:20px;">
                        <!-- Anchor distances will be populated here -->
                    </div>
                </div>
            </div>
    <script>
                // --- Camera Feed Modal Logic ---
                const cameraFeed = document.querySelector('.camera-feed');
                const cameraModal = document.getElementById('cameraModal');
                const closeCameraModal = document.getElementById('closeCameraModal');
                cameraFeed.addEventListener('click', () => {
                    cameraModal.classList.add('active');
                });
                closeCameraModal.addEventListener('click', () => {
                    cameraModal.classList.remove('active');
                });
                cameraModal.addEventListener('click', (e) => {
                    if (e.target === cameraModal) {
                        cameraModal.classList.remove('active');
                    }
                });

                // --- Reel Search Modal Logic ---
                const searchReelBtn = document.getElementById('searchReelBtn');
                const reelSearchModal = document.getElementById('reelSearchModal');
                const closeReelModal = document.getElementById('closeReelModal');
                const reelListContainer = document.getElementById('reelListContainer');
                let scannedReels = [];
                let selectedReel = null;

                function renderReelList(reels) {
                    if (reels.length === 0) {
                        reelListContainer.innerHTML = '<div style="color:#64748b; text-align:center; padding:18px; font-size:18px;">No reels found.</div>';
                    } else {
                        reelListContainer.innerHTML = reels.map(r => `
                            <div class="material-item" style="background:#e0e7ef; color:#1e293b; margin-bottom:10px; cursor:pointer; display:flex; align-items:center; justify-content:space-between;" onclick="selectReel('${r.material_id}')">
                                <div style="flex:1;">
                                    <div class="material-id" style="font-size:14px; color:#d97706;">${r.material_id}</div>
                                    <div class="material-location" style="font-size:13px;"><span>X: ${r.x.toFixed(2)}m</span> <span>Y: ${r.y.toFixed(2)}m</span></div>
                                    <div class="material-time" style="font-size:12px;">Dropped at ${r.datetime}</div>
                                </div>
                                <button onclick="event.stopPropagation(); deleteReel('${r.material_id}')" style="background:#ef4444; color:#fff; border:none; border-radius:4px; padding:6px 10px; font-size:14px; font-weight:600; cursor:pointer; margin-left:8px; white-space:nowrap;">🗑️ Delete</button>
                            </div>
                        `).join('');
                    }
                }
                window.deleteReel = function(material_id) {
                    if (!confirm('Delete reel ' + material_id + '?')) return;
                    fetch('/api/reels/' + encodeURIComponent(material_id), { method: 'DELETE' })
                        .then(res => res.json())
                        .then(data => {
                            if (data.success) {
                                scannedReels = scannedReels.filter(r => r.material_id !== material_id);
                                const query = document.getElementById('reelSearchInput').value.toLowerCase();
                                const filtered = query ? scannedReels.filter(r => r.material_id.toLowerCase().includes(query)) : scannedReels;
                                renderReelList(filtered);
                            }
                        });
                }
                window.filterReels = function() {
                    const query = document.getElementById('reelSearchInput').value.toLowerCase();
                    const filtered = scannedReels.filter(r => r.material_id.toLowerCase().includes(query));
                    renderReelList(filtered);
                }
                searchReelBtn.addEventListener('click', () => {
                    reelSearchModal.style.display = 'flex';
                    document.getElementById('reelSearchInput').value = '';
                    fetch('/api/reels')
                        .then(res => res.json())
                        .then(data => {
                            scannedReels = data.reels || [];
                            renderReelList(scannedReels);
                            document.getElementById('reelSearchInput').focus();
                        });
                });
                closeReelModal.addEventListener('click', () => {
                    reelSearchModal.style.display = 'none';
                });
                window.selectReel = function(material_id) {
                    const reel = scannedReels.find(r => r.material_id === material_id);
                    if (reel) {
                        selectedReel = reel;
                        // Track this reel for live distance display
                        trackedReel = { material_id: reel.material_id, x: reel.x, y: reel.y };
                        updateDistanceTracker();
                        // Add to main materials panel
                        addDroppedMaterial({
                            material_id: reel.material_id,
                            location_x: reel.x,
                            location_y: reel.y,
                            datetime: reel.datetime
                        });
                        showReelOnMap(reel);
                        reelSearchModal.style.display = 'none';
                    }
                }
                function showReelOnMap(reel) {
                    // Move tag to reel position and flash marker
                    const pos = toSvg(reel.x, reel.y);
                    const tag = document.getElementById('tag');
                    tag.setAttribute('transform', `translate(${pos.x}, ${pos.y})`);
                    // Optionally, add a highlight effect
                    tag.querySelector('circle').style.stroke = '#f59e0b';
                    tag.querySelector('circle').style.strokeWidth = '6';
                    setTimeout(() => {
                        tag.querySelector('circle').style.stroke = '';
                        tag.querySelector('circle').style.strokeWidth = '';
                    }, 1200);
                }
                // --- Anchor Distances Modal Logic ---
                const showAnchorBtn = document.getElementById('showAnchorBtn');
                const anchorModal = document.getElementById('anchorModal');
                const closeAnchorModal = document.getElementById('closeAnchorModal');
                const anchorDistancesContainer = document.getElementById('anchorDistancesContainer');

                const ANCHOR_COLORS = {
                    'A': '#ef4444', 'B': '#22c55e', 'C': '#3b82f6', 'D': '#f59e0b',
                    'E': '#a855f7', 'F': '#ec4899', 'G': '#14b8a6'
                };

                function renderAnchorDistances(distances) {
                    anchorDistancesContainer.innerHTML = Object.keys(distances).map(anchor => {
                        const dist = distances[anchor];
                        const distText = dist !== null && dist !== undefined ? `${dist.toFixed(2)}m` : '--';
                        const isAvailable = dist !== null && dist !== undefined;
                        const bgColor = isAvailable ? '#ffffff' : '#f8fafc';
                        const borderColor = isAvailable ? '#e2e8f0' : '#f1f5f9';
                        const textOpacity = isAvailable ? '1' : '0.4';
                        return `
                            <div style="background:${bgColor}; padding:18px 20px; border-radius:6px; border:1px solid ${borderColor}; opacity:${textOpacity};">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <div style="font-size:16px; font-weight:600; color:#475569;">Anchor ${anchor}</div>
                                    <div style="font-size:18px; font-weight:700; color:#1e293b; font-family:monospace;">${distText}</div>
                                </div>
                            </div>
                        `;
                    }).join('');
                }

                showAnchorBtn.addEventListener('click', () => {
                    anchorModal.style.display = 'flex';
                    fetch('/api/anchors')
                        .then(res => res.json())
                        .then(data => {
                            renderAnchorDistances(data.distances);
                        });
                });

                closeAnchorModal.addEventListener('click', () => {
                    anchorModal.style.display = 'none';
                });
        // Load anchor positions, map dimensions, and warehouse boundary from backend config
        const ANCHORS_CONFIG = {{ANCHORS_DATA}};
        const MAP_CONFIG = {{MAP_CONFIG}};
        const WAREHOUSE_BOUNDARY = {{BOUNDARY_DATA}};

        // Use map dimensions from config
        const ROOM_W = MAP_CONFIG.length;
        const ROOM_H = MAP_CONFIG.breadth;

        const SVG_WIDTH = 550;
        const SVG_HEIGHT = 370;
        const PAD = 15; // padding inside SVG

        let trail = [];
        let droppedMaterials = {}; // Track currently dropped materials by ID
        let trackedReel = null; // {material_id, x, y} — reel being tracked for distance
        let currentTagX = ROOM_W / 2, currentTagY = ROOM_H / 2; // latest tag position in meters

        function toSvg(x, y) {
            return {
                x: (SVG_WIDTH - PAD) - (x / ROOM_W) * (SVG_WIDTH - 2 * PAD),
                y: (SVG_HEIGHT - PAD) - (y / ROOM_H) * (SVG_HEIGHT - 2 * PAD)
            };
        }

        // Draw the L-shaped warehouse floor
        (function drawWarehouse() {
            const points = WAREHOUSE_BOUNDARY.map(p => {
                const s = toSvg(p[0], p[1]);
                return `${s.x},${s.y}`;
            }).join(' ');
            document.getElementById('warehouseFloor').setAttribute('points', points);

            // Draw grid lines
            const gridG = document.getElementById('gridLines');
            const gridSpacing = 10; // meters
            for (let gx = 0; gx <= ROOM_W; gx += gridSpacing) {
                const p1 = toSvg(gx, 0), p2 = toSvg(gx, ROOM_H);
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', p1.x); line.setAttribute('y1', p1.y);
                line.setAttribute('x2', p2.x); line.setAttribute('y2', p2.y);
                gridG.appendChild(line);
            }
            for (let gy = 0; gy <= ROOM_H; gy += gridSpacing) {
                const p1 = toSvg(0, gy), p2 = toSvg(ROOM_W, gy);
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', p1.x); line.setAttribute('y1', p1.y);
                line.setAttribute('x2', p2.x); line.setAttribute('y2', p2.y);
                gridG.appendChild(line);
            }

            // Draw anchor dots on SVG
            const dotsG = document.getElementById('anchorDots');
            const ANCHOR_COLORS = {
                'A': '#ef4444', 'B': '#22c55e', 'C': '#3b82f6', 'D': '#f59e0b',
                'E': '#a855f7', 'F': '#ec4899', 'G': '#14b8a6'
            };
            Object.keys(ANCHORS_CONFIG).forEach(key => {
                const a = ANCHORS_CONFIG[key];
                const pos = toSvg(a.x, a.y);
                const color = ANCHOR_COLORS[key] || '#94a3b8';
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', pos.x);
                circle.setAttribute('cy', pos.y);
                circle.setAttribute('r', '4');
                circle.setAttribute('fill', color);
                circle.setAttribute('stroke', '#fff');
                circle.setAttribute('stroke-width', '1');
                dotsG.appendChild(circle);
                // Letter label offset to top-right of dot
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', pos.x + 7);
                label.setAttribute('y', pos.y - 5);
                label.setAttribute('class', 'anchor-letter');
                label.setAttribute('fill', color);
                label.textContent = key;
                dotsG.appendChild(label);
            });
        })();

        function updateUI(data) {
            console.log('Received update:', data);

            if (data.forklift_id) {
                document.getElementById('forkliftId').textContent = data.forklift_id;
            }
            if (data.operator_name) {
                document.getElementById('operatorId').textContent = data.operator_name;
            }

            if (data.location_x != null && data.location_y != null) {
                console.log(`Moving tag to: (${data.location_x}, ${data.location_y})`);
                const pos = toSvg(data.location_x, data.location_y);
                console.log(`SVG coordinates: (${pos.x}, ${pos.y})`);

                const tag = document.getElementById('tag');
                tag.setAttribute('transform', `translate(${pos.x}, ${pos.y})`);

                trail.push(pos);
                if (trail.length > 100) trail.shift();
                document.getElementById('trail').setAttribute('points',
                    trail.map(p => `${p.x},${p.y}`).join(' ')
                );

                // Update distance tracking
                currentTagX = data.location_x;
                currentTagY = data.location_y;
                updateDistanceTracker();
            }
        }

        function addDroppedMaterial(data) {
            droppedMaterials[data.material_id] = {
                id: data.material_id,
                x: data.location_x,
                y: data.location_y,
                time: data.datetime || new Date().toLocaleString(),
                timestamp: Date.now()
            };
            updateMaterialsList();
            updateMapMarkers();
        }

        function removeDroppedMaterial(materialId) {
            delete droppedMaterials[materialId];
            updateMaterialsList();
            updateMapMarkers();
        }

        function updateMaterialsList() {
            const container = document.getElementById('materialsList');
            const materials = Object.values(droppedMaterials);

            if (materials.length === 0) {
                container.innerHTML = '';
                return;
            }

            // Sort by timestamp (most recent first)
            materials.sort((a, b) => b.timestamp - a.timestamp);

            container.innerHTML = materials.map(m => `
                <div class="material-item" style="display:flex; align-items:center; justify-content:space-between;" onclick="highlightMaterialOnMap('${m.id}')">
                    <div style="flex:1;">
                        <div class="material-id">${m.id}</div>
                        <div class="material-location">
                            <span>X: ${m.x.toFixed(2)}m</span>
                            <span>Y: ${m.y.toFixed(2)}m</span>
                        </div>
                        <div class="material-time">${m.time}</div>
                    </div>
                    <button onclick="event.stopPropagation(); removeDroppedMaterial('${m.id}')" style="background:none; border:none; color:#94a3b8; font-size:14px; cursor:pointer; padding:2px 6px;" title="Remove">&times;</button>
                </div>
            `).join('');
        }

        function updateMapMarkers() {
            const container = document.getElementById('materialMarkers');
            const materials = Object.values(droppedMaterials);

            container.innerHTML = materials.map(m => {
                const pos = toSvg(m.x, m.y);
                const shortId = m.id.length > 8 ? m.id.substring(0, 8) + '...' : m.id;
                return `
                    <g class="material-marker-group" onclick="highlightMaterialInList('${m.id}')">
                        <circle class="material-marker" cx="${pos.x}" cy="${pos.y}" r="10"/>
                        <text class="material-label" x="${pos.x}" y="${pos.y + 4}" text-anchor="middle">📦</text>
                    </g>
                `;
            }).join('');
        }

        function highlightMaterialOnMap(materialId) {
            const material = droppedMaterials[materialId];
            if (!material) return;

            // Flash the marker on the map
            const markers = document.querySelectorAll('.material-marker');
            const materials = Object.values(droppedMaterials);
            const index = materials.findIndex(m => m.id === materialId);

            if (index >= 0 && markers[index]) {
                markers[index].style.fill = '#fbbf24';
                markers[index].setAttribute('r', '14');
                setTimeout(() => {
                    markers[index].style.fill = '#f59e0b';
                    markers[index].setAttribute('r', '10');
                }, 800);
            }
        }

        function highlightMaterialInList(materialId) {
            // Scroll to and flash the material in the list
            const container = document.getElementById('materialsList');
            const items = container.querySelectorAll('.material-item');
            const materials = Object.values(droppedMaterials).sort((a, b) => b.timestamp - a.timestamp);
            const index = materials.findIndex(m => m.id === materialId);

            if (index >= 0 && items[index]) {
                items[index].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                items[index].style.background = '#475569';
                setTimeout(() => {
                    items[index].style.background = '#1e293b';
                }, 800);
            }
        }

        // ---- Distance tracking between tag and selected reel ----
        function updateDistanceTracker() {
            const banner = document.getElementById('distanceBanner');
            const line = document.getElementById('distanceLine');
            if (!trackedReel) {
                banner.classList.remove('show');
                line.style.display = 'none';
                return;
            }
            const dx = currentTagX - trackedReel.x;
            const dy = currentTagY - trackedReel.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            document.getElementById('distanceReelId').textContent = trackedReel.material_id + ':';
            document.getElementById('distanceValue').textContent = dist.toFixed(1) + 'm away';
            banner.classList.add('show');

            // Draw dashed line on SVG
            const tagSvg = toSvg(currentTagX, currentTagY);
            const reelSvg = toSvg(trackedReel.x, trackedReel.y);
            line.setAttribute('x1', tagSvg.x);
            line.setAttribute('y1', tagSvg.y);
            line.setAttribute('x2', reelSvg.x);
            line.setAttribute('y2', reelSvg.y);
            line.style.display = '';
        }

        function clearTrackedReel() {
            trackedReel = null;
            updateDistanceTracker();
        }

        document.getElementById('distanceClose').addEventListener('click', clearTrackedReel);

        function showQRNotification(data, type) {
            const notif = document.getElementById('qrNotification');
            notif.classList.remove('drop');

            document.getElementById('qrIcon').textContent = type === 'pickup' ? '📦' : '📤';
            document.getElementById('qrTitle').textContent = type === 'pickup' ? 'QR Code Scanned!' : 'QR Code Dropped!';

            if (type === 'drop') notif.classList.add('drop');

            document.getElementById('qrMaterial').textContent = 'Material: ' + data.material_id;
            document.getElementById('qrPosition').textContent =
                `Position: X=${data.location_x.toFixed(3)}m, Y=${data.location_y.toFixed(3)}m`;
            document.getElementById('qrTime').textContent =
                'Time: ' + (data.datetime || new Date().toLocaleString());

            notif.classList.add('show');
            setTimeout(() => notif.classList.remove('show'), 5000);
        }

        // Initialize tag position at center
        const initPos = toSvg(ROOM_W / 2, ROOM_H / 2);
        document.getElementById('tag').setAttribute('transform', `translate(${initPos.x}, ${initPos.y})`);
        console.log('Tag initialized at:', initPos);
        console.log('Room dimensions:', ROOM_W, 'x', ROOM_H);
        console.log('Anchors loaded:', ANCHORS_CONFIG);

        // Socket.IO connection
        const socket = io();
        socket.on('connect', () => {
            console.log('✓ Connected to server');
            // Request initial materials sync on connect
            fetch('/api/materials')
                .then(res => res.json())
                .then(data => {
                    if (data.materials) {
                        handleMaterialsSync({materials: data.materials});
                    }
                })
                .catch(err => console.error('Failed to fetch initial materials:', err));
        });
        socket.on('disconnect', () => {
            console.log('✗ Disconnected from server');
        });
        socket.on('update', updateUI);
        socket.on('qr_pickup', (data) => showQRNotification(data, 'pickup'));
        socket.on('qr_drop', (data) => showQRNotification(data, 'drop'));

        // ── WeighMate scale events ──────────────────────────────────────────
        let pendingWeight = null;

        socket.on('scale_live', (data) => {
            const el = document.getElementById('weightValue');
            const badge = document.getElementById('weightBadge');
            if (badge.textContent === 'CONFIRMED') return; // don't overwrite confirmed display
            if (data.weight !== null && data.weight !== undefined) {
                el.textContent = data.weight.toFixed(1) + ' kg';
                el.className = 'weight-value';
            } else {
                el.textContent = '---';
                el.className = 'weight-value idle';
            }
        });

        socket.on('scale_state', (data) => {
            const badge   = document.getElementById('weightBadge');
            const buttons = document.getElementById('weightButtons');
            const confirm = document.getElementById('btnWConfirm');
            const stateStr = data.state || 'IDLE';
            const stateBase = stateStr.split(':')[0].toLowerCase();

            badge.textContent = stateStr.split(':')[0];
            badge.className   = 'weight-badge ' + stateBase;

            if (stateStr === 'IDLE') {
                pendingWeight = null;
                buttons.style.display = 'none';
                const el = document.getElementById('weightValue');
                el.textContent = 'No Signal';
                el.className   = 'weight-value idle';
                confirm.style.display = '';
            } else if (stateBase === 'error') {
                buttons.style.display = 'flex';
                confirm.style.display = 'none';
            } else if (stateStr === 'STABILIZING') {
                buttons.style.display = 'none';
            }
        });

        socket.on('scale_stable', (data) => {
            pendingWeight = data.weight;
            const el = document.getElementById('weightValue');
            el.textContent = data.weight.toFixed(1) + ' kg';
            el.className   = 'weight-value';
            document.getElementById('weightButtons').style.display = 'flex';
            document.getElementById('btnWConfirm').style.display   = '';
        });

        socket.on('scale_health', (data) => {
            if (!data.ok && data.issues && data.issues.length > 0) {
                console.warn('[Scale Health]', data.issues.join('; '));
            }
        });

        function confirmWeight() {
            if (pendingWeight === null) return;
            fetch('/api/weight/confirm', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('weightButtons').style.display = 'none';
                        pendingWeight = null;
                    }
                })
                .catch(err => console.error('[Scale] Confirm error:', err));
        }

        function rescanWeight() {
            pendingWeight = null;
            fetch('/api/weight/rescan', { method: 'POST' })
                .catch(err => console.error('[Scale] Rescan error:', err));
            document.getElementById('weightButtons').style.display = 'none';
        }

        // Handle materials synchronization from central server
        socket.on('materials_sync', handleMaterialsSync);

        // Handle material deletions from central server
        socket.on('materials_deleted', (data) => {
            console.log('Materials deleted from central server:', data.material_ids);
            data.material_ids.forEach(materialId => {
                removeDroppedMaterial(materialId);
            });
        });

        function handleMaterialsSync(data) {
            console.log('Received materials sync:', data.materials.length, 'materials');
            // Update local materials from central server
            data.materials.forEach(material => {
                if (material.material_id && material.x != null && material.y != null) {
                    droppedMaterials[material.material_id] = {
                        id: material.material_id,
                        x: material.x,
                        y: material.y,
                        time: material.datetime || new Date().toLocaleString(),
                        timestamp: material.timestamp ? material.timestamp * 1000 : Date.now()
                    };
                }
            });
            updateMaterialsList();
            updateMapMarkers();
        }
    </script>
</body>
</html>
'''
