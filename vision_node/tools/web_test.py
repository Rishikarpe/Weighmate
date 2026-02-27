"""
tools/web_test.py

Local web dashboard for testing the weight detector on uploaded images.
Supports three OCR backends — pick whichever works on your machine.

Run (from the vision_node/ directory):
    python tools/web_test.py
    python tools/web_test.py --port 5050

Then open http://localhost:5000 in your browser.

Backends
--------
tflite   — model_float16.tflite (needs tflite-runtime or tensorflow with Flex support)
tesseract — pytesseract wrapper  (needs: pip install pytesseract  +  Tesseract binary)
easyocr  — EasyOCR               (needs: pip install easyocr)
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weight_detector as _wd
from config import (
    ROI_X, ROI_Y, ROI_W, ROI_H,
    TFLITE_MODEL_PATH,
    TFLITE_INPUT_WIDTH, TFLITE_INPUT_HEIGHT,
    WEIGHT_MIN_KG, WEIGHT_MAX_KG,
)

app        = Flask(__name__)
_DEF_ROI   = (ROI_X, ROI_Y, ROI_W, ROI_H)
_EASYOCR_READER = None   # lazy-init, expensive to load


# ── Backend probe (called once at startup + exposed via /backends) ────────────

def _probe_backends() -> dict[str, dict]:
    """
    Returns a dict keyed by backend name with keys:
        available: bool
        note:      str  (install hint or version)
    """
    results: dict[str, dict] = {}

    # ── TFLite ────────────────────────────────────────────────────────────────
    try:
        try:
            import tflite_runtime.interpreter as _tfl  # noqa: F401
            src = "tflite_runtime"
        except ImportError:
            import tensorflow as _tf  # noqa: F401
            src = "tensorflow"

        if not os.path.isfile(TFLITE_MODEL_PATH):
            results["tflite"] = {
                "available": False,
                "note": "model_float16.tflite not found in vision_node/",
            }
        else:
            # Reset singleton so we always get a fresh load here
            _wd._interp = _wd._input_idx = _wd._output_idx = None
            try:
                from weight_detector import _get_interpreter
                interp, in_idx, _ = _get_interpreter()
                # allocate_tensors() may silently fail; invoke() is where
                # FlexMatMul errors actually raise in Python
                dummy = np.zeros(
                    (1, TFLITE_INPUT_HEIGHT, TFLITE_INPUT_WIDTH, 1), dtype=np.float32
                )
                interp.set_tensor(in_idx, dummy)
                interp.invoke()
                results["tflite"] = {"available": True, "note": f"OK via {src}"}
            except Exception as e:
                # Reset broken singleton so detect calls don't reuse it
                _wd._interp = _wd._input_idx = _wd._output_idx = None
                err = str(e)
                if "Flex" in err or "not supported" in err.lower():
                    results["tflite"] = {
                        "available": False,
                        "note": "FlexMatMul op not supported — pip install tensorflow or use another backend",
                    }
                else:
                    results["tflite"] = {"available": False, "note": err[:160]}
    except ImportError:
        results["tflite"] = {
            "available": False,
            "note": "pip install tflite-runtime  OR  pip install tensorflow",
        }

    # ── Tesseract ─────────────────────────────────────────────────────────────
    try:
        import pytesseract  # noqa: F401
        ver = pytesseract.get_tesseract_version()
        results["tesseract"] = {"available": True, "note": f"Tesseract {ver}"}
    except ImportError:
        results["tesseract"] = {
            "available": False,
            "note": "pip install pytesseract  +  install Tesseract binary",
        }
    except Exception as exc:
        results["tesseract"] = {
            "available": False,
            "note": f"pytesseract installed but Tesseract binary missing: {exc}",
        }

    # ── EasyOCR ───────────────────────────────────────────────────────────────
    # Probe in a subprocess: EasyOCR imports PyTorch which may trigger SIGILL
    # (Illegal instruction) on RPi if the wheel was built for a newer CPU.
    # A subprocess crash only kills the child, not this server process.
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import easyocr"],
            capture_output=True, text=True, timeout=60,
        )
        if probe.returncode == 0:
            results["easyocr"] = {"available": True, "note": "easyocr installed"}
        else:
            stderr = (probe.stderr or "").strip()
            if "No module named" in stderr or "ModuleNotFoundError" in stderr:
                note = "pip install easyocr"
            elif probe.returncode == -4 or "Illegal instruction" in stderr:
                note = "PyTorch SIGILL — CPU lacks required instructions (AVX2/NEON); not usable on this device"
            else:
                note = f"import failed (exit {probe.returncode}): {stderr[:120]}"
            results["easyocr"] = {"available": False, "note": note}
    except subprocess.TimeoutExpired:
        results["easyocr"] = {"available": False, "note": "import timed out (>60 s)"}
    except Exception as exc:
        results["easyocr"] = {"available": False, "note": str(exc)[:120]}

    # ── SSOCR ─────────────────────────────────────────────────────────────────
    try:
        subprocess.run(
            ["ssocr", "--help"], capture_output=True, text=True, timeout=5
        )
        # ssocr prints help to stderr and exits non-zero — that still means it's present
        results["ssocr"] = {"available": True, "note": "ssocr binary found"}
    except FileNotFoundError:
        results["ssocr"] = {
            "available": False,
            "note": "not found — on RPi: sudo apt install ssocr  "
                    "OR  cd ssocr && make && sudo make install",
        }
    except Exception as exc:
        results["ssocr"] = {"available": False, "note": str(exc)[:120]}

    return results


_BACKENDS: dict[str, dict] = {}   # populated at startup


# ── OCR helpers ───────────────────────────────────────────────────────────────

def _run_tflite(frame: np.ndarray, rx: int, ry: int, rw: int, rh: int,
                debug_path: str | None) -> tuple[float | None, str]:
    """Returns (weight, raw_text).  Re-raises RuntimeError so the caller surfaces it."""
    from weight_detector import extract_weight
    _wd.ROI_X, _wd.ROI_Y, _wd.ROI_W, _wd.ROI_H = rx, ry, rw, rh
    w: float | None = None
    raw_text: str   = ""
    try:
        w = extract_weight(frame, debug_image_path=debug_path)
        # Get the raw model text (before weight-range filtering) for display
        roi     = frame[ry:ry + rh, rx:rx + rw]
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (TFLITE_INPUT_WIDTH, TFLITE_INPUT_HEIGHT))
        normed  = resized.astype(np.float32) / 255.0
        tensor  = normed[np.newaxis, :, :, np.newaxis]
        decoded = _wd._run_tflite(tensor)          # type: ignore[attr-defined]
        raw_text = _wd._decode(decoded) if decoded is not None else ""  # type: ignore[attr-defined]
    except RuntimeError:
        raise   # FlexMatMul / model-not-found — let the route handler return the error
    except Exception:
        pass    # decode / resize failures are non-fatal
    finally:
        _wd.ROI_X, _wd.ROI_Y, _wd.ROI_W, _wd.ROI_H = _DEF_ROI
    return w, raw_text


def _run_tesseract(frame: np.ndarray, rx: int, ry: int, rw: int, rh: int,
                   debug_path: str | None) -> tuple[float | None, str]:
    import pytesseract

    roi  = frame[ry:ry + rh, rx:rx + rw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # CLAHE — normalise contrast across the digit strip
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Upscale to a fixed 600-px width so Tesseract always sees large digits
    scale = max(2, 600 // max(roi.shape[1], 1))
    big   = cv2.resize(enhanced, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)

    # Bilateral filter — removes sensor noise while keeping digit edges sharp
    big = cv2.bilateralFilter(big, d=5, sigmaColor=75, sigmaSpace=75)

    # Otsu threshold — binarise
    _, th = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing — fills small gaps in seven-segment strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    if debug_path:
        debug_small = cv2.resize(th, (200, 31), interpolation=cv2.INTER_AREA)
        cv2.imwrite(debug_path, debug_small)

    raw = pytesseract.image_to_string(
        th,
        config="--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789.",
    ).strip()

    w = _parse_float(raw)
    return w, raw


def _run_easyocr(frame: np.ndarray, rx: int, ry: int, rw: int, rh: int,
                 debug_path: str | None) -> tuple[float | None, str]:
    global _EASYOCR_READER
    if not _BACKENDS.get("easyocr", {}).get("available"):
        raise RuntimeError(_BACKENDS.get("easyocr", {}).get("note", "easyocr unavailable"))
    import easyocr

    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)

    roi  = frame[ry:ry + rh, rx:rx + rw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # CLAHE — normalise contrast before upscaling
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # 3× upscale — EasyOCR works better on larger images
    big = cv2.resize(enhanced, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Bilateral filter — suppress noise, keep digit strokes sharp
    big = cv2.bilateralFilter(big, d=5, sigmaColor=75, sigmaSpace=75)

    if debug_path:
        debug_small = cv2.resize(big, (200, 31), interpolation=cv2.INTER_AREA)
        cv2.imwrite(debug_path, debug_small)

    detections = _EASYOCR_READER.readtext(big, allowlist="0123456789.", detail=1)
    if not detections:
        return None, ""

    detections.sort(key=lambda d: d[2], reverse=True)
    raw = detections[0][1]
    w   = _parse_float(raw)
    return w, raw


def _run_ssocr(frame: np.ndarray, rx: int, ry: int, rw: int, rh: int,
               debug_path: str | None) -> tuple[float | None, str]:
    """
    Use the SSOCR C binary — a lightweight, heuristic seven-segment OCR
    purpose-built for digit displays.  On Raspberry Pi:
        sudo apt install ssocr
    OR compile from the ssocr/ source tree in this repo.
    """
    roi  = frame[ry:ry + rh, rx:rx + rw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # CLAHE before handing to ssocr
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, enhanced)

    if debug_path:
        cv2.imwrite(debug_path, cv2.resize(enhanced, (200, 31),
                                           interpolation=cv2.INTER_AREA))

    raw = ""
    try:
        result = subprocess.run(
            [
                "ssocr",
                "-d", "4",            # expect 4 digits
                "--number-pixels=2",  # min 2 lit pixels per segment
                tmp_path,
            ],
            capture_output=True, text=True, timeout=5,
        )
        raw = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    w = _parse_float(raw)
    return w, raw


def _parse_float(raw: str) -> float | None:
    """Parse a numeric string and validate against weight range.

    The scale display has a fixed decimal point before the last (rightmost)
    digit — e.g. the display '4825' means 482.5 kg.  We always strip any
    punctuation, keep only the digit characters, then insert the decimal
    before the last digit regardless of what the OCR produced.
    """
    digits = re.sub(r"[^\d]", "", raw)
    if len(digits) < 2:
        return None
    cleaned = digits[:-1] + '.' + digits[-1]
    try:
        v = float(cleaned)
    except ValueError:
        return None
    return v if WEIGHT_MIN_KG <= v <= WEIGHT_MAX_KG else None


# ── Annotation ────────────────────────────────────────────────────────────────

def _annotate(frame: np.ndarray, weight: float | None,
              rx: int, ry: int, rw: int, rh: int) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    cv2.putText(out, "ROI", (rx + 4, ry + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    label = f"{weight:.1f} kg" if weight is not None else "NO READ"
    color = (0, 230, 0) if weight is not None else (0, 0, 220)
    y0 = max(ry - 12, 30)
    cv2.putText(out, label, (rx + 2, y0 + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (rx, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
    return out


def _encode_jpeg(img: np.ndarray, q: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _encode_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return Response(_HTML, mimetype="text/html")


@app.route("/backends")
def backends():
    return jsonify(_BACKENDS)


@app.route("/config")
def config_info():
    return jsonify(
        roi=dict(x=_DEF_ROI[0], y=_DEF_ROI[1], w=_DEF_ROI[2], h=_DEF_ROI[3]),
        weight_range=dict(min=WEIGHT_MIN_KG, max=WEIGHT_MAX_KG),
        model_input=dict(w=TFLITE_INPUT_WIDTH, h=TFLITE_INPUT_HEIGHT),
    )


def auto_detect_roi(frame: np.ndarray) -> tuple[int, int, int, int]:
    """
    Auto-detect the LCD/7-segment display ROI using edge density.
    Returns (x, y, w, h). Falls back to full frame if detection fails.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_img, w_img = frame.shape[:2]

    # Canny edges + horizontal dilation to merge digits within the display strip
    edges = cv2.Canny(gray, 20, 80)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    dilated = cv2.dilate(edges, kernel_h)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
    dilated = cv2.dilate(dilated, kernel_v)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0
        area = w * h
        # 7-segment displays: wide & narrow, not too small, not too large, not at border
        if (2.0 < aspect < 16.0
                and 0.004 * w_img * h_img < area < 0.5 * w_img * h_img
                and x > 2 and y > 2
                and x + w < w_img - 2 and y + h < h_img - 2):
            # Score by edge density inside the candidate region
            roi_edges = edges[y:y + h, x:x + w]
            density = np.count_nonzero(roi_edges) / float(area)
            candidates.append((x, y, w, h, density))

    if candidates:
        # Prefer the region with highest edge density (most digit strokes)
        x, y, w, h, _ = max(candidates, key=lambda b: b[4])
        # Add 8% padding so clipped display edges are included
        px, py = max(1, int(w * 0.08)), max(1, int(h * 0.08))
        x = max(0, x - px);       y = max(0, y - py)
        w = min(w_img - x, w + 2 * px)
        h = min(h_img - y, h + 2 * py)
        return (x, y, w, h)

    # Fallback: full frame
    return (0, 0, w_img, h_img)


@app.route("/detect", methods=["POST"])
def detect():
  """
  POST /detect
  Form fields:
    image            — image file (multipart)
    roi_x/y/w/h     — ints (optional, default from config.yaml)
    backend          — 'tflite' | 'tesseract' | 'easyocr'  (default: tflite)
  Returns JSON:
    weight, raw_text, annotated (b64 JPEG), debug_crop (b64 PNG),
    roi, image_size, backend, error
  """
  if "image" not in request.files:
    return jsonify(error="No image uploaded"), 400

  data = request.files["image"].read()
  if not data:
    return jsonify(error="Empty file"), 400

  arr   = np.frombuffer(data, dtype=np.uint8)
  frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
  if frame is None:
    return jsonify(error="Cannot decode image — unsupported format?"), 400

  h_img, w_img = frame.shape[:2]

  # Try to auto-detect ROI
  rx, ry, rw, rh = auto_detect_roi(frame)

  backend = request.form.get("backend", "tflite")

  if not _BACKENDS.get(backend, {}).get("available"):
    note = _BACKENDS.get(backend, {}).get("note", "Backend not available")
    return jsonify(error=f"Backend '{backend}' unavailable: {note}"), 400

  with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    debug_path = tmp.name

  try:
    if backend == "tflite":
      weight, raw_text = _run_tflite(frame, rx, ry, rw, rh, debug_path)
    elif backend == "tesseract":
      weight, raw_text = _run_tesseract(frame, rx, ry, rw, rh, debug_path)
    elif backend == "easyocr":
      weight, raw_text = _run_easyocr(frame, rx, ry, rw, rh, debug_path)
    elif backend == "ssocr":
      weight, raw_text = _run_ssocr(frame, rx, ry, rw, rh, debug_path)
    else:
      return jsonify(error=f"Unknown backend: {backend}"), 400
  except Exception as exc:
    return jsonify(error=str(exc)), 500

  annotated = _annotate(frame, weight, rx, ry, rw, rh)
  ann_b64   = _encode_jpeg(annotated)

  crop_b64 = ""
  if os.path.exists(debug_path):
    crop_img = cv2.imread(debug_path, cv2.IMREAD_GRAYSCALE)
    if crop_img is not None:
      scale    = 3
      crop_big = cv2.resize(crop_img,
                  (crop_img.shape[1] * scale, crop_img.shape[0] * scale),
                  interpolation=cv2.INTER_NEAREST)
      crop_b64 = _encode_png(crop_big)
    try:
      os.unlink(debug_path)
    except OSError:
      pass

  return jsonify(
    weight=weight,
    raw_text=raw_text,
    annotated=ann_b64,
    debug_crop=crop_b64,
    roi=dict(x=rx, y=ry, w=rw, h=rh),
    image_size=dict(w=w_img, h=h_img),
    backend=backend,
    error=None,
  )


# ── HTML ──────────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WeightMate — Image Tester</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:      #0f1117;
    --surface: #1a1d27;
    --card:    #21253a;
    --border:  #2e3450;
    --accent:  #4f8ef7;
    --green:   #22c55e;
    --red:     #ef4444;
    --yellow:  #f59e0b;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --radius:  10px;
  }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; display: flex; flex-direction: column;
  }
  header {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 14px 24px; display: flex; align-items: center; gap: 14px;
  }
  header h1 { font-size: 1.15rem; font-weight: 600; }
  header .badge {
    background: var(--accent); color: #fff; font-size: .68rem;
    padding: 2px 8px; border-radius: 99px; font-weight: 700;
  }
  header .sub { color: var(--muted); font-size: .8rem; margin-left: auto; }
  main {
    flex: 1; display: grid;
    grid-template-columns: 340px 1fr; gap: 0;
  }
  .panel {
    background: var(--surface); border-right: 1px solid var(--border);
    padding: 20px; display: flex; flex-direction: column; gap: 18px;
    overflow-y: auto;
  }
  .section-title {
    font-size: .7rem; font-weight: 700; letter-spacing: .1em;
    color: var(--muted); text-transform: uppercase; margin-bottom: 8px;
  }

  /* Drop zone */
  #drop-zone {
    border: 2px dashed var(--border); border-radius: var(--radius);
    padding: 28px 16px; text-align: center; cursor: pointer;
    transition: border-color .2s, background .2s;
  }
  #drop-zone:hover, #drop-zone.drag-over {
    border-color: var(--accent); background: rgba(79,142,247,.06);
  }
  #drop-zone .icon { font-size: 2rem; margin-bottom: 8px; }
  #drop-zone p { font-size: .82rem; color: var(--muted); line-height: 1.5; }
  #drop-zone p strong { color: var(--text); }
  #file-input { display: none; }
  #thumb-wrap { display: none; position: relative; }
  #thumb-wrap img {
    width: 100%; border-radius: 6px; border: 1px solid var(--border); display: block;
  }
  #thumb-wrap .change-btn {
    position: absolute; top: 6px; right: 6px;
    background: rgba(0,0,0,.65); color: #fff; border: none;
    border-radius: 6px; padding: 4px 10px; font-size: .72rem; cursor: pointer;
  }

  /* Backend pills */
  .backend-grid { display: flex; flex-direction: column; gap: 8px; }
  .backend-pill {
    display: flex; align-items: center; gap: 10px;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 12px; cursor: pointer;
    transition: border-color .15s;
  }
  .backend-pill:hover { border-color: var(--accent); }
  .backend-pill input[type=radio] { accent-color: var(--accent); margin: 0; }
  .backend-pill.selected { border-color: var(--accent); background: rgba(79,142,247,.08); }
  .backend-pill.unavailable { opacity: .5; cursor: not-allowed; }
  .bp-info { flex: 1; }
  .bp-name { font-size: .84rem; font-weight: 600; }
  .bp-note { font-size: .71rem; color: var(--muted); margin-top: 2px; }
  .bp-badge {
    font-size: .65rem; font-weight: 700; padding: 2px 7px;
    border-radius: 99px; letter-spacing: .04em;
  }
  .bp-badge.ok  { background: rgba(34,197,94,.15); color: var(--green); }
  .bp-badge.err { background: rgba(239,68,68,.12);  color: var(--red); }

  /* ROI */
  .roi-grid {
    display: grid; grid-template-columns: 36px 1fr 50px;
    gap: 6px 8px; align-items: center;
  }
  .roi-grid label { font-size: .78rem; color: var(--muted); font-weight: 600; }
  .roi-grid input[type=range] { width: 100%; accent-color: var(--accent); }
  .roi-grid .val { font-size: .78rem; text-align: right; font-variant-numeric: tabular-nums; }
  .roi-hint { font-size: .72rem; color: var(--muted); margin-top: -4px; line-height: 1.5; }

  /* Button */
  #detect-btn {
    background: var(--accent); color: #fff; border: none;
    border-radius: var(--radius); padding: 12px; font-size: .95rem;
    font-weight: 600; cursor: pointer; width: 100%;
    transition: opacity .15s, transform .1s;
  }
  #detect-btn:hover:not(:disabled) { opacity: .88; }
  #detect-btn:active:not(:disabled) { transform: scale(.98); }
  #detect-btn:disabled { opacity: .4; cursor: not-allowed; }

  /* Error */
  #error-bar {
    display: none; background: #7f1d1d; border: 1px solid var(--red);
    border-radius: 8px; padding: 10px 16px; font-size: .82rem; color: #fca5a5;
  }

  /* History */
  #history-list {
    display: flex; flex-direction: column; gap: 6px;
    max-height: 220px; overflow-y: auto;
  }
  .hist-item {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 10px;
    display: flex; justify-content: space-between; align-items: center;
    gap: 8px; cursor: pointer; font-size: .8rem;
    transition: border-color .15s;
  }
  .hist-item:hover { border-color: var(--accent); }
  .hist-name { color: var(--muted); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .hist-weight { font-weight: 700; }
  .hist-weight.ok  { color: var(--green); }
  .hist-weight.bad { color: var(--red); }
  .hist-backend { font-size: .65rem; color: var(--muted); text-transform: uppercase; }
  .hist-clear { font-size: .72rem; color: var(--muted); cursor: pointer; text-align: right; }
  .hist-clear:hover { color: var(--red); }

  /* Result area */
  .result-area {
    padding: 20px; display: flex; flex-direction: column;
    gap: 18px; overflow-y: auto;
  }
  .weight-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px 24px;
    display: flex; align-items: center; gap: 24px;
  }
  .weight-num {
    font-size: 3rem; font-weight: 800;
    font-variant-numeric: tabular-nums; line-height: 1;
  }
  .weight-num.ok   { color: var(--green); }
  .weight-num.bad  { color: var(--red); }
  .weight-num.idle { color: var(--muted); }
  .weight-meta { font-size: .8rem; color: var(--muted); line-height: 1.9; }
  .weight-meta strong { color: var(--text); }

  .img-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 900px) { .img-row { grid-template-columns: 1fr; } }
  .img-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
  }
  .img-card-header {
    padding: 10px 14px; font-size: .75rem; font-weight: 700;
    letter-spacing: .07em; text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }
  .img-card img { width: 100%; display: block; background: #000; }
  .img-placeholder {
    height: 200px; display: flex; align-items: center;
    justify-content: center; color: var(--muted); font-size: .82rem;
  }
  #debug-card { display: none; }
  #debug-img { image-rendering: pixelated; width: 100%; background: #000; }

  /* Spinner */
  .spinner {
    width: 16px; height: 16px; border: 2px solid rgba(255,255,255,.3);
    border-top-color: #fff; border-radius: 50%;
    animation: spin .7s linear infinite;
    display: inline-block; vertical-align: middle; margin-right: 6px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header>
  <div style="font-size:1.4rem">⚖️</div>
  <h1>WeightMate — Image Tester</h1>
  <span class="badge">DEV</span>
  <span class="sub">Upload · Pick backend · Tune ROI</span>
</header>

<main>
<!-- ── Left panel ── -->
<div class="panel">

  <!-- Image upload -->
  <div>
    <div class="section-title">Image</div>
    <div id="drop-zone">
      <div class="icon">🖼️</div>
      <p><strong>Click or drag &amp; drop</strong><br>JPG · PNG · BMP · WEBP</p>
    </div>
    <div id="thumb-wrap">
      <img id="thumb" src="" alt="preview">
      <button class="change-btn" id="change-btn">Change</button>
    </div>
    <input type="file" id="file-input" accept="image/*">
  </div>

  <!-- Backend selector -->
  <div>
    <div class="section-title">OCR Backend</div>
    <div class="backend-grid" id="backend-grid">
      <div style="color:var(--muted);font-size:.8rem">Loading…</div>
    </div>
  </div>

  <!-- ROI — auto-detected per image -->
  <div>
    <div class="section-title">Region of Interest (ROI)</div>
    <p class="roi-hint">Detected automatically from each uploaded image.</p>
    <div id="roi-detected" style="
        margin-top:8px; padding:8px 10px;
        background:var(--card); border:1px solid var(--border);
        border-radius:6px; font-size:.8rem;
        font-variant-numeric:tabular-nums; color:var(--muted);">
      Run detection to see the auto-detected ROI.
    </div>
  </div>

  <!-- Detect -->
  <button id="detect-btn" disabled>Detect Weight</button>
  <div id="error-bar"></div>

  <!-- History -->
  <div>
    <div class="section-title" style="display:flex;justify-content:space-between">
      <span>History</span>
      <span class="hist-clear" id="hist-clear">Clear</span>
    </div>
    <div id="history-list">
      <div style="color:var(--muted);font-size:.8rem">No detections yet</div>
    </div>
  </div>

</div><!-- /panel -->

<!-- ── Result area ── -->
<div class="result-area">

  <div class="weight-card">
    <div class="weight-num idle" id="weight-num">—</div>
    <div class="weight-meta" id="weight-meta">
      Upload an image and click <strong>Detect Weight</strong>.
    </div>
  </div>

  <div class="img-row">
    <div class="img-card">
      <div class="img-card-header">
        <span>Annotated Frame</span>
        <span id="img-size-label" style="font-weight:400"></span>
      </div>
      <div id="ann-wrap">
        <div class="img-placeholder">Result will appear here</div>
      </div>
    </div>

    <div class="img-card" id="debug-card">
      <div class="img-card-header">
        <span>Preprocessed ROI</span>
        <span style="font-weight:400;color:var(--text)">model input</span>
      </div>
      <img id="debug-img" src="" alt="debug crop">
    </div>
  </div>

</div><!-- /result-area -->
</main>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
let currentFile   = null;
let history       = [];
let backendsInfo  = {};
let selectedBackend = null;

// ── Boot ──────────────────────────────────────────────────────────────────────
fetch('/backends').then(r => r.json()).then(bk => {
  backendsInfo = bk;
  renderBackends(bk);
});

// ── Backends ──────────────────────────────────────────────────────────────────
const BACKEND_LABELS = {
  tflite:    { name: 'TFLite',    desc: 'model_float16.tflite (seven-segment-ocr)' },
  tesseract: { name: 'Tesseract', desc: 'pytesseract + Tesseract binary' },
  easyocr:   { name: 'EasyOCR',  desc: 'pip install easyocr (no binary needed)' },
  ssocr:     { name: 'SSOCR',    desc: 'Lightweight C seven-segment OCR (best for RPi)' },
};

function renderBackends(bk) {
  const grid = document.getElementById('backend-grid');
  grid.innerHTML = '';

  Object.entries(bk).forEach(([id, info]) => {
    const avail = info.available;
    const meta  = BACKEND_LABELS[id] || { name: id, desc: '' };
    const pill  = document.createElement('label');
    pill.className = 'backend-pill' + (avail ? '' : ' unavailable');
    pill.innerHTML = `
      <input type="radio" name="backend" value="${id}" ${avail ? '' : 'disabled'}>
      <div class="bp-info">
        <div class="bp-name">${meta.name}</div>
        <div class="bp-note">${avail ? meta.desc : info.note}</div>
      </div>
      <span class="bp-badge ${avail ? 'ok' : 'err'}">${avail ? 'READY' : 'UNAVAIL'}</span>
    `;
    grid.appendChild(pill);

    // Auto-select first available
    const radio = pill.querySelector('input');
    radio.addEventListener('change', () => {
      document.querySelectorAll('.backend-pill').forEach(p => p.classList.remove('selected'));
      pill.classList.add('selected');
      selectedBackend = id;
    });
    if (avail && selectedBackend === null) {
      radio.checked = true;
      pill.classList.add('selected');
      selectedBackend = id;
    }
  });
}

// ── ROI display ───────────────────────────────────────────────────────────────
function showRoi(roi) {
  document.getElementById('roi-detected').textContent =
    `x=${roi.x}  y=${roi.y}  w=${roi.w}  h=${roi.h}`;
  document.getElementById('roi-detected').style.color = 'var(--text)';
}

// ── File handling ─────────────────────────────────────────────────────────────
const dropZone  = document.getElementById('drop-zone');
const thumbWrap = document.getElementById('thumb-wrap');
const thumb     = document.getElementById('thumb');
const fileInput = document.getElementById('file-input');
const detectBtn = document.getElementById('detect-btn');

dropZone.addEventListener('click',     () => fileInput.click());
document.getElementById('change-btn').addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) loadFile(fileInput.files[0]); });

function loadFile(file) {
  currentFile = file;
  const url = URL.createObjectURL(file);
  thumb.src = url;
  const img = new Image();
  img.onload = () => {
    document.getElementById('img-size-label').textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
  };
  img.src = url;
  dropZone.style.display  = 'none';
  thumbWrap.style.display = 'block';
  detectBtn.disabled = false;
  document.getElementById('roi-detected').textContent = 'Run detection to see the auto-detected ROI.';
  document.getElementById('roi-detected').style.color = 'var(--muted)';
}

// ── Detect ────────────────────────────────────────────────────────────────────
detectBtn.addEventListener('click', runDetect);

async function runDetect() {
  if (!currentFile || !selectedBackend) return;
  detectBtn.disabled = true;
  detectBtn.innerHTML = '<span class="spinner"></span>Running…';
  showError(null);

  const fd = new FormData();
  fd.append('image',   currentFile);
  fd.append('backend', selectedBackend);

  try {
    const res  = await fetch('/detect', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { showError(data.error); return; }
    renderResult(data);
    addHistory(currentFile.name, data);
  } catch (err) {
    showError('Network error: ' + err.message);
  } finally {
    detectBtn.disabled = false;
    detectBtn.textContent = 'Detect Weight';
  }
}

function renderResult(data) {
  const num  = document.getElementById('weight-num');
  const meta = document.getElementById('weight-meta');

  showRoi(data.roi);

  if (data.weight !== null) {
    num.textContent = data.weight.toFixed(1) + ' kg';
    num.className   = 'weight-num ok';
    meta.innerHTML  =
      `Backend: <strong>${data.backend}</strong><br>` +
      `Raw text: <strong>"${data.raw_text || '—'}"</strong>`;
  } else {
    num.textContent = 'NO READ';
    num.className   = 'weight-num bad';
    meta.innerHTML  =
      `Backend: <strong>${data.backend}</strong><br>` +
      `Raw text: <strong>"${data.raw_text || '—'}"</strong><br>` +
      `ROI auto-detected — check the annotated frame.`;
  }

  document.getElementById('ann-wrap').innerHTML =
    `<img src="data:image/jpeg;base64,${data.annotated}" alt="annotated">`;

  const debugCard = document.getElementById('debug-card');
  const debugImg  = document.getElementById('debug-img');
  if (data.debug_crop) {
    debugImg.src = 'data:image/png;base64,' + data.debug_crop;
    debugCard.style.display = 'block';
  }
}

// ── History ───────────────────────────────────────────────────────────────────
function addHistory(name, data) {
  history.unshift({ name, data });
  renderHistory();
}
function renderHistory() {
  const list = document.getElementById('history-list');
  if (!history.length) {
    list.innerHTML = '<div style="color:var(--muted);font-size:.8rem">No detections yet</div>';
    return;
  }
  list.innerHTML = history.map((item, i) => {
    const w   = item.data.weight;
    const tag = w !== null ? w.toFixed(1) + ' kg' : 'NO READ';
    const cls = w !== null ? 'ok' : 'bad';
    const n   = item.name.length > 20 ? '…' + item.name.slice(-18) : item.name;
    return `<div class="hist-item" onclick="replayHistory(${i})">
      <span class="hist-name" title="${item.name}">${n}</span>
      <span class="hist-backend">${item.data.backend}</span>
      <span class="hist-weight ${cls}">${tag}</span>
    </div>`;
  }).join('');
}
function replayHistory(i) { renderResult(history[i].data); }
document.getElementById('hist-clear').addEventListener('click', () => {
  history = []; renderHistory();
});

// ── Error ──────────────────────────────────────────────────────────────────────
function showError(msg) {
  const bar = document.getElementById('error-bar');
  bar.textContent   = msg ? '⚠ ' + msg : '';
  bar.style.display = msg ? 'block' : 'none';
}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="WeightMate image-tester web dashboard")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="127.0.0.1",
                   help="Use 0.0.0.0 to expose on the network")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\n  WeightMate Image Tester")
    print("  ─────────────────────────────────────────")
    print("  Probing backends…")
    _BACKENDS.update(_probe_backends())
    for name, info in _BACKENDS.items():
        status = "✓" if info["available"] else "✗"
        print(f"    {status} {name:<12} {info['note']}")
    print(f"\n  Open → http://{args.host}:{args.port}")
    print("  Stop → Ctrl-C\n")

    app.run(host=args.host, port=args.port, debug=False)
