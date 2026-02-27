"""
weight_detector.py

Extracts the weight reading from the scale display image using the
renjithsasidharan/seven-segment-ocr TFLite model.

Model:   model_float16.tflite  (float16-quantised CTC OCR)
Source:  https://github.com/renjithsasidharan/seven-segment-ocr
Alphabet: digits 0-9, a-z, and '.' (decimal point)

Preprocessing pipeline:
    1. Crop fixed ROI                — eliminate everything outside the display
    2. Convert to grayscale          — model expects single-channel input
    3. Resize to (200 × 31)          — model input shape: (1, 31, 200, 1)
    4. Normalise to [0, 1]           — divide by 255.0
    5. Add batch + channel dims      — (H, W) → (1, H, W, 1)
    → Run TFLite interpreter
    → Decode CTC output char indices → text string
    → Parse & validate weight

Install the runtime (choose one):
    pip install tflite-runtime          # lightweight, Raspberry Pi / Linux
    pip install tensorflow              # full TF, dev machines
"""

from __future__ import annotations

import logging
import os
import string
from typing import Optional

import cv2
import numpy as np

from config import (
    ROI_X, ROI_Y, ROI_W, ROI_H,
    TFLITE_MODEL_PATH,
    TFLITE_INPUT_WIDTH, TFLITE_INPUT_HEIGHT,
    WEIGHT_MIN_KG, WEIGHT_MAX_KG,
)

logger = logging.getLogger(__name__)

# ─── Alphabet (must match training) ───────────────────────────────────────────
_ALPHABET  = string.digits + string.ascii_lowercase + "."
_BLANK_IDX = len(_ALPHABET)   # CTC blank token index

# ─── CLAHE instance (created once) ────────────────────────────────────────────
# Contrast Limited Adaptive Histogram Equalisation — improves digit contrast
# under varying lighting before passing the image to the model.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ─── Lazy interpreter singleton ───────────────────────────────────────────────
_interp     = None
_input_idx  = None
_output_idx = None


def _get_interpreter():
    global _interp, _input_idx, _output_idx
    if _interp is not None:
        return _interp, _input_idx, _output_idx

    if not os.path.isfile(TFLITE_MODEL_PATH):
        raise RuntimeError(
            f"TFLite model not found: {TFLITE_MODEL_PATH}\n"
            "Download model_float16.tflite from:\n"
            "  https://github.com/renjithsasidharan/seven-segment-ocr\n"
            "and place it in the vision_node/ directory."
        )

    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow as tf
            tflite = tf.lite
        except ImportError:
            raise RuntimeError(
                "Neither tflite_runtime nor tensorflow is installed.\n"
                "Install: pip install tflite-runtime"
            )

    _interp = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    _interp.allocate_tensors()
    inp  = _interp.get_input_details()
    out  = _interp.get_output_details()
    _input_idx  = inp[0]["index"]
    _output_idx = out[0]["index"]
    logger.info("TFLite model loaded: %s", TFLITE_MODEL_PATH)
    return _interp, _input_idx, _output_idx


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_weight(
    frame: np.ndarray,
    debug_image_path: Optional[str] = None,
) -> Optional[float]:
    """
    Extract weight reading from a full camera frame.

    Args:
        frame:            BGR numpy array (full camera frame).
        debug_image_path: If set, saves the preprocessed ROI image here.
                          Use during hardware setup to verify the pipeline.

    Returns:
        Weight as float (e.g., 482.5) or None if extraction failed.
    """
    roi       = _crop_roi(frame)
    processed = _preprocess(roi)

    if debug_image_path:
        try:
            # Save the resized grayscale crop for visual inspection
            cv2.imwrite(debug_image_path, (processed[0, :, :, 0] * 255).astype(np.uint8))
        except Exception as exc:
            logger.debug("Debug image write failed: %s", exc)

    raw = _run_tflite(processed)
    if raw is None:
        return None
    return _parse_weight(raw)


# ─── Preprocessing pipeline ───────────────────────────────────────────────────

def _crop_roi(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, ROI_X);          y1 = max(0, ROI_Y)
    x2 = min(w, ROI_X + ROI_W);  y2 = min(h, ROI_Y + ROI_H)
    return frame[y1:y2, x1:x2]


def _preprocess(roi: np.ndarray) -> np.ndarray:
    """
    Produce a normalised float32 tensor of shape (1, H, W, 1) ready for
    the TFLite interpreter, where H=TFLITE_INPUT_HEIGHT, W=TFLITE_INPUT_WIDTH.

    Pipeline:
        BGR → Grayscale → CLAHE (contrast enhancement) → Resize → Normalise
    CLAHE is applied on the full-size ROI before downscaling so localised
    contrast differences across the digit strip are preserved.
    """
    gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    enhanced = _CLAHE.apply(gray)                  # contrast enhancement
    resized  = cv2.resize(enhanced, (TFLITE_INPUT_WIDTH, TFLITE_INPUT_HEIGHT))
    normed   = resized.astype(np.float32) / 255.0
    return normed[np.newaxis, :, :, np.newaxis]    # (1, H, W, 1)


# ─── TFLite inference ─────────────────────────────────────────────────────────

def _run_tflite(input_data: np.ndarray) -> Optional[str]:
    """Run the TFLite model and return decoded text, or None on failure."""
    try:
        interp, in_idx, out_idx = _get_interpreter()
        interp.set_tensor(in_idx, input_data)
        interp.invoke()
        output = interp.get_tensor(out_idx)
        return _decode(output)
    except RuntimeError:
        raise   # model-not-found errors should propagate
    except Exception as exc:
        logger.debug("TFLite inference failed: %s", exc)
        return None


_N_DIGITS        = 4        # digits on the display (e.g. "4825" → 482.5)
_N_DIGIT_CLASSES = 10       # 0–9
_4DIGIT_OUT_SIZE = _N_DIGITS * _N_DIGIT_CLASSES   # 40 — output size of custom model


def _decode(output: np.ndarray) -> str:
    """
    Dispatch to the right decoder based on the model's output shape.

    • output.shape == (1, 40)  → custom 4-digit CNN  (trained with train_model.py)
    • any other shape          → original CTC model   (model_float16.tflite)
    """
    if output.shape[-1] == _4DIGIT_OUT_SIZE:
        return _decode_4digit(output)
    return _decode_ctc(output)


def _decode_ctc(output: np.ndarray) -> str:
    """
    Greedy CTC decoder.

    Standard CTC rules:
      1. Collapse consecutive repeated indices (e.g. [4,4,8,8,2] → [4,8,2]).
      2. Remove blank tokens (_BLANK_IDX) and -1 padding.

    Without step 1 the raw per-timestep output produces strings like
    '44448882255' instead of '482.5'.
    """
    prev = object()   # sentinel — never equal to any int
    chars: list[str] = []
    for idx in output[0]:
        idx = int(idx)
        if idx != prev:
            if 0 <= idx < _BLANK_IDX:
                chars.append(_ALPHABET[idx])
            prev = idx
    return "".join(chars)


def _decode_4digit(output: np.ndarray) -> str:
    """
    Decode the output of the custom 4-digit CNN (from train_model.py).

    The model outputs a flat (1, 40) tensor: four blocks of 10 softmax
    values, one block per digit position (thousands → ones-of-tenths).
    argmax over each block gives digit 0–9; concatenated → e.g. "4825".
    _parse_weight then inserts the decimal → 482.5 kg.
    """
    flat = output[0]   # shape (40,)
    return "".join(
        str(int(np.argmax(flat[i * _N_DIGIT_CLASSES:(i + 1) * _N_DIGIT_CLASSES])))
        for i in range(_N_DIGITS)
    )


# ─── Weight parsing ───────────────────────────────────────────────────────────

def _parse_weight(raw: str) -> Optional[float]:
    """
    Parse model output text to a validated float weight.

    The scale display always has the decimal point before the last digit
    (e.g. '482.5').  When the OCR misses the dot we insert it automatically.

    Args:
        raw: model output string, e.g. "482.5", "4825", or "500"

    Returns:
        Float weight or None if unparseable / outside the valid range.
    """
    try:
        # Keep only digit characters, then always insert the decimal before
        # the last (rightmost) digit — the scale display has a fixed decimal
        # position before the last digit (e.g. '4825' on the glass = 482.5 kg).
        digits = "".join(c for c in raw if c.isdigit())
        if len(digits) < 2:
            raise ValueError("too few digits")
        value = float(digits[:-1] + "." + digits[-1])
    except ValueError:
        logger.debug("Unparseable TFLite output: %r", raw)
        return None

    if not (WEIGHT_MIN_KG <= value <= WEIGHT_MAX_KG):
        logger.debug(
            "Weight %.1f kg out of range [%.0f, %.0f]",
            value, WEIGHT_MIN_KG, WEIGHT_MAX_KG,
        )
        return None

    return value
