"""
tools/train_model.py

Train a lightweight 4-digit CNN for your specific scale display and export
it as a float-16 TFLite model ready to drop onto the Raspberry Pi.

The model is a drop-in replacement for model_float16.tflite — weight_detector.py
auto-detects the output shape and picks the right decoder.

─── Workflow ─────────────────────────────────────────────────────────────────

1. Collect images ON the RPi/camera (run from vision_node/):
       python tools/collect_training_data.py --target 300

   Images land in  ~/weighmate/training_data/raw/
   Each filename encodes the label, e.g.  482.5kg_20250301_120000_000001.jpg

2. Copy the raw/ folder to your training machine (PC/laptop).

3. Train (run from vision_node/):
       python tools/train_model.py --data /path/to/raw --epochs 40 --augment

4. Copy the output model back to the RPi:
       scp vision_node/model_custom.tflite pi@<rpi-ip>:~/vision_node/

5. Update config.yaml on the RPi:
       tflite:
           model_path: "model_custom.tflite"

─── Requirements (training machine only, NOT required on RPi) ───────────────
    pip install tensorflow numpy opencv-python
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_VISION_NODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _VISION_NODE)

_DEFAULT_DATA   = os.path.expanduser("~/weighmate/training_data/raw")
_DEFAULT_OUTPUT = os.path.join(_VISION_NODE, "model_custom.tflite")

# ── Constants (must match weight_detector.py) ─────────────────────────────────
INPUT_H        = 31
INPUT_W        = 200
N_DIGITS       = 4
N_CLASSES      = 10   # digits 0-9
OUTPUT_SIZE    = N_DIGITS * N_CLASSES   # 40 flat logits

# CLAHE — same as weight_detector.py so train/infer preprocessing is identical
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ── Label extraction ───────────────────────────────────────────────────────────
_FNAME_RE = re.compile(r"^(\d+\.\d)kg_")


def _parse_label(filename: str) -> list[int] | None:
    """
    Parse '482.5kg_...' → [4, 8, 2, 5]  (four digit indices, MSB first).

    The display always shows DDD.D (3 integer digits, 1 fractional).
    value = round(weight × 10) gives a 3-4 digit integer whose digits are
    exactly the four displayed characters.
    """
    m = _FNAME_RE.match(os.path.basename(filename))
    if not m:
        return None
    try:
        value = round(float(m.group(1)) * 10)   # e.g. 482.5 → 4825
    except ValueError:
        return None
    if not (500 <= value <= 5000):               # 50.0 – 500.0 kg
        return None
    return [
        value // 1000,          # thousands  (0–5)
        (value // 100) % 10,    # hundreds
        (value // 10)  % 10,    # tens
        value          % 10,    # ones-of-tenths  (fractional digit)
    ]


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(path: str) -> np.ndarray | None:
    """Load an image and return a (INPUT_H, INPUT_W, 1) float32 array."""
    img = cv2.imread(path)
    if img is None:
        return None
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = _CLAHE.apply(gray)
    resized  = cv2.resize(enhanced, (INPUT_W, INPUT_H))
    normed   = resized.astype(np.float32) / 255.0
    return normed[:, :, np.newaxis]   # (H, W, 1)


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_dataset(
    data_dir: str,
    augment: bool = False,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Returns
        X  — float32 array  (N, H, W, 1)
        Y  — list of 4 one-hot arrays, each (N, 10), one per digit position
    """
    images: list[np.ndarray] = []
    labels: list[list[int]]  = []

    skipped = 0
    for fname in sorted(os.listdir(data_dir)):
        if "_discarded" in fname:
            continue
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(data_dir, fname)
        img  = preprocess(path)
        lbl  = _parse_label(fname)

        if img is None or lbl is None:
            skipped += 1
            continue

        images.append(img)
        labels.append(lbl)

        if augment:
            # Brightness jitter — simulates varying display luminance
            for delta in (-0.15, +0.15):
                images.append(np.clip(img + delta, 0.0, 1.0).astype(np.float32))
                labels.append(lbl)
            # Gaussian noise
            noisy = np.clip(img + np.random.normal(0, 0.04, img.shape), 0, 1)
            images.append(noisy.astype(np.float32))
            labels.append(lbl)

    if not images:
        raise RuntimeError(
            f"No usable images found in {data_dir}\n"
            "Run:  python tools/collect_training_data.py --target 300"
        )
    if skipped:
        print(f"  [skip] {skipped} files with no parseable label")

    X = np.array(images, dtype=np.float32)
    Y = [
        np.eye(N_CLASSES, dtype=np.float32)[np.array(labels)[:, i]]
        for i in range(N_DIGITS)
    ]
    return X, Y


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model():
    """
    Lightweight CNN, ~100-150 KB when float-16 quantised.

    Input : (batch, 31, 200, 1)  — same size as original model
    Output: (batch, 40)          — 4 × 10 softmax logits (concatenated)

    weight_detector._decode_4digit() reads this output.
    """
    import tensorflow as tf

    inp = tf.keras.Input(shape=(INPUT_H, INPUT_W, 1), name="image")

    # ── Feature extraction ───────────────────────────────────────────────────
    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)          # 15 × 100 × 16

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)          # 7 × 50 × 32

    x = tf.keras.layers.Conv2D(48, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)       # 48

    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # ── 4 digit heads ────────────────────────────────────────────────────────
    heads = [
        tf.keras.layers.Dense(N_CLASSES, activation="softmax", name=f"d{i}")(x)
        for i in range(N_DIGITS)
    ]
    # Concatenate so the model has a single (batch, 40) output tensor.
    # weight_detector._decode_4digit detects this shape automatically.
    out = tf.keras.layers.Concatenate(name="digits")(heads)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _split_Y_for_loss(Y: list[np.ndarray]) -> np.ndarray:
    """
    Keras expects a single target array when the model has one output.
    Concatenate the 4 one-hot arrays to match the (N, 40) output.
    """
    return np.concatenate(Y, axis=1)


# ── TFLite export ──────────────────────────────────────────────────────────────

def export_tflite(keras_model, output_path: str) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_bytes)

    kb = os.path.getsize(output_path) / 1024
    print(f"  → {output_path}  ({kb:.0f} KB)")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train lightweight 4-digit scale OCR model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data",    default=_DEFAULT_DATA,   help="Path to raw image directory")
    p.add_argument("--output",  default=_DEFAULT_OUTPUT, help="Output .tflite path")
    p.add_argument("--epochs",  type=int, default=40)
    p.add_argument("--batch",   type=int, default=32)
    p.add_argument("--augment", action="store_true",
                   help="Brightness + noise augmentation (3× dataset size)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading images from  {args.data} …")
    X, Y = load_dataset(args.data, augment=args.augment)
    n = len(X)
    print(f"      {n} images  (augment={'on' if args.augment else 'off'})")
    if n < 100:
        print(f"  [WARN] Only {n} images — collect ≥200 for reliable accuracy.")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    rng   = np.random.default_rng(42)
    idx   = rng.permutation(n)
    split = max(1, int(0.85 * n))

    X_tr, X_va = X[idx[:split]], X[idx[split:]]
    Y_cat       = _split_Y_for_loss(Y)
    Y_tr, Y_va  = Y_cat[idx[:split]], Y_cat[idx[split:]]
    print(f"      train={len(X_tr)}  val={len(X_va)}")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    import tensorflow as tf

    print(f"\n[2/4] Building model …")
    model = build_model()
    model.summary(line_length=65, print_fn=lambda s: print(" ", s))

    print(f"\n[3/4] Training for up to {args.epochs} epochs …")
    model.fit(
        X_tr, Y_tr,
        validation_data=(X_va, Y_va),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5, factor=0.5, verbose=1
            ),
        ],
        verbose=2,
    )

    # ── 4. Export ─────────────────────────────────────────────────────────────
    print("\n[4/4] Exporting to TFLite float-16 …")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_tflite(model, args.output)

    print("\nDone!  Next steps:")
    print(f"  scp {args.output} pi@<rpi-ip>:~/vision_node/")
    print("  Then in config.yaml on the RPi:")
    print(f'      tflite:\n          model_path: "{os.path.basename(args.output)}"')


if __name__ == "__main__":
    main()
