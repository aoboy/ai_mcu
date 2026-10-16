#!/usr/bin/env python3
"""generate_tf_mcu_model.py
================================
End-to-end pipeline: train -> save -> int8 quantize -> validate -> op inventory.

Each original step script in this folder became a function:

    sine_model.py  ->  sine_model_step()      (helpers: generate_training_data,
                                               build_model, train_model,
                                               save_keras, export_saved_model)
    quantize.py    ->  quantize_step()        (helpers: write_calibration_files,
                                               make_representative_dataset,
                                               quantize_model)
    validate.py    ->  validate_step()        (helpers: quantize/dequantize tensors,
                                               compute_metrics)
    inventory.py   ->  inventory_step()       (helpers: none)

main() runs the four steps in order.

Usage
-----
    # Full pipeline on the sine toy model (trains everything from scratch):
    python generate_tf_mcu_model.py

    # Quantize an existing Keras model (needs an existing calibration dir):
    python generate_tf_mcu_model.py --model kws.keras --calib-dir calib

    # Keep float32 I/O instead of int8 I/O:
    python generate_tf_mcu_model.py --float-io

Why full int8 on an MCU
-----------------------
Weights and activations in int8 -> ~4x smaller flash/RAM traffic, integer-only
kernels (no FPU required), and int8 tensors at the boundary mean the model has
no Quantize/Dequantize nodes, so the input feed and output read are raw bytes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
DEFAULT_KERAS_PATH = "sine_model.keras"   # .keras  -> reloading/converting in Python
DEFAULT_SAVEDMODEL_DIR = "sine_model"     # no ext  -> SavedModel (TFLite converter, serving)
DEFAULT_CALIB_DIR = "calib"               # one .npy per calibration sample
DEFAULT_TFLITE_PATH = "model_int8.tflite"  # the artifact that ships to the MCU
DEFAULT_CALIB_SAMPLES = 500               # ~300-500 is usually enough for PTQ
DEFAULT_EPOCHS = 200
DEFAULT_SAMPLES = 1000


# ============================================================================
# Step 1 — sine_model.py : build + train + export the float model
# ============================================================================
def generate_training_data(num_samples: int = DEFAULT_SAMPLES):
    """Return (x, y) for y = sin(x) over [0, 2*pi], float32."""
    x = np.linspace(0, 2 * np.pi, num_samples, dtype=np.float32)
    y = np.sin(x)
    return x, y


def build_model(hidden_units: int = 16, input_shape=(1,)) -> tf.keras.Model:
    """A tiny 2-layer MLP sized for an MCU (Keras 3 style: explicit Input)."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Dense(hidden_units, activation="relu"),
        tf.keras.layers.Dense(hidden_units, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_model(model: tf.keras.Model, x, y, epochs: int = DEFAULT_EPOCHS,
                batch_size: int = 32, verbose: int = 0):
    """Fit and report the final training loss."""
    history = model.fit(x, y, epochs=epochs, batch_size=batch_size,
                        verbose=verbose, shuffle=True)
    final_loss = history.history["loss"][-1]
    print(f"[train] {epochs} epochs, final MSE = {final_loss:.6f}")
    return history


def save_keras(model: tf.keras.Model, path: str = DEFAULT_KERAS_PATH) -> str:
    """Native Keras checkpoint — this is what TFLiteConverter.from_keras_model consumes."""
    if not str(path).endswith((".keras", ".h5")):
        raise ValueError(f"Keras 3 needs a .keras/.h5 extension, got: {path}")
    model.save(path)
    print(f"[save] Keras model -> {path}")
    return path


def load_keras(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(str(path))


def export_saved_model(model: tf.keras.Model, path: str = DEFAULT_SAVEDMODEL_DIR) -> str:
    """SavedModel export (Keras 3 moved this from model.save to model.export)."""
    model.export(path)
    print(f"[save] SavedModel  -> {path}/")
    return path


def sine_model_step(epochs: int = DEFAULT_EPOCHS,
                    keras_path: str = DEFAULT_KERAS_PATH,
                    savedmodel_dir: str = DEFAULT_SAVEDMODEL_DIR,
                    num_samples: int = DEFAULT_SAMPLES):
    """== the whole of sine_model.py ==  Train the model, save both formats.

    Returns (model, x, y) so later steps can reuse the data for calibration
    and validation.
    """
    x, y = generate_training_data(num_samples)
    model = build_model()
    train_model(model, x, y, epochs=epochs)
    save_keras(model, keras_path)
    export_saved_model(model, savedmodel_dir)
    return model, x, y


# ============================================================================
# Step 2 — quantize.py : full-int8 post-training quantization
# ============================================================================
def write_calibration_files(x, calib_dir: str = DEFAULT_CALIB_DIR,
                            num_samples: int = DEFAULT_CALIB_SAMPLES,
                            seed: int = 0) -> Path:
    """Materialise calibration inputs as calib/0.npy .. calib/N.npy.

    quantize.py expects REAL preprocessed inputs here (not random noise): the
    representative dataset is what lets the converter compute correct
    activation ranges, and bad ranges mean bad accuracy on-device.
    """
    calib_dir = Path(calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(x), size=num_samples)
    for i, j in enumerate(picks):
        np.save(calib_dir / f"{i}.npy", x[j])
    print(f"[calib] wrote {num_samples} calibration samples -> {calib_dir}/")
    return calib_dir


def make_representative_dataset(calib_dir: str = DEFAULT_CALIB_DIR,
                                num_samples: int = DEFAULT_CALIB_SAMPLES):
    """Build the callable TFLiteConverter.representative_dataset.

    Yields [batched_input] for each calib/<i>.npy — one batch of 1, matching
    MCU inference, which always runs batch size 1.
    """
    calib_dir = Path(calib_dir)

    def representative_dataset():
        for i in range(num_samples):
            x = np.load(calib_dir / f"{i}.npy").astype(np.float32)
            yield [x.reshape(1, *x.shape)]

    return representative_dataset


def quantize_model(model, calib_dir: str = DEFAULT_CALIB_DIR,
                   tflite_path: str = DEFAULT_TFLITE_PATH,
                   num_samples: int = DEFAULT_CALIB_SAMPLES,
                   full_int8: bool = True,
                   compare_float: bool = True) -> str:
    """Convert a Keras model to TFLite.

    full_int8=True reproduces quantize.py exactly:
      * optimizations          = DEFAULT           (weights + activations int8)
      * representative_dataset = real calib inputs
      * supported_ops          = TFLITE_BUILTINS_INT8   (no float fallback)
      * inference in/out type  = int8              (drops Quantize/Dequantize nodes)
    """
    if isinstance(model, (str, Path)):
        model = load_keras(str(model))

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = make_representative_dataset(calib_dir, num_samples)
    if full_int8:
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8    # int8 in/out avoids
        conv.inference_output_type = tf.int8   # Quantize/Dequantize nodes

    blob = conv.convert()
    Path(tflite_path).write_bytes(blob)
    int8_kib = len(blob) / 1024
    kind = "full int8 (no float fallback)" if full_int8 else "dynamic-range int8"
    print(f"[quantize] {tflite_path}: {int8_kib:.1f} KiB, {kind}")

    if compare_float:
        fconv = tf.lite.TFLiteConverter.from_keras_model(model)
        fblob = fconv.convert()
        float_kib = len(fblob) / 1024
        print(f"[quantize] float32 baseline: {float_kib:.1f} KiB "
              f"(int8 = {100.0 * int8_kib / float_kib:.1f}% of float)")
    return tflite_path


def quantize_step(model, calib_dir: str = DEFAULT_CALIB_DIR,
                  tflite_path: str = DEFAULT_TFLITE_PATH,
                  num_samples: int = DEFAULT_CALIB_SAMPLES,
                  full_int8: bool = True,
                  x_calib=None) -> str:
    """== the whole of quantize.py ==  Produce model_int8.tflite.

    If the calibration dir is missing it is created from x_calib (training
    inputs) — quantize.py assumes calib/ already exists, so the consolidated
    pipeline generates it when it can.
    """
    calib_path = Path(calib_dir)
    if not calib_path.exists() or not any(calib_path.glob("*.npy")):
        if x_calib is None:
            sys.exit(f"[quantize] no calibration data at {calib_dir}/ and no "
                     f"inputs supplied — run the train step or point --calib-dir "
                     f"at real preprocessed inputs.")
        write_calibration_files(x_calib, calib_dir, num_samples)
    return quantize_model(model, calib_dir, tflite_path, num_samples, full_int8)


# ============================================================================
# Step 3 — validate.py : inspect I/O tensors + accuracy vs the float model
# ============================================================================
def _shape_of(details: dict) -> tuple:
    """Plain-python int shape (tensor details give numpy scalars)."""
    return tuple(int(d) for d in details["shape"])


def _quantize_np(arr: np.ndarray, details: dict) -> np.ndarray:
    """float32 -> int8 using the tensor's (scale, zero_point)."""
    scale, zero_point = details["quantization"]
    if scale == 0:                      # never-quantised tensor guard
        return arr.astype(details["dtype"])
    q = np.round(arr / scale) + zero_point
    info = np.iinfo(details["dtype"])
    return np.clip(q, info.min, info.max).astype(details["dtype"])


def _dequantize_np(arr: np.ndarray, details: dict) -> np.ndarray:
    """int8 -> float32 using the tensor's (scale, zero_point)."""
    scale, zero_point = details["quantization"]
    if scale == 0:
        return arr.astype(np.float32)
    return (arr.astype(np.float32) - zero_point) * scale


def compute_metrics(y_true, y_pred) -> dict:
    """MSE/max-err for regression heads, top-1 accuracy for multi-class heads."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred.shape[-1] > 1:                       # classifier: KWS-style logits
        labels = y_true if y_true.ndim == 1 else y_true.argmax(-1)
        return {"accuracy": float(np.mean(y_pred.argmax(-1) == labels))}
    y_true = y_true.reshape(y_pred.shape).astype(np.float32)
    err = y_pred - y_true
    return {"mse": float(np.mean(err ** 2)),
            "max_abs_err": float(np.max(np.abs(err)))}


def _fmt_metrics(m: dict) -> str:
    if "accuracy" in m:
        return f"accuracy = {m['accuracy'] * 100:.2f}%"
    return f"MSE = {m['mse']:.6f}, max|err| = {m['max_abs_err']:.6f}"


def validate_step(tflite_path: str = DEFAULT_TFLITE_PATH,
                  x_val=None, y_val=None,
                  keras_model=None,
                  max_samples: int = 500) -> dict:
    """== the whole of validate.py (completed) ==

    Prints the I/O tensor contract the MCU firmware must honour:
    shape, dtype and (scale, zero_point). Then feeds the validation set
    through the int8 interpreter with explicit quantise/dequantise and
    compares accuracy against the float model, as the original comment
    asked for. Returns the metrics dict.
    """
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    print(f"[validate] input : shape={_shape_of(in_d)} "
          f"dtype={np.dtype(in_d['dtype']).name} quant={in_d['quantization']}")
    print(f"[validate] output: shape={_shape_of(out_d)} "
          f"dtype={np.dtype(out_d['dtype']).name} quant={out_d['quantization']}")
    # in_d["quantization"] == (scale, zero_point) — you'll need these in the firmware:
    #   input_int8 = round(input_float / scale) + zero_point

    metrics: dict = {}
    if x_val is None:
        print("[validate] no validation data supplied — tensor contract only.")
        return metrics

    x_val = np.asarray(x_val, dtype=np.float32)
    n = min(len(x_val), max_samples)
    in_shape = tuple(in_d["shape"])          # e.g. (1, 1) — batch 1 on-device
    preds = []
    for i in range(n):
        # Feed one sample with the batch dim the tensor contract requires.
        sample = x_val[i].reshape(in_shape[1:])[np.newaxis, ...]
        interp.set_tensor(in_d["index"], _quantize_np(sample, in_d))
        interp.invoke()
        out = interp.get_tensor(out_d["index"])
        preds.append(_dequantize_np(out, out_d).reshape(-1))
    preds = np.stack(preds)                  # (n, out_dim)

    if y_val is not None:
        metrics["tflite"] = compute_metrics(np.asarray(y_val)[:n], preds)
        print(f"[validate] TFLite int8 : {_fmt_metrics(metrics['tflite'])}")

    if keras_model is not None:
        float_preds = np.asarray(
            keras_model.predict(x_val[:n], verbose=0))
        if y_val is not None:
            metrics["float"] = compute_metrics(np.asarray(y_val)[:n], float_preds)
            print(f"[validate] float Keras: {_fmt_metrics(metrics['float'])}")
        diff = np.max(np.abs(float_preds.reshape(preds.shape) - preds))
        metrics["max_float_vs_int8_diff"] = float(diff)
        print(f"[validate] max |int8 - float| = {diff:.6f}  "
              f"(quantisation noise; keep labels identical if it flips results)")
    return metrics


# ============================================================================
# Step 4 — inventory.py : list the ops the MCU runtime must support
# ============================================================================
def inventory_step(tflite_path: str = DEFAULT_TFLITE_PATH) -> list:
    """== the whole of inventory.py ==  Unique TFLite op names in the model.

    Every returned op must exist in your MCU runtime (LiteRT-Micro /
    TensorFlow Lite Micro reference kernels), otherwise conversion or
    delegation will fail on-device.
    """
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    ops = sorted({op["op_name"] for op in interp._get_ops_details()})
    print(f"[inventory] {len(ops)} unique ops in {tflite_path}:")
    for name in ops:
        print(f"            - {name}")
    return ops


# ============================================================================
# CLI
# ============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train, fully int8-quantize and validate a model for MCUs.")
    p.add_argument("--model", default=None,
                   help="existing .keras model to quantize instead of training")
    p.add_argument("--keras-out", default=DEFAULT_KERAS_PATH,
                   help="where to save the freshly trained Keras model")
    p.add_argument("--savedmodel-dir", default=DEFAULT_SAVEDMODEL_DIR,
                   help="where to export the SavedModel directory")
    p.add_argument("--calib-dir", default=DEFAULT_CALIB_DIR,
                   help="directory of calib/<i>.npy representative inputs")
    p.add_argument("--num-calib", type=int, default=DEFAULT_CALIB_SAMPLES,
                   help="number of calibration samples")
    p.add_argument("--tflite", default=DEFAULT_TFLITE_PATH,
                   help="output .tflite path")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                   help="training samples generated for the sine model")
    p.add_argument("--float-io", action="store_true",
                   help="keep float32 I/O tensors instead of int8")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # --- Step 1: model -----------------------------------------------------
    if args.model:
        print(f"[load] using existing model {args.model}")
        model = load_keras(args.model)
        x_val = y_val = None      # no ground truth without the dataset
    else:
        model, x, y = sine_model_step(epochs=args.epochs,
                                      keras_path=args.keras_out,
                                      savedmodel_dir=args.savedmodel_dir,
                                      num_samples=args.samples)
        x_val, y_val = x.reshape(-1, 1), y.reshape(-1, 1)

    # --- Step 2: quantize --------------------------------------------------
    tflite_path = quantize_step(model, calib_dir=args.calib_dir,
                                tflite_path=args.tflite,
                                num_samples=args.num_calib,
                                full_int8=not args.float_io,
                                x_calib=x_val)

    # --- Step 3: validate --------------------------------------------------
    validate_step(tflite_path, x_val=x_val, y_val=y_val, keras_model=model)

    # --- Step 4: inventory -------------------------------------------------
    ops = inventory_step(tflite_path)

    print(f"\n[done] MCU artifact: {tflite_path} | {len(ops)} ops")
    if args.float_io:
        print("[note] float I/O selected: the firmware must feed float32 and "
              "dequantise the output itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
