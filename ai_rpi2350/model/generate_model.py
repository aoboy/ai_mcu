#!/usr/bin/env python3
"""generate_model.py — train + int8-quantize the sine model for the MCU.

Thin wrapper around the shared end-to-end pipeline in
``script/generate_tf_mcu_model.py`` (train -> save -> int8 PTQ with a
representative dataset -> validate -> op inventory), redirected at this
application's output locations:

    model/model_int8.tflite   the artifact that gets embedded into the firmware
    model/calib/*.npy         representative-dataset calibration inputs
    model/sine_model.keras    Keras checkpoint (re-runnable without retraining)

Run it with the machine-learning virtualenv (TensorFlow CPU):

    workon machine_learn        # or: make model
    python model/generate_model.py

Then embed the result into C++ sources:

    python model/tflite_to_cc.py model/model_int8.tflite \\
            --source src/sine_model_data.cc --header src/sine_model.h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # <repo>/ai_rpi2350/model
REPO = HERE.parent.parent                        # <repo> = /home/aimlrn/ai_mcu
sys.path.insert(0, str(REPO / "script"))         # shared pipeline lives here

import generate_tf_mcu_model as pipeline         # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=pipeline.DEFAULT_EPOCHS,
                   help=f"training epochs (default {pipeline.DEFAULT_EPOCHS})")
    p.add_argument("--samples", type=int, default=pipeline.DEFAULT_SAMPLES,
                   help="generated training samples")
    p.add_argument("--num-calib", type=int, default=pipeline.DEFAULT_CALIB_SAMPLES,
                   help="representative-dataset size for PTQ")
    p.add_argument("--tflite", default=str(HERE / "model_int8.tflite"),
                   help="output .tflite path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Step 1 — train the float model (sine over [0, 2*pi]) and save both
    # Keras and SavedModel forms so conversion never needs retraining.
    model, x, y = pipeline.sine_model_step(
        epochs=args.epochs,
        keras_path=str(HERE / "sine_model.keras"),
        savedmodel_dir=str(HERE / "sine_model"),
        num_samples=args.samples,
    )
    x_col = x.reshape(-1, 1)
    y_col = y.reshape(-1, 1)

    # Step 2 — full-int8 post-training quantization. The representative
    # dataset (calib/*.npy) is created from the training inputs when absent.
    pipeline.quantize_step(
        model,
        calib_dir=str(HERE / "calib"),
        tflite_path=args.tflite,
        num_samples=args.num_calib,
        full_int8=True,
        x_calib=x_col,
    )

    # Step 3 — print the int8 I/O contract the firmware must honor
    # (shape, dtype, scale, zero_point) and accuracy vs. the float model.
    pipeline.validate_step(args.tflite, x_val=x_col, y_val=y_col,
                           keras_model=model)

    # Step 4 — ops the MCU runtime must provide (FULLY_CONNECTED here);
    # main.cpp registers exactly these in its MicroMutableOpResolver.
    ops = pipeline.inventory_step(args.tflite)

    print(f"\n[done] MCU artifact: {args.tflite} | ops: {ops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
