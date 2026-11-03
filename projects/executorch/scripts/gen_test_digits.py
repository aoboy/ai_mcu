#!/usr/bin/env python3
# Copyright (c) 2026, ExecuTorch Zephyr Pico2W demo authors.
# SPDX-License-Identifier: Apache-2.0
"""
Generate embedded test digits for the firmware (app/src/test_digits.h).

Two modes:

  synthetic (default)
      Hand-drawn 28x28 stroke patterns for the digits 0/1/4/7, identical in
      spirit to the synthetic digits used by the upstream ExecuTorch Pico 2
      example. Works fully offline (no dataset needed). Pair this with
      `train_mnist.py --dataset synthetic` so the model is trained on the
      same kind of patterns.

  mnist
      Pulls one real digit per requested label out of the MNIST test set
      (via torchvision) and embeds it. Pair this with a checkpoint trained
      with `train_mnist.py --dataset mnist`.

The generated header is consumed by app/src/main.cpp and defines:
    TEST_DIGIT_COUNT, test_digit_input[][784] (float, [0..1]),
    test_digit_label[]  (class index), test_digit_names[]  (display string).

Usage:
    python3 scripts/gen_test_digits.py                       # synthetic
    python3 scripts/gen_test_digits.py --dataset mnist
    python3 scripts/gen_test_digits.py --names 0,1,4,7 --out app/src/test_digits.h
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.normpath(os.path.join(_HERE, "..", "python"))
sys.path.insert(0, _PYTHON_DIR)

import embed_header  # noqa: E402
import synth_data  # noqa: E402


def mnist_digits(names):
    import torchvision

    ds = torchvision.datasets.MNIST(
        root=os.path.expanduser("~/.cache/mnistdata"),
        train=False,
        download=True,
    )
    picked = []
    for name in names:
        want = int(name)
        idxs = [i for i in range(len(ds)) if int(ds.targets[i]) == want]
        if not idxs:
            raise SystemExit("no MNIST sample for digit %d" % want)
        img, _ = ds[idxs[len(idxs) // 2]]  # deterministic-ish middle pick
        picked.append((name, [float(v) / 255.0 for v in list(img.getdata())]))
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("synthetic", "mnist"), default="synthetic")
    ap.add_argument(
        "--names", default="0,1,4,7", help="comma-separated digit labels to embed"
    )
    ap.add_argument("--out", default=None, help="output header path")
    args = ap.parse_args()

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if args.dataset == "synthetic":
        if any(n not in synth_data.SYNTHETIC_CLASS_NAMES for n in names):
            raise SystemExit("synthetic dataset only supports digits 0,1,4,7")
        rng_seed = 42  # deterministic output -> stable VCS diffs
        import random

        rng = random.Random(rng_seed)
        digits = [(n, synth_data.flatten(synth_data.synth_digit(n, rng))) for n in names]
        # 4-class model: class index == position in SYNTHETIC_CLASS_NAMES.
        labels = [synth_data.SYNTHETIC_CLASS_NAMES.index(n) for n in names]
    else:
        digits = mnist_digits(names)
        # 10-class model: class index == the digit itself.
        labels = [int(n) for n in names]

    out_path = os.path.abspath(args.out or embed_header.default_out_path())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    embed_header.emit(names, digits, out_path, labels=labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
