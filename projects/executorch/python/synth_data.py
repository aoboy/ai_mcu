# Copyright (c) 2026, ExecuTorch Zephyr Pico2W demo authors.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic stroke digits (0/1/4/7) rendered in pure Python.

This module is the single source of truth for the synthetic data used by:

  * scripts/gen_test_digits.py  (embedded test vectors)
  * python/train_mnist.py       (--dataset synthetic, offline training)
  * python/export_mnist.py      (calibration + embedded test vectors)

It has no third-party dependencies so the firmware's checked-in test
vectors can be regenerated anywhere. The style follows the synthetic
digit patterns of the upstream ExecuTorch Pico 2 example.
"""

import math
import random

GRID = 28
NPIX = GRID * GRID

SYNTHETIC_CLASS_NAMES = ("0", "1", "4", "7")


def _blank():
    return [[0.0] * GRID for _ in range(GRID)]


def _vline(g, x, y0, y1, w=2):
    for y in range(y0, y1 + 1):
        for dx in range(w):
            if 0 <= x + dx < GRID:
                g[y][x + dx] = 1.0


def _hline(g, y, x0, x1, w=2):
    for x in range(x0, x1 + 1):
        for dy in range(w):
            if 0 <= y + dy < GRID:
                g[y + dy][x] = 1.0


def _diag(g, x0, y0, x1, y1, w=2):
    steps = max(abs(x1 - x0), abs(y1 - y0)) * 2
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        for dx in range(w):
            for dy in range(w):
                if 0 <= x + dx < GRID and 0 <= y + dy < GRID:
                    g[y + dy][x + dx] = 1.0


def _ring(g, cx, cy, rx, ry, w=2):
    steps = 240
    for i in range(steps):
        a = 2.0 * math.pi * i / steps
        x = int(round(cx + rx * math.cos(a)))
        y = int(round(cy + ry * math.sin(a)))
        for dx in range(w):
            for dy in range(w):
                if 0 <= x + dx < GRID and 0 <= y + dy < GRID:
                    g[y + dy][x + dx] = 1.0


def synth_digit(name: str, rng: random.Random):
    """Return a 28x28 [0..1] pattern that roughly looks like `name`."""
    g = _blank()
    if name == "0":
        _ring(g, 14, 14, 8 + rng.randint(-1, 1), 11 + rng.randint(-1, 1))
    elif name == "1":
        x = 13 + rng.randint(-2, 2)
        _vline(g, x, 3, 24)
        _diag(g, x - 5, 6, x, 3)            # serif
        _hline(g, 23, x - 4, x + 6)         # base
    elif name == "4":
        _vline(g, 8, 3, 18)
        _hline(g, 16, 3, 22)
        _vline(g, 17, 10, 24)
    elif name == "7":
        _hline(g, 3, 5, 22)
        _diag(g, 22, 5, 10, 24)
    else:
        raise ValueError("synthetic digits support 0/1/4/7 only, got %r" % name)
    return g


def flatten(g):
    return [float(v) for row in g for v in row]


def synth_dataset(n_per_class: int = 150, seed: int = 7):
    """Return (pixels, labels, names) for the synthetic stroke dataset.

    pixels: list of N lists of 784 floats in [0,1]
    labels: list of N ints (index into SYNTHETIC_CLASS_NAMES)
    """
    rng = random.Random(seed)
    pixels, labels = [], []
    for label, name in enumerate(SYNTHETIC_CLASS_NAMES):
        for _ in range(n_per_class):
            g = synth_digit(name, rng)
            # Light augmentation: random 1px shifts + dropout speckle noise.
            sx, sy = rng.randint(-1, 1), rng.randint(-1, 1)
            out = [[0.0] * GRID for _ in range(GRID)]
            for y in range(GRID):
                for x in range(GRID):
                    ny, nx = y + sy, x + sx
                    v = g[y][x]
                    if v and 0 <= ny < GRID and 0 <= nx < GRID:
                        out[ny][nx] = 1.0
            for _ in range(rng.randint(0, 6)):
                oy, ox = rng.randrange(GRID), rng.randrange(GRID)
                out[oy][ox] = rng.choice((0.0, 1.0))
            pixels.append(flatten(out))
            labels.append(label)
    return pixels, labels, list(SYNTHETIC_CLASS_NAMES)
