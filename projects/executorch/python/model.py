# Copyright (c) 2026, ExecuTorch Zephyr Pico2W demo authors.
# SPDX-License-Identifier: Apache-2.0
"""Model definition shared by train_mnist.py and export_mnist.py.

TinyCNN -- a deliberately small LeNet-style CNN for 28x28 grayscale digits:

    input (1, 1, 28, 28) float32 [0..1]
      -> Conv2d(1->8, 3x3)   ReLU   MaxPool(2)      26x26x8  -> 13x13x8
      -> Conv2d(8->16, 3x3)  ReLU   MaxPool(2)      11x11x16 -> 5x5x16
      -> Flatten -> Linear(400 -> num_classes)

Roughly 5.3k parameters (~21 KB as fp32, ~5.5 KB as int8), which is far
beyond enough for ~99 % on MNIST after a few epochs and comfortably inside
the Pico 2 W's 520 KiB SRAM budget: only *activations* live in RAM, the
weights stay in the flash-resident .pte.

The input contract is fixed on purpose: float32 in [0.0, 1.0], shape
(1, 1, 28, 28). The INT8 export (CortexMQuantizer) keeps float I/O and puts
quantize/dequantize nodes inside the graph, so the firmware is identical for
both backends.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_ROWS = 28
IMG_COLS = 28
EXAMPLE_INPUT_SHAPE = (1, 1, IMG_ROWS, IMG_COLS)


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)    # -> 26x26x8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)   # -> 11x11x16 (after pool)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, num_classes)
        self.dropout = nn.Dropout(0.1)  # training only; eval() disables it

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 28->26->13
        x = self.pool(F.relu(self.conv2(x)))   # 13->11->5
        x = torch.flatten(x, 1)                # 16*5*5 = 400
        x = self.dropout(x)
        return self.fc1(x)                     # raw logits (no softmax)


def create_model(num_classes: int = 10) -> TinyCNN:
    return TinyCNN(num_classes=num_classes)


def example_input() -> torch.Tensor:
    """Example tensor matching the deployment input contract."""
    return torch.zeros(EXAMPLE_INPUT_SHAPE, dtype=torch.float32)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_bytes(model: nn.Module) -> int:
    """Storage size of the parameters as fp32 and int8, for docs/planning."""
    fp32 = sum(p.numel() * p.element_size() for p in model.parameters())
    return fp32


if __name__ == "__main__":
    m = create_model(num_classes=10)
    m.eval()
    x = example_input()
    with torch.no_grad():
        y = m(x)
    print("TinyCNN created:", m)
    print("params:", count_parameters(m))
    print("fp32 parameter bytes:", model_size_bytes(m))
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
