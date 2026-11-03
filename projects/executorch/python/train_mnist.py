#!/usr/bin/env python3
# Copyright (c) 2026, ExecuTorch Zephyr Pico2W demo authors.
# SPDX-License-Identifier: Apache-2.0
"""
Train TinyCNN for the Pico 2 W demo and save a checkpoint.

Two dataset modes:

  --dataset mnist (default)
      Real MNIST via torchvision. ~3 epochs on any CPU reaches ~98-99 %.
      The checkpoint is saved with num_classes=10.

  --dataset synthetic
      The offline stroke patterns from synth_data.py (classes 0/1/4/7,
      num_classes=4). No downloads, trains in seconds, and matches the
      checked-in test vectors -- the fastest path to a first end-to-end
      demo run. Do not expect "real" MNIST accuracy from this mode; it
      exists so the whole toolchain can be exercised without internet.

Output:
  <repo>/python/mnist_cnn_state.pt   -- checkpoint consumed by export_mnist.py
                                        (gitignored; regenerated on demand)

Usage:
  python3 train_mnist.py                          # MNIST, 3 epochs
  python3 train_mnist.py --epochs 5 --lr 1e-3
  python3 train_mnist.py --dataset synthetic      # offline quickstart
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from model import create_model, example_input  # noqa: E402

DEFAULT_CHECKPOINT = os.path.join(_HERE, "mnist_cnn_state.pt")


def load_mnist_loaders(batch_size: int, num_workers: int = 0):
    """Returns (train_loader, test_loader, num_classes)."""
    import torchvision
    import torchvision.transforms as T

    tfm = T.ToTensor()  # already scales to [0,1]
    root = os.path.expanduser("~/.cache/mnistdata")
    train = torchvision.datasets.MNIST(root, train=True, download=True, transform=tfm)
    test = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=512, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, 10


def load_synthetic_tensors(n_per_class: int = 150, seed: int = 7):
    """Returns (train_x, train_y, test_x, test_y, num_classes)."""
    import synth_data

    px, labels, _ = synth_data.synth_dataset(n_per_class=n_per_class, seed=seed)
    x = torch.tensor(px, dtype=torch.float32).view(-1, 1, 28, 28)
    y = torch.tensor(labels, dtype=torch.long)

    # Deterministic split: every 10th sample goes to the test set.
    idx = torch.arange(len(y))
    test_mask = idx % 10 == 0
    train_mask = ~test_mask
    return x[train_mask], y[train_mask], x[test_mask], y[test_mask], 4


def evaluate(model, x=None, y=None, loader=None):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        if loader is not None:
            for xb, yb in loader:
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        else:
            pred = model(x).argmax(dim=1)
            correct = (pred == y).sum().item()
            total = y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("mnist", "synthetic"), default="mnist")
    ap.add_argument("--epochs", type=int, default=None,
                    help="default: 3 for mnist, 40 for synthetic")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=DEFAULT_CHECKPOINT)
    args = ap.parse_args()

    torch.manual_seed(0)
    if args.epochs is None:
        args.epochs = 3 if args.dataset == "mnist" else 40

    if args.dataset == "mnist":
        train_loader, test_loader, num_classes = load_mnist_loaders(args.batch_size)
        model = create_model(num_classes=num_classes)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            t0 = time.time()
            for xb, yb in train_loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                running += loss.item() * yb.numel()
            acc = evaluate(model, loader=test_loader)
            print(f"epoch {epoch + 1}/{args.epochs}: "
                  f"loss={running / len(train_loader.dataset):.4f} "
                  f"test_acc={acc * 100:.2f}% ({time.time() - t0:.1f}s)")
    else:
        xtr, ytr, xte, yte, num_classes = load_synthetic_tensors()
        model = create_model(num_classes=num_classes)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        print(f"synthetic dataset: {len(ytr)} train / {len(yte)} test, "
              f"classes={num_classes}")
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(ytr))
            for i in range(0, len(ytr), args.batch_size):
                b = perm[i : i + args.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(xtr[b]), ytr[b])
                loss.backward()
                opt.step()
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                acc = evaluate(model, x=xte, y=yte)
                print(f"epoch {epoch + 1}/{args.epochs}: "
                      f"loss={loss.item():.4f} test_acc={acc * 100:.2f}%")

    # Sanity check the deployment input contract before saving.
    model.eval()
    with torch.no_grad():
        y = model(example_input())
    assert tuple(y.shape) == (1, num_classes), y.shape

    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": num_classes,
            "dataset": args.dataset,
        },
        args.out,
    )
    print(f"checkpoint saved -> {args.out}")
    print("next step: python3 export_mnist.py" +
          ("" if args.dataset == "mnist" else " --dataset synthetic"))


if __name__ == "__main__":
    main()
