#!/usr/bin/env python3
# Copyright (c) 2026, ExecuTorch Zephyr Pico2W demo authors.
# SPDX-License-Identifier: Apache-2.0
"""
Export the trained TinyCNN to ExecuTorch .pte files for the Pico 2 W.

Produces, next to the checkpoint:

  app/model/mnist_cnn_fp32.pte   float32 graph  -> Portable Ops kernels
  app/model/mnist_cnn_int8.pte   INT8 graph     -> Cortex-M / CMSIS-NN kernels
  app/src/test_digits.h          embedded test vectors for the firmware

The INT8 path follows the official ExecuTorch "Arm Cortex-M Backend"
tutorial exactly:

  1. capture the graph with torch.export.export()
  2. quantize with CortexMQuantizer + torchao's PT2E flow
     (prepare_pt2e -> calibrate -> convert_pt2e)
  3. lower to edge with cortex_m_edge_compile_config()
  4. run CortexMPassManager(target_config=CortexMTargetConfig(cpu=CortexM.M33))
     to swap quantized ops for CMSIS-NN implementations
  5. serialize with to_executorch(config=ExecutorchBackendConfig(
     extract_delegate_segments=False))

Both .pte variants keep float32 model I/O (quantize/dequantize nodes live
inside the graph), so the firmware source is identical for both.

Usage:
  python3 export_mnist.py                        # both variants (mnist ckpt)
  python3 export_mnist.py --dataset synthetic    # offline end-to-end
  python3 export_mnist.py --skip-cmsis           # fp32 only
"""

import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

from model import create_model, count_parameters, example_input  # noqa: E402
import embed_header  # noqa: E402
import synth_data  # noqa: E402

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = os.path.join(_HERE, "mnist_cnn_state.pt")
MODEL_OUT_DIR = os.path.normpath(os.path.join(_HERE, "..", "app", "model"))


# ---------------------------------------------------------------------------
# Dataset access (for calibration data and embedded test digits)
# ---------------------------------------------------------------------------
def calibration_data(dataset: str, n: int = 256):
    """Representative samples for the PT2E observers."""
    if dataset == "mnist":
        import torchvision
        import torchvision.transforms as T

        ds = torchvision.datasets.MNIST(
            os.path.expanduser("~/.cache/mnistdata"), train=True,
            download=True, transform=T.ToTensor(),
        )
        idx = torch.linspace(0, len(ds) - 1, n).long()
        return [ds[i][0] for i in idx]  # each: (1, 28, 28) float [0..1]

    px, _, _ = synth_data.synth_dataset(n_per_class=64)
    x = torch.tensor(px, dtype=torch.float32).view(-1, 1, 28, 28)
    idx = torch.linspace(0, len(x) - 1, min(n, len(x))).long()
    return [x[i] for i in idx]


def test_digits(dataset: str, names):
    """Return [(name, [784 floats])] for embedding into the firmware."""
    if dataset == "synthetic":
        import random

        rng = random.Random(42)
        return [
            (n, synth_data.flatten(synth_data.synth_digit(n, rng))) for n in names
        ]

    import torchvision

    ds = torchvision.datasets.MNIST(
        os.path.expanduser("~/.cache/mnistdata"), train=False, download=True
    )
    out = []
    for name in names:
        want = int(name)
        idxs = [i for i in range(len(ds)) if int(ds.targets[i]) == want]
        if not idxs:
            raise SystemExit("no MNIST sample for digit %d" % want)
        img, _ = ds[idxs[len(idxs) // 2]]
        out.append((name, [float(v) / 255.0 for v in list(img.getdata())]))
    return out


# ---------------------------------------------------------------------------
# Export: float32 (Portable Ops)
# ---------------------------------------------------------------------------
def export_float(model, output_path: str):
    from executorch.exir import to_edge

    logger.info("Exporting float32 .pte (Portable Ops path)...")
    exported = torch.export.export(model, (example_input(),))

    edge_program = to_edge(exported)
    exec_program = edge_program.to_executorch()

    with open(output_path, "wb") as f:
        f.write(exec_program.buffer)
    logger.info("Saved %s (%.1f KB)", output_path,
                os.path.getsize(output_path) / 1024)


# ---------------------------------------------------------------------------
# Export: INT8 (Cortex-M / CMSIS-NN)
# ---------------------------------------------------------------------------
def export_int8_cmsis(model, calib, output_path: str):
    from executorch.backends.cortex_m.edge_compile_config import (
        cortex_m_edge_compile_config,
    )
    from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
        CortexMPassManager,
    )
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from executorch.backends.cortex_m.target_config import (
        CortexM,
        CortexMTargetConfig,
    )
    from executorch.exir import ExecutorchBackendConfig, to_edge
    from executorch.extension.export_util.utils import save_pte_program
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    logger.info("Quantizing with CortexMQuantizer (PT2E flow)...")
    quantizer = CortexMQuantizer()
    exported = torch.export.export(model, (example_input(),))
    graph_module = exported.module()

    prepared = prepare_pt2e(graph_module, quantizer)
    logger.info("Calibrating with %d samples...", len(calib))
    with torch.no_grad():
        for i, data in enumerate(calib):
            prepared(data)
            if (i + 1) % 64 == 0:
                logger.info("  calibrated %d/%d", i + 1, len(calib))
    quantized = convert_pt2e(prepared)

    logger.info("Lowering to edge (cortex_m_edge_compile_config)...")
    quantized_exported = torch.export.export(quantized, (example_input(),))
    edge_program = to_edge(
        quantized_exported, compile_config=cortex_m_edge_compile_config()
    )

    logger.info("Applying Cortex-M passes (target: Cortex-M33)...")
    pass_manager = CortexMPassManager(
        target_config=CortexMTargetConfig(cpu=CortexM.M33)
    )
    edge_program = edge_program.transform(pass_manager)

    logger.info("Serializing .pte...")
    exec_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )
    save_pte_program(exec_program, output_path)
    logger.info("Saved %s (%.1f KB)", output_path,
                os.path.getsize(output_path) / 1024)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--dataset", choices=("mnist", "synthetic"), default=None,
                    help="overrides the dataset recorded in the checkpoint")
    ap.add_argument("--out-dir", default=MODEL_OUT_DIR)
    ap.add_argument("--skip-cmsis", action="store_true",
                    help="only export the float32 .pte")
    ap.add_argument("--names", default="0,1,4,7",
                    help="digit labels to embed as firmware test vectors")
    args = ap.parse_args()

    try:
        from executorch.exir import to_edge  # noqa: F401

        del to_edge
    except ImportError as e:
        raise SystemExit(
            "ExecuTorch python package not found. Run the export in the "
            "virtualenvwrapper env that has it:\n"
            "  workon pytorch_cu132 && python python/export_mnist.py\n"
            f"(original error: {e})"
        )

    if not os.path.exists(args.checkpoint):
        raise SystemExit(
            f"checkpoint not found: {args.checkpoint}\n"
            "train first:  python3 train_mnist.py"
        )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_classes = int(ckpt["num_classes"])
    dataset = args.dataset or ckpt.get("dataset", "mnist")
    class_names = (
        synth_data.SYNTHETIC_CLASS_NAMES
        if dataset == "synthetic"
        else tuple(str(d) for d in range(num_classes))
    )
    logger.info("checkpoint: %s (num_classes=%d, dataset=%s)",
                args.checkpoint, num_classes, dataset)

    model = create_model(num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info("model loaded: %d parameters (%.1f KB fp32)",
                count_parameters(model),
                sum(p.numel() * p.element_size() for p in model.parameters()) / 1024)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. float32 -> Portable Ops
    export_float(model, os.path.join(args.out_dir, "mnist_cnn_fp32.pte"))

    # 2. INT8 -> Cortex-M / CMSIS-NN
    if not args.skip_cmsis:
        calib = calibration_data(dataset)
        export_int8_cmsis(
            model, calib, os.path.join(args.out_dir, "mnist_cnn_int8.pte")
        )

    # 3. Embedded test digits for the firmware
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if dataset == "synthetic":
        for n in names:
            if n not in class_names:
                raise SystemExit(
                    "synthetic model only knows classes 0/1/4/7 -- use "
                    "--names 0,1,4,7"
                )
    digits = test_digits(dataset, names)
    if dataset == "synthetic":
        # 4-class model: class index == position in SYNTHETIC_CLASS_NAMES.
        labels = [synth_data.SYNTHETIC_CLASS_NAMES.index(n) for n in names]
    else:
        # 10-class model: class index == the digit itself.
        labels = [int(n) for n in names]
    embed_header.emit(names, digits, embed_header.default_out_path(), labels=labels)

    logger.info("Export complete. Build with e.g.:")
    logger.info(
        "  west build -p -b rpi_pico2/rp2350a/m33_0/w app -- "
        "-DET_PTE_FILE_PATH=%s",
        os.path.join(args.out_dir, "mnist_cnn_int8.pte"),
    )


if __name__ == "__main__":
    main()
