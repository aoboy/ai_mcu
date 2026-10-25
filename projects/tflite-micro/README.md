# TFLite-Micro on the RP2350 (Pico 2 W)

An int8-quantized sine MLP running on [TFLite-Micro](https://github.com/tensorflow/tflite-micro)
under Zephyr RTOS, on a **Raspberry Pi Pico 2 W (RP2350, Cortex-M33)** — plus a
QEMU build so the whole thing can be smoke-tested with no hardware.

The firmware sweeps `x ∈ [0, 2π)` and prints `x_value: …, y_value: …` on the
console, tracing `sin(x)`: ~10k inferences/s under QEMU, ~1k/s on the board.

## Status

| | |
|---|---|
| Board | `rpi_pico2/rp2350a/m33_0/w` (Pico 2 W) |
| Console | USB CDC ACM — enumerates as `/dev/ttyACM0` |
| Model | 3464-byte int8 flatbuffer, quantized MSE 1.7e-4, ops: `FULLY_CONNECTED` |
| Firmware | 68.1 KB flash / 20.6 KB RAM (of 4 MB / 520 KB) |
| QEMU | `qemu_x86` smoke test green |

## Quick start

```sh
workon machine_learn        # TF (CPU) for the training step
make model header           # train -> int8 .tflite -> C array

workon zephyr_rtos          # west + Zephyr SDK for firmware
make build                  # -> build_rpi_pico2_rp2350a_m33_0_w/zephyr.uf2
make qemu                   # run on qemu_x86 (Ctrl-C stops it)
make flash_mcu              # west flash -r uf2 (board in BOOTSEL mode)
make mcu_monitor            # serial console
```

Targets: `model`, `header`, `build`, `qemu`, `flash_mcu`, `mcu_monitor`,
`size`, `rom_report`, `menuconfig`, `clean`, `distclean`, `help`.
Board is overridable: `make build BOARD=rpi_pico2/rp2350a/m33_0`.

## Layout

```
projects/tflite-micro/
├── README.md           # this file
├── IMPLEMENTATION_PLAN.md   # plan + decisions + deviations log
├── docs/               # tutorial.md (9-page PDF too)
├── research/           # original brief + status addendum
├── Makefile            # model / header / build / qemu / flash / monitor
├── CMakeLists.txt      # Zephyr app, C++17, no thread-safe statics
├── prj.conf            # CONFIG_CPP, TFLITE_MICRO, float printf
├── boards/             # board-scoped devicetree + Kconfig (USB CDC ACM)
├── scripts/
│   └── run_qemu.sh     # qemu runner: forwards Ctrl-C to timeout/QEMU
├── src/
│   ├── main.cpp        # interpreter, op resolver, int8 I/O, printout
│   ├── sine_model.h        # generated
│   └── sine_model_data.cc  # generated (model bytes)
└── model/
    ├── generate_model.py   # wraps the shared pipeline in tools/ml/
    ├── tflite_to_cc.py     # .tflite -> C array
    └── model_int8.tflite   # the artifact
```

## Relationship to the repo

- Training pipeline is shared: `tools/ml/generate_tf_mcu_model.py`
  (single source of truth, imported by `model/generate_model.py`).
- Repo layout and conventions: [`../../README.md`](../../README.md).
- `qemu` runs through `scripts/run_qemu.sh`, which keeps QEMU off the
  controlling TTY and makes Ctrl-C shut the emulator down instead of leaving
  it running until the timeout.

Full walkthrough: [`docs/tutorial.md`](docs/tutorial.md).
