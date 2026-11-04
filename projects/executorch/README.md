# ExecuTorch + Zephyr + Raspberry Pi Pico 2 W — Edge AI MNIST demo

Run a PyTorch-trained digit classifier **on the metal** of a Raspberry Pi
Pico 2 W: the model is exported to ExecuTorch's `.pte` format, linked into
the XIP flash of the RP2350, and executed by the ExecuTorch runtime as a
[Zephyr RTOS](https://www.zephyrproject.org/) application — no filesystem, no
malloc at inference time, every activation statically planned.

```
PyTorch (host)                Zephyr (Pico 2 W)
┌──────────────────┐          ┌───────────────────────────────────────┐
│ train_mnist.py   │          │ Zephyr app (main.cpp)                 │
│      │           │          │   Program::load()  ← .pte in XIP flash│
│ export_mnist.py  │  .pte    │   Method::execute()                   │
│  ├─ float32 ─────┼────────► │   ├─ Portable Ops kernels             │
│  └─ INT8 (CMSIS)─┼────────► │   └─ Cortex-M / CMSIS-NN kernels      │
└──────────────────┘          │  results on USB CDC + onboard LED     │
                              └───────────────────────────────────────┘
```

**Why this project is interesting as an Edge AI reference:**

- it is the *official stack*: the [ExecuTorch Zephyr module] is an official
  Zephyr external module; this repo is an out-of-tree application on top of
  it, following the same patterns as the upstream `hello-executorch` sample
- it demonstrates **both execution backends** available on a plain Cortex-M33
  with the *same firmware*: Portable Ops (reference C++ kernels) and the
  Cortex-M backend (INT8 kernels from [CMSIS-NN], using the M33's DSP
  instructions) — the backend is chosen by *which `.pte` you embed*, not by
  recompiling the app
- it documents the whole memory story: 520 KiB SRAM budget, static pools,
  memory-planned activation buffers, weights read **directly from QSPI flash
  via XIP** (zero-copy loader)

---

## Hardware & software requirements

| Item | Version / value | Notes |
| --- | --- | --- |
| Board | Raspberry Pi **Pico 2 W** (RP2350A) | plain Pico 2 works too (`rpi_pico2/rp2350a/m33`) |
| SoC | dual Cortex-M33 @ 150 MHz, FPv5-SP FPU | no NPU — this is pure CPU inference |
| RAM / Flash | 520 KiB SRAM / 4 MiB QSPI (XIP) | model weights stay in flash |
| Zephyr | **v4.4.0 or newer** | first release shipping the ExecuTorch module; also ≥ 4.3 for Pico 2 W WiFi board support |
| Zephyr SDK | current, via `west sdk install` | `arm-zephyr-eabi` toolchain |
| ExecuTorch | **v1.5.x** (module `main`, pin a tag for reproducibility) | runtime built from source by the Zephyr build; Python package used for export |
| Host OS | Linux | macOS can export models but not flash via `west` |
| Python | 3.10+ | for training/export only |

Console: **USB CDC ACM** on the board's native USB port — enumerates as
`/dev/ttyACM*` (Raspberry Pi VID 2E8A), same recipe as `projects/tflite-micro`
(board-scoped `app/boards/*.overlay`). No UART adapter needed. A UART0
console (GP0 TX / GP1 RX, 115200 8N1, 3.3 V logic) is available instead by
editing the overlay.
LED: onboard LED (CYW43 `led0` on the W, GPIO25 on the plain Pico 2).

## Repository layout

```
executorch-zephyr-pico2w/
├── Makefile                      # model / build / flash / monitor / size / clean
├── app/                          # the Zephyr application
│   ├── CMakeLists.txt            #   build glue: selective ops libs, .pte → C header
│   ├── prj.conf                  #   Kconfig: ExecuTorch, C++17, pool sizes
│   ├── boards/rpi_pico2_rp2350a_m33_0_w.conf
│   ├── src/main.cpp              #   inference demo (heavily commented)
│   ├── src/test_digits.h         #   generated test vectors (checked in)
│   └── model/                    #   where the exported .pte files go
├── python/                       # host-side model tooling
│   ├── model.py                  #   TinyCNN (~5.3k params, LeNet-style)
│   ├── train_mnist.py            #   trains on MNIST or offline synthetic strokes
│   ├── export_mnist.py           #   float32 + INT8 (CortexMQuantizer) .pte export
│   ├── synth_data.py             #   dependency-free stroke-digit generator
│   ├── embed_header.py           #   test_digits.h emitter
│   └── requirements.txt
├── scripts/
│   ├── 01_setup_workspace.sh     # verify the pre-provisioned env/workspace (no installs)
│   ├── 02_build.sh               # build portable | cmsis variants
│   └── gen_test_digits.py        # regenerate embedded test vectors
├── west-manifests/executorch.yaml# adds pytorch/executorch as a west module
└── docs/
    ├── ARCHITECTURE.md           # deep dive: runtime, memory, backends
    └── tutorial.pdf              # step-by-step guide with diagrams
```

## Quickstart (TL;DR)

Two virtualenvwrapper environments are used (already provisioned on this
machine): `pytorch_cu132` (torch + executorch 1.5.1, for training/export)
and `zephyr_rtos` (west, for building). The `Makefile` (same target
vocabulary as `projects/tflite-micro`) drives both — no `workon` needed:

```bash
git clone <this-repo> executorch-zephyr-pico2w && cd executorch-zephyr-pico2w

# 1. Verify the pre-provisioned environment (workspace ~/zephyrproject,
#    Zephyr SDK, ExecuTorch module). No installs, no downloads.
make verify

# 2a. Real model: train on MNIST and export both .pte variants
make model

# 2b. (offline quickstart, no dataset) synthetic stroke digits instead:
#     make model DATASET=synthetic

# 3. Smoke-test on QEMU first (fp32/portable path, no hardware needed)
make qemu

# 4. Build the INT8/CMSIS-NN variant for the board (VARIANT=portable is
#    the default; the cmsis variant cannot run under QEMU)
make build VARIANT=cmsis

# 5. Flash: hold BOOTSEL, plug USB, `make flash` (or drop
#    build-cmsis/zephyr/zephyr.uf2 onto the RPI-RP2350 drive).
#    Console: USB CDC ACM (/dev/ttyACM*) on the same USB cable.
make monitor
```

The interactive equivalent, if you prefer the envs by hand:

```bash
workon pytorch_cu132   # model step:  python python/train_mnist.py && python python/export_mnist.py
deactivate && workon zephyr_rtos   # firmware:  ./scripts/02_build.sh cmsis
```

Other targets: `size`, `rom_report`, `menuconfig`, `clean`, `distclean`
(overrides: `VARIANT=portable|cmsis`, `DATASET=mnist|synthetic`,
`BOARD=…`, `SERIAL_PORT=…` — see `make help`).

Expected console output (shape; numbers depend on your model variant):

```
==========================================
 ExecuTorch + Zephyr MNIST demo
 Raspberry Pi Pico 2 W (RP2350A, M33)
==========================================
I [main.cpp] Model PTE at 0x1003a6c0, size 47232 bytes (XIP flash)
I [main.cpp] Method allocator pool: 131072 bytes, temp pool: 4096 bytes
I [main.cpp] Program loaded, 1 method(s)
I [main.cpp] Planned buffer 0: 6216 bytes
I [main.cpp] Total activation (planned) memory: 6216 bytes
I [main.cpp] Method 'forward' loaded. Allocator used: 12736 / 131072 bytes (free 118336)

=== Digit '7' (expected class 3) ===
...............................
#############################..
...........................#...
..........................#....
.........................#....
... (ASCII-art rendering of the embedded digit)
Logits: -2.1 -0.3 1.9 -4.2 0.5 -1.1 0.8 14.2 -0.4 2.3
Predicted: class 7 | CORRECT (312 us)
Expected:  class 7 ('7')
```

after the four digits, a per-round summary, the aggregate over all
rounds and the success LED pattern (three long blinks). The demo runs
**`ROUNDS` rounds** before quitting and releasing the runtime (Kconfig
`APP_DEMO_ROUNDS`, default 10; `make build ROUNDS=3` or
`make qemu ROUNDS=2` to change). The ASCII digits render on the first
round only. Under QEMU the same run prints `Done after N rounds` on the
console — expect **slow** inferences (~2 s/digit under emulation) and,
when the model was trained with `DATASET=synthetic`, poor accuracy: the
synthetic quickstart model exists to exercise the toolchain, not to be
accurate. Train on MNIST for the real 98–99 % classifier.

---

## Step 1 — Zephyr workspace with the ExecuTorch module

The environment is **pre-provisioned and shared** (it also hosts the
`tflite-micro` project): virtualenvwrapper env `zephyr_rtos` with west, the
west workspace `~/zephyrproject` (Zephyr + the ExecuTorch module via the
submanifest below + CMSIS/`cmsis-nn` + `hal_infineon`), and the Zephyr SDK.
`scripts/01_setup_workspace.sh` therefore only **verifies** it — it never
creates a venv, never runs `west init`, and never installs anything:

1. activates `zephyr_rtos` (the script equivalent of `workon zephyr_rtos`)
2. checks the ExecuTorch submanifest `zephyr/submanifests/executorch.yml`
   and copies `west-manifests/executorch.yaml` there only if missing
3. verifies the west modules this app needs: `executorch`, `cmsis`,
   `cmsis-nn`, `cmsis_6`, `hal_infineon` — CMSIS/`cmsis-nn` provide the
   CMSIS-NN kernels for the Cortex-M backend, `hal_infineon` provides the
   CYW43 WiFi/LED support (`ET_DO_UPDATE=1` fetches only missing ones)
4. verifies the Zephyr SDK (`arm-zephyr-eabi` toolchain)
5. verifies the ExecuTorch module and that `import executorch` works in the
   `pytorch_cu132` env — the C++ runtime needs no host build (Zephyr builds
   it during `west build`), and the Python export tooling lives in
   `pytorch_cu132`, not in `zephyr_rtos`
6. checks the **local module patch** (`west-manifests/patches/`) that makes
   the op codegen work from a torch-less west venv: the module's codegen
   imports `torchgen` (ships inside torch) with the build interpreter; the
   patch lets every build pass `-DET_CODEGEN_PYTHON=<pytorch_cu132 python>`
   and turns a failed import into a clear error instead of silently globbing
   every Python file on the machine into the ninja graph. Applied with
   `ET_APPLY_PATCHES=1 ./scripts/01_setup_workspace.sh` or
   `./scripts/patch_module.sh apply` (idempotent; re-run after `west update`)
7. installs the **host flatcc compiler once** into
   `<workspace>/tools/flatcc` (`scripts/install_flatcc.sh`, idempotent):
   the module otherwise reruns its `BUILD_ALWAYS flatcc_ep` external
   project — a full host CMake pass — on *every* firmware build, for a
   binary these builds never use (it only feeds `devtools/etdump`). With
   the prebuilt binary the builds pass the module's own
   `-DFLATCC_EXECUTABLE` hook and skip that step entirely. The in-tree
   `flatccrt` runtime that gets linked into the firmware is unaffected.

Optional toggles: `ET_DO_UPDATE=1` (update missing modules only),
`ET_DO_BLOBS=1` (`west blobs fetch hal_infineon` — CYW43 WiFi firmware;
not needed by this demo).

On a **new machine**, the one-time bootstrap this replaces (Steps 1–3 of the
official [Zephyr tutorial]) is: create the env
(`mkvirtualenv zephyr_rtos && pip install west pyelftools jsonschema`),
`west init --manifest-rev v4.4.0` into `~/zephyrproject`, install the
submanifest, `west update`, and `west sdk install --gnu-toolchains
arm-zephyr-eabi`.

## Step 2 — Train & export the model

`python/train_mnist.py` trains `TinyCNN` (see `python/model.py`):

| Layer | Out | Params |
| --- | --- | --- |
| Conv2d 1→8, 3×3 + ReLU + MaxPool | 13×13×8 | 80 |
| Conv2d 8→16, 3×3 + ReLU + MaxPool | 5×5×16 | 1 168 |
| Linear 400→classes | classes | 4 010 |
| **total** | | **~5.3k (~21 KB fp32)** |

Two dataset modes:

- `--dataset mnist` (default): real MNIST, ~98–99 % test accuracy in 3
  epochs on CPU; export embeds real digits from the test set.
- `--dataset synthetic`: pure-Python stroke digits (0/1/4/7), trains in
  seconds with zero downloads; export embeds synthetic digits. Use this to
  exercise the full toolchain on an offline machine, or as a CI smoke test.

`python/export_mnist.py` (run in the `pytorch_cu132` env) then produces
**both** deployments:

> The INT8 path needs the Arm TOSA serializer in the export env:
> `pip install tosa-tools` (installed 2026.8.0 here — the upstream pin
> 2026.5.0 has no wheel for Python 3.14). Without it only the float32
> `.pte` exports (`make model SKIP_CMSIS=1`).

| File | Export path | Runtime kernels |
| --- | --- | --- |
| `app/model/mnist_cnn_fp32.pte` | `torch.export → to_edge → to_executorch` | Portable Ops (reference C++) |
| `app/model/mnist_cnn_int8.pte` | + `CortexMQuantizer` (PT2E), `cortex_m_edge_compile_config`, `CortexMPassManager(CortexM.M33)` | Cortex-M backend (CMSIS-NN INT8) |

Both keep **float32 model I/O** — quantize/dequantize nodes are inserted
inside the graph — so the firmware input contract is identical
(`(1,1,28,28)` float in `[0,1]`). The script also regenerates
`app/src/test_digits.h` with the digits the firmware will classify.

## Step 3 — Build (portable vs. CMSIS-NN)

```bash
make qemu                     # hardware-free smoke test: qemu_x86, fp32 .pte
make build VARIANT=portable   # float32, Portable Ops -> the Pico 2 W
make build VARIANT=cmsis      # INT8, CMSIS-NN -> the Pico 2 W
```

(Equivalent to `./scripts/02_build.sh portable|cmsis`, which the target
wraps. Run from `workon zephyr_rtos`, or let it activate the env.)

**The QEMU smoke test** builds the same app for `qemu_x86` and runs it with
the float32 `.pte`: the LED code compiles out (no `led0` on the emulated
board), the module skips the Cortex-M kernels on x86, and the console output
proves a real ExecuTorch inference happened. The **cmsis variant cannot run
here** — CMSIS-NN kernels are Cortex-M-only and the INT8 `.pte` dispatches to
them, while QEMU emulates a plain x86 CPU. Same wrapper as tflite-micro
(`scripts/run_qemu.sh`): Ctrl-C stops the emulator, the timeout bounds the
run, and the grep proves the inference.

Equivalently, by hand:

```bash
workon zephyr_rtos && cd ~/zephyrproject
west build -p -b rpi_pico2/rp2350a/m33_0/w <repo>/app -d build-cmsis -- \
    -DET_PTE_FILE_PATH=<repo>/app/model/mnist_cnn_int8.pte
```

The application code never changes. What changes is:

1. the **`.pte`**: the INT8 export records the quantized operators with
   their Cortex-M/CMSIS-NN kernel mappings; the float export records plain
   ATen operators;
2. the **linked kernel libraries**: `app/CMakeLists.txt` builds a
   *selective* operator library from the embedded model
   (`OPS_FROM_MODEL`), pulling INT8 implementations from `cortex_m_kernels`
   (built automatically by the module on every `CONFIG_CPU_CORTEX_M` board)
   and any remaining operators from the portable library.

`CONFIG_EXECUTORCH_BUILD_PORTABLE_OPS=n` in `app/prj.conf` keeps the
"include every portable op" registry off so only the operators the model
actually uses are linked. Set it to `y` if you experiment with other models
and hit "operator missing" (runtime error `0x14`) and don't want to think
about selective builds — at the cost of a much larger binary.

## Step 4 — Flash and run

- **UF2 drag-and-drop**: hold **BOOTSEL**, plug in USB, copy
  `build-cmsis/zephyr/zephyr.uf2` onto the mounted `RPI-RP2350` drive.
- or `west flash -d build-cmsis` (needs OpenOCD/picotool setup).

Open the console: the board's USB port enumerates as **USB CDC ACM**
(`/dev/ttyACM*`, Raspberry Pi VID 2E8A) — just plug the cable and watch the
digits classify themselves; the onboard LED blinks *(predicted class + 1)*
times per digit. `make monitor` opens picocom/minicom/screen for you.

## Memory budget (Pico 2 W)

| Region | Use | Demo numbers |
| --- | --- | --- |
| QSPI flash (4 MiB, XIP) | Zephyr code + rodata + `model_pte[]` (the .pte) | fp32 ≈ 90 KB, int8 ≈ 25 KB model |
| SRAM — `method_allocation_pool` | method metadata + planned activation buffers + I/O tensors | `CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE=131072`, ~13–46 KB used |
| SRAM — `temp_allocation_pool` | per-op scratch, reset each op | 4 KB |
| SRAM — Zephyr + stacks + heap | RTOS + app | ~40–60 KB |

Print after method load in the console log tells you exactly how much of the
pool is used — size it from measurement, not guesswork. The upstream Pico 2
MNIST example reports ~46 KB used for a tiny MLP with the same pool
mechanism.

## Benchmarking the two backends

Fill this in on your board (`method->execute()` time, `k_cycle_get_32()`):

| Build | Model | avg execute time | flash used | RAM used |
| --- | --- | --- | --- | --- |
| portable | `mnist_cnn_fp32.pte` | ______ us | ______ | ______ |
| cmsis | `mnist_cnn_int8.pte` | ______ us | ______ | ______ |

Reference point from the upstream bare-metal Pico 2 example: their tiny INT8
MLP runs a forward pass in ~245–312 µs on the same 150 MHz Cortex-M33 (see
the [Pico2 MNIST tutorial]).

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| CMake: `ET_PTE_FILE_PATH must point to...` | no model exported yet | run `python3 python/export_mnist.py` |
| `Program::load failed` / garbage output | .pte mismatched to build variant | rebuild with matching `-DET_PTE_FILE_PATH`, `west build -p` |
| `load_method ... MemoryAllocationFailed` / OOM in planned buffer | pool too small | raise `CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE` |
| runtime error `0x14` (operator missing) | selective build missed an op | check `python/export_mnist.py` output; set `CONFIG_EXECUTORCH_BUILD_PORTABLE_OPS=y` |
| linker: `region FLASH overflowed` | all-portable-ops build too big | keep `=n`, selective build only |
| no console output | no CDC ACM device | check `ls /dev/ttyACM*` after flashing; the console is the board's USB port (UART0 GP0/GP1 only with a custom overlay) |
| LED doesn't blink | using plain Pico 2 | LED alias differs; build board `rpi_pico2/rp2350a/m33` (still uses `led0`) |
| export: `torchao`/`executorch` import error | wrong environment | `workon pytorch_cu132` — torch + executorch live there, not in `zephyr_rtos` |

## Version compatibility & provenance

Verified against (Nov 2026): ExecuTorch **v1.5.1** docs and sources, Zephyr
**v4.4.0+** (workspace currently on v4.5.0-rc3), board
`rpi_pico2/rp2350a/m33_0/w`. The heavy lifting here is done
by the official components; this repo is an out-of-tree application + tooling
that follows their documented patterns:

- [ExecuTorch Zephyr module] (`zephyr/` in pytorch/executorch) and its
  [`hello-executorch` sample] — source of the CMake integration pattern
- [Pico2 MNIST tutorial] and [`examples/raspberry_pi/pico2`] — model export
  pattern (`CortexMQuantizer`), memory numbers, UF2 flow
- [Arm Cortex-M Backend] docs — INT8/CMSIS-NN lowering details
- [Zephyr: MobileNetV2 with Ethos-U NPU] — official Zephyr workspace setup
- [Zephyr board docs: rpi_pico2] — board targets incl. the W variant

## License

Apache-2.0 (see `LICENSE`). ExecuTorch itself is BSD-style; CMSIS-NN is
Apache-2.0; Zephyr is Apache-2.0.

[ExecuTorch Zephyr module]: https://github.com/pytorch/executorch/tree/main/zephyr
[`hello-executorch` sample]: https://github.com/pytorch/executorch/tree/main/zephyr/samples/hello-executorch
[Pico2 MNIST tutorial]: https://docs.pytorch.org/executorch/stable/pico2_tutorial.html
[`examples/raspberry_pi/pico2`]: https://github.com/pytorch/executorch/tree/main/examples/raspberry_pi/pico2
[Arm Cortex-M Backend]: https://docs.pytorch.org/executorch/stable/embedded-arm-cortex-m.html
[Zephyr: MobileNetV2 with Ethos-U NPU]: https://docs.pytorch.org/executorch/stable/zephyr_alif_tutorial.html
[Zephyr board docs: rpi_pico2]: https://docs.zephyrproject.org/latest/boards/raspberrypi/rpi_pico2/doc/index.html
[CMSIS-NN]: https://github.com/ARM-software/CMSIS-NN
