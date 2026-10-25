# Implementation Plan — TFLite-Micro on Raspberry Pi Pico 2 W (RP2350) with Zephyr

Source of truth: `research/project-plan.txt`. This file records the plan, the
decisions taken while executing it, and the deviations from the original text.

## 0. Environment facts (verified, not assumed)

| Item | Value |
|---|---|
| West workspace (topdir) | `/home/agonga/zephyrproject` |
| Zephyr | main, `v4.4.0-9527-g8f3807f5578` |
| Zephyr SDK | `/home/agonga/zephyr-sdk-1.0.1` |
| Board (Pico 2 W) | `rpi_pico2/rp2350a/m33_0/w` (from `zephyr/AGO_NOTES.txt`) |
| tflite-micro module | west project `tflite-micro`, fork `zephyrproject-rtos/tflite-micro` @ `fcc760af` |
| Module glue (CMake+Kconfig) | in-tree: `zephyr/modules/tflite-micro/{CMakeLists.txt,Kconfig}` — auto-discovered, defines `CONFIG_TENSORFLOW_LITE_MICRO` |
| Flash / serial | `west flash -r uf2`, `minicom -D /dev/ttyAC0 -b115200` |
| venv (build) | `workon zephyr_rtos` → west 1.5.0 |
| venv (ML) | `workon machine_learn` → TensorFlow 2.21.0 (CPU, already installed) |
| PDF toolchain | pandoc 3.1.3 + pdflatex |

## 1. Phase 1 — Reconfigure the tflite-micro west module

Problem: the module checkout is at `optional/modules/lib/tflite-micro` (path set
by the upstream `zephyr/submanifests/optional.yaml`). A hand-added duplicate
entry in `zephyr/west.yml` (`path: modules/tflite-micro`) is **shadowed by the
submanifest**, so it had no effect — `west list tflite-micro` still resolves the
`optional/...` path.

Steps (in order):
1. Revert `zephyr/west.yml` to upstream (`git checkout -- west.yml`) — the only
   diff was the ineffective duplicate entry.
2. Edit `zephyr/submanifests/optional.yaml`: change
   `path: optional/modules/lib/tflite-micro` → `path: modules/tflite-micro`.
3. `mv /home/agonga/zephyrproject/optional/modules/lib/tflite-micro
   /home/agonga/zephyrproject/modules/tflite-micro` (git history and the pinned
   revision `fcc760af` travel with it; remove the now-empty `optional/` dirs).
4. Verify: `west list tflite-micro` shows `modules/tflite-micro` +
   `west status` clean; a real build (Phase 4) is the definitive proof.

No changes inside the module repo are required: `zephyr/module.yml`
(`cmake-ext`/`kconfig-ext`) + Zephyr's in-tree glue is the official mechanism.

## 2. Phase 2 — The Zephyr application (`projects/tflite-micro/`)

Correction: the plan says `/home/aimlrn/ai_cmu`; that directory does not exist.
The project is implemented under **`/home/aimlrn/ai_mcu/`** (the working
directory that already held `research/` and `script/`). Since then the repo was
reorganized for multiple projects — see deviation 8 for the current layout.

```
projects/tflite-micro/
├── CMakeLists.txt            # standard Zephyr app, C++17, no-threadsafe-statics
├── prj.conf                  # CONFIG_CPP, CONFIG_TENSORFLOW_LITE_MICRO, float printf
├── Makefile                  # model / header / build / qemu / flash_mcu / mcu_monitor
├── boards/                   # board-specific, auto-applied for rpi_pico2/.../w only
│   ├── rpi_pico2_rp2350a_m33_0_w.overlay  # USB CDC ACM console (devicetree)
│   └── rpi_pico2_rp2350a_m33_0_w.conf     # USB device stack + CDC ACM (Kconfig)
├── scripts/
│   └── run_qemu.sh           # qemu runner: forwards Ctrl-C to timeout/QEMU
├── src/
│   ├── main.cpp              # MicroInterpreter + MicroMutableOpResolver, int8 I/O
│   ├── sine_model.h          # extern declarations for the embedded model
│   └── sine_model_data.cc    # GENERATED: model bytes as C array (alignas(16))
└── model/
    ├── generate_model.py     # wrapper around tools/ml/generate_tf_mcu_model.py pipeline
    └── tflite_to_cc.py       # .tflite -> sine_model_data.cc + sine_model.h
```

Firmware design (mirrors Zephyr's `samples/modules/tflite-micro/hello_world`,
which targets this exact module/glue):
- Tiny 2×16-ReLU MLP trained on `y = sin(x)`, fully int8-quantized.
- `tflite::MicroMutableOpResolver` registers exactly the ops the model needs
  (from the generator's op inventory — `FULLY_CONNECTED`).
- Input quantization / output dequantization use the tensor's
  `(scale, zero_point)` — the contract printed by the generator's validate step.
- Sweeps `x ∈ [0, 2π)` printing `x_value: …, y_value: …` (float printk via
  `CONFIG_REQUIRES_FLOAT_PRINTF`) every 100 ms, forever.

## 3. Phase 3 — Model generation

- Wrapper `model/generate_model.py` imports the existing pipeline in
  `tools/ml/generate_tf_mcu_model.py` (single source of truth) and runs
  train → int8-quantize (representative dataset from the same sine inputs) →
  validate → op inventory, writing `model/model_int8.tflite`.
- `model/tflite_to_cc.py` converts the flatbuffer into `src/sine_model_data.cc`
  + `src/sine_model.h` (`alignas(16) const unsigned char g_sine_model[]`).
- Runs with `workon machine_learn` python (TF CPU — already installed; the
  plan's "may need to install TF CPU" is not needed).

## 4. Phase 4 — Build & verify

1. **QEMU smoke test** (`qemu_x86`): proves model + interpreter + console
   output end-to-end without hardware. Console must show the `x_value: …,
   y_value: …` sine pattern.
2. **Pico 2 W build** (`rpi_pico2/rp2350a/m33_0/w`, pristine): produces
   `zephyr.uf2` + RAM/Flash usage. Flashing is left to the user
   (`make flash_mcu`, alias `make flash` -> BOOTSEL UF2 via `west flash -r uf2`;
   `make mcu_monitor` -> serial console, minicom/picocom/screen, defaults
   /dev/ttyACM0 @115200, SERIAL_PORT/SERIAL_BAUD overridable).

## 5. Phase 5 — Tutorial (.md + .pdf)

`docs/tutorial.md` → `docs/tutorial.pdf` (pandoc + pdflatex). Contents:
overview & architecture, environment & venvs, the west-module reconfiguration
(with the why), model training/quantization theory, embedding the model,
firmware walkthrough, build/QEMU/flash/run, expected output, troubleshooting,
next steps (CMSIS-NN kernels, sensors, networking).

## 6. Phase 6 — Update the plan files

- Append a STATUS/deviations section to `research/project-plan.txt` (per its
  "Extras" instruction) and keep this file as the detailed record.

## Deviations log

1. `/home/aimlrn/ai_cmu` → `/home/aimlrn/ai_mcu` (typo; target dir did not exist).
2. `west.yml` duplicate tflite-micro entry removed; the real path switch is done
   in `submanifests/optional.yaml` (the submanifest is what west actually uses).
3. No TF install performed — TF 2.21.0 (CPU) was already present in the
   `machine_learn` venv.
4. USB CDC ACM console added on top of the base project (user request): the
   USB-CDC-only part of `zephyr/samples/basic/pico2w_gpio16_blinky/app.overlay`
   + its `prj.conf` USB options, applied as board-scoped files
   (`boards/rpi_pico2_rp2350a_m33_0_w.{overlay,conf}`) so the qemu_x86 smoke test
   keeps its regular console. Console now enumerates natively as `/dev/ttyACM*`
   (VID 2E8A). Cost: +11.4 KB flash, +8.9 KB RAM; firmware 67.4 KB flash /
   20.4 KB RAM. Legacy-USB-stack deprecation warnings during build are expected
   (same as in the inspiring sample).
5. `make qemu` smoke-test quirk fixed: `timeout` runs the child in a background
   process group, so QEMU reading the controlling TTY raises SIGTTIN and stops
   before its first serial byte — the test passed with piped stdin (CI/agent)
   but showed an empty console in an interactive terminal. Fix: run
   `west build -t run < /dev/null` (output still streams via tee; the timeout
   stops the run — Ctrl-C did not, see item 7). `timeout --foreground` was
   rejected: the orphaned QEMU then holds the tee pipe open and make hangs.
6. Zephyr main moved forward (user rebased onto origin/main, Oct 2026;
   conflict in `submanifests/optional.yaml` resolved as the union of
   upstream's new tflite-micro pin `43b2a106` and our
   `path: modules/tflite-micro`). Follow-ups required by the jump: `west update`
   synced every module (a stale `hal/rpi_pico` broke the hwinfo driver build),
   and the RP2350 board qualifier was renamed upstream `m33/w` → `m33_0/w` —
   board-scoped fragments match the *canonical* board id, so they silently
   stopped applying (USB CDC vanished; caught via firmware size dropping back
   to 56 KB). Fixed by renaming `boards/` fragments to
   `rpi_pico2_rp2350a_m33_0_w.*` and switching Makefile BOARD to the canonical
   `rpi_pico2/rp2350a/m33_0/w`. Verified: pico build 68.1 KB flash with CDC
   restored, qemu smoke test green.
7. `make qemu` Ctrl-C fixed (follow-up to 5): `timeout` puts west/ninja/QEMU
   in their own process group, so the terminal SIGINT never reached QEMU, and
   the `-` recipe prefix made make ignore the interrupt. On this host
   `timeout` is uutils coreutils, which moves *itself* into that group too, so
   it never received the terminal signal either. Added
   `projects/tflite-micro/scripts/run_qemu.sh`: it traps INT/TERM, signals `timeout`
   (which forwards to the whole group so QEMU gets a graceful SIGINT) and
   falls back to the `qemu.pid` written by Zephyr's QEMU runner. Dropped the
   `-` prefix so make aborts on Ctrl-C. Verified in a PTY: make exits
   immediately, no QEMU/timeout/west processes are left behind; the timeout
   path still passes the inference grep.
8. Repo reorganized for multiple projects (the working directory now hosts
   more than this one). One project's files had been spread over four
   top-level directories, and the shared training pipeline sat next to
   them. New layout: `projects/<name>/` holds each project root, `tools/ml/`
   holds reusable tooling. Concretely: `ai_rpi2350/` → `projects/tflite-micro/`,
   `script/generate_tf_mcu_model.py` → `tools/ml/`, and the repo-level
   `IMPLEMENTATION_PLAN.md`, `docs/`, `research/` moved into the project.
   The stale `build_*/` trees were deleted rather than moved — CMakeCache
   and build.ninja bake in absolute paths, so they were invalid the moment
   anything moved (regenerate with `make build` / `make qemu`).
   References fixed: Makefile `PIPELINE_PY`, `generate_model.py`'s
   `sys.path`, `.vscode` cmake.sourceDirectory, `.gitignore` globs (now
   `projects/*/…` so they cover future projects), tutorial paths and file
   maps, and the generated-file source comments in `src/sine_model.{h,cc}`
   (a `make header` regen now writes the new path itself).
