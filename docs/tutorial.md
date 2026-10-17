---
title: "Machine Learning on a $15 MCU: TensorFlow Lite Micro meets the Raspberry Pi Pico 2 W"
author: "AI MCU Project"
date: "October 2026"
---

# Machine Learning on a Raspberry Pi Pico 2 W with Zephyr and TensorFlow Lite Micro

This tutorial takes you end-to-end: train a neural network on your PC, compress
it to full 8-bit integer format, embed it into a Zephyr firmware, and run the
inference on a Raspberry Pi Pico 2 W (RP2350, Arm Cortex-M33) -- with every
command, file, and number from a real, tested build.

**What you get at the end**

```
model: 3464 bytes | arena: 1060 bytes used of 4096
input  tensor: int8, scale=0.024591 zero_point=-128
output tensor: int8, scale=0.007899 zero_point=1
x_value: 0.000, y_value: 0.016
x_value: 0.049, y_value: 0.063
x_value: 0.098, y_value: 0.087
...
```

A 3.4 KiB neural network that computes `sin(x)` on the microcontroller itself,
sweeping the full curve forever, printing one inference per line.

---

## 1. Architecture at a glance

```
 PC (machine_learn venv)                    Target (Pico 2 W / RP2350)
+----------------------------+             +---------------------------------+
| generate_model.py          |   .tflite   |  Zephyr RTOS (Cortex-M33)       |
|  train float MLP           | ----------> |  + tflite-micro module          |
|  int8 post-training quant  |  (3.4 KiB)  |    - MicroInterpreter           |
|  validate + op inventory   |             |    - MicroMutableOpResolver     |
|                            |  C array    |    - FULLY_CONNECTED kernel     |
| tflite_to_cc.py            | ----------> |  + app: src/main.cpp            |
+----------------------------+ sine_model_ |    static tensor arena (4 KiB)  |
                                          |    int8 in -> invoke -> int8 out|
                                          +---------------------------------+
```

Two worlds, one artifact. The Python side produces a flatbuffer
(`model_int8.tflite`); the C++ side embeds it as a byte array and interprets
it. No OS-level ML runtime on the MCU: TensorFlow Lite *Micro* links the
interpreter and only the kernels your model uses.

---

## 2. Prerequisites

### Hardware

* Raspberry Pi Pico 2 W (RP2350, 520 KB SRAM, 4 MB flash)
* USB cable (data-capable)
* Optional: a USB-UART adapter on GPIO 0/1 (UART0) if you do not use the USB CDC

### Software (already installed on this machine -- listed for reproduction)

| Component | Where | Version / notes |
|---|---|---|
| Zephyr RTOS | `/home/agonga/zephyrproject/zephyr` | main, v4.4.0-9527 |
| Zephyr SDK | `/home/agonga/zephyr-sdk-1.0.1` | arm toolchain + picotool |
| west | `zephyr_rtos` virtualenv | v1.5.0 |
| TensorFlow | `machine_learn` virtualenv | 2.21.0 (CPU) |
| tflite-micro | `/home/agonga/zephyrproject/modules/tflite-micro` | zephyrproject-rtos fork |

Virtual environments (virtualenvwrapper):

```
workon zephyr_rtos      # enter: west, Zephyr build tooling
workon machine_learn    # enter: TensorFlow (CPU already installed)
deactivate              # leave either environment
```

### Board support in Zephyr

Zephyr main ships the Pico 2 board family in `zephyr/boards/raspberrypi/`.
The Pico 2 **W** (with the Infineon/Cypress CYW43439 radio) is selected via a
board *variant*:

```
rpi_pico2/rp2350a/m33_0/w     # Pico 2 W: RP2350, Cortex-M33 core, WiFi variant
rpi_pico2/rp2350a/m33       # plain Pico 2 (no radio)
```

This exact qualifier was confirmed working on this setup (see
`zephyr/AGO_NOTES.txt`).

---

## 3. Step 1 -- Put the tflite-micro module under `modules/`

Zephyr consumes external code through **west projects** declared in the
manifest. TensorFlow Lite Micro is an *optional* Zephyr project; upstream
declares it in `zephyr/submanifests/optional.yaml`:

```yaml
- name: tflite-micro
  revision: fcc760af130f3a595b5802cdebcc77461e54f382
  path: optional/modules/lib/tflite-micro   # <- upstream default location
  repo-path: tflite-micro
  remote: upstream
  groups:
    - optional
```

We want it in the standard tree at `zephyrproject/modules/tflite-micro`.
A note on a trap: simply adding a second `tflite-micro` entry to the main
`west.yml` does **not** work -- west resolves the project from the submanifest,
so the duplicate is silently shadowed (`west list tflite-micro` keeps pointing
at the old path). The path must be changed where west actually reads it:

```
# 1. Revert the ineffective duplicate (if you added one) -- inside the zephyr repo:
git checkout -- west.yml

# 2. Point the submanifest at the new location:
#    zephyr/submanifests/optional.yaml:
#      path: optional/modules/lib/tflite-micro
#    becomes
#      path: modules/tflite-micro

# 3. Move the git checkout (history and pinned revision travel with it):
mv /home/agonga/zephyrproject/optional/modules/lib/tflite-micro \
   /home/agonga/zephyrproject/modules/tflite-micro
rmdir /home/agonga/zephyrproject/optional/modules/lib \
      /home/agonga/zephyrproject/optional/modules \
      /home/agonga/zephyrproject/optional

# 4. Verify west now resolves the new path:
west list tflite-micro
# tflite-micro  modules/tflite-micro  fcc760af...  https://github.com/zephyrproject-rtos/tflite-micro
```

### Where do the CMake and Kconfig files come from?

You do **not** need to write any CMake/Kconfig glue inside the module.
The fork only ships `zephyr/module.yml`:

```yaml
name: tflite-micro
build:
  cmake-ext: True
  kconfig-ext: True
```

and Zephyr main provides the glue **in-tree** at
`zephyr/modules/tflite-micro/{CMakeLists.txt,Kconfig}`. The build system globs
that directory, wires `ZEPHYR_TFLITE_MICRO_CMAKE_DIR` /
`ZEPHYR_TFLITE_MICRO_KCONFIG`, and the module's CMakeLists compiles ~170
tflite-micro sources from wherever `module.yml` was discovered -- including the
pre-exported third-party headers in `third_party_static/` (flatbuffers,
gemmlowp, ruy, kissfft). The Kconfig file defines the switches your
`prj.conf` uses:

* `CONFIG_TENSORFLOW_LITE_MICRO` -- the library (selects full C++ support)
* `CONFIG_TENSORFLOW_LITE_MICRO_CMSIS_NN_KERNELS` -- Arm CMSIS-NN optimized
  kernels instead of reference kernels (Cortex-M only; see Section 11)

Because the module is registered by west, `ZEPHYR_BASE`-based builds find it
automatically -- no `ZEPHYR_EXTRA_MODULES` needed.

---

## 4. Step 2 -- Train and quantize the model (PC side)

All under `ai_rpi2350/model/`. The wrapper
`generate_model.py` drives the shared pipeline
`script/generate_tf_mcu_model.py` (train -> save -> quantize -> validate ->
inventory) with this application's output paths.

### Why full int8?

On an MCU, 8-bit arithmetic is roughly 4x cheaper in flash bandwidth and RAM
traffic than float32, integer kernels need no FPU, and -- crucially --
*quantized tensors at the model boundary* mean the model contains no
Quantize/Dequantize wrapper nodes: the firmware feeds raw int8 bytes and reads
raw int8 bytes. Post-training quantization (PTQ) needs no retraining: the
converter observes a **representative dataset** and picks activation ranges.

### Run it

```
workon machine_learn
cd /home/aimlrn/ai_mcu/ai_rpi2350
make model            # = python model/generate_model.py  (800 epochs)
```

What happens, step by step:

1. **Train** a 2x16-ReLU fully-connected MLP on `y = sin(x)`, x in [0, 2pi],
   1000 samples, 800 epochs (final MSE 0.000057). Save as `.keras` +
   `SavedModel` so quantization never needs retraining.
2. **Calibrate**: write 500 representative inputs to `model/calib/*.npy`.
3. **Quantize** with `TFLiteConverter`:
   `optimizations=[DEFAULT]`, `supported_ops=[TFLITE_BUILTINS_INT8]`,
   `inference_input/output_type=int8`, representative dataset attached.
   Result: `model_int8.tflite`, **3464 bytes** (vs 3.2 KiB float32 -- tiny
   models are dominated by schema overhead, so int8 wins big only at larger
   sizes; the win here is integer-only inference).
4. **Validate**: the pipeline prints the *tensor contract* the firmware must
   honor and the accuracy cost of quantization:

```
[validate] input : shape=(1, 1) dtype=int8 quant=(0.024590613320469856, -128)
[validate] output: shape=(1, 1) dtype=int8 quant=(0.007898937910795212, 1)
[validate] TFLite int8 : MSE = 0.000169, max|err| = 0.057487
[validate] float Keras: MSE = 0.000060, max|err| = 0.013927
```

Quantization noise is 0.0001 MSE -- invisible on a serial console.

5. **Inventory** the ops the firmware must provide:

```
[inventory] 2 unique ops: DELEGATE, FULLY_CONNECTED
```

`DELEGATE` is the host-side XNNPACK artifact (ignore it); the only real kernel
is `FULLY_CONNECTED` -- ReLU gets fused into its activation options. This
tells us exactly what to register in the firmware.

### The quantization contract

For each I/O tensor, `(scale, zero_point)` defines the float<->int8 map:

```
x_int8 = round(x_float / scale) + zero_point      # feed the model
y_float = (y_int8 - zero_point) * scale           # read the answer
```

The firmware reads these values from the *live tensors* at runtime
(`input->params.scale`), so it never hard-codes them -- retrain the model and
the firmware keeps working unchanged.

---

## 5. Step 3 -- Embed the model into C++

```
make header
# = python model/tflite_to_cc.py model/model_int8.tflite \
#       --source src/sine_model_data.cc --header src/sine_model.h
```

`tflite_to_cc.py` checks the flatbuffer magic (`TFL3`) and emits a byte
array -- the same thing the upstream sample does with `xxd -i`, plus 16-byte
alignment:

```cpp
/* src/sine_model_data.cc (generated) */
alignas(16) const unsigned char g_sine_model[] = {
    0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
};
const int g_sine_model_len = 3464;
```

`tflite::GetModel()` maps this array in place -- zero copy, no filesystem.

---

## 6. Step 4 -- The firmware

Layout of `ai_rpi2350/`:

```
ai_rpi2350/
+-- CMakeLists.txt     # standard Zephyr app; C++17, no thread-safe statics
+-- prj.conf           # CONFIG_CPP, CONFIG_TENSORFLOW_LITE_MICRO, float printf
+-- Makefile           # model / header / build / qemu / flash_mcu / mcu_monitor
+-- boards/            # board-specific; auto-applied for rpi_pico2/.../w only
|   +-- rpi_pico2_rp2350a_m33_0_w.overlay   # USB CDC ACM console (devicetree)
|   \-- rpi_pico2_rp2350a_m33_0_w.conf      # USB device stack + CDC ACM (Kconfig)
+-- src/
|   +-- main.cpp               # interpreter, resolver, int8 I/O, printout
|   +-- sine_model.h           # extern decls (generated)
|   \-- sine_model_data.cc     # model bytes (generated)
\-- model/
    +-- generate_model.py      # training pipeline wrapper
    +-- tflite_to_cc.py        # .tflite -> C array
    \-- model_int8.tflite      # the artifact
```

The interesting part of `src/main.cpp`:

```cpp
/* 1. Map the flatbuffer. */
model = tflite::GetModel(g_sine_model);

/* 2. Register exactly the kernels the op inventory listed. */
static tflite::MicroMutableOpResolver<1> resolver;
resolver.AddFullyConnected();

/* 3. Interpreter over a static arena (no heap). */
static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
static_interpreter.AllocateTensors();      // arena: 1060 bytes used

/* 4. One inference: quantize -> invoke -> dequantize -> print. */
input->data.int8[0] = (int8_t)lroundf(x / input->params.scale)
                    + input->params.zero_point;
interpreter->Invoke();
float y = (output->data.int8[0] - output->params.zero_point)
        * output->params.scale;
printk("x_value: %.3f, y_value: %.3f\n", x, y);
```

Design notes:

* **Static arena** (`uint8_t tensor_arena[4096] __aligned(16)`): all
  activations, operator temporaries and the graph live here. TFLite-Micro
  never allocates from the heap at runtime; `arena_used_bytes()` reports the
  high-water mark (1060 B -- shrink the array or grow your model accordingly).
* **Op resolver size** (`<1>`): the template argument is the *capacity* of the
  registration table; each registration costs a little flash. Match it to the
  inventory.
* **Batch of 1**: the input shape is `[1,1]`; MCUs always run batch 1, which
  keeps the arena tiny.
* `CONFIG_REQUIRES_FLOAT_PRINTF=y` enables `%.3f` in `printk` without pulling
  in a heavyweight libc formatter.

---

## 7. Step 5 -- Smoke-test on QEMU (no hardware needed)

The same application builds for `qemu_x86` -- this validates model, kernel
registrations and console output in seconds:

```
workon zephyr_rtos
cd /home/aimlrn/ai_mcu/ai_rpi2350
make qemu          # builds for qemu_x86, runs it for 30 s, greps the console
```

Observed output (abridged):

```
*** Booting Zephyr OS build v4.4.0-9527-g8f3807f55780 ***
model: 3464 bytes | arena: 1060 bytes used of 4096
input  tensor: int8, scale=0.024591 zero_point=-128
output tensor: int8, scale=0.007899 zero_point=1
x_value: 0.000, y_value: 0.016
x_value: 0.049, y_value: 0.063
...
x_value: 6.234, y_value: -0.039
[qemu] sine inference verified on qemu_x86
```

The runner feeds QEMU's stdin from `/dev/null` on purpose: under `timeout`
the child runs in a *background* process group, and a QEMU that reads the
controlling TTY would be stopped by SIGTTIN before printing its first
serial byte -- the console looks dead and the smoke test fails with
"no inference output found" (it only happens in an interactive terminal;
piped stdin masks the problem). Output still streams live to the
terminal; stop early with `Ctrl-C`, or let the 30 s timeout end the run.

The `y` values trace `sin(x)`: up to +1 near `x = 1.57`, down to -1 near
`x = 4.71`, back to 0 at `2*pi`. (~200k inferences in a 20-second run.)

---

## 8. Step 6 -- Build and flash the Pico 2 W

```
workon zephyr_rtos
cd /home/aimlrn/ai_mcu/ai_rpi2350
make build         # BOARD defaults to rpi_pico2/rp2350a/m33_0/w
```

Result from this build:

```
Memory region         Used Size  Region Size  %age Used
           FLASH:       55924 B         4 MB      1.33%
             RAM:       11544 B       520 KB      2.17%
...
Converted to uf2, output size: 112128, start address: 0x10000000
Wrote 112128 bytes to zephyr.uf2
```

~56 KB of flash (the interpreter + kernels dominate), ~11.5 KB RAM -- about
2% of both budgets, leaving room for your real application around the model.

### Flashing (`make flash_mcu`)

Hold the BOOTSEL button while plugging in USB; the board appears as a mass
storage device `RPI-RP2`:

```
make flash_mcu      # west flash -r uf2 (uses picotool); `make flash` is an alias
```

The target refuses to run if `build_rpi_pico2_rp2350a_m33_0_w/zephyr.uf2` is
missing (run `make build` first), reminds you about BOOTSEL, writes the UF2,
and the board reboots straight into the inference firmware.

### How the USB console is wired (board-specific)

Two small files, auto-applied by Zephyr **only** when building for
`rpi_pico2/rp2350a/m33_0/w` (the `qemu_x86` smoke test keeps its regular
console). They take the USB CDC ACM part of
`zephyr/samples/basic/pico2w_gpio16_blinky/app.overlay` and its `prj.conf`,
nothing else (the GPIO16 LED / buzzer nodes were dropped).

Note the `_m33_0_` in the fragment names: the RP2350 is dual-core, and since
the upstream qualification rework (Oct 2026) the canonical board id is
`rpi_pico2/rp2350a/m33_0/w` -- previously `m33/w`, which still resolves via
an alias for builds but *not* for fragment matching. Board-scoped files are
matched against the canonical id and fail silently when it changes: the
symptom is a missing USB CDC console and a ~11 KB smaller firmware, so after
a Zephyr update check `CONFIG_USB_CDC_ACM=y` in `build_*/zephyr/.config`.

`boards/rpi_pico2_rp2350a_m33_0_w.overlay` (devicetree):

```dts
/ {
    /* Route the console to the CDC ACM UART instance below. */
    chosen {
        zephyr,console = &cdc_acm_uart0;
    };
};

/* CDC ACM UART instance under the Pico's USB device controller. */
&zephyr_udc0 {
    cdc_acm_uart0: cdc_acm_uart0 {
        compatible = "zephyr,cdc-acm-uart";
    };
};
```

`boards/rpi_pico2_rp2350a_m33_0_w.conf` (Kconfig):

```
CONFIG_USB_DEVICE_STACK=y
CONFIG_USB_CDC_ACM=y
CONFIG_USB_DEVICE_INITIALIZE_AT_BOOT=y
CONFIG_USB_DEVICE_VID=0x2E8A      # Raspberry Pi
CONFIG_USB_DEVICE_PID=0x000A
CONFIG_USB_DEVICE_PRODUCT="Pico2W TFLM Sine"
CONFIG_SERIAL=y / CONFIG_CONSOLE=y / CONFIG_UART_CONSOLE=y
CONFIG_UART_INTERRUPT_DRIVEN=y / CONFIG_UART_LINE_CTRL=y
```

The CMakeLists also silences the legacy-USB-stack deprecation warnings
(`-Wno-deprecated-declarations`, only when `CONFIG_USB_CDC_ACM` is on) --
same as the sample. Cost of the USB console: **+11.4 KB flash, +8.9 KB RAM**
(see Section 9). Prefer a UART adapter on GPIO 0/1 instead? Remove/rename the
two `boards/` files and the console falls back to UART0 @115200.

Expect a few *"Deprecated symbol USB_DEVICE_STACK is enabled"* /
*"Macro is deprecated"* **warnings** during the build: on Zephyr main the
legacy USB device stack is marked deprecated (the new stack uses a different
configuration scheme). They are harmless -- the build completes and the
firmware runs; the sample that inspired this setup shows the same messages.

### Watching the sine curve (`make mcu_monitor`)

The firmware enumerates on the host as a Raspberry Pi CDC serial device
(VID `2E8A`) on the board's **own USB port** -- plug in a plain USB cable,
no UART adapter needed:

```
make mcu_monitor                  # minicom on /dev/ttyACM0 @ 115200
make mcu_monitor SERIAL_PORT=/dev/ttyUSB0          # other adapter names
make mcu_monitor SERIAL_PORT=/dev/ttyACM0 SERIAL_BAUD=115200
```

The target picks whichever serial tool is installed (minicom, then picocom,
then screen) and, if the port is missing, lists the serial devices that *are*
present so you can pick the right `SERIAL_PORT`. If you open the terminal
after the board has booted, the startup banner may already be gone -- the
sweep repeats every second, so data flows continuously. (Exit minicom with
`Ctrl-A` then `X`.)

You will see the same banner and the `x_value/y_value` sweep as in QEMU,
repeating every second (128 inferences per sweep, then a 1 s pause).

---

## 9. Performance and sizing cheat-sheet

| Quantity | Value here | How to read it |
|---|---|---|
| Model flatbuffer | 3464 B | lives in flash as a `const` array |
| Tensor arena | 1060 B of 4096 B | RAM; set size from `arena_used_bytes()` |
| Firmware flash | 67.4 KB (55.9 KB w/o USB CDC console) | interpreter + kernels + app + RTOS (+ USB stack) |
| Firmware RAM | 20.4 KB (11.5 KB w/o USB CDC console) | arena + stacks + RTOS objects (+ USB buffers) |
| Ops registered | 1 (`FULLY_CONNECTED`) | from the inventory step |
| Inference rate (QEMU x86) | ~10k/s | RP2350 M33 will be far slower but ample |

Rules of thumb: arena ~= 2-3x total tensor bytes; each extra registered op
adds a few hundred bytes of flash; `west build -t rom_report` attributes flash
per component.

---

## 10. Troubleshooting

**`west build` cannot find the tflite-micro module / Kconfig.**
Run west commands from inside the workspace (`cd /home/agonga/zephyrproject`)
or -- as this project's Makefile does -- `cd` there and pass absolute app and
build paths. Check `west list tflite-micro` shows `modules/tflite-micro`.

**`AllocateTensors() failed`.**
The arena is too small for the model's tensors. Print
`interpreter->arena_used_bytes()` on a working setup or bump
`kTensorArenaSize`.

**`Invoke failed`.**
An op is missing from the resolver -- rerun the inventory step and add the
matching `Add*()` calls (and grow the resolver template parameter).

**Garbage output values.**
Almost always the quantization contract: feeding floats into an int8 tensor or
mixing up scale/zero_point of input vs output. Use the *live* tensor's
`input->params` / `output->params`.

**`%f` prints as raw `%f`.**
`CONFIG_REQUIRES_FLOAT_PRINTF=y` missing from `prj.conf`.

**Board not enumerating for `make flash_mcu` / `west flash`.**
BOOTSEL while plugging in; if already flashed, hold BOOTSEL 3 s to reboot to
mass storage. `picotool` must be on `PATH` (it is, at `/usr/local/bin`).

**`make mcu_monitor` says the serial port does not exist.**
The target lists the devices it found; pick the right one, e.g.
`make mcu_monitor SERIAL_PORT=/dev/ttyUSB0`. No entries at all means the
board/adapter is not plugged in (or you are still in BOOTSEL, which exposes
mass storage, not a serial port).

**Duplicate `tflite-micro` in west.**
Never declare the same project in both `west.yml` and a submanifest; west
honours the submanifest and ignores the shadowed entry.

---

## 11. Where to go next

* **CMSIS-NN kernels**: set
  `CONFIG_TENSORFLOW_LITE_MICRO_CMSIS_NN_KERNELS=y` in `prj.conf`. The same
  model then routes `FULLY_CONNECTED` (and conv/pool/etc. for bigger models)
  through Arm's CMSIS-NN library -- several times faster on int8 on Cortex-M.
  Available on this build tree; drop-in for this project.
* **Bigger models**: keyword spotting (DS-CNN), gesture recognition
  (`samples/modules/tflite-micro/magic_wand`), image classification with the
  RP2350's second core for pre-processing.
* **Sensor input**: replace the `sin(x)` sweep with real data (I2C IMU,
  microphone via PDM); keep the same int8 contract by feeding quantized
  samples computed with the input tensor's scale.
* **Networking**: the Pico 2 W's CYW43439 is supported by Zephyr's WiFi
  stack (see `zephyr/AGO_NOTES.txt` for the airoc driver build) -- stream
  inference results to MQTT/HTTP.
* **Power**: run one inference per wakeup from deep sleep; the 4 KiB arena
  keeps retention RAM needs small.

---

## Appendix A -- File map (everything created by this project)

```
/home/aimlrn/ai_mcu/
+-- IMPLEMENTATION_PLAN.md              # plan + decisions + deviations
+-- docs/tutorial.md                    # this document
+-- docs/tutorial.pdf                   # PDF rendering
+-- research/project-plan.txt           # original brief (+ status addendum)
+-- script/generate_tf_mcu_model.py     # shared train/quantize pipeline (pre-existing)
\-- ai_rpi2350/
    +-- CMakeLists.txt
    +-- prj.conf
    +-- Makefile
    +-- boards/{rpi_pico2_rp2350a_m33_0_w.overlay, rpi_pico2_rp2350a_m33_0_w.conf}
    +-- src/{main.cpp, sine_model.h, sine_model_data.cc}
    \-- model/{generate_model.py, tflite_to_cc.py, model_int8.tflite, calib/, sine_model*}
```

Workspace change (outside the repo, one-time):

```
zephyrproject/modules/tflite-micro/                 # moved from optional/modules/lib/
zephyrproject/zephyr/submanifests/optional.yaml     # path: modules/tflite-micro
```

## Appendix B -- Makefile targets

| Target | Effect |
|---|---|
| `make model` | train + int8-quantize (machine_learn venv python) |
| `make header` | embed `.tflite` into `src/sine_model_data.cc` |
| `make build` | west build for `rpi_pico2/rp2350a/m33_0/w` -> `zephyr.uf2` |
| `make qemu` | build + run on `qemu_x86`, verify console output |
| `make flash_mcu` | `west flash -r uf2` (BOOTSEL mode; alias: `make flash`) |
| `make mcu_monitor` | serial console, `SERIAL_PORT=/dev/ttyACM0` `SERIAL_BAUD=115200` (minicom/picocom/screen) |
| `make size` / `rom_report` | RAM/ROM usage |
| `make menuconfig` | interactive Kconfig |
| `make clean` / `distclean` | drop build trees / also drop model artifacts |
