# ARCHITECTURE — ExecuTorch on Zephyr on RP2350

This document explains *why* the demo is built the way it is: the model
lifecycle, the runtime object graph, the memory plan, the backend dispatch
mechanics, and the trade-offs an Edge AI engineer has to reason about when
bringing PyTorch models to a Cortex-M33 with 520 KiB of SRAM.

---

## 1. The four planes

```
┌─────────────────────────────────────────────────────────────────────────┐
│ A. MODEL PLANE (host)                                                   │
│    PyTorch eager model                                                  │
│      → torch.export.export()          ATen graph, functionalized        │
│      → [INT8: CortexMQuantizer]       PT2E: prepare_pt2e → calibrate    │
│                                       → convert_pt2e (QDQ graph)        │
│      → to_edge()                      edge dialect (CORE ATen ops)      │
│      → [INT8: CortexMPassManager]     swap quantized ops for CMSIS-NN   │
│      → to_executorch()                .pte flatbuffer (default method   │
│                                       "forward", constant segments,     │
│                                       memory-planned buffer layout)     │
├─────────────────────────────────────────────────────────────────────────┤
│ B. ARTIFACT PLANE                                                       │
│    mnist_cnn_fp32.pte / mnist_cnn_int8.pte                              │
│      → pte_to_header.py at build time → const model_pte[] in section    │
│        "network_model_sec" → linked into FLASH (XIP)                    │
├─────────────────────────────────────────────────────────────────────────┤
│ C. RUNTIME PLANE (device, app/src/main.cpp)                             │
│    runtime_init() → BufferDataLoader(flash span) → Program::load()      │
│      → MethodMeta (planned buffer sizes) → Method::load()               │
│      → per inference: set_input() → execute() → get_outputs()           │
├─────────────────────────────────────────────────────────────────────────┤
│ D. KERNEL PLANE                                                         │
│    cpu_portable_ops_lib   (portable functions.yaml, selected by model)  │
│    zephyr_cortex_m_ops_lib(backends/cortex_m/ops/operators.yaml, INT8,  │
│                            implemented by cortex_m_kernels = CMSIS-NN)  │
└─────────────────────────────────────────────────────────────────────────┘
```

The planes meet at one file: `app/CMakeLists.txt` decides *which kernels*
are linked; the `.pte` decides *which implementations* an operator
dispatches to at runtime.

## 2. What is inside a `.pte`?

An ExecuTorch program is a flatbuffer containing:

- **Method(s)** — the "forward" function as a flatbuffer instruction list
  over the edge dialect (CORE ATen operators, e.g. `aten::convolution.out`,
  `aten::addmm.out`, `aten::relu.out`);
- **Constant data** — weights/biases laid out in constant segments. With
  `extract_delegate_segments=False` (what the Cortex-M flow uses) they stay
  in the plain constant section;
- **Memory-planned buffer layout** — the sizes of the activation buffers,
  already lifetime-overlapped by the AOT memory planner. The device does
  *zero* runtime planning: `MethodMeta::memory_planned_buffer_size(id)`
  reports each span's size and the app just hands over memory;
- **Backend-specific op registrations** — for the INT8 build, custom
  operators like `quantized_conv2d` from `backends/cortex_m/ops/operators.yaml`
  that the runtime resolves against the linked kernel libraries.

Two consequences:

1. The dispatch decision is **baked into the .pte**. A float graph executed
   on the same firmware uses portable kernels; a quantized graph uses
   CMSIS-NN kernels — the application source is unchanged. This is why the
   demo can offer "both paths" with a single `west build` flag pointing at a
   different file.
2. Selective builds are safe: `arm_runner_create_selected_ops_lib(...,
   OPS_FROM_MODEL=model.pte)` reads the operator table from the serialized
   program itself (including ops introduced by lowering passes), so nothing
   that the graph needs can be missed.

## 3. Runtime object graph (device)

From `app/src/main.cpp`, mirroring the official `hello-executorch` runner:

| Object | Lifetime | Backing memory |
| --- | --- | --- |
| `BufferDataLoader` | whole program | none — returns spans into `model_pte[]` (flash) |
| `Program` | whole run | parsed in method pool |
| `Method` | whole run | method pool |
| `HierarchicalAllocator` (planned memory) | whole run | spans carved from `method_allocation_pool` |
| input tensor | re-filled per digit | 3 136 B from method pool |
| `temp_allocator` | per-op | `temp_allocation_pool`, reset each kernel |

Call sequence:

```cpp
executorch::runtime::runtime_init();                       // platform layer
BufferDataLoader loader(model_pte, sizeof(model_pte));     // zero-copy
auto program  = Program::load(&loader);                    // parse flatbuffer
auto mmeta    = program->method_meta("forward");           // sizes, dtypes
// carve planned buffers out of the static pool...
auto method   = program->load_method("forward", &memory_manager, nullptr);
method->set_input(tensor, 0);
method->execute();
method->get_outputs(&value, 1);
```

Error handling uses `Result<T>` + `ET_CHECK_MSG` — ExecuTorch builds with
exceptions and RTTI disabled (`-fno-exceptions -fno-rtti`), which is also
why the Zephyr module Kconfig selects `REQUIRES_FULL_LIBCPP` and builds with
`CONFIG_STD_CPP17`.

## 4. Memory map on the Pico 2 W

| Address range | Region | Contents in this demo |
| --- | --- | --- |
| `0x10000000…` | 4 MiB QSPI flash, XIP-mapped | Zephyr `.text`/`.rodata`, then `model_pte[]` in `network_model_sec` (GNU ld places the read-only orphan section with rodata → flash) |
| `0x20000000…` | 520 KiB SRAM | `.data`/`.bss` incl. both pools, Zephyr heaps/stacks, `main` thread stack (8 KiB) |

Key design point — **weights never touch SRAM**:

- upstream Ethos-U samples memcpy `model_pte` into SRAM because the NPU's
  DMA master cannot read through the XIP cache;
- the Cortex-M33 CPU *can*, so this demo wraps the flash-resident array
  directly in a `BufferDataLoader`. Constant tensors are returned as
  pointers into flash, and only activations live in SRAM. For the fp32
  model (~90 KB) that is ~20 % of the entire SRAM budget saved.

The rp2350 XIP cache makes flash-resident weights fast enough for tiny
models; measure with the printed latencies before drawing conclusions for
your own model.

Pool sizing method (documented in `prj.conf`):

1. boot once with a generous `CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE`;
2. read `Allocator used: N / M bytes` from the console after method load;
3. set the Kconfig to ~1.5×N. The upstream Pico 2 example measured
   45 632 B for a tiny MLP.

## 5. Backend dispatch mechanics

```
                    ┌────────────────────────────┐
                    │  .pte (INT8 variant)       │
                    │  op: cortex_m.quantized_   │
                    │      conv2d.out            │
                    └─────────────┬──────────────┘
                                  │ runtime op lookup
             registered operators │
   ┌──────────────────────────────┼───────────────────────────────┐
   │ cpu_portable_ops_lib         │ zephyr_cortex_m_ops_lib       │
   │ (from portable functions.yaml│ (from backends/cortex_m/ops/  │
   │  ∩ ops used by the model)    │  operators.yaml ∩ model ops)  │
   │  float + generic ops         │  INT8: conv2d, dw_conv2d,     │
   │                              │  transpose_conv2d, linear,    │
   │                              │  add, mul, relu, pooling…     │
   └──────────────┬───────────────┴───────────────┬───────────────┘
                  │                               │
        portable_kernels                cortex_m_kernels (CMSIS-NN)
        (reference C++)                 arm_convolve_s8, arm_fully_connected_s8…
```

- The module's `zephyr/CMakeLists.txt` sets `EXECUTORCH_BUILD_CORTEX_M=ON`
  on every `CONFIG_CPU_CORTEX_M` board → the `cortex_m_kernels` target
  (CMSIS-NN) is always built for the RP2350; the app links a *selected* ops
  library against it.
- Ops the Cortex-M kernels don't implement fall through to the portable
  library — graceful degradation, one binary.
- CMSIS-NN exploits the Cortex-M33's DSP SIMD instructions (SMLAD et al.);
  this is the main speedup source over reference kernels. The upstream
  Cortex-M backend docs also describe 4w8a/8w16a modes (not yet wired into
  the quantizer configs — 8w8a is the supported path today).

## 6. Quantization flow (INT8 export)

```
float graph
  → torch.export.export()                      functional ATen
  → prepare_pt2e(graph, CortexMQuantizer())    insert observers
  → run N calibration batches                  activation min/max stats
  → convert_pt2e(prepared)                     observers → QDQ nodes
      graph now: float in → quantize → int8 conv/linear → dequantize → float out
  → torch.export.export() again                freeze
  → to_edge(compile_config=cortex_m_edge_compile_config())
      NOTE: this config keeps ops like relu/maxpool alive through lowering;
      using a hand-written config instead silently degrades some ops to
      float portable kernels (documented upstream gotcha)
  → CortexMPassManager(CortexMTargetConfig(cpu=CortexM.M33)).transform()
      QDQ pattern-matched onto CMSIS-NN kernels
  → to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
```

Why float I/O on purpose: sensors/UI code keeps working in float, the
quantize/dequantize pairs fuse around the int8 cores, and the *same*
firmware input contract serves both backends. PT2E also gives you
per-tensor scale/zp computed from real calibration data — the observers see
MNIST-like ranges, not synthetic ones (use `--dataset mnist` calibration for
real workloads).

## 7. Build integration (out-of-tree app ↔ Zephyr module)

```
<repo>/app/                                <ws>/modules/lib/executorch/
  CMakeLists.txt  ── ZEPHYR_EXECUTORCH_MODULE_DIR ──►  zephyr/module.yml
  prj.conf (Kconfig)                                    zephyr/Kconfig
      CONFIG_EXECUTORCH=y        ◄──────────  CONFIG_EXECUTORCH_* options
      ..._METHOD_ALLOCATOR_POOL_SIZE ── cmake ──►  ET_ARM_METHOD_ALLOCATOR_POOL_SIZE
                                                   (compile definition)
  -DET_PTE_FILE_PATH=model.pte ─► pte_to_header.py ─► build/model_pte.h
```

- `west-manifests/executorch.yaml` mounts `pytorch/executorch` as
  `modules/lib/executorch`; Zephyr's module discovery exports
  `ZEPHYR_EXECUTORCH_MODULE_DIR` to any application CMake.
- The module compiles the runtime against Zephyr's toolchain flags
  (`zephyr_interface`), swaps the PAL (platform abstraction layer) to the
  Zephyr one (`ET_LOG` → Zephyr logging), and exposes
  `CONFIG_EXECUTORCH_*` Kconfig knobs.
- The pool-size Kconfigs flow into the app as preprocessor defines; the
  *app* (not the module) owns the static pools. That keeps memory policy in
  the board-specific layer where it belongs.

## 8. Design decisions & trade-offs

| Decision | Rationale | Trade-off |
| --- | --- | --- |
| Zero-copy flash loader | save ~20 % SRAM; RP2350 CPU reads XIP flash | per-access cache misses; for hot weights a memcpy to SRAM can win — measure |
| Static pools, no malloc at inference | deterministic, certifiable | pool sized per model; documented sizing method |
| Same firmware for both backends | one app to validate; dispatch lives in the .pte | both kernel libraries linked → bigger binary than a single-path build |
| Selective ops (`OPS_FROM_MODEL`) | minimum binary; catches lowering-introduced ops | rebuild needed when the model changes (that's fine — it's automated) |
| float model I/O | identical input contract for both backends; simpler host/sensor glue | extra Q/DQ ops inside the graph (negligible here) |
| MLP-friendly tiny CNN (~5.3k params) | fits SRAM/flash with huge margin; teaches the pipeline | not a "production" accuracy story; swap in your own model via `python/model.py` |
| synthetic dataset mode | offline quickstart + deterministic CI test vectors | 0/1/4/7 only; real MNIST for real accuracy |

## 9. Extending the demo

- **Swap the model**: edit `python/model.py`, retrain, re-export, rebuild.
  Bump `CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE` per the console
  printout. Watch flash: the fp32 `.pte` grows ~4× faster than int8.
- **ETDump profiling**: `CONFIG_EXECUTORCH_ENABLE_EVENT_TRACER=y` and pass a
  tracer to `load_method` to get op-level timings on device (upstream docs:
  "ETDump — ExecuTorch Dump").
- **WiFi telemetry (the "W")**: the Pico 2 W board target supports the
  CYW4343W via `CONFIG_WIFI=y` + `west blobs fetch hal_infineon` (Zephyr
  ≥ 4.3); a natural next step is POSTing predictions/MEM stats to a host.
  Kept out of this demo deliberately: networking multiplies the failure
  surface and the core lesson is the inference path.
- **Second core**: RP2350 is dual-core; Zephyr supports the second M33 as
  a child image. Inference on core 0, sensor/UI work on core 1 is the
  canonical split.

## 10. References (verified Nov 2026)

- ExecuTorch v1.5.1 docs — *Pico2: A simple MNIST Tutorial*,
  *Arm Cortex-M Backend*, *Zephyr: MobileNetV2 with Ethos-U NPU (Corstone
  FVP & Alif E8)* — docs.pytorch.org/executorch/stable
- `pytorch/executorch` sources — `zephyr/` module (Kconfig, CMake,
  `samples/hello-executorch`), `examples/raspberry_pi/pico2/`,
  `examples/arm/common/{pte_to_header.py,arm_memory_allocator.*}`
- Zephyr docs — boards `rpi_pico2` / `rpi_pico2/rp2350a/m33_0/w`,
  sample `samples/boards/raspberrypi/pico_w_wifi_led`
- Zephyr Project announcement — *ExecuTorch Integrated with Zephyr RTOS*
  (module requirements: Zephyr 4.4.0, ExecuTorch 1.5.1+)
- RP2350 datasheet — 520 KiB SRAM, 4 MiB flash boards, Cortex-M33+FPU,
  150 MHz
