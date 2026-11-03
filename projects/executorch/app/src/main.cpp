/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * main.cpp -- ExecuTorch MNIST inference demo on the Raspberry Pi Pico 2 W
 *             (RP2350A, dual Cortex-M33) running Zephyr RTOS.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS APP DOES
 * ---------------------------------------------------------------------------
 * 1. Wakes up, prints a banner and the exact address of the .pte model blob
 *    that the build system linked into XIP flash (memory-mapped QSPI NOR).
 * 2. Loads the ExecuTorch program DIRECTLY FROM FLASH. The Pico 2 W maps its
 *    4 MiB QSPI flash at 0x10000000 and executes/reads from it in place
 *    (XIP), so unlike the upstream Ethos-U samples we do NOT copy the model
 *    into SRAM first -- the Cortex-M33 can dereference flash pointers like
 *    any other memory and we save hundreds of KiB of precious RAM.
 * 3. Plans the runtime memory: one static "method allocator" pool backs the
 *    method metadata, the memory-planned activation buffers and the I/O
 *    tensors. One small "temp allocator" pool backs per-kernel scratch.
 * 4. For every embedded test digit (see test_digits.h):
 *       - renders it as ASCII art over the console
 *       - copies it into the input tensor (float32, [0..1], 1x1x28x28)
 *       - times method->execute() with k_cycle_get_32()
 *       - prints the logits, argmaxes, compares with the expected label
 *       - blinks the onboard LED (predicted class index + 1) times
 * 5. Prints a latency summary and gives a "success" LED blink pattern.
 *
 * The model itself was exported on a host with python/export_mnist.py either
 * as float32 (executed by the Portable Ops kernels) or as INT8 with the
 * CortexMQuantizer (executed by the Cortex-M / CMSIS-NN kernels). The
 * firmware source is IDENTICAL for both -- the backend that runs an op is
 * decided at export time and recorded inside the .pte; this app only needs
 * matching kernel libraries linked in (see app/CMakeLists.txt).
 *
 * ---------------------------------------------------------------------------
 * DATA FLOW (host -> flash -> tensors)
 * ---------------------------------------------------------------------------
 *   PyTorch model (python/model.py)
 *        |  torch.export.export()                      [host]
 *        |  (+ CortexMQuantizer PT2E flow for INT8)
 *        v
 *   model .pte flatbuffer -----------------> app/model/*.pte
 *        |  pte_to_header.py (build step)            [CMake]
 *        v
 *   const uint8_t model_pte[] in section "network_model_sec" -> XIP flash
 *        |  BufferDataLoader (zero-copy span)
 *        v
 *   Program::load() -> Method::load() -> Method::execute()
 *
 * ---------------------------------------------------------------------------
 * REFERENCES (verified against ExecuTorch v1.5.x sources)
 *   - zephyr/samples/hello-executorch (official Zephyr module sample)
 *   - examples/raspberry_pi/pico2 (official Pico 2 bare-metal example)
 *   - docs.pytorch.org/executorch/stable: "Pico2: A simple MNIST Tutorial",
 *     "Arm Cortex-M Backend", "Zephyr: MobileNetV2 with Ethos-U NPU"
 */

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <executorch/examples/arm/common/arm_memory_allocator.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <memory> /* @lint-ignore NOSTDINC */

/*
 * Generated at build time from ET_PTE_FILE_PATH by
 * executorch/examples/arm/common/pte_to_header.py. It declares:
 *
 *   __attribute__((section("network_model_sec"), aligned(16)))
 *   const unsigned char model_pte[];
 *
 * The GNU linker places this read-only "orphan" section with the rest of the
 * read-only data inside the FLASH region, so on the Pico 2 W it ends up in
 * the 4 MiB external QSPI flash, read through the XIP cache.
 */
#include "model_pte.h"

/* Generated test vectors (scripts/gen_test_digits.py or python/export_mnist.py). */
#include "test_digits.h"

/* ---------------------------------------------------------------------------
 * Tensor geometry -- must match the exported model's input contract.
 * export_mnist.py always exports input shape (1, 1, 28, 28) float32 [0..1],
 * for BOTH the float32 and the INT8 (Cortex-M/CMSIS-NN) .pte variants:
 * the quantized graphs keep float I/O and put quantize/dequantize nodes
 * inside the graph.
 * ------------------------------------------------------------------------- */
#define IMG_ROWS 28
#define IMG_COLS 28
#define IMG_PIXELS (IMG_ROWS * IMG_COLS)

/* ---------------------------------------------------------------------------
 * Static memory pools
 * ---------------------------------------------------------------------------
 * method_allocation_pool
 *   Backs a bump allocator used for: method metadata produced by
 *   Program::load_method(), the memory-planned activation buffers ("planned
 *   buffers"), and the input tensors we set before each execute(). Its size
 *   is plumbed from Kconfig: CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE is
 *   turned into the ET_ARM_METHOD_ALLOCATOR_POOL_SIZE compile definition by
 *   app/CMakeLists.txt. Start with ~128 KiB for a tiny CNN/MLP and raise it
 *   if load_method() reports MemoryAllocationFailed.
 *
 * temp_allocation_pool
 *   Scratch memory handed to kernels for transient data; reset after every
 *   op. The upstream samples place it in a ".bss.tensor_arena" section --
 *   that name only matters for Ethos-U boards whose linker scripts put the
 *   tensor arena in dedicated SRAM banks. On the RP2350 there is no NPU, so
 *   a plain static array in .bss is all we need (we keep the upstream
 *   attribute so the same source works on Corstone boards too).
 * ------------------------------------------------------------------------- */
#if !defined(ET_ARM_METHOD_ALLOCATOR_POOL_SIZE)
#define ET_ARM_METHOD_ALLOCATOR_POOL_SIZE (128 * 1024)
#endif
static const size_t kMethodPoolSize = ET_ARM_METHOD_ALLOCATOR_POOL_SIZE;
static unsigned char
    method_allocation_pool[ET_ARM_METHOD_ALLOCATOR_POOL_SIZE]
    __attribute__((aligned(16)));

#if !defined(ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
#define ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE 4096
#endif
static const size_t kTempPoolSize =
    ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE;
static __attribute__((section(".bss.tensor_arena"), aligned(16)))
unsigned char temp_allocation_pool[ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE];

/* ---------------------------------------------------------------------------
 * Onboard LED
 * ---------------------------------------------------------------------------
 * On the Pico 2 W the LED hangs off the CYW43 WiFi chip (WL_GPIO0); the
 * Zephyr board exposes it through the standard `led0` devicetree alias, so
 * the plain GPIO API works. On a plain Pico 2 (non-W) the alias points at
 * GPIO25 instead -- same code, no #ifdef needed.
 * ------------------------------------------------------------------------- */
#if DT_NODE_EXISTS(DT_ALIAS(led0))
static const struct gpio_dt_spec led =
    GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
#define HAVE_LED 1
#else
#define HAVE_LED 0
#endif

static int led_init(void)
{
#if HAVE_LED
        if (!gpio_is_ready_dt(&led)) {
                return -ENODEV;
        }
        return gpio_pin_configure_dt(&led, GPIO_OUTPUT_INACTIVE);
#else
        return 0;
#endif
}

static void led_set(bool on)
{
#if HAVE_LED
        (void)gpio_pin_set_dt(&led, on ? 1 : 0);
#else
        ARG_UNUSED(on);
#endif
}

/* Blink `n` times, then keep the LED off for a moment. */
static void led_blink(uint32_t n, uint32_t on_ms, uint32_t off_ms)
{
        for (uint32_t i = 0; i < n; i++) {
                led_set(true);
                k_msleep(on_ms);
                led_set(false);
                k_msleep(off_ms);
        }
        k_msleep(400);
}

/* Fast-blink-forever pattern so a failure is recognisable without a console. */
static void led_fail_forever(void)
{
        for (;;) {
                led_blink(10, 50, 50);
                k_msleep(1000);
        }
}

/* ---------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */
static void print_ascii_digit(const float *px)
{
        for (int r = 0; r < IMG_ROWS; r++) {
                for (int c = 0; c < IMG_COLS; c++) {
                        putchar(px[r * IMG_COLS + c] > 0.35f ? '#' : '.');
                }
                putchar('\n');
        }
}

static size_t argmax(const float *v, size_t n)
{
        size_t best = 0;
        for (size_t i = 1; i < n; i++) {
                if (v[i] > v[best]) {
                        best = i;
                }
        }
        return best;
}

static uint32_t now_us(void)
{
        return (uint32_t)k_cyc_to_us_floor64(k_cycle_get_32());
}

/* ---------------------------------------------------------------------------
 * main()
 * ------------------------------------------------------------------------- */
int main(void)
{
        printk("\n\n");
        printk("==========================================\n");
        printk(" ExecuTorch + Zephyr MNIST demo\n");
        printk(" Raspberry Pi Pico 2 W (RP2350A, M33)\n");
        printk("==========================================\n");

        if (led_init() != 0) {
                printk("LED init failed (continuing without it)\n");
        }

        ET_LOG(Info, "Model PTE at %p, size %u bytes (XIP flash)",
               model_pte, (unsigned int)sizeof(model_pte));
        ET_LOG(Info, "Method allocator pool: %u bytes, temp pool: %u bytes",
               (unsigned int)kMethodPoolSize, (unsigned int)kTempPoolSize);

        /* ---- 1. Initialize the ExecuTorch platform layer ------------------- */
        executorch::runtime::runtime_init();

        /* ---- 2. Wrap the flash-resident model in a zero-copy loader --------
         *
         * BufferDataLoader never copies: when the runtime asks for a span of
         * the flatbuffer it hands back pointers into `model_pte` itself, i.e.
         * into XIP flash. Constant tensors (weights) therefore stay in flash
         * for the whole lifetime of the program -- only activations live in
         * SRAM. (The upstream Ethos-U samples copy the blob into SRAM because
         * the NPU's DMA controller cannot fetch through the XIP cache; the
         * Cortex-M33 CPU can, so we skip the copy.)
         */
        executorch::extension::BufferDataLoader loader(model_pte, sizeof(model_pte));
        ET_LOG(Info, "BufferDataLoader ready over %u bytes",
               (unsigned int)sizeof(model_pte));

        /* ---- 3. Parse the program (immutable; reuse for every inference) --- */
        auto program = executorch::runtime::Program::load(&loader);
        if (!program.ok()) {
                ET_LOG(Error, "Program::load failed: 0x%" PRIx32,
                       (uint32_t)program.error());
                led_fail_forever();
                return 1;
        }
        ET_LOG(Info, "Program loaded, %zu method(s)", program->num_methods());

        auto method_name = program->get_method_name(0);
        ET_CHECK_MSG(method_name.ok(), "Program has no methods");

        /* ---- 4. Plan the runtime memory ------------------------------------
         *
         * method_allocator  -> bump allocator over the big static pool
         * planned_memory    -> one span per memory-planned buffer; the AOT
         *                      memory planner has already laid out every
         *                      activation tensor lifetimes-overlapped inside
         *                      these spans
         * temp_allocator    -> small per-op scratch pool
         */
        /* ArmMemoryAllocator is declared at global scope in
         * examples/arm/common/arm_memory_allocator.h (not inside
         * executorch::runtime). */
        ArmMemoryAllocator method_allocator(
            (uint32_t)kMethodPoolSize, method_allocation_pool);
        ArmMemoryAllocator temp_allocator(
            (uint32_t)kTempPoolSize, temp_allocation_pool);

        auto method_meta_result = program->method_meta(*method_name);
        ET_CHECK_MSG(method_meta_result.ok(), "method_meta() failed");

        const size_t num_planned =
            method_meta_result->num_memory_planned_buffers();
        size_t planned_total = 0;
        for (size_t id = 0; id < num_planned; id++) {
                size_t sz = static_cast<size_t>(
                    method_meta_result->memory_planned_buffer_size(id).get());
                planned_total += sz;
                ET_LOG(Info, "Planned buffer %zu: %zu bytes", id, sz);
        }
        ET_LOG(Info, "Total activation (planned) memory: %zu bytes", planned_total);

        /*
         * The planned buffers must stay alive as long as the Method does, and
         * their number is known only after parsing the program, so allocate
         * them out of the method pool (which IS static storage) and keep the
         * span bookkeeping in a small static array. 16 spans is far above
         * anything a tiny graph produces (one span is typical).
         */
        static constexpr size_t kMaxPlannedSpans = 16;
        static executorch::runtime::Span<uint8_t> planned_spans[kMaxPlannedSpans];
        ET_CHECK_MSG(num_planned <= kMaxPlannedSpans,
                     "model has %zu planned buffers, raise kMaxPlannedSpans",
                     num_planned);
        for (size_t id = 0; id < num_planned; id++) {
                size_t sz = static_cast<size_t>(
                    method_meta_result->memory_planned_buffer_size(id).get());
                uint8_t *buf = (uint8_t *)method_allocator.allocate(sz);
                ET_CHECK_MSG(buf != nullptr,
                             "OOM: planned buffer %zu (%zu bytes). Raise "
                             "CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE.",
                             id, sz);
                planned_spans[id] = {buf, sz};
        }

        executorch::runtime::HierarchicalAllocator planned_memory(
            {planned_spans, num_planned});

        executorch::runtime::MemoryManager memory_manager(
            &method_allocator, &planned_memory, &temp_allocator);

        /* ---- 5. Load the method -------------------------------------------- */
        auto method = program->load_method(*method_name, &memory_manager,
                                           /*event_tracer=*/nullptr);
        if (!method.ok()) {
                ET_LOG(Error, "load_method('%s') failed: 0x%" PRIx32,
                       *method_name, (uint32_t)method.error());
                led_fail_forever();
                return 1;
        }
        ET_LOG(Info, "Method '%s' loaded. Allocator used: %zu / %u bytes "
                     "(free %u)", *method_name, method_allocator.used_size(),
               (unsigned int)kMethodPoolSize,
               (unsigned int)method_allocator.free_size());

        /* ---- 6. Prepare the input tensor once, refill it per digit ---------
         *
         * We allocate the input buffer from the method allocator and wrap it
         * in a TensorImpl whose sizes/dim-order come from the model metadata
         * (never hard-coded against the .pte!).
         */
        auto input_meta = method_meta_result->input_tensor_meta(0);
        ET_CHECK_MSG(input_meta.ok(), "input 0 is not a tensor");
        ET_CHECK_MSG(input_meta->nbytes() == IMG_PIXELS * sizeof(float),
                     "model expects %zu input bytes, embedded digits are %zu",
                     (size_t)input_meta->nbytes(), (size_t)(IMG_PIXELS * sizeof(float)));

        float *input_data =
            (float *)method_allocator.allocate(input_meta->nbytes());
        ET_CHECK_MSG(input_data != nullptr, "OOM: input tensor");

        /*
         * TensorImpl stores POINTERS to the sizes/dim-order arrays, so the
         * arrays must outlive the TensorImpl -- use static storage, not
         * compound literals. The ctor takes mutable pointers, so the arrays
         * cannot be constexpr (const would not convert).
         */
        static executorch::aten::TensorImpl::SizesType
            kInputSizes[4] = {1, 1, IMG_ROWS, IMG_COLS};
        static executorch::aten::TensorImpl::DimOrderType
            kInputDimOrder[4] = {0, 1, 2, 3};

        executorch::aten::TensorImpl input_impl(
            executorch::aten::ScalarType::Float,
            /*dim=*/4,
            /*sizes=*/kInputSizes,
            input_data,
            /*dim_order=*/kInputDimOrder);
        executorch::aten::Tensor input(&input_impl);

        ET_CHECK_MSG(method->set_input(input, 0) ==
                         executorch::runtime::Error::Ok,
                     "set_input failed");

        /* ---- 7. Inference loop over the embedded digits -------------------- */
        const size_t num_outputs = method->outputs_size();
        float logits[16];
        ET_CHECK_MSG(num_outputs >= 1 && num_outputs <= 16,
                     "unexpected output count %zu", num_outputs);

        uint32_t lat_us[TEST_DIGIT_COUNT];
        size_t correct = 0;

        for (size_t k = 0; k < TEST_DIGIT_COUNT; k++) {
                printk("\n=== Digit '%s' (expected class %u) ===\n",
                       test_digit_names[k], test_digit_label[k]);
                print_ascii_digit(test_digit_input[k]);

                memcpy(input_data, test_digit_input[k],
                       IMG_PIXELS * sizeof(float));

                uint32_t t0 = now_us();
                auto status = method->execute();
                uint32_t t1 = now_us();
                lat_us[k] = t1 - t0;

                if (status != executorch::runtime::Error::Ok) {
                        ET_LOG(Error, "execute() failed: 0x%" PRIx32,
                               (uint32_t)status);
                        led_fail_forever();
                        return 1;
                }

                executorch::runtime::EValue out_value;
                ET_CHECK_MSG(method->get_outputs(&out_value, 1) ==
                                 executorch::runtime::Error::Ok,
                             "get_outputs failed");
                executorch::aten::Tensor out = out_value.toTensor();
                ET_CHECK_MSG(out.scalar_type() ==
                                     executorch::aten::ScalarType::Float,
                             "output is not float32 (got %s)",
                             executorch::runtime::toString(out.scalar_type()));

                const size_t n =
                    out.numel() < 16 ? (size_t)out.numel() : 16;
                for (size_t i = 0; i < n; i++) {
                        logits[i] = out.const_data_ptr<float>()[i];
                }

                printk("Logits:");
                for (size_t i = 0; i < n; i++) {
                        printk(" %.1f", (double)logits[i]);
                }
                printk("\n");

                size_t pred = argmax(logits, n);
                bool ok = (pred == (size_t)test_digit_label[k]);
                correct += ok ? 1 : 0;

                /*
                 * NOTE: test_digit_names[] names the EMBEDDED digits, not all
                 * model classes, and the class index is NOT always the digit:
                 *   - 4-class synthetic model: class index = position in the
                 *     training class list ("0","1","4","7"), so class 2 is
                 *     digit '4' and class 3 is digit '7';
                 *   - 10-class MNIST model: class index == the digit itself.
                 * Either way test_digit_label[] (generated together with the
                 * vectors) already carries the correct expected index, and the
                 * line below simply prints the raw argmax for inspection.
                 */
                printk("Predicted: class %zu | %s (%lu us)\n", pred,
                       ok ? "CORRECT" : "WRONG", (unsigned long)lat_us[k]);
                printk("Expected:  class %u ('%s')\n", test_digit_label[k],
                       test_digit_names[k]);

                /* LED feedback: (predicted class index + 1) blinks, max 10. */
                led_blink((uint32_t)(pred % 10) + 1, 120, 120);
        }

        /* ---- 8. Summary ----------------------------------------------------- */
        uint32_t total = 0;
        for (size_t k = 0; k < TEST_DIGIT_COUNT; k++) {
                total += lat_us[k];
        }
        printk("\n------------------------------------------\n");
        printk("Latency summary (method->execute only):\n");
        for (size_t k = 0; k < TEST_DIGIT_COUNT; k++) {
                printk("  '%s': %lu us\n", test_digit_names[k],
                       (unsigned long)lat_us[k]);
        }
        printk("  average: %lu us\n",
               (unsigned long)(total / TEST_DIGIT_COUNT));
        printk("Accuracy: %zu/%zu\n", correct, (size_t)TEST_DIGIT_COUNT);
        printk("Allocator after run: %zu / %u bytes\n",
               method_allocator.used_size(), (unsigned int)kMethodPoolSize);
        printk("------------------------------------------\n");

        /* Success pattern: three long blinks, then the LED mirrors the
         * average latency in ~10 ms units. Watch it breathe! */
        led_blink(3, 400, 400);

        ET_LOG(Info, "SUCCESS: ExecuTorch MNIST demo complete.");
        return 0;
}
