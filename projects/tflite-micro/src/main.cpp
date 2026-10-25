/*
 * main.cpp — TensorFlow Lite for Microcontrollers sine-regression demo.
 *
 * Target : Raspberry Pi Pico 2 W (RP2350, Cortex-M33) running Zephyr.
 * Model  : tiny 2x16-ReLU MLP trained on y = sin(x), fully int8-quantized
 *          (post-training quantization with a representative dataset).
 * Flow   : sweep x over [0, 2*pi), quantize to int8 using the input tensor's
 *          (scale, zero_point), invoke the interpreter, dequantize the int8
 *          output and print "x_value: ..., y_value: ...".
 *
 * Inspired by Zephyr's samples/modules/tflite-micro/hello_world, adapted as a
 * standalone out-of-tree application with the model embedded from the
 * training pipeline in model/generate_model.py.
 */

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "sine_model.h"

namespace {

/* Model + interpreter state. */
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

/* Static tensor arena — all activations live here, no heap involved.
 * The full-int8 2x16 MLP needs ~3 KiB; 4 KiB leaves head-room.
 */
constexpr int kTensorArenaSize = 4096;
uint8_t tensor_arena[kTensorArenaSize] __aligned(16);

/* Sweep shape: how many points per period of the sine curve. */
constexpr int kInferencesPerCycle = 128;
constexpr float kXrange = 6.283185f; /* 2*pi, matches training range */

/* Setup the interpreter; returns 0 on success, -1 on failure. */
int setup()
{
	/* Map the flatbuffer into the schema structure — no copy, no parsing. */
	model = tflite::GetModel(g_sine_model);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		MicroPrintf("Model schema %d != supported %d",
			    model->version(), TFLITE_SCHEMA_VERSION);
		return -1;
	}

	/* Register exactly the kernels the model needs (op inventory from the
	 * generator: FULLY_CONNECTED only — ReLU is fused into its options).
	 */
	static tflite::MicroMutableOpResolver<1> resolver;
	resolver.AddFullyConnected();

	static tflite::MicroInterpreter static_interpreter(
		model, resolver, tensor_arena, kTensorArenaSize);
	interpreter = &static_interpreter;

	if (interpreter->AllocateTensors() != kTfLiteOk) {
		MicroPrintf("AllocateTensors() failed");
		return -1;
	}

	input = interpreter->input(0);
	output = interpreter->output(0);

	printk("model: %d bytes | arena: %d bytes used of %d\n",
	       g_sine_model_len, interpreter->arena_used_bytes(),
	       kTensorArenaSize);
	printk("input  tensor: int8, scale=%.6f zero_point=%d\n",
	       (double)input->params.scale, input->params.zero_point);
	printk("output tensor: int8, scale=%.6f zero_point=%d\n",
	       (double)output->params.scale, output->params.zero_point);
	return 0;
}

/* One full sweep of the sine curve. */
static size_t infer_log_counter = 0;
void run_inference_cycle()
{
	for (int i = 0; i < kInferencesPerCycle; i++) {
		float x = (static_cast<float>(i) /
			   static_cast<float>(kInferencesPerCycle)) * kXrange;

		/* float -> int8 using the input tensor's quantization params:
		 *   x_q = round(x / scale) + zero_point
		 */
		input->data.int8[0] = static_cast<int8_t>(
			lroundf(x / input->params.scale) +
			input->params.zero_point);

		if (interpreter->Invoke() != kTfLiteOk) {
			MicroPrintf("Invoke failed at x=%f", (double)x);
			return;
		}

		/* int8 -> float using the output tensor's quantization params:
		 *   y = (y_q - zero_point) * scale
		 */
		float y = (static_cast<float>(output->data.int8[0]) -
			   output->params.zero_point) * output->params.scale;
		if (infer_log_counter %2 == 0) {
			printk("x_value: %.3f, y_value: %.3f\n", (double)x, (double)y);
		}
	}
}

} /* namespace */

int main(void)
{
	if (setup() != 0) {
		return -1;
	}

	while (true) {
		run_inference_cycle();
		k_sleep(K_MSEC(1000));
		infer_log_counter++;
	}

	return 0;
}
