# =============================================================================
# TFLite-Micro sine demo — Raspberry Pi Pico 2 W (RP2350) + Zephyr
# =============================================================================
# The application lives OUTSIDE the west workspace, so every west invocation
# runs from $(WEST_TOPDIR) with absolute paths (west needs the workspace
# context for module discovery — the tflite-micro module among others).
#
# Typical flow:
#   workon machine_learn        # TF (CPU) for the training step   (make model)
#   make model header           # train -> int8 .tflite -> C array
#   workon zephyr_rtos          # west + Zephyr SDK for firmware   (make build)
#   make build                  # -> build_rpi_pico2_rp2350a_m33_0_w/zephyr.uf2
#   make qemu                   # smoke-test inference on qemu_x86
#   make flash_mcu              # west flash -r uf2 (board in BOOTSEL mode)
#   make mcu_monitor            # serial console (minicom/picocom/screen)
#
# Override examples:
#   make build BOARD=rpi_pico2/rp2350a/m33_0        (Pico 2, no WiFi radio)
#   make build BUILD_DIR=/tmp/tflm_build PRISTINE=always
# =============================================================================

APP_DIR      := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
WEST_TOPDIR  ?= /home/agonga/zephyrproject
ML_PYTHON    ?= /home/agonga/.virtualenvs/tensorflow/bin/python

BOARD        ?= rpi_pico2/rp2350a/m33_0/w
PRISTINE     ?= auto
BUILD_DIR    ?= $(APP_DIR)/build_$(subst /,_,$(BOARD))
QEMU_BOARD   ?= qemu_x86
QEMU_BUILD   ?= $(APP_DIR)/build_$(subst /,_,$(QEMU_BOARD))
QEMU_TIMEOUT ?= 30

# Serial console for mcu_monitor (UART0 on GPIO0/1 via a USB adapter, or the
# board's USB CDC when the firmware enables it). Override as needed:
#   make mcu_monitor SERIAL_PORT=/dev/ttyUSB0
SERIAL_PORT  ?= /dev/ttyACM0
SERIAL_BAUD  ?= 115200

MODEL_TFLITE := $(APP_DIR)/model/model_int8.tflite
MODEL_CC     := $(APP_DIR)/src/sine_model_data.cc
MODEL_H      := $(APP_DIR)/src/sine_model.h
GEN_PY       := $(APP_DIR)/model/generate_model.py
EMBED_PY     := $(APP_DIR)/model/tflite_to_cc.py
PIPELINE_PY  := $(APP_DIR)/../script/generate_tf_mcu_model.py

WEST := cd $(WEST_TOPDIR) && west

.DEFAULT_GOAL := help
.PHONY: all model header build qemu flash_mcu flash mcu_monitor size rom_report menuconfig clean distclean help

all: build

# -- ML pipeline ---------------------------------------------------------------

model: $(MODEL_TFLITE)

$(MODEL_TFLITE): $(GEN_PY) $(PIPELINE_PY)
	$(ML_PYTHON) $(GEN_PY)

header: $(MODEL_CC)

$(MODEL_CC): $(MODEL_TFLITE) $(EMBED_PY)
	$(ML_PYTHON) $(EMBED_PY) $(MODEL_TFLITE) --source $(MODEL_CC) --header $(MODEL_H)

# -- Firmware ------------------------------------------------------------------

build: $(MODEL_CC)
	$(WEST) build -b $(BOARD) -d $(BUILD_DIR) -p $(PRISTINE) $(APP_DIR)

# QEMU smoke test: build the same app for qemu_x86 and run it, capturing the
# console. The timeout inside scripts/run_qemu.sh ends the run after
# QEMU_TIMEOUT seconds (the firmware loops forever); the grep proves real
# inferences happened.
#
# timeout(1) puts west/ninja/QEMU in a separate process group, so a terminal
# Ctrl-C would never reach the emulator. The wrapper forwards SIGINT and falls
# back to QEMU's pidfile, so Ctrl-C stops the run immediately instead of
# waiting for the timeout.
qemu: $(MODEL_CC)
	$(WEST) build -b $(QEMU_BOARD) -d $(QEMU_BUILD) -p $(PRISTINE) $(APP_DIR)
	@$(APP_DIR)/scripts/run_qemu.sh "$(WEST_TOPDIR)" "$(QEMU_BUILD)" "$(QEMU_TIMEOUT)" \
		2>&1 | tee $(APP_DIR)/qemu_output.log
	@grep -m1 "x_value:" $(APP_DIR)/qemu_output.log \
		&& echo "[qemu] sine inference verified on $(QEMU_BOARD)" \
		|| (echo "[qemu] FAILED: no inference output found" && exit 1)

# Flash the Pico 2 W (RP2350) over UF2. The board must be in BOOTSEL mode:
# hold the BOOTSEL button while plugging in USB (it mounts as mass storage
# RPI-RP2); picotool then writes zephyr.uf2 and the board reboots into it.
flash_mcu:
	@test -f $(BUILD_DIR)/zephyr/zephyr.uf2 || { \
		echo "[flash] $(BUILD_DIR)/zephyr/zephyr.uf2 not found - run 'make build' first"; \
		exit 1; }
	@echo "[flash] $(BOARD) <- $(BUILD_DIR)/zephyr/zephyr.uf2"
	@echo "[flash] board must be in BOOTSEL mode (hold BOOTSEL while plugging in USB)"
	$(WEST) flash -r uf2 -d $(BUILD_DIR)

# Backwards-compatible alias.
flash: flash_mcu

# Live serial console of the RPI2350. Uses minicom, picocom or screen —
# whichever is installed (exit minicom with Ctrl-A then X).
mcu_monitor:
	@if [ ! -e "$(SERIAL_PORT)" ]; then \
		echo "[monitor] $(SERIAL_PORT) not found. Serial devices present:"; \
		ls -1 /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | sed 's/^/          /' || true; \
		echo "[monitor] plug the board in, or override: make mcu_monitor SERIAL_PORT=/dev/ttyUSB0"; \
		exit 1; \
	fi
	@if command -v picocom >/dev/null 2>&1; then \
		echo "[monitor] picocom on $(SERIAL_PORT) @ $(SERIAL_BAUD) baud (exit: Ctrl-A then Ctrl-X)"; \
		exec picocom -b $(SERIAL_BAUD) $(SERIAL_PORT); \
	elif command -v minicom >/dev/null 2>&1; then \
		echo "[monitor] minicom on $(SERIAL_PORT) @ $(SERIAL_BAUD) baud (exit: Ctrl-A then X)"; \
		exec minicom -D $(SERIAL_PORT) -b $(SERIAL_BAUD); \
	elif command -v screen >/dev/null 2>&1; then \
		echo "[monitor] screen on $(SERIAL_PORT) @ $(SERIAL_BAUD) baud (exit: Ctrl-A then K, Y)"; \
		exec screen $(SERIAL_PORT) $(SERIAL_BAUD); \
	else \
		echo "[monitor] no serial tool found - install minicom, picocom or screen"; \
		exit 1; \
	fi

size:
	$(WEST) build -d $(BUILD_DIR) -t rom_report
	$(WEST) build -d $(BUILD_DIR) -t ram_report

rom_report:
	$(WEST) build -d $(BUILD_DIR) -t rom_report

menuconfig:
	$(WEST) build -d $(BUILD_DIR) -t menuconfig

clean:
	rm -rf $(BUILD_DIR) $(QEMU_BUILD)

distclean: clean
	rm -rf $(APP_DIR)/model/calib $(APP_DIR)/model/sine_model \
	       $(APP_DIR)/model/sine_model.keras $(MODEL_TFLITE) \
	       $(APP_DIR)/qemu_output.log \
	       $(APP_DIR)/src/sine_model_data.cc $(APP_DIR)/src/sine_model.h

help:
	@echo "TFLite-Micro on $(BOARD) — targets:"
	@echo "  make model       train + int8-quantize ($(notdir $(ML_PYTHON)))"
	@echo "  make header      embed .tflite into src/sine_model_data.cc"
	@echo "  make build       west build for $(BOARD) -> $(notdir $(BUILD_DIR))/zephyr.uf2"
	@echo "  make qemu        build + run on $(QEMU_BOARD) (console smoke test)"
	@echo "  make flash_mcu   west flash -r uf2 (board in BOOTSEL mode; alias: make flash)"
	@echo "  make mcu_monitor serial console on $(SERIAL_PORT) @ $(SERIAL_BAUD) baud"
	@echo "                   (override: SERIAL_PORT=/dev/ttyUSB0 SERIAL_BAUD=115200)"
	@echo "  make size        RAM/ROM usage of the last build"
	@echo "  make menuconfig  interactive Kconfig"
	@echo "  make clean       remove build trees (keeps the trained model)"
	@echo "  make distclean   also remove trained/quantized model artifacts"
