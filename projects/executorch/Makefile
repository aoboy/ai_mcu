# =============================================================================
# ExecuTorch MNIST demo — Raspberry Pi Pico 2 W (RP2350) + Zephyr
# =============================================================================
# Same target vocabulary as projects/tflite-micro so the projects feel
# identical from the shell. The heavy lifting lives in the scripts:
#   scripts/01_setup_workspace.sh   verify the pre-provisioned environment
#   scripts/02_build.sh             west build for one variant (portable|cmsis)
#
# Typical flow:
#   make verify              # check envs, workspace, SDK, ExecuTorch module
#   make model               # train -> mnist_cnn_{fp32,int8}.pte (torch env)
#   make build               # VARIANT=portable|cmsis (default: portable)
#   make flash               # west flash -r uf2 (board in BOOTSEL mode)
#   make monitor             # serial console (UART0 GP0/GP1 @ 115200)
#
# Interactive equivalents: `workon pytorch_cu132` for the model step,
# `workon zephyr_rtos` for firmware. Environment names and paths come from
# tools/env.sh (repo-wide single source of truth); the fallbacks below only
# apply if it moved.
#
# Override examples:
#   make model DATASET=synthetic     # offline quickstart, no MNIST download
#   make build VARIANT=cmsis         # INT8 .pte -> CMSIS-NN kernels
#   make build VARIANT=portable BOARD=rpi_pico2/rp2350a/m33_0
#   make monitor SERIAL_PORT=/dev/ttyUSB1
# =============================================================================

APP_DIR      := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ENV_SH       := $(abspath $(APP_DIR)/../../tools/env.sh)
WORKON_HOME  ?= $(shell . $(ENV_SH) 2>/dev/null; echo "$${AI_WORKON_HOME:-$$HOME/.virtualenvs}")
WEST_TOPDIR  ?= $(shell . $(ENV_SH) 2>/dev/null; echo "$${AI_WEST_TOPDIR:-$$HOME/zephyrproject}")
ZEPHYR_ENV   ?= $(shell . $(ENV_SH) 2>/dev/null; echo "$${AI_ENV_ZEPHYR:-zephyr_rtos}")
TORCH_PYTHON ?= $(shell { . $(ENV_SH) 2>/dev/null && ai_python "$$AI_ENV_TORCH"; } 2>/dev/null || echo "$$HOME/.virtualenvs/pytorch_cu132/bin/python")

VARIANT      ?= portable
BOARD        ?= rpi_pico2/rp2350a/m33_0/w
BUILD_DIR    := $(APP_DIR)/build-$(VARIANT)
SERIAL_PORT  ?= /dev/ttyUSB0
SERIAL_BAUD  ?= 115200

PTE_FP32     := $(APP_DIR)/app/model/mnist_cnn_fp32.pte
PTE_INT8     := $(APP_DIR)/app/model/mnist_cnn_int8.pte
CKPT         := $(APP_DIR)/python/mnist_cnn_state.pt
EXPORT_STAMP := $(APP_DIR)/build/.export.stamp

SETUP_SH     := $(APP_DIR)/scripts/01_setup_workspace.sh
BUILD_SH     := $(APP_DIR)/scripts/02_build.sh
TRAIN_PY     := $(APP_DIR)/python/train_mnist.py
EXPORT_PY    := $(APP_DIR)/python/export_mnist.py
PYTHON_SRCS  := $(APP_DIR)/python/model.py $(APP_DIR)/python/synth_data.py \
                $(APP_DIR)/python/embed_header.py

# west runs from the workspace with the zephyr_rtos venv bin on PATH, so
# flash/size work whether or not the caller did `workon zephyr_rtos`.
ZEPHYR_BIN   := $(WORKON_HOME)/$(ZEPHYR_ENV)/bin
WEST         := cd $(WEST_TOPDIR) && export PATH="$(ZEPHYR_BIN):$$PATH" && west

.DEFAULT_GOAL := help
.PHONY: all verify model build flash flash_mcu monitor mcu_monitor size rom_report menuconfig clean distclean help

all: build

# -- Environment ----------------------------------------------------------------

# Verify the pre-provisioned envs/workspace/SDK/module (no installs).
verify:
	"$(SETUP_SH)" "$(WEST_TOPDIR)"

# -- Model (pytorch_cu132 env) ----------------------------------------------------

# Train TinyCNN and export BOTH deployments (float32 portable + INT8
# CMSIS-NN) plus the embedded firmware test vectors. The checkpoint is
# reused when present; DATASET only applies on the run that (re)trains —
# `make distclean` first to switch datasets.
model: $(EXPORT_STAMP)

$(CKPT): $(TRAIN_PY) $(PYTHON_SRCS)
	$(TORCH_PYTHON) $(TRAIN_PY) $(if $(DATASET),--dataset $(DATASET),)

$(EXPORT_STAMP): $(CKPT) $(EXPORT_PY) $(PYTHON_SRCS)
	$(TORCH_PYTHON) $(EXPORT_PY)
	@mkdir -p $(dir $(EXPORT_STAMP)) && touch $(EXPORT_STAMP)

# -- Firmware (zephyr_rtos env) ---------------------------------------------------

# Delegates to scripts/02_build.sh: picks the .pte matching VARIANT and
# activates the zephyr_rtos env itself.
build:
	BOARD="$(BOARD)" "$(BUILD_SH)" "$(VARIANT)" "$(WEST_TOPDIR)"

flash:
	@test -f "$(BUILD_DIR)/zephyr/zephyr.uf2" || { \
		echo "[flash] $(BUILD_DIR)/zephyr/zephyr.uf2 not found - run 'make build VARIANT=$(VARIANT)' first"; \
		exit 1; }
	@echo "[flash] $(BOARD) <- $(BUILD_DIR)/zephyr/zephyr.uf2"
	@echo "[flash] board must be in BOOTSEL mode (hold BOOTSEL while plugging in USB)"
	$(WEST) flash -r uf2 -d "$(BUILD_DIR)"

# Backwards-compatible aliases with the tflite-micro target names.
flash_mcu: flash

# Live serial console. This app prints on UART0 (GP0 TX / GP1 RX) through a
# 3.3 V USB-UART adapter -- NOT on the board's USB connector. Uses picocom,
# minicom or screen — whichever is installed.
monitor:
	@if [ ! -e "$(SERIAL_PORT)" ]; then \
		echo "[monitor] $(SERIAL_PORT) not found. Serial devices present:"; \
		ls -1 /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | sed 's/^/          /' || true; \
		echo "[monitor] plug the adapter in, or override: make monitor SERIAL_PORT=/dev/ttyUSB1"; \
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

mcu_monitor: monitor

size:
	$(WEST) build -d "$(BUILD_DIR)" -t rom_report
	$(WEST) build -d "$(BUILD_DIR)" -t ram_report

rom_report:
	$(WEST) build -d "$(BUILD_DIR)" -t rom_report

menuconfig:
	$(WEST) build -d "$(BUILD_DIR)" -t menuconfig

clean:
	rm -rf $(APP_DIR)/build-portable $(APP_DIR)/build-cmsis $(APP_DIR)/build

distclean: clean
	rm -f $(PTE_FP32) $(PTE_INT8) $(CKPT)

help:
	@echo "ExecuTorch MNIST on $(BOARD) — targets:"
	@echo "  make verify      check the pre-provisioned env/workspace (no installs)"
	@echo "  make model       train + export fp32/int8 .pte ($(notdir $(TORCH_PYTHON)))"
	@echo "  make build       west build VARIANT=$(VARIANT) -> $(notdir $(BUILD_DIR))/zephyr.uf2"
	@echo "  make flash       west flash -r uf2 (board in BOOTSEL mode; alias: make flash_mcu)"
	@echo "  make monitor     serial console on $(SERIAL_PORT) @ $(SERIAL_BAUD) baud"
	@echo "                   (override: SERIAL_PORT=/dev/ttyUSB1 SERIAL_BAUD=115200)"
	@echo "  make size        RAM/ROM usage of the last $(VARIANT) build"
	@echo "  make menuconfig  interactive Kconfig"
	@echo "  make clean       remove build trees (keeps the trained model)"
	@echo "  make distclean   also remove .pte models, checkpoint and export stamp"
	@echo "Overrides: VARIANT=portable|cmsis  DATASET=mnist|synthetic  BOARD=...  SERIAL_PORT=..."
