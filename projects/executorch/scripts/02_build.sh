#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 02_build.sh -- build the firmware for the Raspberry Pi Pico 2 W.
#
# Usage:
#   ./scripts/02_build.sh [portable|cmsis] [workspace-dir]
#
#   portable  float32 .pte -> Portable Ops kernels   (default)
#   cmsis     INT8 .pte     -> Cortex-M / CMSIS-NN kernels
#
# Both binaries contain the SAME application code; the executing backend is
# determined by the operators recorded inside the embedded .pte.
#
# The build runs in the pre-provisioned `zephyr_rtos` virtualenvwrapper env
# (where west lives); activate it with `workon zephyr_rtos` first, or let the
# script do it (default workspace: ~/zephyrproject).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-portable}"
WS="${2:-$HOME/zephyrproject}"
ENV_NAME="${ENV_NAME:-zephyr_rtos}"
WORKON_HOME="${WORKON_HOME:-$HOME/.virtualenvs}"
BOARD="${BOARD:-rpi_pico2/rp2350a/m33_0/w}"
BUILD_DIR="${REPO_DIR}/build-${VARIANT}"

case "${VARIANT}" in
  portable)
    PTE="${REPO_DIR}/app/model/mnist_cnn_fp32.pte"
    ;;
  cmsis)
    PTE="${REPO_DIR}/app/model/mnist_cnn_int8.pte"
    ;;
  *)
    echo "unknown variant '${VARIANT}' (use: portable|cmsis)" >&2
    exit 2
    ;;
esac

if [ ! -f "${PTE}" ]; then
  echo "model not found: ${PTE}" >&2
  echo "export it first:  workon pytorch_cu132 && python python/export_mnist.py" >&2
  exit 1
fi

if [ ! -d "${WS}/.west" ]; then
  echo "no west workspace at ${WS} (run scripts/01_setup_workspace.sh first)" >&2
  exit 1
fi
cd "${WS}"
# west lives in the zephyr_rtos env; if it is not already active, activate it
# (same effect as `workon ${ENV_NAME}`, which also works non-interactively).
if ! command -v west >/dev/null; then
  # shellcheck disable=SC1091
  source "${WORKON_HOME}/${ENV_NAME}/bin/activate"
fi

echo "==> west build (${VARIANT}) board=${BOARD}"
west build -p -b "${BOARD}" "${REPO_DIR}/app" \
  -d "${BUILD_DIR}" -- -DET_PTE_FILE_PATH="${PTE}"

echo
echo "Build OK."
echo "  ELF : ${BUILD_DIR}/zephyr/zephyr.elf"
echo "  UF2 : ${BUILD_DIR}/zephyr/zephyr.uf2"
echo
echo "Flash (either way works):"
echo "  1) hold BOOTSEL, plug USB, copy zephyr.uf2 onto the RPI-RP2350 drive"
echo "  2) west flash -d ${BUILD_DIR}"
echo
echo "Console: UART0 on GP0(TX)/GP1(RX) at 115200 8N1, e.g."
echo "  picocom -b 115200 /dev/ttyUSB0"
