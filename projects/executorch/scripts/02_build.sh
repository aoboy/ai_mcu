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
# Environment names / paths come from tools/env.sh (repo-wide single source of
# truth); the fallbacks below only apply if that file moved.
if [ -f "${REPO_DIR}/../../tools/env.sh" ]; then
  # shellcheck disable=SC1091
  . "${REPO_DIR}/../../tools/env.sh"
fi
WS="${2:-${AI_WEST_TOPDIR:-$HOME/zephyrproject}}"
ENV_NAME="${ENV_NAME:-${AI_ENV_ZEPHYR:-zephyr_rtos}}"
PYTORCH_ENV="${PYTORCH_ENV:-${AI_ENV_TORCH:-pytorch_cu132}}"
WORKON_HOME="${WORKON_HOME:-${AI_WORKON_HOME:-$HOME/.virtualenvs}}"
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
# ET_CODEGEN_PYTHON: the ExecuTorch op codegen imports torchgen, which ships
# inside torch; the west/build env does not have it. Requires the module
# patch (scripts/patch_module.sh). See app/CMakeLists.txt and
# west-manifests/patches/.
TORCH_PY="${WORKON_HOME}/${PYTORCH_ENV}/bin/python"
if [ ! -x "${TORCH_PY}" ]; then
  echo "torch env python not found: ${TORCH_PY}" >&2
  echo "needed for the model codegen step (tools/env.sh: ${PYTORCH_ENV})" >&2
  exit 1
fi
west build -p -b "${BOARD}" "${REPO_DIR}/app" \
  -d "${BUILD_DIR}" -- -DET_PTE_FILE_PATH="${PTE}" -DET_CODEGEN_PYTHON="${TORCH_PY}"

echo
echo "Build OK."
echo "  ELF : ${BUILD_DIR}/zephyr/zephyr.elf"
echo "  UF2 : ${BUILD_DIR}/zephyr/zephyr.uf2"
echo
echo "Flash (either way works):"
echo "  1) hold BOOTSEL, plug USB, copy zephyr.uf2 onto the RPI-RP2350 drive"
echo "  2) west flash -d ${BUILD_DIR}"
echo
echo "Console: USB CDC ACM on the board's USB port (RPI VID 2E8A,"
echo "  enumerates as /dev/ttyACM*), e.g."
echo "  picocom /dev/ttyACM0"
