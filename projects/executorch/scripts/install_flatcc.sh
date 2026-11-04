#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# install_flatcc.sh -- build the ExecuTorch host flatcc compiler ONCE into
# the west workspace (${AI_WEST_TOPDIR}/tools/flatcc) so the firmware builds
# stop running the module's `flatcc_ep` external project on every compile.
#
# Why: the module builds a HOST flatcc via ExternalProject_Add(flatcc_ep)
# with BUILD_ALWAYS TRUE, but the binary is only consumed by
# devtools/etdump, which our Zephyr builds do not enable -- a full host
# cmake pass per firmware build for an unused tool. The module's official
# escape hatch is -DFLATCC_EXECUTABLE=<path>: when set, flatcc_ep is
# skipped and an imported target is used instead. This script produces
# that prebuilt binary with the exact same flags the flatcc_ep would use.
#
# The install lives OUTSIDE this repo (like the Zephyr SDK); `west update`
# does not manage it. The in-tree flatccrt runtime library that gets linked
# into the firmware is unaffected -- it is built by the module itself from
# third-party/flatcc as a normal CMake target.
#
# Usage:
#   ./scripts/install_flatcc.sh [install|status|force]
#
#     install  build + install if missing (idempotent; default)
#     status   report whether the prebuilt binary exists
#     force    rebuild + reinstall even if present
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${REPO_DIR}/../../tools/env.sh" ]; then
  # shellcheck disable=SC1091
  . "${REPO_DIR}/../../tools/env.sh"
fi
WS="${AI_WEST_TOPDIR:-$HOME/zephyrproject}"
MODULE="${MODULE:-${WS}/modules/lib/executorch}"
PREFIX="${PREFIX:-${WS}/tools/flatcc}"
FLATCC_BIN="${PREFIX}/bin/flatcc"

cmd="${1:-install}"

case "${cmd}" in
  status)
    if [ -x "${FLATCC_BIN}" ]; then
      echo "installed: ${FLATCC_BIN}"
      "${FLATCC_BIN}" --version 2>/dev/null | head -1 || true
    else
      echo "not installed: ${FLATCC_BIN} missing"
      echo "install with: $0   (or ET_APPLY_PATCHES-style: 01_setup_workspace.sh does it automatically)"
    fi
    exit 0
    ;;
  install|force)
    ;;
  *)
    echo "unknown command '${cmd}' (use: install|status|force)" >&2
    exit 2
    ;;
esac

if [ -x "${FLATCC_BIN}" ] && [ "${cmd}" = "install" ]; then
  echo "flatcc already installed: ${FLATCC_BIN}"
  exit 0
fi

for d in "${MODULE}/third-party/flatcc-host" "${MODULE}/third-party/flatcc"; do
  if [ ! -f "${d}/CMakeLists.txt" ]; then
    echo "ExecuTorch module incomplete: ${d} missing" >&2
    echo "run scripts/01_setup_workspace.sh first" >&2
    exit 1
  fi
done

echo "==> Building host flatcc (one-time) -> ${PREFIX}"
BUILD_DIR="${PREFIX}/.build"
rm -rf "${BUILD_DIR}"
cmake -S "${MODULE}/third-party/flatcc-host" -B "${BUILD_DIR}" -GNinja \
  -DEXECUTORCH_FLATCC_SOURCE_DIR:PATH="${MODULE}/third-party/flatcc" \
  -DFLATCC_RTONLY=OFF \
  -DFLATCC_TEST=OFF \
  -DFLATCC_REFLECTION=OFF \
  -DFLATCC_DEBUG_CLANG_SANITIZE=OFF \
  -DFLATCC_ALLOW_WERROR=OFF \
  -DFLATCC_INSTALL=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_INSTALL_PREFIX:PATH="${PREFIX}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_TOOLCHAIN_FILE= > /dev/null
cmake --build "${BUILD_DIR}" > /dev/null
cmake --install "${BUILD_DIR}" > /dev/null
rm -rf "${BUILD_DIR}"

if [ ! -x "${FLATCC_BIN}" ]; then
  echo "install finished but ${FLATCC_BIN} not found" >&2
  exit 1
fi
echo "    installed: ${FLATCC_BIN} ($("${FLATCC_BIN}" --version 2>/dev/null | head -1 || echo '?'))"
echo "Firmware builds now pass -DFLATCC_EXECUTABLE and skip the module's"
echo "BUILD_ALWAYS flatcc_ep external project."
