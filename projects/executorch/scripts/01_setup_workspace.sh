#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 01_setup_workspace.sh -- attach this project to the ALREADY PROVISIONED
# Zephyr environment. This script no longer creates or installs anything:
#
#   * Python env     : the virtualenvwrapper env `zephyr_rtos` is activated
#                      (west, pyelftools, jsonschema, ...). No venv is
#                      created, nothing is pip-installed.
#   * West workspace : the existing `~/zephyrproject` is used as-is. No
#                      `west init`, no Zephyr checkout, no `west sdk install`.
#   * ExecuTorch     : the module is already cloned at
#                      ~/zephyrproject/modules/lib/executorch (submanifest in
#                      zephyr/submanifests/executorch.yml). The C++ runtime is
#                      built by Zephyr itself during `west build`; the Python
#                      export tooling (executorch 1.5.1 + torch) lives in the
#                      `pytorch_cu132` env, NOT in zephyr_rtos.
#
# The script only VERIFIES the pieces this project needs and tops up the one
# glue file it owns (the west submanifest) if it is missing.
#
# Usage:
#   ./scripts/01_setup_workspace.sh [workspace-dir]   (default: ~/zephyrproject)
#
# Optional toggles (off by default so nothing is ever downloaded):
#   ET_DO_UPDATE=1   run `west update` -- but only for the modules below that
#                    are missing, never for the whole workspace
#   ET_DO_BLOBS=1    run `west blobs fetch hal_infineon` (CYW43 WiFi firmware;
#                    the MNIST demo does not need it)
#
# Requirements: the `zephyr_rtos` virtualenvwrapper env and the `~/zephyrproject`
# west workspace with the Zephyr SDK already set up (they are -- see the repo
# README). If a `west build` ever complains about a missing host Python
# package, top up manually with `west packages pip --install` in the workspace.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${1:-$HOME/zephyrproject}"
ENV_NAME="${ENV_NAME:-zephyr_rtos}"
PYTORCH_ENV="${PYTORCH_ENV:-pytorch_cu132}"
WORKON_HOME="${WORKON_HOME:-$HOME/.virtualenvs}"
# west project names this project needs (path != name, e.g. hal_infineon
# lives at modules/hal/infineon, which is why we check via `west list`).
REQUIRED_MODULES="executorch cmsis cmsis-nn cmsis_6 hal_infineon"

echo "==> Python environment: activating '${ENV_NAME}' (no install)"
if [ -n "${VIRTUAL_ENV:-}" ] && [ "$(basename "${VIRTUAL_ENV}")" = "${ENV_NAME}" ]; then
  echo "    already active: ${VIRTUAL_ENV}"
elif [ -f "${WORKON_HOME}/${ENV_NAME}/bin/activate" ]; then
  # Equivalent of `workon ${ENV_NAME}` that also works non-interactively
  # (virtualenvwrapper's shell functions are not loaded in scripts).
  # shellcheck disable=SC1091
  source "${WORKON_HOME}/${ENV_NAME}/bin/activate"
else
  echo "    env '${ENV_NAME}' not found under ${WORKON_HOME}" >&2
  echo "    use: workon ${ENV_NAME}   (or create it once with mkvirtualenv)" >&2
  exit 1
fi
command -v west >/dev/null || { echo "    west not on PATH after activation" >&2; exit 1; }

echo "==> Workspace: ${WS}"
if [ ! -d "${WS}/.west" ]; then
  echo "    no west workspace at ${WS} -- this script does not create one" >&2
  echo "    point it at the existing workspace: $0 ~/zephyrproject" >&2
  exit 1
fi
cd "${WS}"

echo "==> ExecuTorch module submanifest"
if [ -f zephyr/submanifests/executorch.yml ]; then
  echo "    present: zephyr/submanifests/executorch.yml (left untouched)"
elif [ -f zephyr/submanifests/executorch.yaml ]; then
  echo "    present: zephyr/submanifests/executorch.yaml (left untouched)"
else
  mkdir -p zephyr/submanifests
  cp "${REPO_DIR}/west-manifests/executorch.yaml" zephyr/submanifests/executorch.yml
  echo "    copied west-manifests/executorch.yaml -> zephyr/submanifests/executorch.yml"
fi

echo "==> Verifying required west modules (executorch + CMSIS-NN + Infineon HAL)"
MISSING=""
for m in ${REQUIRED_MODULES}; do
  MPATH="$(west list --format '{path}' "${m}" 2>/dev/null || true)"
  if [ -z "${MPATH}" ]; then
    echo "    MISSING: ${m} (no manifest entry)"
    MISSING="${MISSING} ${m}"
  elif [ ! -d "${MPATH}" ]; then
    echo "    MISSING: ${m} (in manifest but not cloned at ${MPATH})"
    MISSING="${MISSING} ${m}"
  else
    echo "    ok: ${m} -> ${MPATH}"
  fi
done
if [ -n "${MISSING}" ] && [ "${ET_DO_UPDATE:-0}" != "1" ]; then
  echo "    fix with: ET_DO_UPDATE=1 $0   (runs: west update${MISSING})" >&2
  exit 1
fi
if [ -n "${MISSING}" ]; then
  echo "==> west update (ONLY the missing modules, never the whole workspace)"
  west update${MISSING}
fi

echo "==> Zephyr SDK (verified, not installed)"
SDK_DIR="${ZEPHYR_SDK_INSTALL_DIR:-}"
if [ -z "${SDK_DIR}" ]; then
  # Same default locations Zephyr itself searches ($HOME/zephyr-sdk-* first).
  SDK_DIR="$(ls -d "${HOME}"/zephyr-sdk-* 2>/dev/null | head -1 || true)"
fi
if [ -n "${SDK_DIR}" ] && [ -d "${SDK_DIR}/gnu/arm-zephyr-eabi" ]; then
  echo "    found: ${SDK_DIR} (arm-zephyr-eabi)"
else
  echo "    NOT FOUND under ZEPHYR_SDK_INSTALL_DIR or ${HOME}/zephyr-sdk-*" >&2
  echo "    install it once manually: west sdk install --gnu-toolchains arm-zephyr-eabi" >&2
  exit 1
fi

echo "==> ExecuTorch module (verified, not rebuilt)"
if [ -f modules/lib/executorch/CMakeLists.txt ]; then
  echo "    module OK: modules/lib/executorch"
  echo "    (C++ runtime is built by Zephyr during west build; no"
  echo "     ./install_executorch.sh needed -- the Python export tooling"
  echo "     already lives in the '${PYTORCH_ENV}' env)"
else
  echo "    module not populated at modules/lib/executorch" >&2
  echo "    fix with: ET_DO_UPDATE=1 $0" >&2
  exit 1
fi

if [ "${ET_DO_BLOBS:-0}" = "1" ]; then
  echo "==> WiFi blobs for the Pico 2 W (optional, only for WiFi features)"
  west blobs fetch hal_infineon || echo "    (skipped: not fatal for this demo)"
fi

# Soft check: the export step needs executorch+torch, which live in
# ${PYTORCH_ENV}, not in the build env activated above.
if [ -x "${WORKON_HOME}/${PYTORCH_ENV}/bin/python" ] \
    && "${WORKON_HOME}/${PYTORCH_ENV}/bin/python" -c "import executorch" 2>/dev/null; then
  echo "==> Export env '${PYTORCH_ENV}': executorch import OK"
else
  echo "==> WARNING: cannot import executorch in '${PYTORCH_ENV}'" >&2
  echo "    train/export will fail; check with: workon ${PYTORCH_ENV} && python -c 'import executorch'" >&2
fi

ZV="$(awk '/^VERSION_MAJOR|^VERSION_MINOR|^PATCHLEVEL|^EXTRAVERSION/ {gsub(/[[:space:]]/, "", $3); if ($3 == "") next; v = (v == "" ? $3 : v ($1 ~ /EXTRAVERSION/ ? "-" : ".") $3)} END {print v}' zephyr/VERSION)"
echo
echo "Environment verified (nothing was installed or modified):"
echo "  workspace  : ${WS}  (Zephyr ${ZV})"
echo "  build env  : ${ENV_NAME}  (west build)"
echo "  export env : ${PYTORCH_ENV}  (torch + executorch python)"
echo "Next:"
echo "  workon ${PYTORCH_ENV}"
echo "  cd ${REPO_DIR}"
echo "  python python/train_mnist.py            # or --dataset synthetic"
echo "  python python/export_mnist.py"
echo "  deactivate && workon ${ENV_NAME}"
echo "  ./scripts/02_build.sh cmsis ${WS}"
