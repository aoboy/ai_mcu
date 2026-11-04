#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# patch_module.sh -- apply / check the local patches this project needs on
# the ExecuTorch west module (~/zephyrproject/modules/lib/executorch).
#
# Why: the module's op-codegen (tools/cmake/Codegen.cmake) runs
# `python -c "import torchgen; ..."` with the Zephyr build interpreter and
# then GLOB_RECURSEs the printed directory. With a torch-less west venv the
# import fails, the directory is empty, and the glob degenerates to "/*.py" —
# every Python file on the machine becomes a ninja dependency. The patch lets
# a torch-equipped interpreter be passed as -DET_CODEGEN_PYTHON=<path> and
# fails with a clear message instead of globbing the filesystem.
#
# The west manifest tracks pytorch/executorch@main, so `west update` may
# rewrite the module and drop the patch — re-run this script afterwards (it
# refuses to double-apply).
#
# Usage:
#   ./scripts/patch_module.sh [apply|check|status|unapply]   (default: apply)
#
# west update refuses to touch a checkout with local modifications, so
# updating the module requires the dance:
#   ./scripts/patch_module.sh unapply && west update <projects> \
#     && ./scripts/patch_module.sh apply
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE="${MODULE:-${AI_WEST_TOPDIR:-$HOME/zephyrproject}/modules/lib/executorch}"
PATCHES_DIR="${REPO_DIR}/west-manifests/patches"

cmd="${1:-apply}"

if [ ! -d "${MODULE}/.git" ]; then
  echo "ExecuTorch module not found at ${MODULE}" >&2
  echo "run scripts/01_setup_workspace.sh first" >&2
  exit 1
fi

case "${cmd}" in
  status)
    echo "module  : ${MODULE}"
    echo "patches : $(ls -1 "${PATCHES_DIR}"/*.patch 2>/dev/null | wc -l)"
    git -C "${MODULE}" status --short
    for p in "${PATCHES_DIR}"/*.patch; do
      if git -C "${MODULE}" apply --check "${p}" 2>/dev/null; then
        echo "NOT APPLIED: $(basename "${p}")"
      elif git -C "${MODULE}" apply --reverse --check "${p}" 2>/dev/null; then
        echo "applied    : $(basename "${p}")"
      else
        echo "CONFLICTS  : $(basename "${p}") (upstream changed? west update, then re-check)"
      fi
    done
    ;;
  check)
    for p in "${PATCHES_DIR}"/*.patch; do
      if git -C "${MODULE}" apply --reverse --check "${p}" 2>/dev/null; then
        echo "ok: $(basename "${p}") already applied"
      elif git -C "${MODULE}" apply --check "${p}" 2>/dev/null; then
        echo "ok: $(basename "${p}") applicable"
      else
        echo "PATCH CONFLICTS with the current module checkout: $(basename "${p}")" >&2
        echo "upstream (executorch@main) likely changed; update the patch in west-manifests/patches/" >&2
        exit 1
      fi
    done
    ;;
  apply)
    for p in "${PATCHES_DIR}"/*.patch; do
      if git -C "${MODULE}" apply --reverse --check "${p}" 2>/dev/null; then
        echo "already applied: $(basename "${p}")"
      elif git -C "${MODULE}" apply --check "${p}" 2>/dev/null; then
        git -C "${MODULE}" apply "${p}"
        echo "applied: $(basename "${p}")"
      else
        echo "PATCH CONFLICTS with the current module checkout: $(basename "${p}")" >&2
        echo "upstream (executorch@main) likely changed; update the patch in west-manifests/patches/" >&2
        exit 1
      fi
    done
    ;;
  unapply)
    # Reverse the patches so west update can move the checkout. Refuses to
    # run when OTHER uncommitted changes are present (they would be lost by
    # a later hard reset; we only ever revert what our patches touched).
    for p in "${PATCHES_DIR}"/*.patch; do
      if git -C "${MODULE}" apply --check "${p}" 2>/dev/null; then
        echo "not applied: $(basename "${p}")"
      elif git -C "${MODULE}" apply --reverse --check "${p}" 2>/dev/null; then
        git -C "${MODULE}" apply --reverse "${p}"
        echo "unapplied: $(basename "${p}")"
      else
        echo "cannot unapply: $(basename "${p}") conflicts with the working tree" >&2
        echo "resolve manually in ${MODULE} (git diff / git checkout -- <file>)" >&2
        exit 1
      fi
    done
    ;;
  *)
    echo "unknown command '${cmd}' (use: apply|check|status)" >&2
    exit 2
    ;;
esac
