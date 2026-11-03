# tools/env.sh -- single source of truth for the shared host environments.
#
# Sourced (never executed) by project Makefiles and scripts so the venv names
# and paths live in exactly one place. Nothing here installs anything: it only
# locates the pre-provisioned virtualenvs and the west workspace.
#
# Interactive shells use virtualenvwrapper directly:
#   workon tensorflow      # TensorFlow (CPU)  -- tflite-micro training/quantization
#   workon pytorch_cu132   # torch + executorch -- executorch training/export
#   workon zephyr_rtos     # west + Zephyr SDK -- firmware builds, every project
#   deactivate             # leave
#
# Non-interactive code (Makefiles, scripts) uses the AI_* variables and the
# ai_python / ai_activate helpers instead of `workon`, which needs
# virtualenvwrapper's shell functions. Variables honor pre-set values:
#   AI_WORKON_HOME=... AI_WEST_TOPDIR=... . tools/env.sh

AI_WORKON_HOME="${AI_WORKON_HOME:-$HOME/.virtualenvs}"
AI_WEST_TOPDIR="${AI_WEST_TOPDIR:-$HOME/zephyrproject}"

# The three environments, by role:
AI_ENV_TF="tensorflow"        # TensorFlow (CPU): train + int8 quantize
AI_ENV_TORCH="pytorch_cu132"  # PyTorch + ExecuTorch: train + .pte export
AI_ENV_ZEPHYR="zephyr_rtos"   # west: build, flash, QEMU, size reports

# Path to a venv's interpreter:  py="$(ai_python tensorflow)"
ai_python() { echo "$AI_WORKON_HOME/$1/bin/python"; }

# Activate a venv in the current shell (script-equivalent of workon):
ai_activate() { . "$AI_WORKON_HOME/$1/bin/activate"; }
