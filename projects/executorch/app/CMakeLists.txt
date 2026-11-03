# SPDX-License-Identifier: Apache-2.0
#
# Zephyr application that runs an ExecuTorch .pte MNIST model on the
# Raspberry Pi Pico 2 W. Structurally identical to the official
# zephyr/samples/hello-executorch sample of the ExecuTorch repo (v1.5.x);
# see docs/ARCHITECTURE.md for a walkthrough of every block.
#
# The .pte to embed is chosen at configure time:
#
#   west build -b rpi_pico2/rp2350a/m33_0/w app -- \
#       -DET_PTE_FILE_PATH=$PWD/models/mnist_cnn_fp32.pte    # portable ops
#   west build -b rpi_pico2/rp2350a/m33_0/w app -- \
#       -DET_PTE_FILE_PATH=$PWD/models/mnist_cnn_int8.pte    # CMSIS-NN path
#
cmake_minimum_required(VERSION 3.24)

# Disable install(rule) generation while building under Zephyr. The app never
# runs `cmake --install`, and skipping the rules avoids export dependency
# checks when Zephyr provides targets such as cmsis-nn.
set(CMAKE_SKIP_INSTALL_RULES
    ON
    CACHE BOOL "" FORCE
)

# Use configuration overlay (prj.conf sets CONFIG_EXECUTORCH_BUILD_PORTABLE_OPS=n)
# to disable the default portable-ops registry so we can use a selective
# operator build: only the operators the .pte actually needs are linked.
set(ET_PTE_FILE_PATH
    ""
    CACHE FILEPATH "Path to the ExecuTorch .pte (or .bpte) model to embed"
)
set(ET_PTE_SECTION
    "network_model_sec"
    CACHE STRING "Section attribute used for the generated model data"
)
if(NOT ET_PTE_FILE_PATH)
  message(
    FATAL_ERROR
      "ET_PTE_FILE_PATH must point to the ExecuTorch .pte model to embed. "
      "Produce one with: python3 python/export_mnist.py"
  )
endif()
if(NOT IS_ABSOLUTE "${ET_PTE_FILE_PATH}")
  get_filename_component(
    ET_PTE_FILE_PATH "${ET_PTE_FILE_PATH}" ABSOLUTE BASE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()
if(NOT EXISTS "${ET_PTE_FILE_PATH}")
  message(
    FATAL_ERROR
      "Could not find ExecuTorch model at ET_PTE_FILE_PATH: ${ET_PTE_FILE_PATH}"
  )
endif()
set(ET_PTE_FILE_PATH
    "${ET_PTE_FILE_PATH}"
    CACHE FILEPATH "Path to the ExecuTorch .pte (or .bpte) model to embed"
    FORCE
)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(executorch_mnist_pico2w)

find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED
)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wno-switch -Wno-float-conversion -ffunction-sections -fdata-sections"
)

# Always use selective build since embedded devices are resource constrained.
# Locate the ExecuTorch module: Zephyr exports ZEPHYR_EXECUTORCH_MODULE_DIR
# for every module discovered through the west manifest (see
# west-manifests/executorch.yaml).
if(NOT DEFINED EXECUTORCH_DIR)
  message(STATUS "ZEPHYR_EXECUTORCH_MODULE_DIR: ${ZEPHYR_EXECUTORCH_MODULE_DIR}")
  if(DEFINED ZEPHYR_EXECUTORCH_MODULE_DIR)
    set(EXECUTORCH_DIR ${ZEPHYR_EXECUTORCH_MODULE_DIR})
    message(STATUS "Using Zephyr module discovery: EXECUTORCH_DIR=${EXECUTORCH_DIR}")
  else()
    message(
      FATAL_ERROR
        "ExecuTorch module not found. Ensure the west submanifest "
        "west-manifests/executorch.yaml is installed and `west update` ran."
    )
  endif()
else()
  message(STATUS "Using predefined EXECUTORCH_DIR=${EXECUTORCH_DIR}")
endif()

include(${EXECUTORCH_DIR}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_DIR}/backends/arm/cmake/ArmRunnerUtils.cmake)

# NOTE: the module's op codegen imports torchgen (ships inside torch) with
# the build interpreter. The west/build venv has no torch, so every build
# passes -DET_CODEGEN_PYTHON=<pytorch_cu132 python> — supported by the local
# module patch west-manifests/patches/0001-*.patch (apply with
# scripts/patch_module.sh; it fails with a clear message if the patch is
# missing, instead of globbing every Python file into the build graph).

# Ensure the portable kernels target exists even when portable ops are
# disabled globally (CONFIG_EXECUTORCH_BUILD_PORTABLE_OPS=n).
if(NOT TARGET portable_kernels)
  set(EXECUTORCH_PORTABLE_BUILD_KERNELS_ONLY ON)
  add_subdirectory(
    ${EXECUTORCH_DIR}/kernels/portable
    ${CMAKE_CURRENT_BINARY_DIR}/executorch/kernels/portable
  )
  unset(EXECUTORCH_PORTABLE_BUILD_KERNELS_ONLY)
endif()

set(_executorch_zephyr_selected_ops_libs)

# Portable operator library restricted to the operators present in the .pte.
# arm_runner_create_selected_ops_lib reads the operator table out of the
# serialized model (EXECUTORCH_SELECT_OPS_MODEL mechanism), so operators
# introduced by the edge-IR lowering passes are not missed.
arm_runner_create_selected_ops_lib(
  LIB_NAME
  "cpu_portable_ops_lib"
  FUNCTIONS_YAML
  "${EXECUTORCH_DIR}/kernels/portable/functions.yaml"
  OPS_FROM_MODEL
  "${ET_PTE_FILE_PATH}"
  KERNEL_LIBS
  portable_kernels
  DEPS
  executorch
)
list(APPEND _executorch_zephyr_selected_ops_libs cpu_portable_ops_lib)

# The Zephyr module builds `cortex_m_kernels` (CMSIS-NN kernels) on every
# CONFIG_CPU_CORTEX_M board (see the module's zephyr/CMakeLists.txt:
# `if(CONFIG_CPU_CORTEX_M) set(EXECUTORCH_BUILD_CORTEX_M ON)`). Linking the
# cortex-m custom-ops library makes the INT8 .pte exported with the
# CortexMQuantizer dispatch to CMSIS-NN implementations; everything the
# cortex-m kernels do not cover transparently falls back to the portable
# library above. This is what turns the "both paths" switch: same firmware,
# only the .pte (and hence the dispatch) differs.
if(TARGET cortex_m_kernels)
  arm_runner_create_selected_ops_lib(
    LIB_NAME
    "zephyr_cortex_m_ops_lib"
    CUSTOM_OPS_YAML
    "${EXECUTORCH_DIR}/backends/cortex_m/ops/operators.yaml"
    OPS_FROM_MODEL
    "${ET_PTE_FILE_PATH}"
    KERNEL_LIBS
    cortex_m_kernels
    DEPS
    executorch
  )
  list(APPEND _executorch_zephyr_selected_ops_libs zephyr_cortex_m_ops_lib)
endif()

# HACK (same as upstream): the executorch CMake scripts generate a relative
# include path ("../") that does not resolve in an out-of-tree module build,
# so force the parent directory of executorch into the global include path.
set(_local_flatcc_root ${CMAKE_BINARY_DIR}/flatcc_src)
if(NOT EXISTS ${_local_flatcc_root}/CMakeLists.txt)
  file(MAKE_DIRECTORY ${_local_flatcc_root})
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${EXECUTORCH_DIR}/third-party/flatcc ${_local_flatcc_root}
  )
endif()
set(EXECUTORCH_FLATCC_SOURCE_ROOT
    ${_local_flatcc_root}
    CACHE PATH "" FORCE
)
set(EXECUTORCH_FLATCC_INSTALL_ROOT
    ${_local_flatcc_root}
    CACHE PATH "" FORCE
)

set(app_sources
    src/main.cpp
    # Official bump allocator used by all upstream Arm samples (also provides
    # the used_size()/free_size() reporting our demo prints).
    ${EXECUTORCH_DIR}/examples/arm/common/arm_memory_allocator.cpp
)
target_sources(app PRIVATE ${app_sources})

# Convert the .pte into a C header: a `const unsigned char model_pte[]` in
# section "network_model_sec", which the linker places in FLASH. On the Pico
# 2 W that is the external QSPI flash, read in place through XIP.
set(_model_pte_header ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h)
add_custom_command(
  OUTPUT ${_model_pte_header}
  COMMAND
    ${Python3_EXECUTABLE} ${EXECUTORCH_DIR}/examples/arm/common/pte_to_header.py
    --pte ${ET_PTE_FILE_PATH} --outdir ${CMAKE_CURRENT_BINARY_DIR} --section
    ${ET_PTE_SECTION}
  DEPENDS ${ET_PTE_FILE_PATH}
  ${EXECUTORCH_DIR}/examples/arm/common/pte_to_header.py
  COMMENT "Converting ${ET_PTE_FILE_PATH} to model_pte.h"
)
add_custom_target(gen_model_header DEPENDS ${_model_pte_header})
add_dependencies(app gen_model_header)

# Plumb the Kconfig-configured pool sizes into the source.
if(DEFINED CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(
    app
    PRIVATE
    ET_ARM_METHOD_ALLOCATOR_POOL_SIZE=${CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE}
  )
endif()
if(DEFINED CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(
    app
    PRIVATE
    ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=${CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE}
  )
endif()

# Link the Zephyr-agnostic ExecuTorch runtime (libexecutorch is the module's
# zephyr_library that wraps executorch_core + optional portable_ops_lib) plus
# our selective operator libraries.
target_link_libraries(app PRIVATE libexecutorch)
target_link_libraries(app PRIVATE ${_executorch_zephyr_selected_ops_libs})

target_include_directories(app PRIVATE src ${CMAKE_CURRENT_BINARY_DIR})
