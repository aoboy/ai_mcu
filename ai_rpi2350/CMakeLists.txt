# SPDX-License-Identifier: Apache-2.0
# TFLite-Micro sine-demo application — Zephyr / Raspberry Pi Pico 2 W (RP2350)

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(ai_mcu_tflm_sine)

# tflite-micro relies on local static initialization; Zephyr does not provide
# the C++ ABI for thread-safe initialization of local statics, and the
# constructors here do not require thread safety (same approach as Zephyr's
# samples/modules/tflite-micro/hello_world).
set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

# The Pico's USB device driver still uses the legacy USB stack API, which
# emits deprecation warnings on Zephyr main (same as the gpio16_blinky
# sample they inspired this setup). Only silence them when the CDC ACM
# console is actually enabled (i.e. building for the Pico 2 W).
if(CONFIG_USB_CDC_ACM)
  zephyr_compile_options(-Wno-deprecated-declarations)
endif()

target_sources(app PRIVATE
  src/main.cpp
  src/sine_model_data.cc
)
