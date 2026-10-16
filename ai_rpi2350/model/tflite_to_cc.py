#!/usr/bin/env python3
"""tflite_to_cc.py — embed a .tflite flatbuffer into C++ sources.

Mirrors what Zephyr's hello_world sample does with ``xxd -i model.tflite``
(plus 16-byte alignment, which some MCU ports want for tensor data):

    python tflite_to_cc.py model/model_int8.tflite \\
            --source ../src/sine_model_data.cc --header ../src/sine_model.h

Emits:
  <source>  alignas(16) const unsigned char g_sine_model[] = { ... };
            const int g_sine_model_len = <N>;
  <header>  extern declarations guarded for C and C++.

Note: the header overwrites src/sine_model.h with a generated banner; the
symbology (g_sine_model / g_sine_model_len) is fixed so main.cpp never needs
to change when the model is retrained.
"""

from __future__ import annotations

import argparse
import pathlib

BANNER = """\
/*
 * {name} — embedded TFLite-Micro model (GENERATED FILE, do not edit).
 *
 * Source : {src}
 * Bytes  : {nbytes}
 * Regen  : make header        (model/tflite_to_cc.py)
 */

"""

HEADER_TEMPLATE = BANNER + """
#ifndef AI_MCU_SINE_MODEL_H_
#define AI_MCU_SINE_MODEL_H_

#ifdef __cplusplus
extern "C" {{
#endif

/* Raw TFLite flatbuffer; consumed with tflite::GetModel(g_sine_model). */
extern const unsigned char g_sine_model[];

/* Size of the flatbuffer in bytes. */
extern const int g_sine_model_len;

#ifdef __cplusplus
}}
#endif

#endif /* AI_MCU_SINE_MODEL_H_ */
"""

SOURCE_TEMPLATE = BANNER + """
#include "sine_model.h"

alignas(16) const unsigned char g_sine_model[] = {{
{body}
}};

const int g_sine_model_len = {nbytes};
"""


def to_c_array(data: bytes, per_line: int = 12) -> str:
    lines = []
    for i in range(0, len(data), per_line):
        chunk = data[i:i + per_line]
        lines.append("\t" + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    # Drop the trailing comma on the very last line for tidiness.
    lines[-1] = lines[-1].rstrip(",")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("tflite", type=pathlib.Path, help="input .tflite model")
    p.add_argument("--source", type=pathlib.Path, required=True,
                   help="output .cc file")
    p.add_argument("--header", type=pathlib.Path, required=True,
                   help="output .h file")
    args = p.parse_args(argv)

    data = args.tflite.read_bytes()
    # Sanity check: a TFLite flatbuffer starts with a "TFL3" offset-polytable
    # magic at bytes [4:8].
    if data[4:8] != b"TFL3":
        raise SystemExit(f"{args.tflite} does not look like a .tflite file")

    banner_vars = dict(name=args.tflite.name, src=str(args.tflite),
                       nbytes=len(data))

    args.source.parent.mkdir(parents=True, exist_ok=True)
    args.source.write_text(
        SOURCE_TEMPLATE.format(body=to_c_array(data), **banner_vars))
    args.header.write_text(HEADER_TEMPLATE.format(**banner_vars))

    print(f"[embed] {args.tflite} ({len(data)} bytes)")
    print(f"        -> {args.source}")
    print(f"        -> {args.header}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
