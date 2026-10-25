# ai_mcu

ML/AI models on microcontrollers (MCUs). Each project lives in its own
directory under `projects/`, so unrelated experiments can share one repo
without stepping on each other.

## Layout

```
ai_mcu/
├── README.md            # this file: repo map + conventions
├── projects/            # one directory per project
│   └── tflite-micro/    # TFLite-Micro sine demo on the RP2350 (Pico 2 W)
├── tools/               # code shared across projects
│   └── ml/              # Python training/quantization tooling
├── .vscode/             # editor config (cmake.sourceDirectory)
└── .gitignore           # globs written as projects/*/<artifact> …
```

## Projects

| Project | What it is |
|---|---|
| [`projects/tflite-micro`](projects/tflite-micro/README.md) | TFLite-Micro on Zephyr: an int8 sine MLP on the Raspberry Pi Pico 2 W (RP2350), smoke-tested in QEMU without hardware. |

## Conventions

- **A project owns its own directory**: source, Makefile, docs, plans, and its
  build trees all live under `projects/<name>/`. Nothing project-specific goes
  at the repo root.
- **Shared tooling goes in `tools/<area>/`**. It is imported by path from a
  project's Makefile / Python wrapper, never copied. Example:
  `tools/ml/generate_tf_mcu_model.py` is the single source of truth for the
  train → quantize → validate pipeline; `projects/tflite-micro` wraps it.
- **Build outputs are ignored by glob**, not by name, so a new project is
  covered as soon as it is created: `projects/*/build_*/`,
  `projects/*/model/calib/`, …
- **Zephyr stays out of tree.** The west workspace (`~/zephyrproject`) is
  outside this repo; projects invoke `west build` from there with absolute
  paths. Moving a project directory is therefore safe, but build trees must be
  regenerated afterwards — `CMakeCache.txt` and `build.ninja` bake in the old
  absolute paths (run `make clean` or delete `build_*/`).

## Adding a project

```
projects/<new-name>/
├── README.md        # what it is, hardware, status
├── docs/            # its own docs
└── …sources
```

If it reuses the training pipeline, import it — do not fork it:

```python
# in projects/<name>/model/generate_model.py
HERE = Path(__file__).resolve().parent       # <repo>/projects/<name>/model
sys.path.insert(0, str(HERE.parents[2] / "tools" / "ml"))
import generate_tf_mcu_model as pipeline
```
