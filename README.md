# ai_mcu

ML/AI models on microcontrollers (MCUs). Each project lives in its own
directory under `projects/`, so unrelated experiments can share one repo
without stepping on each other.

## Layout

```
ai_mcu/
├── README.md            # this file: repo map + conventions
├── projects/            # one directory per project
│   ├── tflite-micro/    # TFLite-Micro sine demo on the RP2350 (Pico 2 W)
│   └── executorch/      # ExecuTorch MNIST demo on the same board
├── tools/
│   ├── env.sh           # single source of truth: env names + paths
│   └── ml/              # shared Python training/quantization tooling
├── .vscode/             # editor config (cmake.sourceDirectory)
└── .gitignore           # globs written as projects/*/<artifact> …
```

## Projects

| Project | What it is |
|---|---|
| [`projects/tflite-micro`](projects/tflite-micro/README.md) | TFLite-Micro on Zephyr: an int8 sine MLP on the Raspberry Pi Pico 2 W (RP2350), smoke-tested in QEMU without hardware. |
| [`projects/executorch`](projects/executorch/README.md) | ExecuTorch on Zephyr: a PyTorch MNIST CNN exported to `.pte` (float32 Portable Ops **and** INT8 CMSIS-NN), executed from XIP flash on the same board; QEMU smoke test for the portable path. |

## Environments

Everything is pre-provisioned on this machine — nothing in this repo installs
or downloads environments. Interactive use goes through virtualenvwrapper:

| venv (`workon …`) | Contents | Used for |
|---|---|---|
| `zephyr_rtos` | west 1.5, Zephyr SDK | firmware builds (all projects) |
| `tensorflow` | TensorFlow (CPU) | tflite-micro training/quantization |
| `pytorch_cu132` | torch + executorch | executorch training/export |

`deactivate` leaves a venv. Non-interactive code (Makefiles, scripts) does
not use `workon`; they read [`tools/env.sh`](tools/env.sh) instead — the
single source of truth for venv names, `WORKON_HOME` and the west workspace
(`~/zephyrproject`). Changing an env name or path is a one-file edit there.

## Conventions

- **A project owns its own directory**: source, Makefile, docs, plans, and its
  build trees all live under `projects/<name>/`. Nothing project-specific goes
  at the repo root.
- **Every project has a Makefile with the same target vocabulary**
  (`model`, `build`, `flash`, `monitor`, `size`, `clean`, …), wrapping its
  scripts, so the projects feel identical from the shell.
- **Shared tooling goes in `tools/<area>/`** and is imported by path, never
  copied — `tools/ml/generate_tf_mcu_model.py` is the single source of truth
  for the train → quantize → validate pipeline; `projects/tflite-micro`
  wraps it. A project may keep its **own** ML tooling when the framework
  differs (`projects/executorch/python/` is PyTorch/ExecuTorch, not TF).
- **Environment knowledge lives only in `tools/env.sh`** — venv names, the
  west workspace path and the helpers to use them non-interactively.
  Projects source it instead of hardcoding paths.
- **Build outputs are ignored by glob**, not by name, so a new project is
  covered as soon as it is created: `projects/*/build_*/`, `build-*/`,
  `projects/*/model/calib/`, … A project may add its own `.gitignore` for
  artifacts the root globs don't fit.
- **Zephyr stays out of tree.** The west workspace (`~/zephyrproject`) is
  outside this repo; projects invoke `west build` from there with absolute
  paths. Moving a project directory is therefore safe, but build trees must be
  regenerated afterwards — `CMakeCache.txt` and `build.ninja` bake in the old
  absolute paths (run `make clean` or delete the build tree).

## Adding a project

```
projects/<new-name>/
├── README.md        # what it is, hardware, status
├── Makefile         # the shared target vocabulary (see an existing project)
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

If it needs new venv names or paths, add them to `tools/env.sh` — never
hardcode them in the project.
