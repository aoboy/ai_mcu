# app/model/

Drop the exported ExecuTorch program files here (or anywhere; pass an
explicit path to CMake):

| File                    | Produced by                              | Runs on                      |
| ----------------------- | ---------------------------------------- | ---------------------------- |
| `mnist_cnn_fp32.pte`    | `python3 python/export_mnist.py`         | Portable Ops kernels (float) |
| `mnist_cnn_int8.pte`    | `python3 python/export_mnist.py --cmsis` | Cortex-M / CMSIS-NN kernels  |

Example build:

```bash
west build -p -b rpi_pico2/rp2350a/m33_0/w app -- \
    -DET_PTE_FILE_PATH=$PWD/app/model/mnist_cnn_int8.pte
```

The `.pte` never needs to be inside this folder -- the build only takes the
path you give via `-DET_PTE_FILE_PATH=`. The file is converted to a C header
(`model_pte.h`) at build time and linked into the firmware image; nothing is
read from the filesystem at runtime.
