# RX 6600 setup on native Windows

This project uses an experimental, pinned ZLUDA stack for the RX 6600 (gfx1032,
8 GiB). AMD's [HIP SDK 6.2.4 support table](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-6.2.4/reference/system-requirements.html)
supports this GPU's HIP runtime, but **not its SDK libraries**. The community
gfx1032 rocBLAS package below is therefore required. This is not official AMD
PyTorch support.

## 1. Install the matching native components

- Install AMD's Windows HIP SDK **6.2.4**. Its usual root is
  `C:\Program Files\AMD\ROCm\6.2`.
- Extract the **ROCm6** archive from [ZLUDA v3.9.5](https://github.com/lshqqytiger/ZLUDA/releases/tag/rel.5e717459179dc272b7d7d23391f0fad66c7459cf)
  (`ZLUDA-windows-rocm6-amd64.zip`) to `$env:USERPROFILE\zluda`, with `zluda.exe`
  directly in that directory. This release adds support for Adrenalin 25.5.1;
  compatibility with every newer driver is not established.
- Download `rocm.gfx1032.for.hip.sdk.6.2.4.navi21.logic.7z` from the
  [community rocBLAS v0.6.2.4 release](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.6.2.4).
  Back up the SDK's `bin\rocblas.dll` and `bin\rocblas\library` directory, then
  replace them with the package's matching DLL and library directory. Keep the
  matching SDK's `amdhip64_6.dll` and `rocsolver.dll` in `bin`.

The launchers check for the required files, Tensile metadata, and gfx1032 kernel
artifacts. File presence does not prove driver or binary compatibility; run the
numerical probes in step 4.

## 2. Create the Python environment

From this repository, using **Python 3.12 x64**:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-rx6600.txt
```

The requirements select **torch 2.4.1+cu118** and **NumPy 1.26.4** as a reproducible
baseline. torchvision is not needed by this trainer. Use the CUDA 11.8 wheel;
CPU, ROCm and newer CUDA wheels are not interchangeable with this setup. See
[PyTorch's 2.4.1 wheel instructions](https://pytorch.org/get-started/previous-versions/#v241)
and the [pinned ZLUDA PyTorch instructions](https://github.com/lshqqytiger/ZLUDA/blob/5e717459179dc272b7d7d23391f0fad66c7459cf/README.md#pytorch).

## 3. Configure paths and apply the cuBLAS fix

These optional environment variables override the defaults for the current shell:

```powershell
$env:ZLUDA_PATH = 'D:\GPU tools\zluda'          # directory containing zluda.exe
$env:HIP_PATH = 'C:\Program Files\AMD\ROCm\6.2' # SDK root, not bin
$env:RX6600_PYTHON = 'D:\Python envs\trainer\Scripts\python.exe'
```

Without overrides, launchers use `$env:USERPROFILE\zluda`, the HIP path above,
and this repository's `.venv\Scripts\python.exe`. Relative overrides resolve
from the calling directory. Launchers run the script from the repository, so
the corpus, cache, training log and checkpoints live there.

PyTorch's CUDA 11.8 wheel ships NVIDIA's `torch\lib\cublas64_11.dll`. In the
original RX 6600 setup, device discovery succeeded but GEMM returned
`CUBLAS_STATUS_NOT_SUPPORTED` until that DLL was replaced with ZLUDA's `cublas.dll`.
Apply the same fix using the Python environment you will train with:

```powershell
.venv\Scripts\python.exe setup_rx6600.py --apply
.venv\Scripts\python.exe setup_rx6600.py
```

For an external environment, use `& $env:RX6600_PYTHON setup_rx6600.py --apply`.
The helper checks the torch distribution without importing it, backs up the
original as `cublas64_11.dll.nv_backup`, and verifies the copy by SHA-256. Repeating
it preserves the backup. Without `--apply` it only checks and returns nonzero
when the replacement is missing. It refuses to overwrite a conflicting backup
after a reinstall; preserve and resolve that backup manually first.

Close Python processes before changing DLLs. To undo the replacement, copy the
preserved `.nv_backup` over `cublas64_11.dll` in the same environment. Re-check
the replacement after reinstalling torch. This fix covers the dense operations
used by this trainer; it is not a claim of general PyTorch compatibility.

## 4. Verify GPU computation, then train

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\zluda_run_probe.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\zluda_run_train_smoke.ps1
```

The first run may take time while ZLUDA compiles kernels. The probe reports the
device and checks fp32 matmul and gradients against CPU references. The training
smoke checks batched matmul, biased linear outputs and gradients, math attention,
and five production GPT/AdamW steps with finite values and updated parameters.
Expect `ALL GPU TESTS PASSED` and `ALL TRAINING TESTS PASSED`, respectively.
Both the Python scripts and launchers return nonzero on failure.

Place `TinyStories.txt` in the repository, then start with a small workload:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\zluda_run_train.ps1 --steps 10 --batch-size 4 --eval-iters 2
```

GPU training is the default and does not fall back silently to CPU. Startup
verifies numerical GPU work before reading the corpus. If it fails, inspect the
reported missing file, wheel version, or numerical error and revisit steps 1-3.
For a direct cuBLAS diagnostic, run `scripts\zluda_run_cublas_ctypes.ps1`.

## Limits and development checks

- Use fp32 eager execution. cuDNN, fused attention backends, TF32, fused AdamW
  and foreach AdamW are disabled. Both BLASLt preference variables are cleared
  before importing torch, and `DISABLE_ADDMM_CUDA_LT=1` selects legacy cuBLAS.
- Half precision, compilation, other GPU architectures and stack upgrades are
  unvalidated. Reduce batch/block/model sizes if you run out of 8 GiB VRAM.
  Sampling disables autograd to avoid retaining its generation graph.
- Throughput depends on the community rocBLAS kernels. The original setup
  reported about 0.5 TFLOPS for a 4096-cubed fp32 GEMM; no new performance claim
  is made by the portable support changes.
- `rocblas_direct_test*.py` are historical experiments, not supported verification
  entry points. Use the launchers above.

For CPU development, explicitly run `python train_tinystories.py --device cpu`.
The tests use Python 3.12, torch 2.4.1+cpu and NumPy 1.26.4:

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
powershell -NoProfile -ExecutionPolicy Bypass -File tests\test_launchers.ps1
pwsh -NoProfile -File tests\test_launchers.ps1
```

CI runs these tests on Windows. They cover setup backup behavior, backend and
failure handling, CPU training through checkpoint creation, and native launcher
arguments, paths and exit codes. **They do not validate RX 6600 hardware.** Run
both GPU probes on the target card and record GPU, driver and stack versions
before treating a revision as hardware validated.
