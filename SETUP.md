# Setup: PyTorch CUDA wheels on AMD RX 6600 via ZLUDA (native Windows)

This documents the working setup. **The critical fix is #4** (the cuBLAS DLL swap) — without it,
`torch.cuda.is_available()` returns True but *every* matmul fails with
`CUBLAS_STATUS_NOT_SUPPORTED`.

## 1. Install ZLUDA (rocm6 build)

Download and extract to `C:\Users\HTD\zluda`:

```
https://github.com/lshqqytiger/ZLUDA/releases/download/rel.5e717459179dc272b7d7d23391f0fad66c7459cf/ZLUDA-windows-rocm6-amd64.zip
```

The folder contains `zluda.exe`, `cublas.dll`, `nvcuda.dll`, `zluda_redirect.dll`, etc.

## 2. Install ROCm 6.2 with patched gfx1032 rocBLAS

Install the ROCm 6.2 Windows SDK from AMD. Then replace the stock `rocblas.dll` and the
`rocblas\library` folder under `C:\Program Files\AMD\ROCm\6.2\bin` with the gfx1032 build that
contains kernels for the RX 6600:

```
https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/download/v0.6.2.4/rocm.gfx1032.for.hip.sdk.6.2.4.navi21.logic.7z
```

(Back up the stock files before replacing.)

## 3. Install Python + PyTorch CUDA 11.8 wheels

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## 4. THE cuBLAS FIX (resolves CUBLAS_STATUS_NOT_SUPPORTED)

PyTorch's cu118 wheel statically imports cuBLAS from `cublas64_11.dll`. That file, shipped inside
`torch\lib`, is NVIDIA's real cuBLAS. It loads against ZLUDA's fake driver but returns
`CUBLAS_STATUS_NOT_SUPPORTED` from every GEMM.

Fix: replace torch's `cublas64_11.dll` with ZLUDA's working implementation
(a copy of `C:\Users\HTD\zluda\cublas.dll`). Back up the original first:

```powershell
$lib = ".venv\Lib\site-packages\torch\lib"
Copy-Item "$lib\cublas64_11.dll" "$lib\cublas64_11.dll.nv_backup"
Copy-Item "C:\Users\HTD\zluda\cublas.dll" "$lib\cublas64_11.dll" -Force
```

ZLUDA's `cublas.dll` exports every symbol torch imports (including `cublasSgemmStridedBatched`,
`GemmEx`, `*_v2`, batched/async variants), and it depends only on `rocblas.dll` + `rocsolver.dll`,
which resolve via the PATH entries in the launcher.

### Verify
```powershell
powershell -ExecutionPolicy Bypass -File scripts\zluda_run_probe.ps1
```
Expect: `matmul ok`, `elementwise ok`, `autograd ok`, `ALL GPU TESTS PASSED`.

## 5. Run everything through zluda.exe

ZLUDA's CUDA driver is `nvcuda.dll` in the ZLUDA folder. Launchers in `scripts/` do three things:

1. Prepend `C:\Program Files\AMD\ROCm\6.2\bin` and `C:\Users\HTD\zluda` to `PATH`.
2. Set `DISABLE_ADDMM_CUDA_LT=1` (keeps matmul on the legacy cuBLAS path; cuBLASLt is not
   available on this stack).
3. Start `zluda.exe -- python <script>` and stream output live.

Example:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\zluda_run_train.ps1 --steps 100 --batch-size 32
```

## Known limitations

- fp32 eager mode only. Half-precision GEMM and cuBLASLt paths are unvalidated; keep
  `DISABLE_ADDMM_CUDA_LT=1` and `TORCH_BLAS_PREFER_CUBLASLT` unset.
- cuDNN is disabled (`torch.backends.cudnn.enabled = False`); SDP uses the math backend
  (flash/mem-efficient are disabled in code).
- GEMM throughput is currently far below hardware peak (~0.5 TFLOPS measured on 4096^3 fp32).
  Kernel tuning for gfx1032 is an open item.
- If you later reinstall/upgrade torch, you must re-apply the DLL swap from step 4.