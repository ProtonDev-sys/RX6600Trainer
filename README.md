# RX6600Trainer

From-scratch GPT-style language model training on an AMD RX 6600 (gfx1032, 8 GB) on native Windows,
running PyTorch CUDA 11.8 wheels through [ZLUDA](https://github.com/lshqqytiger/ZLUDA).

This is an experimental, pinned setup that requires community gfx1032 rocBLAS kernels.
The original environment reported working GPU training; run the numerical probes on your
installation before training. The portable support changes require RX 6600 hardware validation.

## Contents

- `train_tinystories.py` – from-scratch GPT trainer (char-level tokenizer, causal attention,
  AdamW + warmup/cosine LR, eval loss, text samples, checkpoints).
- `training_smoke.py` – smoke test of training-shaped ops on GPU.
- `smoke_test.py`, `zluda_probe.py` – numerical GPU matmul/backward checks against CPU references.
- `setup_rx6600.py` – checks or applies the cuBLAS DLL fix while preserving the original.
- `cublas_ctypes_test.py` – direct ctypes test against ZLUDA's `cublas.dll`.
- `scripts/` – PowerShell launchers that start Python under `zluda.exe` with the required env.
- `SETUP.md` – the exact environment setup and the cuBLAS fix that makes torch work.

## Quick start

1. Follow `SETUP.md` to prepare the environment (ZLUDA, patched rocBLAS, DLL swap).
2. Place `TinyStories.txt` (or any text corpus) in this folder.
3. Verify the GPU and run via the launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\zluda_run_probe.ps1
powershell -ExecutionPolicy Bypass -File scripts\zluda_run_train_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\zluda_run_train.ps1 --steps 10 --batch-size 4 --eval-iters 2
```

Live progress goes to the console and `train_log.txt`.
Launchers use the cloned repository and its `.venv`, with optional `ZLUDA_PATH`,
`HIP_PATH` and `RX6600_PYTHON` overrides. Failures return nonzero; training requires
a working GPU unless you explicitly run Python with `--device cpu` for development.

## Tuning knobs

`--n-layer` `--n-head` `--n-embd` `--block-size` `--batch-size` `--lr` `--steps`
`--warmup-steps` `--eval-iters` `--log-every`

Example (bigger model, longer run):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\zluda_run_train.ps1 --n-layer 8 --n-head 8 --n-embd 512 --block-size 512 --batch-size 16 --steps 1000
```

## Notes

- fp32 eager mode only. fp16/bf16 and cuBLASLt paths are NOT validated under this setup;
  the runtime selects legacy cuBLAS and math SDP, with fused/foreach AdamW disabled.
- Peak synthetic GEMM throughput measured so far is unexpectedly low (~0.5 TFLOPS on a 4096^3
  matmul); performance tuning of the patched rocBLAS kernels is an open item.
