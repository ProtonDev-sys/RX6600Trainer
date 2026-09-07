# RX6600Trainer

From-scratch GPT-style language model training on an AMD RX 6600 (gfx1032, 8 GB) on native Windows,
running PyTorch CUDA 11.8 wheels through [ZLUDA](https://github.com/lshqqytiger/ZLUDA).

All core ops (matmul, autograd, bmm, linear, layernorm, embedding, attention, cross-entropy,
AdamW train steps) are verified working on GPU in this setup.

## Contents

- `train_tinystories.py` – from-scratch GPT trainer (char-level tokenizer, causal attention,
  AdamW + warmup/cosine LR, eval loss, text samples, checkpoints).
- `training_smoke.py` – smoke test of training-shaped ops on GPU.
- `smoke_test.py`, `zluda_probe.py` – minimal GPU sanity checks.
- `cublas_ctypes_test.py` – direct ctypes test against ZLUDA's `cublas.dll`.
- `scripts/` – PowerShell launchers that start Python under `zluda.exe` with the required env.
- `SETUP.md` – the exact environment setup and the cuBLAS fix that makes torch work.

## Quick start

1. Follow `SETUP.md` to prepare the environment (ZLUDA, patched rocBLAS, DLL swap).
2. Place `TinyStories.txt` (or any text corpus) in this folder.
3. Run via the launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\zluda_run_train.ps1 --steps 100 --batch-size 32
```

Live progress goes to the console and `train_log.txt`.

## Tuning knobs

`--n-layer` `--n-head` `--n-embd` `--block-size` `--batch-size` `--lr` `--steps`
`--warmup-steps` `--eval-iters` `--log-every`

Example (bigger model, longer run):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\zluda_run_train.ps1 --n-layer 8 --n-head 8 --n-embd 512 --block-size 512 --batch-size 16 --steps 1000
```

## Notes

- fp32 eager mode only. fp16/bf16 and cuBLASLt paths are NOT validated under this setup;
  keep `DISABLE_ADDMM_CUDA_LT=1` and math SDP.
- Peak synthetic GEMM throughput measured so far is unexpectedly low (~0.5 TFLOPS on a 4096^3
  matmul); performance tuning of the patched rocBLAS kernels is an open item.