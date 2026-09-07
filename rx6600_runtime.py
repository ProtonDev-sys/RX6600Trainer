"""Conservative fp32 runtime for the pinned Windows RX 6600/ZLUDA stack."""

import os

# These are read during torch/CUDA initialization, so set them before import.
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ.pop("TORCH_BLAS_PREFER_CUBLASLT", None)
os.environ.pop("TORCH_BLAS_PREFER_HIPBLASLT", None)

import torch


def configure_torch():
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def require_gpu():
    if str(torch.__version__).split("+")[0] != "2.4.1" or torch.version.cuda != "11.8":
        raise RuntimeError(
            f"RX 6600/ZLUDA requires torch 2.4.1+cu118; found {torch.__version__} "
            f"(CUDA {torch.version.cuda}). Follow SETUP.md in this Python environment."
        )
    configure_torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU is available through ZLUDA. Run scripts/zluda_run_probe.ps1 "
            "and check the driver, HIP_PATH and ZLUDA_PATH in SETUP.md. "
            "Use --device cpu only for an explicit CPU development run."
        )
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}, {props.total_memory / 1024**3:.1f} GiB; "
          f"torch {torch.__version__}, fp32 eager", flush=True)
    return device


def verify_gpu():
    """Check numerical results and backward; device discovery alone is insufficient."""
    a = (torch.arange(48, dtype=torch.float32).reshape(8, 6) / 48).requires_grad_()
    b = (torch.arange(24, dtype=torch.float32).reshape(6, 4) / 24).requires_grad_()
    ga = a.detach().to("cuda").requires_grad_()
    gb = b.detach().to("cuda").requires_grad_()
    try:
        expected = a @ b
        actual = ga @ gb
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.cuda.synchronize()
        for result, reference in ((actual, expected), (ga.grad, a.grad), (gb.grad, b.grad)):
            torch.testing.assert_close(result.cpu(), reference, rtol=1e-4, atol=1e-5)
    except Exception as exc:
        raise RuntimeError(
            "GPU matmul/backward verification failed. Check the gfx1032 rocBLAS "
            "files and run setup_rx6600.py --apply after installing torch; see SETUP.md."
        ) from exc
    print("matmul ok; autograd ok (GPU values match CPU reference)", flush=True)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
