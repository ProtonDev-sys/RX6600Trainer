"""Compare torch and direct ZLUDA cuBLAS GEMM against a CPU reference."""

import ctypes
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import traceback

from rx6600_runtime import require_gpu, torch


STATUS = {
    0: "SUCCESS",
    1: "NOT_INITIALIZED",
    3: "ALLOC_FAILED",
    7: "INVALID_VALUE",
    8: "ARCH_MISMATCH",
    11: "MAPPING_ERROR",
    13: "EXECUTION_FAILED",
    14: "INTERNAL_ERROR",
    15: "NOT_SUPPORTED",
    16: "LICENSE_ERROR",
}


def check_status(name, status):
    if status != 0:
        raise RuntimeError(f"{name}: {STATUS.get(status, 'UNKNOWN')} ({status})")


def check_cublas(device):
    zluda_dir = Path(os.environ.get("ZLUDA_PATH", str(Path.home() / "zluda")))
    hip_dir = Path(os.environ.get("HIP_PATH", r"C:\Program Files\AMD\ROCm\6.2"))
    with ExitStack() as stack:
        for directory in (hip_dir / "bin", zluda_dir):
            stack.enter_context(os.add_dll_directory(str(directory.resolve())))
        cublas = ctypes.WinDLL(str((zluda_dir / "cublas.dll").resolve()))
        handle_type = ctypes.c_void_p
        cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(handle_type)]
        cublas.cublasCreate_v2.restype = ctypes.c_int
        cublas.cublasDestroy_v2.argtypes = [handle_type]
        cublas.cublasDestroy_v2.restype = ctypes.c_int
        cublas.cublasSetStream_v2.argtypes = [handle_type, ctypes.c_void_p]
        cublas.cublasSetStream_v2.restype = ctypes.c_int
        cublas.cublasSgemm_v2.argtypes = [
            handle_type, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int,
        ]
        cublas.cublasSgemm_v2.restype = ctypes.c_int

        a_cpu = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16) / 128
        b_cpu = torch.arange(16 * 12, dtype=torch.float32).reshape(16, 12) / 192
        expected = a_cpu @ b_cpu
        a, b = a_cpu.to(device), b_cpu.to(device)
        c = torch.empty((8, 12), device=device, dtype=torch.float32)
        torch.testing.assert_close((a @ b).cpu(), expected, rtol=1e-4, atol=1e-5)
        print("PASS: torch GEMM agrees with CPU", flush=True)

        handle = handle_type()
        check_status("cublasCreate_v2", cublas.cublasCreate_v2(ctypes.byref(handle)))
        try:
            stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
            check_status("cublasSetStream_v2", cublas.cublasSetStream_v2(handle, stream))
            alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
            # cuBLAS is column-major: for row-major tensors compute C^T = B^T A^T.
            status = cublas.cublasSgemm_v2(
                handle, 0, 0, 12, 8, 16,
                ctypes.byref(alpha), ctypes.c_void_p(b.data_ptr()), 12,
                ctypes.c_void_p(a.data_ptr()), 16,
                ctypes.byref(beta), ctypes.c_void_p(c.data_ptr()), 12,
            )
            check_status("cublasSgemm_v2", status)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(c.cpu(), expected, rtol=1e-4, atol=1e-5)
            print("PASS: direct ZLUDA cuBLAS GEMM agrees with CPU", flush=True)
        finally:
            check_status("cublasDestroy_v2", cublas.cublasDestroy_v2(handle))


def main():
    try:
        device = require_gpu()
        check_cublas(device)
        return 0
    except Exception:
        traceback.print_exc(file=sys.stdout)
        return 1


if __name__ == "__main__":
    sys.exit(main())
