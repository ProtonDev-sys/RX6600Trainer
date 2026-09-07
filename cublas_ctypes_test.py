import os
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

LOG = r"C:\Users\HTD\AppData\Local\Temp\opencode\cublas_ctypes_log.txt"

def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
        f.flush()
    print(msg, flush=True)

import sys
import ctypes
import traceback

log("=== cublas ctypes test start ===")

try:
    import torch
except Exception as e:
    log("torch import failed: " + repr(e))
    sys.exit(1)

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

try:
    log("cuda available: " + str(torch.cuda.is_available()))
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    c = torch.zeros(2048, 2048, device="cuda", dtype=torch.float32)
    log("torch cuda alloc ok, a_ptr=%#x b_ptr=%#x c_ptr=%#x" % (a.data_ptr(), b.data_ptr(), c.data_ptr()))
except Exception as e:
    log("ALLOC FAILED: " + repr(e)); log(traceback.format_exc()); sys.exit(1)

log("PATH in child: " + os.environ.get("PATH", "<none>"))

def tryload(path, mode=0):
    try:
        d = ctypes.WinDLL(path, mode=mode)
        log("LOADED: " + path + ("" if mode == 0 else " (mode=0x8)"))
        return d
    except Exception as e:
        log("FAILED %s: %s" % (path, repr(e)))
        return None

for dep in [r"C:\Program Files\AMD\ROCm\6.2\bin\amdhip64_6.dll",
            r"C:\Program Files\AMD\ROCm\6.2\bin\rocblas.dll",
            r"C:\Program Files\AMD\ROCm\6.2\bin\rocsolver.dll"]:
    if tryload(dep) is None:
        if tryload(dep, 0x00000008) is None:
            log("cannot load dep: " + dep)
            sys.exit(1)

cublas = None
for mode in [0, 0x00000008]:
    cublas = tryload(r"C:\Users\HTD\zluda\cublas.dll", mode)
    if cublas is not None:
        break
if cublas is None:
    sys.exit(1)

for name in ["cublasCreate_v2", "cublasSgemm_v2", "cublasSetStream_v2", "cublasGetVersion_v2", "cublasSgemmStridedBatched_v2", "cublasGetStream_v2", "cublasSetPointerMode_v2"]:
    try:
        getattr(cublas, name)
        log("symbol %s exported" % name)
    except AttributeError:
        log("symbol %s MISSING" % name)

cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
cublas.cublasCreate_v2.restype = ctypes.c_int
cublas.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
cublas.cublasSetStream_v2.restype = ctypes.c_int
cublas.cublasGetStream_v2.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
cublas.cublasGetStream_v2.restype = ctypes.c_int
cublas.cublasGetVersion_v2.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
cublas.cublasGetVersion_v2.restype = ctypes.c_int
cublas.cublasSgemm_v2.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int,
]
cublas.cublasSgemm_v2.restype = ctypes.c_int

try:
    log("torch a@b (torch internal cublas) BEFORE manual handle...")
    c2 = a @ b
    torch.cuda.synchronize()
    log("torch a@b OK, sum: " + str(c2.sum().item()))
except Exception as e:
    log("TORCH a@b FAILED: " + repr(e)); log(traceback.format_exc())

handle = ctypes.c_void_p()
st = cublas.cublasCreate_v2(ctypes.byref(handle))
log("cublasCreate_v2 = %d (%s), handle=%d" % (st, STATUS.get(st, "?"), handle.value or 0))
if st != 0:
    sys.exit(1)

ver = ctypes.c_int(0)
st = cublas.cublasGetVersion_v2(handle, ctypes.byref(ver))
log("cublasGetVersion_v2 = %d (%s), version=%d" % (st, STATUS.get(st, "?"), ver.value))

try:
    cur_stream = torch.cuda.current_stream(0).cuda_stream
    log("torch current stream id: %d" % cur_stream)
except Exception as e:
    cur_stream = 0
    log("current_stream query failed: " + repr(e))

st = cublas.cublasSetStream_v2(handle, ctypes.c_void_p(cur_stream))
log("cublasSetStream_v2 = %d (%s)" % (st, STATUS.get(st, "?")))

got_stream = ctypes.c_void_p()
st = cublas.cublasGetStream_v2(handle, ctypes.byref(got_stream))
log("cublasGetStream_v2 = %d (%s), stream=%d" % (st, STATUS.get(st, "?"), got_stream.value or 0))

alpha = ctypes.c_float(1.0)
beta = ctypes.c_float(0.0)
m = n = k = 2048

try:
    log("calling cublasSgemm_v2 directly (2048x2048x2048, with set stream)...")
    st = cublas.cublasSgemm_v2(
        handle, 0, 0, m, n, k,
        ctypes.byref(alpha), ctypes.c_void_p(a.data_ptr()), 2048,
        ctypes.c_void_p(b.data_ptr()), 2048,
        ctypes.byref(beta), ctypes.c_void_p(c.data_ptr()), 2048,
    )
    log("cublasSgemm_v2 = %d (%s)" % (st, STATUS.get(st, "?")))
    torch.cuda.synchronize()
    log("c sum (should be ~0 if gemm ran): " + str(c.sum().item()))
except Exception as e:
    log("SGEM DIRECT RAISED: " + repr(e)); log(traceback.format_exc())

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
kernel32.GetModuleHandleW.restype = ctypes.c_void_p
kernel32.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
kernel32.GetProcAddress.restype = ctypes.c_void_p

zluda_cublas_h = cublas._handle
log("ctypes ZLUDA cublas.dll module base: %d" % zluda_cublas_h)
for name in ["cublas.dll", "cublas64_11.dll", "cublasLt64_11.dll", "cublasLt.dll", "cudart64_11.dll", "rocblas.dll"]:
    h = kernel32.GetModuleHandleW(name)
    pa = kernel32.GetProcAddress(h, b"cublasSgemm_v2") if h else None
    log("module %-20s handle=%d  sgemm_v2 addr=%s" % (name, h or 0, hex(pa) if pa else "n/a"))

log("=== cublas ctypes test done ===")