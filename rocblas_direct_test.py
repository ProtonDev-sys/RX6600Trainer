import ctypes, ctypes.util, os, sys, struct

OUT = r"C:\Users\HTD\AppData\Local\Temp\opencode\rocblas_direct_log.txt"
def log(m):
    with open(OUT, "a") as f:
        f.write(m + "\n"); f.flush()
    print(m, flush=True)

log("=== rocblas direct test start ===")

try:
    rb = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.2\bin\rocblas.dll")
    log("loaded rocblas.dll ok")
except Exception as e:
    log("LOAD FAILED: " + repr(e))
    sys.exit(1)

def check(name, status):
    # status codes: rocblas_status_success=0, ... not_supported = 8
    log(f"{name} -> status {status}")

rb.rocblas_create_handle.restype = ctypes.c_size_t
rb.rocblas_create_handle.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
handle = ctypes.c_size_t(0)
s = rb.rocblas_create_handle(ctypes.byref(handle))
check("rocblas_create_handle", s)
if s != 0:
    log("could not create handle, exiting")
    sys.exit(1)

n = 4
A = (ctypes.c_float * (n*n))(*([1.0]*(n*n)))
B = (ctypes.c_float * (n*n))(*([2.0]*(n*n)))
C = (ctypes.c_float * (n*n))(*([0.0]*(n*n)))
alpha = ctypes.c_float(1.0)
beta = ctypes.c_float(0.0)

rb.rocblas_sgemm.restype = ctypes.c_int
rb.rocblas_sgemm.argtypes = [
    ctypes.c_size_t,        # handle
    ctypes.c_int, ctypes.c_int,  # opA, opB
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # m n k
    ctypes.POINTER(ctypes.c_float),  # alpha
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # A, lda
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # B, ldb
    ctypes.POINTER(ctypes.c_float),  # beta
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # C, ldc
]
s = rb.rocblas_sgemm(
    handle.value, 0, 0,
    n, n, n,
    ctypes.byref(alpha),
    A, n,
    B, n,
    ctypes.byref(beta),
    C, n,
)
check("rocblas_sgemm 4x4", s)

rb.rocblas_destroy_handle.restype = ctypes.c_int
rb.rocblas_destroy_handle.argtypes = [ctypes.c_size_t]
rb.rocblas_destroy_handle(handle.value)

if s == 0:
    log("C[0][0] = " + str(C[0]))
    log("ROCBLAS DIRECT TEST PASSED")
else:
    log("ROCBLAS DIRECT TEST FAILED status=" + str(s))
log("=== rocblas direct test done ===")