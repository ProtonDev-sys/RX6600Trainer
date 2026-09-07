import ctypes, sys

OUT = r"C:\Users\HTD\AppData\Local\Temp\opencode\rocblas_direct_log.txt"
def log(m):
    with open(OUT, "a") as f:
        f.write(m + "\n"); f.flush()
    print(m, flush=True)

rb = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.2\bin\rocblas.dll")

# rocblas_status_to_string
rb.rocblas_status_to_string.restype = ctypes.c_char_p
rb.rocblas_status_to_string.argtypes = [ctypes.c_int]

buf = ctypes.create_string_buffer(64)
rb.rocblas_get_version_string_size.restype = ctypes.c_size_t
rb.rocblas_get_version_string_size.argtypes = [ctypes.c_char_p]
sz = rb.rocblas_get_version_string_size(buf)
log("version string size: " + str(sz))

rb.rocblas_get_version_string.restype = ctypes.c_int
rb.rocblas_get_version_string.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
s = rb.rocblas_get_version_string(buf, len(buf))
log("version string: " + buf.value.decode('latin1'))

handle = ctypes.c_void_p(0)
rb.rocblas_create_handle.restype = ctypes.c_int
rb.rocblas_create_handle.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
s = rb.rocblas_create_handle(ctypes.byref(handle))
log("create_handle status: " + str(s) + " = " + (rb.rocblas_status_to_string(s) or b'').decode('latin1'))
if s != 0:
    sys.exit(1)

n = 4
A = (ctypes.c_float * (n*n))(*([1.0]*(n*n)))
B = (ctypes.c_float * (n*n))(*([2.0]*(n*n)))
C = (ctypes.c_float * (n*n))(*([0.0]*(n*n)))
alpha = ctypes.c_float(1.0)
beta = ctypes.c_float(0.0)

rb.rocblas_sgemm.restype = ctypes.c_int
rb.rocblas_sgemm.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
]
s = rb.rocblas_sgemm(
    handle, 0, 0, n, n, n,
    ctypes.byref(alpha), A, n, B, n,
    ctypes.byref(beta), C, n,
)
log("sgemm status: " + str(s) + " = " + (rb.rocblas_status_to_string(s) or b'').decode('latin1'))
if s == 0:
    # needs a device C-value read... can't easily; just log success
    log("ROCBLAS DIRECT GEMM CALL SUCCEEDED (status 0)")
else:
    log("ROCBLAS DIRECT GEMM FAILED")
log("done")