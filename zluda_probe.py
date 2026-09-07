import os
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

LOG = r"C:\Users\HTD\AppData\Local\Temp\opencode\probe_log.txt"

def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
        f.flush()
    print(msg, flush=True)

import sys
log("=== probe start ===")
log("python: " + sys.version.split("(")[0].strip())
log("argv trace: cwd=" + os.getcwd())

try:
    import torch
    log("torch imported: " + torch.__version__)
except Exception as e:
    import traceback
    log("IMPORT FAILED: " + repr(e))
    log(traceback.format_exc())
    sys.exit(1)

log("cuda available check...")
try:
    avail = torch.cuda.is_available()
    log("torch.cuda.is_available() = " + str(avail))
except Exception as e:
    import traceback
    log("IS_AVAILABLE RAISED: " + repr(e))
    log(traceback.format_exc())
    sys.exit(1)

if not avail:
    log("no CUDA - exiting probe")
    sys.exit(0)

try:
    log("device count: " + str(torch.cuda.device_count()))
    log("device name: " + str(torch.cuda.get_device_name(0)).encode("ascii", "replace").decode())
    log("capability: " + str(torch.cuda.get_device_capability(0)))
    props = torch.cuda.get_device_properties(0)
    log("total mem MB: " + str(round(props.total_memory / 1024 / 1024)))
except Exception as e:
    import traceback
    log("DEVICE QUERY RAISED: " + repr(e))
    log(traceback.format_exc())
    sys.exit(1)

torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)

try:
    log("allocating 2048x2048...")
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    log("allocated, matmul...")
    c = a @ b
    torch.cuda.synchronize()
    log("matmul ok, sum: " + str(c.sum().item()))
except Exception as e:
    import traceback
    log("MATMUL RAISED: " + repr(e))
    log(traceback.format_exc())
    sys.exit(1)

try:
    x = torch.randn(4096, device="cuda")
    y = torch.randn(4096, device="cuda")
    log("elementwise ok: " + str((x + y).sum().item()))
except Exception as e:
    import traceback
    log("ELEMENTWISE RAISED: " + repr(e))
    log(traceback.format_exc())
    sys.exit(1)

try:
    grad = torch.randn(512, 512, device="cuda", requires_grad=True)
    loss = (grad * grad).sum()
    loss.backward()
    log("autograd ok, grad sample: " + str(grad.grad.reshape(-1)[:3].tolist()))
except Exception as e:
    import traceback
    log("AUTOGRAD RAISED: " + repr(e))
    log(traceback.format_exc())
    sys.exit(1)

log("ALL GPU TESTS PASSED")
log("=== probe done ===")