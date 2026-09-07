import os
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

import torch

lines = []
lines.append("torch version: " + torch.__version__)
lines.append("cuda available: " + str(torch.cuda.is_available()))

if torch.cuda.is_available():
    lines.append("device: " + torch.cuda.get_device_name(0))
    lines.append("capability: " + str(torch.cuda.get_device_capability(0)))
    props = torch.cuda.get_device_properties(0)
    lines.append("total memory MB: " + str(round(props.total_memory / 1024 / 1024)))

torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)

ok = True
if torch.cuda.is_available():
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
        c = a @ b
        torch.cuda.synchronize()
        lines.append("matmul ok, sum: " + str(c.sum().item()))
        x = torch.randn(4096, device="cuda")
        y = torch.randn(4096, device="cuda")
        lines.append("elementwise ok: " + str((x + y).sum().item()))
        grad = torch.randn(512, 512, device="cuda", requires_grad=True)
        loss = (grad * grad).sum()
        loss.backward()
        lines.append("autograd ok, grad sample: " + str(grad.grad.reshape(-1)[:3].tolist()))
        lines.append("ALL GPU TESTS PASSED")
    except Exception as e:
        import traceback
        lines.append("GPU test FAILED: " + repr(e))
        lines.append(traceback.format_exc())
        ok = False

with open(r"C:\Users\HTD\AppData\Local\Temp\opencode\smoke_result.txt", "w") as f:
    f.write("\n".join(lines))

print("wrote result, ok=", ok)