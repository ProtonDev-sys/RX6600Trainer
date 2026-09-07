import os
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

import torch
import torch.nn as nn

torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)

out = []

def check(name, fn):
    try:
        v = fn()
        out.append("PASS: " + name + ((" -> " + str(v)) if v is not None else ""))
    except Exception as e:
        import traceback
        out.append("FAIL: " + name + ": " + repr(e))
        out.append(traceback.format_exc())

dtype = torch.float32
dev = "cuda"

def bmm_test():
    a = torch.randn(32, 64, 128, device=dev, dtype=dtype)
    b = torch.randn(32, 128, 96, device=dev, dtype=dtype)
    c = torch.bmm(a, b)
    torch.cuda.synchronize()
    return round(c.sum().item(), 4)

def linear_test():
    m = nn.Linear(256, 512).to(dev)
    x = torch.randn(16, 128, 256, device=dev, dtype=dtype)
    y = m(x)
    torch.cuda.synchronize()
    return str(y.shape)

def layernorm_test():
    ln = nn.LayerNorm(256).to(dev)
    x = torch.randn(16, 128, 256, device=dev, dtype=dtype)
    y = ln(x)
    torch.cuda.synchronize()
    return round(y.mean().item(), 4)

def embed_test():
    emb = nn.Embedding(1000, 128).to(dev)
    idx = torch.randint(0, 1000, (16, 64), device=dev)
    y = emb(idx)
    torch.cuda.synchronize()
    return str(y.shape)

def attn_test():
    q = torch.randn(4, 8, 64, 64, device=dev, dtype=dtype)
    k = torch.randn(4, 8, 64, 64, device=dev, dtype=dtype)
    v = torch.randn(4, 8, 64, 64, device=dev, dtype=dtype)
    att = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    return str(att.shape)

def softmax_ce_test():
    logits = torch.randn(16, 1000, device=dev, dtype=dtype)
    targets = torch.randint(0, 1000, (16,), device=dev)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    torch.cuda.synchronize()
    return round(loss.item(), 4)

def train_step_test():
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(1000, 64)
            self.attn = nn.MultiheadAttention(64, 4, batch_first=True, dropout=0.0)
            self.ln1 = nn.LayerNorm(64)
            self.ff = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 64))
            self.ln2 = nn.LayerNorm(64)
            self.out = nn.Linear(64, 1000)
        def forward(self, x):
            x = self.emb(x)
            a, _ = self.attn(x, x, x)
            x = self.ln1(x + a)
            x = self.ln2(x + self.ff(x))
            return self.out(x)

    torch.manual_seed(0)
    model = TinyTransformer().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for i in range(5):
        idx = torch.randint(0, 1000, (8, 32), device=dev)
        target = torch.randint(0, 1000, (8, 32), device=dev)
        logits = model(idx)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        losses.append(round(loss.item(), 4))
    return str(losses)

check("bmm", bmm_test)
check("linear", linear_test)
check("layernorm", layernorm_test)
check("embedding", embed_test)
check("attention", attn_test)
check("softmax+cross_entropy", softmax_ce_test)
check("5 train steps (mini transformer, AdamW)", train_step_test)

with open(r"C:\Users\HTD\AppData\Local\Temp\opencode\train_smoke_result.txt", "w") as f:
    f.write("\n".join(out))
print("\n".join(out))