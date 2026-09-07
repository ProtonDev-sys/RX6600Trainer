import os
import sys
import json
import math
import time
import argparse
import pathlib

os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled = False

DATA_FILE = "TinyStories.txt"
SENTINEL = "\x00"
SENTINEL_HTML = "<|endoftext|>"
CACHE_DIR = pathlib.Path("cache")


def build_vocab(text):
    stoi = {SENTINEL: 0}
    for ch in text:
        if ch not in stoi:
            stoi[ch] = len(stoi)
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def tokenize_file(path):
    print("tokenizing TinyStories (char vocab + encode)...", flush=True)
    raw = pathlib.Path(path).read_text(encoding="utf-8").replace(SENTINEL_HTML, SENTINEL)
    stoi, itos = build_vocab(raw)
    table = {ord(c): chr(i) for c, i in stoi.items()}
    buf = raw.translate(table).encode("utf-16-be")
    ids = np.frombuffer(buf, dtype=">u2").astype(np.uint16)
    return ids, stoi, itos


def load_data():
    if CACHE_DIR.exists() and (CACHE_DIR / "train.bin").exists() and (CACHE_DIR / "meta.json").exists():
        train = np.fromfile(CACHE_DIR / "train.bin", dtype=np.uint16)
        val = np.fromfile(CACHE_DIR / "val.bin", dtype=np.uint16)
        meta = json.loads((CACHE_DIR / "meta.json").read_text())
        itos = {int(k): v for k, v in meta["itos"].items()}
        return train, val, meta["vocab_size"], itos
    t0 = time.time()
    train, stoi, itos = tokenize_file(DATA_FILE)
    n = len(train)
    split = int(n * 0.95)
    val_arr = train[split:]
    train = train[:split]
    CACHE_DIR.mkdir(exist_ok=True)
    train.tofile(CACHE_DIR / "train.bin")
    val_arr.tofile(CACHE_DIR / "val.bin")
    meta = {"vocab_size": len(stoi), "itos": {str(i): c for c, i in stoi.items()}}
    (CACHE_DIR / "meta.json").write_text(json.dumps(meta))
    print(f"tokenized {n} chars in {time.time() - t0:.1f}s, vocab_size={len(stoi)}", flush=True)
    return train, val_arr, len(stoi), itos


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.n_head)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, itos, device, out):
    model.eval()
    res = {}
    for name in ["train", "val"]:
        data = train_data if name == "train" else val_data
        losses = []
        for _ in range(eval_iters):
            ix = torch.randint(len(data) - block_size - 1, (batch_size,))
            bx = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
            by = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
            bx, by = bx.to(device), by.to(device)
            _, loss = model(bx, by)
            losses.append(loss.item())
        res[name] = float(np.mean(losses))
    model.train()
    text = f"step {out['step']}: train loss {res['train']:.4f}, val loss {res['val']:.4f}"
    print(text, flush=True)
    with open(out["log"], "a", buffering=1) as f:
        f.write("EVAL " + text + "\n")


def sample_text(model, seed, itos, n_tokens, device, block_size):
    model.eval()
    idx = torch.tensor([[seed]], dtype=torch.long).to(device)
    idx = model.generate(idx, n_tokens)
    chars = []
    for t in idx[0].tolist():
        c = itos[t]
        chars.append(SENTINEL_HTML if c == SENTINEL else c)
    model.train()
    return "".join(chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--eval-iters", type=int, default=30)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    train_data, val_data, vocab_size, itos = load_data()
    print(f"device {device}, train {len(train_data)} chars, val {len(val_data)} chars", flush=True)

    model = GPT(vocab_size, args.n_layer, args.n_head, args.n_embd, args.block_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M", flush=True)

    if device == "cuda":
        gpu_diag = torch.randn(4096, 4096, device="cuda")
        t_b = time.time()
        gpu_diag = gpu_diag @ gpu_diag
        torch.cuda.synchronize()
        gpu_ms = (time.time() - t_b) * 1000
        gflops = 2 * 4096**3 / 1e9
        print(f"synthetic 4096^3 fp32 GEMM on GPU: {gpu_ms:.0f} ms = {gflops / (gpu_ms/1000) / 1000:.2f} TFLOPS", flush=True)
        del gpu_diag
        torch.cuda.empty_cache()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        prog = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    log_file = "train_log.txt"
    eval_log_interval = max(1, args.log_every)
    start = time.time()
    t0 = start
    tokens_so_far = 0
    model.train()
    for step in range(1, args.steps + 1):
        ix = torch.randint(len(train_data) - args.block_size - 1, (args.batch_size,))
        bx = torch.stack([torch.from_numpy(train_data[i : i + args.block_size].astype(np.int64)) for i in ix])
        by = torch.stack([torch.from_numpy(train_data[i + 1 : i + 1 + args.block_size].astype(np.int64)) for i in ix])
        bx, by = bx.to(device), by.to(device)
        t_fwd = time.time()
        _, loss = model(bx, by)
        torch.cuda.synchronize()
        t_fwd = time.time() - t_fwd
        opt.zero_grad(set_to_none=True)
        t_bwd = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.time() - t_bwd
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        t_opt = time.time()
        opt.step()
        torch.cuda.synchronize()
        t_opt = time.time() - t_opt
        scheduler.step()

        tokens_so_far += args.batch_size * args.block_size
        if step % eval_log_interval == 0 or step == args.steps:
            mem = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0.0
            line = f"step {step}/{args.steps} loss {loss.item():.4f} lr {lr_at(step):.2e} fwd {t_fwd*1000:.0f}ms bwd {t_bwd*1000:.0f}ms opt {t_opt*1000:.0f}ms mem {mem:.2f}GB"
            print(line, flush=True)
            with open(log_file, "a", buffering=1) as f:
                f.write(line + "\n")

    estimate_loss(model, train_data, val_data, args.batch_size, args.block_size, args.eval_iters, itos, device, {"step": args.steps, "log": log_file})
    for seed in (itos.get(SENTINEL, 0), 65):
        text = sample_text(model, seed, itos, 120, device, args.block_size)
        text = text.replace(SENTINEL_HTML, "\n<|endoftext|>\n")
        print("--- sample ---\n" + text + "\n-------------", flush=True)

    ckpt_dir = pathlib.Path("train_ckpt")
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "itos": itos, "config": vars(args)}, ckpt_dir / f"ckpt_step{args.steps}.pt")
    print(f"checkpoint saved to {ckpt_dir / f'ckpt_step{args.steps}.pt'}", flush=True)
    print(f"DONE in {time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()