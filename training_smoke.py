"""Exercise the training operations and the production GPT on the GPU."""

from copy import deepcopy
import sys
import traceback

from rx6600_runtime import require_gpu, synchronize, torch, verify_gpu

from train_tinystories import GPT


def check_finite(value, name):
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"{name} contains non-finite values")


def check_ops(device):
    a_cpu = torch.arange(4 * 16 * 32, dtype=torch.float32).reshape(4, 16, 32) / 1024 - 1
    b_cpu = torch.arange(4 * 32 * 24, dtype=torch.float32).reshape(4, 32, 24) / 1536 - 1
    actual = torch.bmm(a_cpu.to(device), b_cpu.to(device))
    torch.testing.assert_close(actual.cpu(), torch.bmm(a_cpu, b_cpu), rtol=1e-4, atol=1e-5)
    print("PASS: bmm agrees with CPU", flush=True)

    # Bias exercises addmm, whose default cuBLASLt path is unavailable here.
    cpu_linear = torch.nn.Linear(32, 24)
    with torch.no_grad():
        cpu_linear.weight.copy_(torch.arange(24 * 32).reshape(24, 32) / 768 - 0.5)
        cpu_linear.bias.copy_(torch.arange(24) / 24 - 0.5)
    linear = deepcopy(cpu_linear).to(device)
    cpu_input = (torch.arange(2 * 8 * 32, dtype=torch.float32).reshape(2, 8, 32) / 256 - 1).requires_grad_()
    gpu_input = cpu_input.detach().to(device).clone().requires_grad_()
    expected = cpu_linear(cpu_input)
    actual = linear(gpu_input)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for result, reference in (
        (actual, expected),
        (gpu_input.grad, cpu_input.grad),
        (linear.weight.grad, cpu_linear.weight.grad),
        (linear.bias.grad, cpu_linear.bias.grad),
    ):
        torch.testing.assert_close(result.cpu(), reference, rtol=1e-4, atol=1e-5)
    print("PASS: biased linear output and gradients agree with CPU", flush=True)

    q, k, v = [
        torch.randn(2, 2, 16, 16, device=device, dtype=torch.float32)
        for _ in range(3)
    ]
    attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    check_finite(attention, "math scaled dot product attention")
    print("PASS: math scaled dot product attention", flush=True)


def check_train_steps(device):
    # Use the real trainer so this checks its embedding, linear layers, layernorm,
    # causal attention, cross entropy and backward path together.
    torch.manual_seed(0)
    model = GPT(128, n_layer=1, n_head=2, n_embd=32, block_size=16, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False, fused=False)
    indices = torch.randint(0, 128, (2, 16), device=device)
    targets = torch.randint(0, 128, (2, 16), device=device)
    initial_weight = model.token_embedding.weight.detach().clone()
    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(indices, targets)
        check_finite(logits, "GPT logits")
        check_finite(loss, "GPT loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"{name} did not receive a gradient")
            check_finite(parameter.grad, f"{name} gradient")
        optimizer.step()
        for name, parameter in model.named_parameters():
            check_finite(parameter, f"{name} after AdamW")
        losses.append(round(loss.item(), 4))
    synchronize(device)
    if torch.equal(initial_weight, model.token_embedding.weight):
        raise RuntimeError("AdamW did not update the model weights")
    print(f"PASS: 5 production GPT train steps (fp32, AdamW): {losses}", flush=True)


def main():
    try:
        device = require_gpu()
        verify_gpu()
        check_ops(device)
        check_train_steps(device)
        print("ALL TRAINING TESTS PASSED", flush=True)
        return 0
    except Exception:
        traceback.print_exc(file=sys.stdout)
        return 1


if __name__ == "__main__":
    sys.exit(main())
