"""Check that the configured RX 6600 stack can compute and backpropagate."""

import os
import sys
import traceback

from rx6600_runtime import require_gpu, torch, verify_gpu


def main():
    try:
        print("=== RX 6600 probe ===", flush=True)
        print(f"python: {sys.version.split()[0]}", flush=True)
        print(f"working directory: {os.getcwd()}", flush=True)
        print(f"torch: {torch.__version__}", flush=True)
        require_gpu()
        verify_gpu()
        print("ALL GPU TESTS PASSED", flush=True)
        return 0
    except Exception:
        traceback.print_exc(file=sys.stdout)
        return 1


if __name__ == "__main__":
    sys.exit(main())
