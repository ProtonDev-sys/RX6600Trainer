import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import rx6600_runtime as runtime
from setup_rx6600 import configure_cublas
from train_tinystories import GPT, sample_text
import training_smoke

torch = runtime.torch
ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lib = self.root / "torch lib"
        self.zluda = self.root / "zluda path"
        self.lib.mkdir()
        self.zluda.mkdir()
        self.target = self.lib / "cublas64_11.dll"
        self.backup = self.lib / "cublas64_11.dll.nv_backup"
        self.target.write_bytes(b"original cuBLAS")
        (self.zluda / "cublas.dll").write_bytes(b"ZLUDA cuBLAS")

    def test_check_does_not_modify_dlls(self):
        with self.assertRaisesRegex(RuntimeError, "--apply"):
            configure_cublas(self.lib, self.zluda)
        self.assertEqual(self.target.read_bytes(), b"original cuBLAS")
        self.assertFalse(self.backup.exists())

    def test_apply_is_idempotent_and_preserves_original(self):
        with contextlib.redirect_stdout(io.StringIO()):
            configure_cublas(self.lib, self.zluda, apply=True)
            configure_cublas(self.lib, self.zluda, apply=True)
            configure_cublas(self.lib, self.zluda)
        self.assertEqual(self.target.read_bytes(), b"ZLUDA cuBLAS")
        self.assertEqual(self.backup.read_bytes(), b"original cuBLAS")

    def test_stale_backup_is_not_overwritten_after_reinstall(self):
        self.backup.write_bytes(b"older original")
        with self.assertRaisesRegex(RuntimeError, "Existing backup differs"):
            configure_cublas(self.lib, self.zluda, apply=True)
        self.assertEqual(self.target.read_bytes(), b"original cuBLAS")
        self.assertEqual(self.backup.read_bytes(), b"older original")

    def test_missing_source_does_not_modify_target(self):
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            configure_cublas(self.lib, self.root / "missing", apply=True)
        self.assertEqual(self.target.read_bytes(), b"original cuBLAS")
        self.assertFalse(self.backup.exists())


class RuntimeTests(unittest.TestCase):
    def test_environment_is_configured_before_torch_import(self):
        env = dict(os.environ, TORCH_BLAS_PREFER_CUBLASLT="1", TORCH_BLAS_PREFER_HIPBLASLT="1")
        code = (
            "import rx6600_runtime, os, json; "
            "print(json.dumps([os.getenv(k) for k in "
            "['DISABLE_ADDMM_CUDA_LT', 'TORCH_BLAS_PREFER_CUBLASLT', "
            "'TORCH_BLAS_PREFER_HIPBLASLT', 'NVIDIA_TF32_OVERRIDE']]))"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), ["1", None, None, "0"])

    def test_gpu_rejects_wrong_wheel_before_backend_calls(self):
        with patch.object(torch, "__version__", "2.3.1+cu118"), \
                patch.object(runtime, "configure_torch") as configure:
            with self.assertRaisesRegex(RuntimeError, "requires torch 2.4.1"):
                runtime.require_gpu()
            configure.assert_not_called()

    def test_missing_gpu_cannot_silently_select_cpu(self):
        with patch.object(torch, "__version__", "2.4.1+cu118"), \
                patch.object(torch.version, "cuda", "11.8"), \
                patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "No GPU is available"):
                runtime.require_gpu()

    def test_conservative_backends_and_cpu_synchronization(self):
        runtime.configure_torch()
        self.assertFalse(torch.backends.cudnn.enabled)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
        self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
        self.assertFalse(torch.backends.cuda.cudnn_sdp_enabled())
        self.assertTrue(torch.backends.cuda.math_sdp_enabled())
        with patch.object(torch.cuda, "synchronize") as synchronize:
            runtime.synchronize("cpu")
            synchronize.assert_not_called()

    def test_probe_returns_failure_for_missing_gpu(self):
        import zluda_probe
        with patch.object(zluda_probe, "require_gpu", side_effect=RuntimeError("No GPU")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(zluda_probe.main(), 1)
        self.assertNotIn("ALL GPU TESTS PASSED", output.getvalue())

    def test_probe_returns_failure_for_bad_numerical_results(self):
        import zluda_probe
        with patch.object(zluda_probe, "require_gpu", return_value=torch.device("cuda")), \
                patch.object(zluda_probe, "verify_gpu", side_effect=RuntimeError("GPU values differ")) as verify, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(zluda_probe.main(), 1)
            verify.assert_called_once()
        self.assertIn("GPU values differ", output.getvalue())
        self.assertNotIn("ALL GPU TESTS PASSED", output.getvalue())


class TrainingTests(unittest.TestCase):
    def test_production_smoke_on_cpu(self):
        runtime.configure_torch()
        with contextlib.redirect_stdout(io.StringIO()):
            training_smoke.check_ops(torch.device("cpu"))
            training_smoke.check_train_steps(torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            training_smoke.check_finite(torch.tensor(float("nan")), "loss")

    def test_sampling_disables_gradient_graphs(self):
        model = GPT(3, 1, 1, 8, 4, dropout=0.0)
        original_generate = model.generate

        def checked_generate(*args):
            self.assertFalse(torch.is_grad_enabled())
            return original_generate(*args)

        with patch.object(model, "generate", side_effect=checked_generate):
            result = sample_text(model, 0, {0: "a", 1: "b", 2: "c"}, 4, "cpu", 4)
        self.assertEqual(len(result), 5)
        self.assertTrue(model.training)

    def test_cpu_training_evaluation_sampling_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "TinyStories.txt").write_text("a tiny story.\n" * 100, encoding="utf-8")
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            result = subprocess.run([
                sys.executable, str(ROOT / "train_tinystories.py"), "--device", "cpu",
                "--steps", "1", "--n-layer", "1", "--n-head", "1", "--n-embd", "8",
                "--block-size", "8", "--batch-size", "1", "--eval-iters", "1",
            ], cwd=path, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("device cpu", result.stdout)
            self.assertIn("val loss", result.stdout)
            self.assertIn("checkpoint saved", result.stdout)
            saved = torch.load(path / "train_ckpt" / "ckpt_step1.pt", weights_only=True)
            self.assertEqual(saved["config"]["device"], "cpu")
            self.assertTrue(all(torch.isfinite(t).all() for t in saved["model"].values()))


if __name__ == "__main__":
    unittest.main()
