"""Check or apply the cuBLAS DLL replacement without importing torch."""

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure_cublas(torch_lib, zluda_path, *, apply=False):
    source = Path(zluda_path) / "cublas.dll"
    target = Path(torch_lib) / "cublas64_11.dll"
    backup = target.with_name(target.name + ".nv_backup")
    for path in (source, target):
        if not path.is_file():
            raise RuntimeError(f"Missing {path}. Follow SETUP.md before applying the DLL fix.")
    if digest(source) == digest(target):
        print(f"cuBLAS is configured: {target}")
        return
    if not apply:
        raise RuntimeError("torch still has a different cuBLAS DLL. Run this command with --apply.")
    if backup.exists() and digest(backup) != digest(target):
        raise RuntimeError(
            f"Existing backup differs from the installed DLL: {backup}. "
            "Preserve and resolve that backup manually before retrying."
        )
    if not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(source, target)
    if digest(source) != digest(target):
        raise RuntimeError(f"DLL copy verification failed; original is preserved at {backup}.")
    print(f"Installed ZLUDA cuBLAS: {target}\nOriginal preserved: {backup}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zluda-path", type=Path,
                        default=Path(os.environ.get("ZLUDA_PATH", Path.home() / "zluda")))
    parser.add_argument("--apply", action="store_true", help="back up and replace torch's cuBLAS DLL")
    args = parser.parse_args()
    try:
        if sys.platform != "win32":
            raise RuntimeError("This setup helper targets native Windows. See SETUP.md.")
        dist = importlib.metadata.distribution("torch")
        if dist.version != "2.4.1+cu118":
            raise RuntimeError(f"Expected torch 2.4.1+cu118 in this interpreter; found {dist.version}.")
        configure_cublas(dist.locate_file("torch/lib"), args.zluda_path, apply=args.apply)
    except (RuntimeError, OSError, importlib.metadata.PackageNotFoundError) as exc:
        print(f"RX 6600 setup: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
