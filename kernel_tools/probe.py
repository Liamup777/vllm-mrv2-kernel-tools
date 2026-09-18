from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def environment(device=None):
    info = {"python": sys.version.split()[0], "executable": sys.executable,
            "hostname": platform.node(), "packages": {}, "imports": {}}
    for name in ("torch", "torch-npu", "triton", "triton-ascend", "vllm", "vllm-ascend"):
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = None
    for name in ("vllm", "vllm_ascend", "triton", "torch_npu"):
        try:
            spec = importlib.util.find_spec(name)
            info["imports"][name] = spec.origin if spec else None
        except (ValueError, ImportError) as error:
            info["imports"][name] = str(error)
    cann_root = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    info["cann"] = {"root": str(cann_root), "version": None}
    for path in [cann_root / "version.cfg", cann_root / "aarch64-linux/ascend_toolkit_install.info",
                 cann_root / "x86_64-linux/ascend_toolkit_install.info"]:
        if path.is_file():
            lines = [line.strip() for line in path.read_text(errors="replace").splitlines()
                     if line.strip().lower().startswith(("version=", "version ="))]
            if lines:
                info["cann"]["version"] = lines[0].split("=", 1)[1].strip()
                break
    try:
        import torch
        import torch_npu  # noqa: F401
        import triton
        info["triton_module_version"] = triton.__version__
        info["npu_count"] = torch.npu.device_count()
        info["npu_available"] = bool(torch.npu.is_available())
        if device and info["npu_available"]:
            torch.npu.set_device(device)
            info["device"] = device
            info["device_name"] = torch.npu.get_device_name(device)
    except Exception as error:
        info["npu_available"] = False
        info["npu_error"] = f"{type(error).__name__}: {error}"
    if shutil.which("npu-smi"):
        try:
            result = subprocess.run(["npu-smi", "info"], text=True, capture_output=True, timeout=15)
            info["npu_smi"] = result.stdout[-16000:]
        except (subprocess.TimeoutExpired, OSError) as error:
            info["npu_smi_error"] = str(error)
    return info


def main():
    print(json.dumps(environment(sys.argv[1] if len(sys.argv) > 1 else None), ensure_ascii=False))


if __name__ == "__main__":
    main()
