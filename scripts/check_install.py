#!/usr/bin/env python3
"""Build/install in an isolated venv and check first-use commands without AI/NPU.

Requires Python 3.11+ and pip. Building may download setuptools. No private
configuration is read; all generated files live in a temporary directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import venv


def main():
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="kernel-tools-install-check-") as tmp:
        root = Path(tmp).resolve()
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        bin_dir = root / "venv" / ("Scripts" if os.name == "nt" else "bin")
        python = bin_dir / ("python.exe" if os.name == "nt" else "python")
        cli = bin_dir / ("kernel-tools.exe" if os.name == "nt" else "kernel-tools")

        def run(*args):
            result = subprocess.run([str(x) for x in args], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            return result.stdout

        print("Building wheel in a temporary environment (build dependencies may be downloaded)...", flush=True)
        run(python, "-m", "pip", "wheel", repo, "--no-deps", "--wheel-dir", root / "wheels")
        wheel, = (root / "wheels").glob("*.whl")
        run(python, "-m", "pip", "install", "--no-deps", wheel)
        run(cli, "--help")
        run(python, "-m", "kernel_tools", "--help")
        run(cli, "init")
        config = json.loads((root / "kernel-tools.json").read_text())
        assert config["targets"]["my-npu"]["host"] == "npu-lab"
        run(cli, "doctor", "--npu", "my-npu", "--dry-run")
        run(cli, "pipeline", "--repo", root, "--base", "v1", "--target", "v2",
            "--config", root / "missing.json", "--prepare-only", "--dry-run")
        resources = Path(run(python, "-c",
            "from kernel_tools.review import resource_root; print(resource_root())").strip())
        assert resources.is_relative_to(root / "venv"), resources
        assert len(list((resources / "skills").rglob("SKILL.md"))) == 4
        entry = resources / "skills/vllm-mrv2-kernel-workflow"
        assert len(list((entry / "references").glob("*.md"))) == 4
        assert (entry / "agents/openai.yaml").is_file()
        for file in (resources / "skills").rglob("*.md"):
            prose = re.sub(r"`[^`]*`", "", file.read_text())
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", prose):
                if "://" not in target and not target.startswith("#"):
                    assert (file.parent / target.split("#", 1)[0]).exists(), (file, target)
        run(cli, "cases", "validate", resources / "examples/fill_num_accepted.json")
        run(python, "-c", "import sys; from kernel_tools.cli import template; template(); "
            "assert not {'torch', 'torch_npu', 'triton'} & sys.modules.keys()")
        print("PASS: isolated wheel installation, both CLI entries, init, doctor dry-run, "
              "prepare-only dry-run, case validation, four skills and references.")
        print("Scope: no real AI request, SSH connection or NPU workload was executed.")


if __name__ == "__main__":
    main()
