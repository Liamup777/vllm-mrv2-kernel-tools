from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path

from .common import save_json


def load_target(config, name):
    data = json.loads(Path(config).read_text())
    targets = data.get("targets", {})
    if name not in targets:
        raise ValueError(f"Unknown target {name!r}; configured: {', '.join(targets)}")
    target = targets[name]
    required = ("host", "cwd", "python", "result_root")
    if any(not isinstance(target.get(k), str) or not target[k] for k in required):
        raise ValueError("Target requires host, cwd, python and result_root")
    if target["host"].startswith("-") or any(x.isspace() for x in target["host"]):
        raise ValueError("host must be an SSH alias or user@host")
    for key in ("cwd", "result_root"):
        if not target[key].startswith("/"):
            raise ValueError(f"Remote {key} must be absolute")
    if not isinstance(target.get("pythonpath", []), list) or not all(isinstance(p, str) and p.startswith("/") for p in target.get("pythonpath", [])):
        raise ValueError("Remote pythonpath must be a list of absolute paths")
    if not isinstance(target.get("env", {}), dict) or any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", k) for k in target.get("env", {})):
        raise ValueError("env must be an object with valid environment variable names")
    return target


def ssh_command(target, command):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target["host"], command]


def target_command(target, argv, *, tool_root=None, setup_stderr=False):
    parts = []
    if target.get("setup"):
        parts.append("source " + shlex.quote(target["setup"]) + (" >&2" if setup_stderr else ""))
    parts.append("cd " + shlex.quote(target["cwd"]))
    paths = ([tool_root] if tool_root else []) + target.get("pythonpath", [])
    environment = dict(target.get("env", {}))
    if paths:
        environment["PYTHONPATH"] = ":".join(paths)
    assignments = [f"{key}={value}" for key, value in environment.items()]
    parts.append(shlex.join(["env", *assignments, *argv]))
    shell = ["bash", "-lc", " && ".join(parts)]
    if target.get("container"):
        shell = ["docker", "exec", target["container"], *shell]
    return shlex.join(shell)


def checked_ssh(target, command, **kwargs):
    result = subprocess.run(ssh_command(target, command), **kwargs)
    if result.returncode:
        raise ValueError(f"Remote command failed (exit {result.returncode}); check SSH key, host fingerprint and target config")
    return result


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    for member in archive.getmembers():
        path = (destination / member.name).resolve()
        if not path.is_relative_to(destination) or not (member.isdir() or member.isfile()):
            raise ValueError(f"Unsafe result archive member: {member.name}")
    if hasattr(tarfile, "data_filter"):
        archive.extractall(destination, filter="data")
    else:
        archive.extractall(destination)


def fetch_snapshot(target, destination, *, device="npu:0", failure_log=None):
    """Download actual import sources and environment without launching kernels."""
    destination = Path(destination)
    probe_source = Path(__file__).with_name("probe.py").read_text().split('\nif __name__ == "__main__":')[0]
    snapshot_source = Path(__file__).with_name("snapshot.py").read_text().replace(
        "from __future__ import annotations", "")
    script = probe_source + "\n" + snapshot_source + (
        "\nimport contextlib\nsaved_stdout = os.dup(1)\nos.dup2(2, 1)\n"
        "try:\n    with contextlib.redirect_stdout(sys.stderr):\n"
        f"        info = environment({device!r})\n"
        "finally:\n    sys.stdout.flush()\n    os.dup2(saved_stdout, 1)\n    os.close(saved_stdout)\n"
        "write_snapshot(info, sys.stdout.buffer)\n")
    with tempfile.TemporaryDirectory(prefix="kernel-tools-snapshot-") as tmp:
        archive_path, log = Path(tmp) / "sources.tar.gz", Path(tmp) / "capture.log"
        try:
            with archive_path.open("wb") as stream, log.open("wb") as errors:
                command = target_command(target, [target["python"], "-c", script], setup_stderr=True)
                checked_ssh(target, command, stdout=stream, stderr=errors, timeout=180)
            destination.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path) as archive:
                safe_extract(archive, destination)
            return json.loads((destination / "snapshot.json").read_text())
        except (ValueError, OSError, tarfile.TarError, subprocess.TimeoutExpired) as error:
            if failure_log:
                failure_log = Path(failure_log)
                failure_log.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(log, failure_log)
                with failure_log.open("a") as stream:
                    stream.write(f"\n{error}\n")
            raise ValueError(f"Remote source snapshot failed: {error}") from error


def run_remote(target, cases, output, *, device="npu:0", warmup=10, rounds=100,
               timeout=600, dry_run=False, doctor=False, expected_identity=None):
    output = Path(output).resolve()
    if not doctor and output.exists() and any(output.iterdir()):
        raise ValueError(f"Output already exists: {output}")
    token = uuid.uuid4().hex
    stage = "/tmp/kernel-tools-" + token
    remote_output = target["result_root"].rstrip("/") + "/" + output.name + "-" + token[:6]
    argv = [target["python"], "-m", "kernel_tools"]
    if doctor:
        argv += ["doctor", "--device", device]
    else:
        argv += ["run", stage + "/input.json", "--cwd", target["cwd"], "--output", remote_output,
                 "--device", device, "--warmup", str(warmup), "--rounds", str(rounds), "--timeout", str(timeout)]
        for path in target.get("pythonpath", []):
            argv += ["--pythonpath", path]
        if expected_identity:
            argv += ["--expected-source-fingerprint", expected_identity]
    command = target_command(target, argv, tool_root=stage)
    if dry_run:
        print(json.dumps({"host": target["host"], "container": target.get("container"),
                          "command": command, "download_to": str(output),
                          "note": "No SSH connection or NPU execution performed"}, ensure_ascii=False, indent=2))
        return 0
    staged = False
    try:
        with tempfile.TemporaryDirectory(prefix="kernel-tools-upload-") as tmp:
            bundle = Path(tmp) / "bundle.tar"
            payload = Path(tmp) / "input.json"
            save_json(payload, cases)
            with tarfile.open(bundle, "w") as tar:
                for source in sorted(Path(__file__).parent.glob("*.py")):
                    tar.add(source, arcname="kernel_tools/" + source.name)
                tar.add(payload, arcname="input.json")
            checked_ssh(target, shlex.join(["mkdir", "-p", stage]))
            staged = True
            with bundle.open("rb") as file:
                checked_ssh(target, shlex.join(["tar", "-xf", "-", "-C", stage]), stdin=file)
            if target.get("container"):
                # Docker cp copies the tool, not the user's production checkout.
                checked_ssh(target, shlex.join(["docker", "cp", stage, target["container"] + ":" + stage]))
            result = subprocess.run(ssh_command(target, command))
            if doctor:
                return result.returncode
            if result.returncode == 255:
                raise ValueError(f"SSH interrupted. Do not restart blindly; inspect remote processes and {remote_output}")
            print(f"Remote results retained: {remote_output}", flush=True)
            archive_path = Path(tmp) / "results.tar"
            collect = ["tar", "-cf", "-", "-C", remote_output, "."]
            if target.get("container"):
                collect = ["docker", "exec", target["container"], *collect]
            with archive_path.open("wb") as file:
                checked_ssh(target, shlex.join(collect), stdout=file)
            output.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path) as archive:
                safe_extract(archive, output)
            print(f"Local report: {output / 'report.md'}", flush=True)
            return result.returncode
    finally:
        if staged:
            # A broken SSH session might have left workers running. Preserve the
            # staging path for diagnosis in that case rather than deleting code.
            if 'result' in locals() and result.returncode != 255:
                commands = []
                if target.get("container"):
                    commands.append(shlex.join(["docker", "exec", target["container"], "rm", "-rf", stage]))
                commands.append(shlex.join(["rm", "-rf", stage]))
                subprocess.run(ssh_command(target, " && ".join(commands)), timeout=20)
