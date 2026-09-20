from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .common import digest, file_key, save_json, source_snapshot, user_data_root

METRICS = ("mean", "p50", "p90", "p99", "min", "max")


def runtime_env(cwd, pythonpath=()):
    env = os.environ.copy()
    roots = [str(Path(__file__).resolve().parent.parent), *map(str, pythonpath), str(cwd)]
    if env.get("PYTHONPATH"):
        roots.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(roots)
    return env


def probe(cwd, env, device):
    proc = subprocess.run([sys.executable, "-m", "kernel_tools.probe", device],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    for line in reversed(proc.stdout.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict) and "npu_available" in value:
                return value
        except json.JSONDecodeError:
            pass
    raise ValueError("Environment probe failed: " + (proc.stderr or proc.stdout)[-1500:])


@contextlib.contextmanager
def device_lock(device):
    # Cooperative lock only; it cannot reserve the NPU against external programs.
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    index = int(device.split(":")[-1])
    mapping = [v.strip() for v in visible.split(",")] if visible else []
    physical = mapping[index] if mapping and index < len(mapping) else str(index)
    path = Path(tempfile.gettempdir()) / ("kernel-tools-device-" + file_key(physical) + ".lock")
    with path.open("a+") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"Another kernel-tools job holds {device}; retry later") from None
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def failure_from(log, phase="unknown", reason="case process failed"):
    lower = log.lower()
    markers = ("compilationerror", "mlircompilationerror", "converttritonir", "ub overflow",
               "llvm error", "failed to run bisheng", "failed to legalize unresolved")
    if phase not in ("input", "import", "environment", "correctness", "timeout") and any(x in lower for x in markers):
        phase = "compile"
    lines = []
    for line in log.splitlines():
        if any(x in line.lower() for x in ("error", "failed", "overflow", "assertion", "unsupported")):
            clean = line.strip()[:500]
            if clean and clean not in lines:
                lines.append(clean)
    return {"phase": phase, "reason": " ".join(reason.split())[:700], "error": lines[:3]}


def validate_result(data, case, device, warmup, rounds):
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError("worker must return exactly one result")
    row = data[0]
    for key, expected in {**{k: case[k] for k in ("name", "kernel", "target")},
                          "device": device, "warmup": warmup, "profiling_rounds": rounds}.items():
        if row.get(key) != expected:
            raise ValueError(f"worker result identity mismatch: {key}")
    samples = row.get("latencies_ms")
    if not isinstance(samples, list) or len(samples) != rounds:
        raise ValueError("worker returned incomplete samples")
    values = {k: row.get("summary", {}).get(k + "_ms") for k in METRICS}
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in [*samples, *values.values()]):
        raise ValueError("worker returned invalid latency")
    correctness = row.get("correctness", "not_checked")
    if correctness not in ("passed", "not_checked") or (case.get("check") and correctness != "passed"):
        raise ValueError("worker did not confirm the requested correctness check")
    return {"status": "success", "correctness": correctness,
            "latency_us": {k: v * 1000 for k, v in values.items()},
            "binding": row.get("binding"), "benchmark_scope": row.get("benchmark_scope")}


def run_one(case, cwd, env, device, warmup, rounds, timeout, log_path, worker_command=None):
    with tempfile.TemporaryDirectory(prefix="kernel-tools-case-") as tmp:
        input_path, output = Path(tmp) / "case.json", Path(tmp) / "raw.json"
        save_json(input_path, [case])
        command = worker_command or [sys.executable, "-m", "kernel_tools.benchmark"]
        command = [*command, "--input-file", str(input_path), "--kernel", case["kernel"],
                   "--case-name", case["name"], "--target", case["target"],
                   "--device", device,
                   "--warmup", str(warmup), "--profiling-rounds", str(rounds), "--output", str(output)]
        capture = Path(tmp) / "process.log"
        phase, reason = "unknown", "case process failed"
        interrupted = False
        with capture.open("wb") as stream:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
                reason = f"worker exited with code {code}"
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                interrupted = isinstance(error, KeyboardInterrupt)
                phase = "interrupted" if interrupted else "timeout"
                reason = "run interrupted" if interrupted else f"case exceeded {timeout:g} seconds"
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait()
        data = None
        if output.exists():
            try:
                data = json.loads(output.read_text())
                if isinstance(data, dict) and "error" in data and phase == "unknown":
                    phase = data["error"].get("phase", "unknown")
                    reason = data["error"].get("reason", reason)
            except (ValueError, OSError) as error:
                reason = f"invalid worker JSON: {error}"
        if code == 0 and phase == "unknown":
            try:
                result = validate_result(data, case, device, warmup, rounds)
                log_path.unlink(missing_ok=True)
                return result
            except (ValueError, KeyError, TypeError) as error:
                phase, reason = "result", str(error)
        # Full failed log remains on disk. Read only its tail for the short report.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(capture, log_path)
        with log_path.open("rb") as file:
            file.seek(max(0, log_path.stat().st_size - 128_000))
            tail = file.read().decode(errors="replace")
        with log_path.open("a") as file:
            file.write(f"\n[kernel-tools] {phase}: {reason}\n")
        return {"status": "interrupted" if interrupted else "failed",
                "correctness": "failed" if phase == "correctness" else "not_checked",
                "failure": failure_from(tail, phase, reason)}


def render_report(root):
    root = Path(root)
    files = sorted((root / "results").glob("*.json"))
    results = [json.loads(p.read_text()) for p in files]
    lines = ["# 单算子测试结果", "", "单位：μs。success 表示执行成功；correctness 单列。", ""]
    if results:
        context = results[0]["context"]
        environment = context["environment"]
        versions = ", ".join(f"{k}={v}" for k, v in environment.get("packages", {}).items() if v)
        lines += [f"设备：`{context['device']}`；warmup={context['warmup']}，rounds={context['rounds']}。", "",
                  f"环境：{environment.get('hostname', '?')} / Python {environment.get('python', '?')} / {versions}", "",
                  "设备隔离：协作锁；外部进程占用未自动确认。详细环境和绑定见 results/*.json。", ""]
        for source in context["sources"]:
            lines.append(f"- 源码 `{source['path']}`：HEAD `{source['head']}`，dirty={source['dirty']}")
        lines.append("")
    for item in results:
        lines += [f"## {item['kernel']}", "",
                  "| case | 执行 | 精度 | mean | p50 | p90 | p99 | min | max |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in item["scenarios"]:
            metrics = row.get("latency_us", {})
            safe = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
            lines.append("| " + " | ".join([safe(row["name"]), row["status"], row["correctness"],
                         *[f"{metrics[k]:.3f}" if k in metrics else "—" for k in METRICS]]) + " |")
        lines.append("")
    failures = [(item["kernel"], r) for item in results for r in item["scenarios"] if r.get("failure")]
    if failures:
        lines += ["## 失败与阻塞", ""]
        for kernel, row in failures:
            failure = row["failure"]
            lines += [f"- `{kernel}/{row['name']}`：{failure['phase']} — {failure['reason']}"]
            if row.get("log"):
                lines += [f"  [完整报错日志]({row['log']})"]
            if failure.get("error"):
                lines += ["", "```text", *[x.replace("```", "'''") for x in failure["error"]], "```", ""]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def run_suite(cases, *, cwd, output, device="npu:0", warmup=10, rounds=100, timeout=600,
              pythonpath=(), resume=False, probe_info=None, worker_command=None,
              expected_identity=None):
    if warmup < 0 or rounds < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Require warmup >= 0, rounds > 0 and a finite positive timeout")
    if not device.startswith("npu:") or not device[4:].isdigit():
        raise ValueError("device must be npu:<logical index>")
    cwd, output = Path(cwd).resolve(), Path(output).resolve()
    if not cwd.is_dir():
        raise ValueError(f"Working directory not found: {cwd}")
    env = runtime_env(cwd, list(pythonpath))
    try:
        info = probe_info if probe_info is not None else probe(cwd, env, device)
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        info = {"npu_available": False, "npu_error": str(error)}
    if expected_identity:
        from .snapshot import source_identity
        actual = source_identity(info)["fingerprint"]
        info["source_fingerprint"] = actual
        if actual != expected_identity:
            info = {**info, "npu_available": False,
                    "npu_error": "Source/environment changed since AI case generation; start a new pipeline"}
    sources = [source_snapshot(p) for p in dict.fromkeys([str(cwd), *map(str, pythonpath)])]
    imported_sources = {}
    for name in ("vllm", "vllm_ascend"):
        path = info.get("imports", {}).get(name)
        if path and Path(path).is_file():
            imported_sources[name] = source_snapshot(Path(path).parent)
    # The verbose npu-smi table is shown by doctor, not duplicated in every report.
    context = {"tool_version": __version__, "device": device, "warmup": warmup,
               "run_dir": str(output),
               "rounds": rounds, "timeout_seconds": timeout,
               "environment": {k: v for k, v in info.items() if k != "npu_smi"}, "sources": sources,
               "imported_sources": imported_sources,
               "visible_devices": env.get("ASCEND_RT_VISIBLE_DEVICES", "all"),
               "isolation": "cooperative_lock_only"}
    tool_hash = digest({p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")})
    signature = digest({"cases": cases, "context": context, "tool_hash": tool_hash})
    if resume and any(s["dirty"] is not False for s in [*sources, *imported_sources.values()]):
        raise ValueError("Resume requires clean Git source roots; start a new output directory for dirty/unversioned sources")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError(f"Output already exists: {output}; choose a new path or use --resume")
    output.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for case in cases:
        grouped.setdefault(case["kernel"], []).append(case)
    documents = {}
    for kernel, group in grouped.items():
        key = file_key(kernel)
        path = output / "results" / (key + ".json")
        previous = json.loads(path.read_text()) if resume and path.exists() else None
        if previous and previous.get("run_signature") != signature:
            raise ValueError("Resume refused: cases, source commits, tool, or environment changed")
        old = {r["name"]: r for r in previous["scenarios"]} if previous else {}
        documents[kernel] = {"schema_version": 1, "kernel": kernel, "run_signature": signature,
                             "context": context, "scenarios": [old.get(c["name"], {
                                 "name": c["name"], "status": "pending", "correctness": "not_checked",
                                 "attempts": 0}) for c in group]}
    def persist():
        for kernel, document in documents.items():
            save_json(output / "results" / (file_key(kernel) + ".json"), document)
        render_report(output)
    for kernel, group in grouped.items():
        save_json(output / "cases" / (file_key(kernel) + ".json"), group)
    persist()
    stopped = False
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(device_lock(device))
        except ValueError as error:
            for document in documents.values():
                for row in document["scenarios"]:
                    if row["status"] != "success":
                        row.update(status="blocked", failure={"phase": "device_busy", "reason": str(error), "error": []})
            persist()
            print(f"Blocked: {error}. Report: {output / 'report.md'}", flush=True)
            return 1
        total, index = len(cases), 0
        for kernel, group in grouped.items():
            for case, row in zip(group, documents[kernel]["scenarios"]):
                index += 1
                if row["status"] == "success" and resume:
                    print(f"[{index}/{total}] {kernel}/{case['name']}: reused", flush=True)
                    continue
                if stopped or not info.get("npu_available"):
                    row.update(status="blocked", failure={"phase": "environment" if not stopped else "interrupted",
                               "reason": info.get("npu_error", "NPU unavailable") if not stopped else "previous case interrupted",
                               "error": []})
                    persist()
                    continue
                row["attempts"] += 1
                row["status"] = "running"
                persist()
                print(f"[{index}/{total}] {kernel}/{case['name']}", flush=True)
                log = Path("logs") / file_key(kernel) / (file_key(case["name"]) + ".log")
                result = run_one(case, cwd, env, device, warmup, rounds, timeout, output / log, worker_command)
                for field in ("failure", "log", "latency_us", "binding"):
                    row.pop(field, None)
                row.update(result)
                if result["status"] != "success":
                    row["log"] = str(log)
                stopped = result["status"] == "interrupted"
                persist()
    print(f"Report: {output / 'report.md'}", flush=True)
    return 0 if all(r["status"] == "success" for d in documents.values() for r in d["scenarios"]) else 1


def new_run_path(kind="runs"):
    return user_data_root() / kind / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
