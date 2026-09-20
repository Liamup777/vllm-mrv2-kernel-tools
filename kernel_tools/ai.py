"""Non-interactive Codex adapter. Model output is data, never a shell command."""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


STRING = {"type": "string"}
REVIEW_SCHEMA = object_schema({
    "operators": {"type": "array", "items": object_schema({
        "id": STRING, "kernel": STRING, "definition": STRING,
        "classification": {"type": "string", "enum": ["new", "not_new", "needs_review"]},
        "reason": STRING, "evidence": STRING})},
    "unresolved": {"type": "array", "items": STRING},
    "summary": STRING})
CASE_SCHEMA = object_schema({
    "status": {"type": "string", "enum": ["ready", "blocked"]},
    "reason": STRING, "analysis": STRING, "cases_json": STRING})
DIAGNOSIS_SCHEMA = object_schema({"analysis": STRING})


def describe_codex(executable, model=None, timeout=1800):
    configured = {}
    config = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
    try:
        configured = tomllib.loads(config.read_text()) if config.is_file() else {}
    except (OSError, tomllib.TOMLDecodeError):
        pass
    return {
        "executable": str(executable),
        "model": model or configured.get("model") or "Codex configured default",
        "model_source": "command line" if model else "Codex config",
        "reasoning_effort": configured.get("model_reasoning_effort") if not model else None,
        "timeout_seconds": timeout,
    }


def resolve_codex(config=None, executable=None):
    if executable:
        return executable
    data = json.loads(Path(config).read_text()) if config and Path(config).exists() else {}
    ai_config = data.get("ai", {})
    if not isinstance(ai_config, dict):
        raise ValueError("Configuration ai must be an object")
    value = ai_config.get("codex", "codex")
    if not isinstance(value, str) or not value:
        raise ValueError("ai.codex must be an executable name or path")
    if "/" in value:
        value = str((Path(config).resolve().parent / Path(value).expanduser()).resolve())
    return value


def validate_schema(value, schema, path="response"):
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict) or set(value) != set(schema["properties"]):
            raise ValueError(f"{path}: incorrect JSON object fields")
        for key, child in schema["properties"].items():
            validate_schema(value[key], child, path + "." + key)
    elif kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array")
        for item in value:
            validate_schema(item, schema["items"], path + "[]")
    elif kind == "string" and not isinstance(value, str):
        raise ValueError(f"{path}: expected string")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: invalid enum value")


class Codex:
    def __init__(self, executable="codex", model=None, timeout=1800):
        self.executable = shutil.which(executable)
        if not self.executable:
            raise ValueError(f"Codex CLI not found: {executable}; install it and sign in first")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("AI timeout must be finite and positive")
        self.model, self.timeout = model, timeout

    def description(self):
        """Return the effective user-visible AI settings without invoking Codex."""
        return describe_codex(self.executable, self.model, self.timeout)

    def ask(self, prompt, schema, *, workspace, failure_log, label="AI task"):
        """Retain full failed CLI output, discard successful event chatter."""
        failure_log = Path(failure_log)
        with tempfile.TemporaryDirectory(prefix="kernel-tools-ai-") as tmp:
            tmp = Path(tmp)
            spec, result, capture = tmp / "schema.json", tmp / "answer.json", tmp / "events.log"
            spec.write_text(json.dumps(schema))
            command = [self.executable, "exec", "--sandbox", "read-only",
                       "-c", 'approval_policy="never"', "--skip-git-repo-check", "--ephemeral",
                       "--color", "never", "--cd", str(Path(workspace).resolve()),
                       "--output-schema", str(spec), "--output-last-message", str(result)]
            if self.model:
                command += ["--model", self.model]
            command += ["-"]
            shown_model = self.description()["model"]
            print(f"[ai] start: {label}; model={shown_model}; prompt={len(prompt)} chars; "
                  f"timeout={self.timeout:g}s", flush=True)
            started = time.monotonic()
            try:
                with capture.open("wb") as stream:
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        process.stdin.write(prompt.encode())
                        process.stdin.close()
                        next_notice = started + 30
                        while process.poll() is None:
                            now = time.monotonic()
                            if now - started >= self.timeout:
                                raise subprocess.TimeoutExpired(command, self.timeout)
                            if now >= next_notice:
                                print(f"[ai] running: {label}; elapsed={int(now - started)}s", flush=True)
                                next_notice = now + 30
                            time.sleep(min(1, max(0.05, self.timeout - (now - started))))
                    except (subprocess.TimeoutExpired, KeyboardInterrupt):
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise
                print(f"[ai] finished: {label}; elapsed={int(time.monotonic() - started)}s", flush=True)
                if process.returncode:
                    with capture.open("rb") as stream:
                        stream.seek(max(0, capture.stat().st_size - 32000))
                        lines = stream.read().decode(errors="replace").splitlines()
                    errors = [line for line in lines if line.startswith("ERROR:")]
                    detail = errors[-1][:1200] if errors else "see full failure log"
                    raise ValueError(f"Codex exited with code {process.returncode}: {detail}")
                value = json.loads(result.read_text())
                validate_schema(value, schema)
                return value
            except (ValueError, OSError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                failure_log.parent.mkdir(parents=True, exist_ok=True)
                if capture.exists():
                    shutil.copyfile(capture, failure_log)
                with failure_log.open("a") as stream:
                    stream.write(f"\n[kernel-tools AI] {type(error).__name__}: {error}\n")
                    if result.exists():
                        stream.write(result.read_text(errors="replace"))
                if isinstance(error, KeyboardInterrupt):
                    raise
                raise ValueError(f"AI stage failed; see {failure_log}: {error}") from error
