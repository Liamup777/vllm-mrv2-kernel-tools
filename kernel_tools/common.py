from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def file_key(value):
    # Include the digest so sanitized/truncated names cannot collide.
    stem = re.sub(r"[^a-zA-Z0-9_.-]", "_", value).strip(".")[:90] or "kernel"
    return stem + "-" + hashlib.sha256(value.encode()).hexdigest()[:8]


def git(path, *args):
    result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True,
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}, timeout=300)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def source_snapshot(path):
    path = Path(path).resolve()
    try:
        return {"path": str(path), "root": git(path, "rev-parse", "--show-toplevel"),
                "head": git(path, "rev-parse", "HEAD"),
                "dirty": bool(git(path, "status", "--porcelain"))}
    except ValueError:
        return {"path": str(path), "head": None, "dirty": None}
