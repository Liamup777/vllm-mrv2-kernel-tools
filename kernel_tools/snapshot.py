"""Read-only inventory of the actual imported source, usable on a remote host."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path


def source_files(info):
    files = {}
    for name in ("vllm", "vllm_ascend"):
        origin = info.get("imports", {}).get(name)
        if not origin or not Path(origin).is_file():
            continue
        root = Path(origin).resolve().parent
        for file in sorted(root.rglob("*.py")):
            if file.is_file() and not file.is_symlink() and "__pycache__" not in file.parts:
                files[name + "/" + file.relative_to(root).as_posix()] = file.read_bytes()
    return files


def source_identity(info, files=None):
    if files is None:
        files = source_files(info)
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    identity = {"files": hashes, "packages": info.get("packages", {}), "python": info.get("python"),
                "triton_module_version": info.get("triton_module_version"), "cann": info.get("cann")}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"fingerprint": fingerprint, **identity}


def write_snapshot(info, stream):
    files = source_files(info)
    metadata = {"environment": info, "identity": source_identity(info, files)}
    with tarfile.open(fileobj=stream, mode="w|gz") as archive:
        payloads = {**files, "snapshot.json": json.dumps(metadata, ensure_ascii=False).encode()}
        for name, data in payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
