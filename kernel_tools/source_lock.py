"""Build and verify reproducible source identities without importing projects."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .common import digest, git


PACKAGES = ("vllm", "vllm_ascend")
GENERATED_SOURCE_FILES = {"vllm/_version.py"}


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _git_bytes(repo, commit, relative):
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{relative}"],
        capture_output=True,
        timeout=300,
    )
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip() or
                         f"Cannot read {relative} at {commit}")
    return result.stdout


def _package_for(relative):
    first = Path(relative).parts[0] if Path(relative).parts else ""
    return first if first in PACKAGES else None


def _resolve_repo(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Local source path not found: {path}")
    root = Path(git(path.parent if path.is_file() else path,
                    "rev-parse", "--show-toplevel")).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Source path is outside its Git repository: {path}") from error
    return path, root, relative


def _worktree_package_files(root, package):
    package_root = root / package
    if not package_root.is_dir():
        return {}
    files = {}
    for path in sorted(package_root.rglob("*.py")):
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
            relative = path.relative_to(root).as_posix()
            if relative not in GENERATED_SOURCE_FILES:
                files[relative] = path.read_bytes()
    return files


def _revision_package_files(root, commit, package):
    names = git(root, "ls-tree", "-r", "--name-only", commit, "--", package).splitlines()
    return {name: _git_bytes(root, commit, name) for name in names
            if name.endswith(".py") and name not in GENERATED_SOURCE_FILES}


def _selected_paths(relative, available):
    path = Path(relative)
    if relative in ("", "."):
        return set(available)
    if path.suffix == ".py":
        if relative not in available:
            raise ValueError(f"Python source does not exist at selected revision: {relative}")
        return {relative}
    prefix = relative.rstrip("/") + "/"
    selected = {name for name in available if name.startswith(prefix)}
    if not selected:
        raise ValueError(f"No Python sources under selected path: {relative}")
    return selected


def local_source(path, *, ref=None, selectors=()):
    """Read bounded AI context and a complete package hash lock from a local Git repo."""
    path, root, relative = _resolve_repo(path)
    if ref and ref.startswith("-"):
        raise ValueError("--ref must name a Git revision")
    commit = git(root, "rev-parse", "--verify", f"{ref or 'HEAD'}^{{commit}}")
    packages = []
    direct_package = _package_for(relative)
    if direct_package:
        packages = [direct_package]
    elif path == root:
        packages = [name for name in PACKAGES if (root / name).is_dir()]
    if not packages:
        raise ValueError("--source must be inside a vllm or vllm_ascend package, or its repository root")

    package_files = {}
    locks = {}
    for package in packages:
        files = (_revision_package_files(root, commit, package) if ref else
                 _worktree_package_files(root, package))
        if not files:
            raise ValueError(f"No Python files found for package {package} in {root}")
        package_files.update(files)
        locks[package] = {
            "commit": commit,
            "requested_ref": ref,
            "dirty": False if ref else bool(git(root, "status", "--porcelain", "--", package)),
            "files": {name: _hash(data) for name, data in sorted(files.items())},
            "local_root": str(root),
        }

    chosen = _selected_paths(relative, package_files)
    needles = {name.rsplit(".", 1)[-1] for name in selectors}
    if needles:
        for name, data in package_files.items():
            if any(needle.encode() in data for needle in needles):
                chosen.add(name)
    context = {name: package_files[name].decode("utf-8") for name in sorted(chosen)}
    return context, locks


def package_lock(package, commit, sources, *, ref=None, local_root=None):
    files = {name: _hash(text.encode()) for name, text in sorted(sources.items())
             if name.startswith(package + "/") and name not in GENERATED_SOURCE_FILES}
    if not files:
        raise ValueError(f"No {package} Python sources available for source lock")
    return {"commit": commit, "requested_ref": ref, "dirty": False,
            "files": files, "local_root": str(Path(local_root).resolve()) if local_root else None}


def make_source_lock(packages):
    if not packages:
        raise ValueError("At least one source package is required")
    normalized = {}
    for name, value in sorted(packages.items()):
        if name not in PACKAGES:
            raise ValueError(f"Unsupported source package: {name}")
        normalized[name] = {
            "commit": value["commit"],
            "requested_ref": value.get("requested_ref"),
            "dirty": bool(value.get("dirty")),
            "files": dict(sorted(value["files"].items())),
        }
        if value.get("local_root"):
            normalized[name]["local_root"] = value["local_root"]
    identity = {"schema_version": 1, "packages": {
        name: {key: value[key] for key in ("commit", "requested_ref", "dirty", "files")}
        for name, value in normalized.items()
    }}
    return {**identity, "fingerprint": digest(identity)}


def validate_source_lock(lock):
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise ValueError("Unsupported source lock format")
    packages = lock.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ValueError("Source lock has no packages")
    rebuilt = make_source_lock(packages)
    if rebuilt["fingerprint"] != lock.get("fingerprint"):
        raise ValueError("Source lock fingerprint is invalid")
    return lock


def verify_source_lock(lock, info):
    """Compare a lock with the sources actually imported by this Python environment."""
    validate_source_lock(lock)
    mismatches = []
    actual = {}
    for package, expected in lock["packages"].items():
        origin = info.get("imports", {}).get(package)
        if not origin or not Path(origin).is_file():
            mismatches.append(f"{package}: actual import path is unavailable ({origin!r})")
            continue
        package_root = Path(origin).resolve().parent
        imported_from = str(Path(origin).resolve())
        revision = info.get("source_revisions", {}).get(package, {})
        head = revision.get("head")
        if not head:
            mismatches.append(f"{package}: imported from {imported_from}; source is not in a detectable Git checkout")
        elif head != expected["commit"]:
            mismatches.append(f"{package}: expected commit {expected['commit']}, actual {head}; "
                              f"imported from {imported_from}")
        files = _worktree_package_files(package_root.parent, package)
        hashes = {name: _hash(data) for name, data in sorted(files.items())}
        expected_files = expected["files"]
        changed = sorted(name for name in expected_files.keys() & hashes.keys()
                         if expected_files[name] != hashes[name])
        missing = sorted(expected_files.keys() - hashes.keys())
        extra = sorted(hashes.keys() - expected_files.keys())
        if changed or missing or extra:
            examples = [*(f"changed {name}" for name in changed[:5]),
                        *(f"missing {name}" for name in missing[:5]),
                        *(f"extra {name}" for name in extra[:5])]
            mismatches.append(f"{package}: {len(changed)} changed, {len(missing)} missing, "
                              f"{len(extra)} extra Python files ({'; '.join(examples)}); "
                              f"imported from {imported_from}")
        actual[package] = {"import": imported_from, "root": str(package_root.parent),
                           "commit": head, "dirty": revision.get("dirty"),
                           "file_count": len(hashes)}
    if mismatches:
        raise ValueError("Source verification failed: " + " | ".join(mismatches))
    return {"status": "verified", "fingerprint": lock["fingerprint"], "packages": actual}
