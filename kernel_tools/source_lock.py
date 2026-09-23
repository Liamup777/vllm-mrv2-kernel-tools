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
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
         "--exclude-standard", "--", package],
        capture_output=True,
        timeout=300,
    )
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip() or
                         f"Cannot enumerate Git sources under {package}")
    files = {}
    names = sorted(name.decode(errors="surrogateescape")
                   for name in result.stdout.split(b"\0") if name)
    for relative in names:
        path = root / relative
        if (relative.endswith(".py") and relative not in GENERATED_SOURCE_FILES and
                path.is_file() and not path.is_symlink()):
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
    """Build the persisted commit gate used by verify and pipeline.

    ``packages`` may contain file hashes used internally while selecting AI
    context, but the persisted lock intentionally records only repository
    commits.  Standalone ``run`` does not consume this lock.
    """
    if not packages:
        raise ValueError("At least one source package is required")
    normalized = {}
    for name, value in sorted(packages.items()):
        if name not in PACKAGES:
            raise ValueError(f"Unsupported source package: {name}")
        commit = value.get("commit")
        if not isinstance(commit, str) or not commit:
            raise ValueError(f"Source package {name} has no commit")
        normalized[name] = {"commit": commit}
    identity = {"schema_version": 2, "packages": normalized}
    return {**identity, "fingerprint": digest(identity)}


def validate_source_lock(lock):
    if not isinstance(lock, dict) or lock.get("schema_version") not in {1, 2}:
        raise ValueError("Unsupported source lock format")
    packages = lock.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ValueError("Source lock has no packages")
    if lock["schema_version"] == 1:
        # Read old artifacts, but all new locks use the compact schema above.
        identity = {"schema_version": 1, "packages": {}}
        for name, value in sorted(packages.items()):
            if name not in PACKAGES:
                raise ValueError(f"Unsupported source package: {name}")
            identity["packages"][name] = {
                key: value[key]
                for key in ("commit", "requested_ref", "dirty", "files")
            }
        fingerprint = digest(identity)
    else:
        fingerprint = make_source_lock(packages)["fingerprint"]
    if fingerprint != lock.get("fingerprint"):
        raise ValueError("Source lock fingerprint is invalid")
    return lock


def verify_source_lock(lock, info):
    """Compare locked commits with the actually imported Git checkouts."""
    validate_source_lock(lock)
    mismatches = []
    actual = {}
    for package, expected in lock["packages"].items():
        origin = info.get("imports", {}).get(package)
        if not origin or not Path(origin).is_file():
            mismatches.append(f"{package}: actual import path is unavailable ({origin!r})")
            continue
        imported_from = str(Path(origin).resolve())
        revision = info.get("source_revisions", {}).get(package, {})
        head = revision.get("head")
        if not head:
            mismatches.append(f"{package}: imported from {imported_from}; source is not in a detectable Git checkout")
        elif head != expected["commit"]:
            mismatches.append(f"{package}: expected commit {expected['commit']}, actual {head}; "
                              f"imported from {imported_from}")
        package_root = Path(origin).resolve().parent
        actual[package] = {"import": imported_from, "root": str(package_root.parent),
                           "commit": head, "dirty": revision.get("dirty")}
    if mismatches:
        raise ValueError("Source verification failed: " + " | ".join(mismatches))
    return {"status": "verified", "fingerprint": lock["fingerprint"], "packages": actual}
