"""Conservative static candidates, without importing the inspected project.

Known local/imported functions and self.method calls are resolved. Dynamic
dispatch is explicitly outside the proof boundary; review.md hands it to AI.
"""
from __future__ import annotations

import ast
import collections
import copy
import hashlib
import subprocess
import tarfile
import tempfile
from pathlib import Path

from .common import digest, git, save_json


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return base + "." + node.attr if base else None
    return None


def module_name(path):
    value = path.removesuffix(".py").replace("/", ".")
    return value.removesuffix(".__init__")


def node_hash(node, ignore_name=False):
    value = copy.deepcopy(node)
    if ignore_name and isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef)):
        value.name = "__kernel__"
    return hashlib.sha256(ast.dump(value, include_attributes=False).encode()).hexdigest()


def tagged_sources(repo, tag):
    # Always resolve refs/tags explicitly; a same-named branch is not equivalent.
    if tag.startswith("-") or ":" in tag or ".." in tag or "^" in tag or "~" in tag:
        raise ValueError("Expected a literal Git tag")
    commit = git(repo, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    with tempfile.TemporaryFile() as archive:
        process = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit, "vllm"],
                                 stdout=archive, stderr=subprocess.PIPE)
        if process.returncode:
            raise ValueError(process.stderr.decode(errors="replace"))
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r|") as tar:
            files = {m.name: tar.extractfile(m).read().decode("utf-8")
                     for m in tar if m.isfile() and m.name.endswith(".py")}
    return commit, files


class Module:
    def __init__(self, path, source):
        self.path, self.name = path, module_name(path)
        self.tree = ast.parse(source, filename=path)
        self.hash = node_hash(self.tree)
        self.imports = collections.defaultdict(set)
        self.functions = {}
        self.calls = collections.defaultdict(list)
        self.launches = []
        self.aliases = collections.defaultdict(set)
        self.scope = self.name
        self.class_name = None
        self.conditions = []
        self.walk(self.tree)

    def walk(self, node):
        if isinstance(node, ast.ClassDef):
            old_scope, old_class = self.scope, self.class_name
            self.scope, self.class_name = self.scope + "." + node.name, node.name
            for child in node.body:
                self.walk(child)
            self.scope, self.class_name = old_scope, old_class
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            old = self.scope
            self.scope = old + "." + node.name
            jit = any((dotted(d.func) if isinstance(d, ast.Call) else dotted(d) or "").endswith(".jit")
                      for d in node.decorator_list if dotted(d.func if isinstance(d, ast.Call) else d))
            self.functions[self.scope] = {"node": node, "jit": jit, "class": self.class_name}
            for child in node.body:
                self.walk(child)
            self.scope = old
            return
        if isinstance(node, (ast.If, ast.IfExp)):
            condition = ast.unparse(node.test)
            self.walk(node.test)
            branches = [(node.body, condition), (node.orelse, "not (" + condition + ")")]
            for children, text in branches:
                self.conditions.append(text)
                for child in children if isinstance(children, list) else [children]:
                    self.walk(child)
                self.conditions.pop()
            return
        if isinstance(node, ast.ImportFrom):
            package = self.name if self.path.endswith("/__init__.py") else self.name.rpartition(".")[0]
            if node.level:
                pieces = package.split(".")
                base = ".".join(pieces[:len(pieces) - node.level + 1])
                base = ".".join(x for x in (base, node.module) if x)
            else:
                base = node.module or ""
            for alias in node.names:
                self.imports[(self.scope, alias.asname or alias.name)].add(base + "." + alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                self.imports[(self.scope, alias.asname or alias.name.split(".")[0])].add(
                    alias.name if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            key, value = dotted(node.targets[0]), dotted(node.value)
            if key and value:
                self.aliases[(self.scope, key)].add(value)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Subscript):
                self.launches.append({"scope": self.scope, "symbol": dotted(node.func.value),
                                      "line": node.lineno, "grid": ast.unparse(node.func.slice),
                                      "call": ast.unparse(node), "conditions": list(self.conditions)})
            else:
                name = dotted(node.func)
                if name:
                    self.calls[self.scope].append(name)
        for child in ast.iter_child_nodes(node):
            self.walk(child)


def scan(repo, tag, scope="vllm/v1/worker/gpu"):
    repo = Path(repo).resolve()
    commit, sources = tagged_sources(repo, tag)
    scope = scope.strip("/")
    if not any(p.startswith(scope + "/") for p in sources):
        raise ValueError(f"Scope {scope} does not exist at {tag}")
    modules, parse_errors = {}, []
    for path, source in sources.items():
        try:
            module = Module(path, source)
            modules[module.name] = module
        except SyntaxError as error:
            parse_errors.append({"path": path, "reason": str(error)})
    functions = {name: (module, info) for module in modules.values() for name, info in module.functions.items()}

    def resolve(module, scope_name, name, seen=frozenset()):
        key = (module.name, scope_name, name)
        if not name or key in seen:
            return set()
        seen = seen | {key}
        if name in functions:
            return {name}
        if name.startswith("self.") or name.startswith("cls."):
            owner = module.functions.get(scope_name, {}).get("class")
            if owner:
                candidate = module.name + "." + owner + "." + name.split(".", 1)[1]
                return {candidate} if candidate in functions else set()
        head, _, tail = name.partition(".")
        current = scope_name
        while current.startswith(module.name):
            targets = module.imports.get((current, head), set())
            if targets:
                found = set()
                for target in targets:
                    full = target + ("." + tail if tail else "")
                    if full in functions:
                        found.add(full)
                    else:
                        # Follow explicit module re-exports.
                        for boundary in range(len(full.split(".")) - 1, 0, -1):
                            parts = full.split(".")
                            origin = modules.get(".".join(parts[:boundary]))
                            if origin:
                                found.update(resolve(origin, origin.name, ".".join(parts[boundary:]), seen))
                                break
                return found
            aliases = module.aliases.get((current, name), set())
            if aliases:
                return set().union(*(resolve(module, current, a, seen) for a in aliases))
            candidate = current + "." + name
            if candidate in functions:
                return {candidate}
            if current == module.name:
                break
            current = current.rpartition(".")[0]
        return set()

    graph = collections.defaultdict(set)
    for module in modules.values():
        for caller, names in module.calls.items():
            for name in names:
                graph[caller].update(resolve(module, caller, name))
    seeds = {name for name, (m, _) in functions.items() if m.path.startswith(scope + "/")}
    seeds.update(m.name for m in modules.values() if m.path.startswith(scope + "/"))
    reached, parent = set(seeds), {}
    pending = collections.deque(sorted(seeds))
    while pending:
        source = pending.popleft()
        for target in sorted(graph[source]):
            if target not in reached:
                reached.add(target)
                parent[target] = source
                pending.append(target)
    records, unresolved = {}, []
    launched_anywhere = set()
    for module in modules.values():
        for launch in module.launches:
            targets = resolve(module, launch["scope"], launch["symbol"])
            candidates = {t for t in targets if functions[t][1]["jit"]}
            launched_anywhere.update(candidates)
            if launch["scope"] not in reached:
                continue
            if len(candidates) != 1:
                unresolved.append({"path": module.path, **launch,
                                   "reason": "launch target is dynamic, ambiguous, or not a recognized Triton JIT",
                                   "candidate_targets": sorted(candidates)})
                continue
            name = next(iter(candidates))
            origin, definition = functions[name]
            node = definition["node"]
            path = [launch["scope"]]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            path.reverse()
            record = records.setdefault(name, {"id": name, "kernel": node.name,
                "definition": {"path": origin.path, "line": node.lineno},
                "source_type": "local" if origin.path.startswith(scope + "/") else "imported",
                "body_hash": node_hash(node), "rename_hash": node_hash(node, ignore_name=True),
                "parameters": [{"name": arg.arg, "annotation": ast.unparse(arg.annotation) if arg.annotation else None}
                               for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]],
                "launches": [], "correctness": "not_checked", "runtime_coverage": "not_checked"})
            wrapper_node = module.functions.get(launch["scope"], {}).get("node", module.tree)
            record["launches"].append({"path": module.path, **launch, "call_path": path,
                                        "wrapper_hash": node_hash(wrapper_node)})
    for name, record in records.items():
        dependency_modules, visited = set(), set()
        queue = [name, *[x["scope"] for x in record["launches"]]]
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in functions:
                dependency_modules.add(functions[current][0].name)
            queue.extend(graph[current] - visited)
        record["dependency_hash"] = digest({m: modules[m].hash for m in sorted(dependency_modules)})
        record["wrapper_hash"] = digest(sorted(x["wrapper_hash"] for x in record["launches"]))
    unlaunched = [{"id": name, "path": module.path, "line": info["node"].lineno,
                  "reason": "no statically resolved direct launch found; helper or dynamic launch needs review"}
                 for name, (module, info) in functions.items()
                 if info["jit"] and module.path.startswith(scope + "/") and name not in launched_anywhere]
    return {"schema_version": 1, "tag": tag, "commit": commit, "scope": scope,
            "validation": "static_candidates_only", "complete_inventory": False,
            "limitations": ["Dynamic dispatch, monkey patches and conditional runtime binding require review.",
                            "All functions under scope are static roots, not proof of runtime reachability.",
                            "Only one discovered call path is retained per direct launcher.",
                            "Module dependency hashes conservatively include unrelated changes in the same module."],
            "kernels": sorted(records.values(), key=lambda r: r["id"]),
            "unresolved_launches": unresolved, "unlaunched_jit": unlaunched, "parse_errors": parse_errors}


def compare(before, after):
    if before["scope"] != after["scope"]:
        raise ValueError("Inventories must use the same scan scope")
    old, new = ({r["id"]: r for r in x["kernels"]} for x in (before, after))
    removed, added = set(old) - set(new), set(new) - set(old)
    moves = []
    for previous in sorted(removed.copy()):
        matches = [current for current in added if new[current]["rename_hash"] == old[previous]["rename_hash"]]
        peers = [x for x in removed if old[x]["rename_hash"] == old[previous]["rename_hash"]]
        if len(matches) == len(peers) == 1:
            current = matches[0]
            moves.append({"from": previous, "to": current, "evidence": "unique identical normalized AST body",
                          "requires_review": True})
            removed.remove(previous)
            added.remove(current)
    changed = []
    for name in sorted(set(old) & set(new)):
        changes = [key.removesuffix("_hash") for key in ("body_hash", "wrapper_hash", "dependency_hash")
                   if old[name][key] != new[name][key]]
        if changes:
            changed.append({"id": name, "changes": changes})
    return {"schema_version": 1, "base": {"tag": before["tag"], "commit": before["commit"]},
            "target": {"tag": after["tag"], "commit": after["commit"]},
            "validation": "static_candidates_only", "added": sorted(added), "removed": sorted(removed),
            "moved_or_renamed": moves, "changed": changed,
            "requires_review": bool(after["unresolved_launches"] or after["parse_errors"] or after["unlaunched_jit"])}


def write_scan(repo, base, target, output, scope="vllm/v1/worker/gpu", *, announce=True):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Scan output already exists: {output}")
    before, after = scan(repo, base, scope), scan(repo, target, scope)
    delta = compare(before, after)
    save_json(output / "base.json", before)
    save_json(output / "target.json", after)
    save_json(output / "delta.json", delta)
    selected = set(delta["added"]) | {r["id"] for r in delta["changed"]} | {r["to"] for r in delta["moved_or_renamed"]}
    lines = [f"# {base} → {target}", "", "这是静态候选清单，尚未证明完整覆盖、Ascend 绑定或数值正确性。", "",
             f"新增候选 {len(delta['added'])}；变化 {len(delta['changed'])}；移动/改名 {len(delta['moved_or_renamed'])}；移除 {len(delta['removed'])}。", "",
             f"目标 commit：`{after['commit']}`。", "",
             "逐项核实：真实 wrapper/调用条件、Ascend 有效绑定、参数关联与布局、grid、case 和 reference。", ""]
    for record in after["kernels"]:
        if record["id"] not in selected:
            continue
        lines += [f"## {record['id']}", "", f"定义：`{record['definition']['path']}:{record['definition']['line']}`", ""]
        for launch in record["launches"]:
            lines += [f"入口：`{' → '.join(launch['call_path'])}`", "", "```python", launch["call"], "```", ""]
    lines += ["## 未解决项", "", f"动态/未解析 launch：{len(after['unresolved_launches'])}；无已解析 launch 的 JIT：{len(after['unlaunched_jit'])}；解析失败：{len(after['parse_errors'])}。", "",
              "详细证据见 target.json。不要仅凭候选清单自动宣称新增算子已完整盘点。"]
    (output / "review.md").write_text("\n".join(lines) + "\n")
    if announce:
        print(f"Scan: {output / 'review.md'}")
    return delta
