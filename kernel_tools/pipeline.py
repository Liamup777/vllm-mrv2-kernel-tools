"""Scan → AI review/cases → deterministic remote run → evidence-based report."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .ai import CASE_SCHEMA, DIAGNOSIS_SCHEMA, Codex, describe_codex, resolve_codex
from .cases import validate_cases
from .common import file_key, save_json
from .remote import load_target, run_remote
from .scan import Module, write_scan
from .review import (REVIEW_POLICY_VERSION, instructions, load_scan_result,
                     resource_root, review_sources, write_sources)
from .source_lock import local_source, make_source_lock, package_lock


CASE_POLICY_VERSION = 4


def validate_bundle(bundle, operator, runtime_sources):
    if bundle["status"] == "blocked":
        if not bundle["reason"].strip():
            raise ValueError("Blocked generation requires a reason")
        return []
    if not bundle["analysis"].strip():
        raise ValueError("Case generation requires contract and coverage analysis")
    cases = json.loads(bundle["cases_json"])
    json.dumps(cases, allow_nan=False)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Ready generation requires a nonempty JSON case array")
    validate_cases(cases)
    for case in cases:
        if case["kernel"] != operator["kernel"]:
            raise ValueError("Generated case belongs to a different kernel")
        if not isinstance(case.get("scenario"), str) or not case["scenario"].strip():
            raise ValueError("Each generated case needs a scenario description")
        target_module, symbol = case["target"].split(":", 1)
        path = target_module.replace(".", "/") + ".py"
        if path not in runtime_sources:
            path = target_module.replace(".", "/") + "/__init__.py"
        if path not in runtime_sources:
            raise ValueError(f"Case target absent from actual NPU source snapshot: {case['target']}")
        definition = Module(path, runtime_sources[path]).functions.get(target_module + "." + symbol)
        if not definition or not definition["jit"]:
            raise ValueError(f"Case target is not a Triton JIT kernel in NPU source: {case['target']}")
    return cases


def resolve_named_kernels(target_sources, selectors):
    """Resolve explicit names to Triton JIT definitions in supplied sources."""
    definitions = []
    for path, source in target_sources.items():
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue
        module_name = path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")

        class Finder(ast.NodeVisitor):
            def __init__(self):
                self.scope = [module_name]

            def visit_ClassDef(self, node):
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            def visit_FunctionDef(self, node):
                def dotted(value):
                    if isinstance(value, ast.Name):
                        return value.id
                    if isinstance(value, ast.Attribute):
                        parent = dotted(value.value)
                        return parent + "." + value.attr if parent else None
                    return None
                jit = any((dotted(item.func) if isinstance(item, ast.Call) else dotted(item) or "").endswith(".jit")
                          for item in node.decorator_list)
                if jit:
                    definitions.append((".".join([*self.scope, node.name]), path, node))
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

        Finder().visit(tree)
    selected, seen = [], set()
    for selector in selectors:
        matches = [item for item in definitions
                   if item[0] == selector or item[2].name == selector]
        if not matches:
            raise ValueError(f"Triton JIT kernel not found in selected local sources: {selector}")
        if len(matches) > 1:
            identities = ", ".join(sorted(item[0] for item in matches))
            raise ValueError(f"Kernel name is ambiguous; use a full Python ID for {selector}: {identities}")
        identity, path, definition = matches[0]
        if identity in seen:
            raise ValueError(f"Kernel selected more than once: {identity}")
        seen.add(identity)
        selected.append({
            "id": identity,
            "kernel": definition.name,
            "definition": path,
            "classification": "selected",
            "reason": "用户显式指定该本地 Triton JIT kernel 生成单算子用例。",
            "evidence": f"本地源码定义：{path}:{definition.lineno}",
        })
    return selected


def _add_local_sources(paths, *, selectors=(), ref=None):
    context, packages = {}, {}
    roots = []
    for path in paths:
        selected, locked = local_source(path, ref=ref, selectors=selectors)
        for name, source in selected.items():
            if name in context and context[name] != source:
                raise ValueError(f"Conflicting local source contents for {name}")
            context[name] = source
        for package, identity in locked.items():
            if package in packages and packages[package]["files"] != identity["files"]:
                raise ValueError(f"Package {package} was selected from different local source states")
            packages[package] = identity
            roots.append(identity.get("local_root"))
    return context, packages, sorted(set(x for x in roots if x))


def _write_context(destination, sources):
    for name, source in sources.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


def render_pipeline(root, state, execution_report=""):
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    version = ("源码：本地选择的 `vllm` / `vllm_ascend`" if state.get("selection_mode") == "manual"
               else f"版本：`{state['base']}` → `{state['target']}`")
    target = f"；NPU 目标：`{state['npu']}`" if state.get("npu") else ""
    lines = ["# Triton 单算子测试报告", "",
             f"{version}{target}。", "",
             f"流程状态：**{state['status']}**；当前/最后阶段：`{state['stage']}`。", "",
             f"生成用例目录：`{state['cases_path']}`。", "",
             "静态扫描和 AI 复核仍可能遗漏动态路径；数值检查通过仅代表所列输入。", ""]
    if state.get("error"):
        lines += ["失败原因：" + state["error"], ""]
    if state.get("source_drift"):
        drift = state["source_drift"]
        if isinstance(drift, dict):
            lines += ["运行源码与目标 tag 存在差异："
                      f"修改 {drift['changed_count']}、远端缺少 {drift['missing_count']}、"
                      f"远端新增 {drift['extra_count']} 个 Python 文件。", "",
                      f"远端 vLLM 包版本：`{drift.get('remote_package_version')}`；"
                      f"远端源码 HEAD：`{drift.get('remote_head') or '未检测到'}`；"
                      f"目标 commit：`{drift.get('target_commit')}`。", "",
                      "差异示例：" + ", ".join(f"`{p}`" for p in drift["examples"]), ""]
        else:
            lines += ["运行源码与目标 tag 存在差异：" + drift, ""]
    if state.get("ai"):
        ai = state["ai"]
        lines += ["AI："
                  f"`{ai.get('model')}`（{ai.get('model_source')}），"
                  f"推理强度 `{ai.get('reasoning_effort') or '由 Codex 配置决定'}`；"
                  "release review、逐算子 case 生成和失败分析是独立调用。", ""]
    if state.get("source_packages"):
        lines += ["源码锁：`" + state.get("source_lock", "") + "`。", ""]
        for name, package in state["source_packages"].items():
            lines += [f"- `{name}`：commit `{package['commit']}`；"
                      f"Python 文件 {package['file_count']}；"
                      f"生成时源码状态 `{'dirty' if package['dirty'] else 'clean/tag snapshot'}`。"]
        lines.append("")
    review = state.get("review", {})
    if review:
        lines += [review["summary"], "", "| 算子 | AI 判断 | 用例生成 | 原因 |", "|---|---|---|---|"]
        for row in review["operators"]:
            generation = state.get("generation", {}).get(row["id"], {})
            lines.append("| " + " | ".join(map(safe, [row["id"], row["classification"],
                         generation.get("status", "未生成"), generation.get("reason") or row["reason"]])) + " |")
        lines += ["", *["- 未解决：" + x for x in review["unresolved"]], ""]
    if execution_report:
        lines += [execution_report, ""]
    if state.get("diagnosis"):
        lines += ["## AI 失败分析", "", state["diagnosis"], ""]
    if state.get("correctness_not_checked"):
        lines += [f"数值正确性未检查的场景：{state['correctness_not_checked']}；"
                  "这些场景只证明输入构造、编译和运行状态。", ""]
    logs = sorted((root / "logs").rglob("*.log")) if (root / "logs").exists() else []
    if logs:
        lines += ["## 完整失败日志", "", *[f"- [{p.name}]({p.relative_to(root).as_posix()})" for p in logs], ""]
    (root / "report.md").write_text("\n".join(lines))
    save_json(root / "workflow.json", state)


def _execute_workflow(repo, base, target, npu, config, output, *, scope="vllm/v1/worker/gpu",
                      codex=None, model=None, reasoning_effort=None, ai_timeout=1800,
                      device=None, warmup=10, rounds=100, timeout=600,
                      prepare_only=False, dry_run=False, ai_client=None,
                      cases_output=None, resume=False, scan_input=None, selected_kernels=None,
                      source_paths=(), source_ref=None, operation="pipeline"):
    selected_kernels = list(selected_kernels or [])
    source_paths = [Path(path).expanduser().resolve() for path in source_paths]
    if selected_kernels and not source_paths:
        raise ValueError("generate --kernel requires at least one local --source file or directory")
    if source_ref and not selected_kernels:
        raise ValueError("--ref is only supported with generate --kernel")
    target_config = load_target(config, npu) if npu else None
    configured_codex = resolve_codex(config, codex)
    device = device or (target_config.get("device", "npu:0") if target_config else "npu:0")
    if warmup < 0 or rounds < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Require warmup >= 0, rounds > 0 and a finite positive timeout")
    if not device.startswith("npu:") or not device[4:].isdigit():
        raise ValueError("device must be npu:<logical index>")
    root = Path(output).resolve()
    workflow = root / "workflow.json"
    prior = None
    if resume:
        if not workflow.is_file():
            raise ValueError(f"Cannot resume without {workflow}")
        prior = json.loads(workflow.read_text())
        expected_run = {"base": base, "target": target, "npu": npu,
                        "source_paths": [str(path) for path in source_paths],
                        "source_ref": source_ref}
        if selected_kernels or "selected_kernels" in prior:
            expected_run["selected_kernels"] = selected_kernels
        mismatched = [k for k, value in expected_run.items() if prior.get(k) != value]
        if mismatched:
            raise ValueError("Resume arguments differ from workflow.json: " + ", ".join(mismatched))
        if prior.get("status") in {"completed", "no_new_operators"}:
            raise ValueError(f"Pipeline is already {prior['status']}; nothing to resume")
        stored_cases = Path(prior.get("cases_path", root / "cases")).resolve()
        if cases_output is not None and Path(cases_output).resolve() != stored_cases:
            raise ValueError("--cases-output differs from the original workflow")
        cases_dir = stored_cases
    else:
        cases_dir = Path(cases_output).resolve() if cases_output is not None else root / "cases"
    if cases_dir == root or (cases_dir.is_relative_to(root) and
                             cases_dir.relative_to(root).parts[0] in {"adapters", "results", "logs"}):
        raise ValueError("--cases-output must be a separate case directory")
    if dry_run:
        ai_plan = describe_codex(configured_codex, model, ai_timeout, reasoning_effort)
        steps = (["read selected local sources", "resolve explicitly selected kernels", "skip release review"]
                 if selected_kernels else
                 ["verify reusable scan result" if scan_input else "scan",
                  "reuse AI review" if scan_input or (resume and prior.get("review"))
                  else "AI review with release skill", "read local generation sources"])
        plan = {"device": device,
            "steps": [*steps, "AI JSON cases with case skill", "validate",
                      "stop after generation" if prepare_only else
                      "verify locked remote sources, NPU run and failure analysis"],
            "AI": {**ai_plan, "command": [configured_codex, "exec"]},
            "output": str(root), "cases_output": str(cases_dir),
            "resume": resume,
            "note": "No AI request, SSH connection or NPU job started"}
        if npu:
            plan["npu"] = npu
        if base is not None:
            plan["base"] = base
        if target is not None:
            plan["target"] = target
        if source_paths:
            plan["sources"] = [str(path) for path in source_paths]
        if selected_kernels:
            plan["kernels"] = selected_kernels
            if source_ref:
                plan["ref"] = source_ref
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if not resume and root.exists() and any(root.iterdir()):
        raise ValueError(f"Pipeline output already exists: {root}; use a new output directory")
    if not resume and cases_dir.exists() and (not cases_dir.is_dir() or any(cases_dir.iterdir())):
        raise ValueError(f"Case output already exists: {cases_dir}; use a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    state = prior or {"schema_version": 3, "base": base, "target": target, "npu": npu,
                      "status": "running", "stage": operation, "operation": operation, "generation": {},
                      "cases_path": str(cases_dir), "scope": scope, "stage_history": [],
                      "selected_kernels": selected_kernels,
                      "source_paths": [str(path) for path in source_paths], "source_ref": source_ref,
                      "selection_mode": "manual" if selected_kernels else "release_scan"}
    state["operation"] = operation
    if resume:
        state["status"] = "running"
        state.pop("error", None)
        state.setdefault("resumes", []).append(datetime.now(timezone.utc).isoformat())
    execution_report = ""
    def persist():
        render_pipeline(root, state, execution_report)
    def stage(name, progress=None):
        state["stage"] = name
        state.setdefault("stage_history", []).append({"stage": name,
            "time": datetime.now(timezone.utc).isoformat()})
        prefix = f"[{operation} {progress}]" if progress else f"[{operation}]"
        print(f"{prefix} {name}", flush=True)
        persist()
    persist()
    try:
        ai = ai_client or Codex(configured_codex, model, ai_timeout, reasoning_effort)
        state["ai"] = (ai.description() if hasattr(ai, "description") else {
            "executable": configured_codex, "model": model or "test/configured AI",
            "model_source": "injected/configured",
            "reasoning_effort": reasoning_effort or "injected/configured",
            "reasoning_effort_source": "injected/configured",
            "timeout_seconds": ai_timeout})
        state["model"] = state["ai"]["model"]
        print(f"[{operation}] output={root}", flush=True)
        print(f"[{operation}] AI model={state['ai']['model']} ({state['ai']['model_source']}), "
              f"reasoning={state['ai']['reasoning_effort']} "
              f"({state['ai'].get('reasoning_effort_source', 'Codex config')}), "
              f"executable={state['ai']['executable']}", flush=True)
        with tempfile.TemporaryDirectory(prefix="kernel-tools-pipeline-") as tmp:
            workspace = Path(tmp)
            if selected_kernels:
                stage("Read selected local sources", "1/3")
                runtime_sources, package_locks, local_roots = _add_local_sources(
                    source_paths, selectors=selected_kernels, ref=source_ref)
                _write_context(workspace / "source", runtime_sources)
                stage("Resolve explicitly selected kernels from local sources", "2/3")
                operators = resolve_named_kernels(runtime_sources, selected_kernels)
                review = {"operators": operators, "unresolved": [],
                          "summary": f"用户显式选择 {len(operators)} 个本地 Triton JIT kernel；未执行版本差异扫描。"}
                state["review"] = review
                state["local_source_roots"] = local_roots
                incomplete = False
                commit = None
            else:
                stage("Verify scan result against exact tag sources" if operation == "generate"
                      else "Static scan and exact tag source extraction",
                      "1/3" if operation == "generate" else "1/6")
                delta = write_scan(repo, base, target, workspace / "scan", scope, announce=False)
                state["delta"] = delta
                before = json.loads((workspace / "scan/base.json").read_text())
                write_sources(repo, base, workspace / "base")
                commit, target_sources = write_sources(repo, target, workspace / "target")
                if scan_input:
                    scan_file, scan_document = load_scan_result(scan_input)
                    expected_scan = {
                        "base tag": (scan_document["base"]["tag"], base),
                        "base commit": (scan_document["base"]["commit"], delta["base"]["commit"]),
                        "target tag": (scan_document["target"]["tag"], target),
                        "target commit": (scan_document["target"]["commit"], delta["target"]["commit"]),
                        "scope": (scan_document.get("scope"), scope),
                    }
                    mismatch = [name for name, (actual_value, expected_value) in expected_scan.items()
                                if actual_value != expected_value]
                    if mismatch:
                        raise ValueError("Scan result differs from requested sources: " + ", ".join(mismatch))
                    review = scan_document["review"]
                    from .review import validate_review
                    validate_review(review, delta, target_sources, before)
                    state["review"] = review
                    state["review_policy_version"] = REVIEW_POLICY_VERSION
                    state["scan_source"] = str(scan_file)
                    if operation != "generate":
                        stage("Reuse independent scan result", "2/6")
                elif (resume and state.get("review") and
                        state.get("review_policy_version") == REVIEW_POLICY_VERSION):
                    review = state["review"]
                    from .review import validate_review
                    validate_review(review, delta, target_sources, before)
                    stage("Reuse completed AI release review", "2/6")
                else:
                    stage("AI release review with vllm-triton-release-scan", "2/6")
                    review = review_sources(ai, delta, target_sources, before, workspace,
                                            root / "logs/ai-review.log")
                    state["review"] = review
                    state["review_policy_version"] = REVIEW_POLICY_VERSION
                operators = [r for r in review["operators"] if r["classification"] == "new"]
                incomplete = bool(review["unresolved"] or any(
                    r["classification"] == "needs_review" for r in review["operators"]))
                stage("Read local generation sources",
                      "2/3" if operation == "generate" else "3/6")
                runtime_sources = dict(target_sources)
                package_locks = {"vllm": package_lock(
                    "vllm", commit, target_sources, ref=target, local_root=repo)}
                extra_sources, extra_locks, local_roots = _add_local_sources(source_paths)
                for name, source in extra_sources.items():
                    if name in runtime_sources and runtime_sources[name] != source:
                        raise ValueError(f"Local --source conflicts with target-tag source: {name}")
                    runtime_sources[name] = source
                for package, identity in extra_locks.items():
                    if package in package_locks and package_locks[package]["files"] != identity["files"]:
                        raise ValueError(f"Local --source conflicts with locked {package} target sources")
                    package_locks[package] = identity
                _write_context(workspace / "source", extra_sources)
                state["local_source_roots"] = [str(Path(repo).resolve()), *local_roots]
            source_lock = make_source_lock(package_locks)
            state["source_lock"] = source_lock["fingerprint"]
            state["source_packages"] = {name: {
                "commit": package["commit"], "dirty": package["dirty"],
                "file_count": len(package["files"]),
            } for name, package in source_lock["packages"].items()}
            save_json(root / "source-lock.json", source_lock)
            (workspace / "source").mkdir(exist_ok=True)
            if cases_dir.parent != root:
                save_json(cases_dir / "source-lock.json", source_lock)
            operator_ids = {r["id"] for r in operators}
            # Old pipeline revisions generated per-run adapter modules. They are
            # incompatible with the JSON-only case policy and must not leak into
            # a resumed run or keep out-of-scope operators executable.
            for identity in list(state.get("generation", {})):
                saved = state["generation"][identity]
                if identity in operator_ids and saved.get("case_policy_version") == CASE_POLICY_VERSION:
                    continue
                (cases_dir / (file_key(identity) + ".json")).unlink(missing_ok=True)
                legacy_module = "kt_generated_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
                (root / "adapters" / (legacy_module + ".py")).unlink(missing_ok=True)
                state["generation"].pop(identity)
            if (root / "adapters").is_dir() and not any((root / "adapters").iterdir()):
                (root / "adapters").rmdir()
            if not operators:
                state["status"] = "needs_review" if incomplete else "no_new_operators"
                persist()
                return 1 if incomplete else 0
            cases = []
            for index, operator in enumerate(operators, 1):
                key = file_key(operator["id"])
                saved = state.get("generation", {}).get(operator["id"], {})
                case_file = cases_dir / (key + ".json")
                if (resume and saved.get("status") == "ready" and
                        saved.get("case_policy_version") == CASE_POLICY_VERSION and case_file.is_file()):
                    generated = json.loads(case_file.read_text())
                    fingerprints = {c.get("source", {}).get("source_lock") for c in generated}
                    if fingerprints != {source_lock["fingerprint"]}:
                        raise ValueError(f"Saved cases for {operator['kernel']} use a different local source "
                                         "lock; start a new output directory")
                    bundle = {"status": "ready", "reason": saved.get("reason", ""),
                              "analysis": saved.get("analysis", ""),
                              "cases_json": json.dumps(generated)}
                    validate_bundle(bundle, operator, runtime_sources)
                    cases.extend(generated)
                    stage(f"Reuse cases {index}/{len(operators)}: {operator['kernel']}",
                          "3/3" if operation == "generate" else "4/6")
                    continue
                stage(f"Generate cases {index}/{len(operators)}: {operator['kernel']}",
                      "3/3" if operation == "generate" else "4/6")
                source_guidance = (
                    "Read source/ as the authoritative selected LOCAL sources. "
                    if selected_kernels else
                    "Read target/ for exact upstream target-tag sources and source/ for additional LOCAL binding sources. "
                )
                task = ("Generate multi-scenario cases for this operator:\n" + json.dumps(operator, ensure_ascii=False)
                    + "\n" + source_guidance +
                    "The controller records a source lock outside this AI workspace; do not contact or inspect an NPU host. "
                    "Resolve Ascend patches and imports only from supplied local sources. Explain the selected "
                    "binding, shape/index/stride/pointer/alias/reset contract, coverage and measurement scope in analysis. "
                    "Return cases_json as a JSON array string. Every case must have "
                    f"kernel={operator['kernel']!r} and a scenario description. Use only the JSON materializer "
                    "capabilities documented in the case format and point target at the real runtime Triton kernel. "
                    "Every case directly launches target[grid](...); do not emit mode or wrapper fields. "
                    "Do not generate Python adapters, helper wrappers, references or checkers; all generated cases "
                    "must omit check and will report correctness=not_checked. If legal inputs "
                    "cannot be expressed by the framework, return blocked with a specific missing framework "
                    "capability instead of generating auxiliary code or simplifying the operator. "
                    "Include smoke, typical and relevant boundaries based on actual branches, not arbitrary "
                    "Cartesian products. Avoid stress/OOM sizes; all cases should fit the selected NPU. "
                    "No NPU execution at this stage.\nCase format:\n" + (resource_root() / "docs/cases.md").read_text())
                bundle = None
                for attempt in range(2):
                    try:
                        bundle = ai.ask(instructions("vllm-ascend-kernel-cases", task), CASE_SCHEMA,
                            workspace=workspace, failure_log=root / "logs" / f"ai-cases-{key}-{attempt + 1}.log",
                            label=f"case generation {index}/{len(operators)}: {operator['kernel']} "
                                  "(vllm-ascend-kernel-cases)")
                        generated = validate_bundle(bundle, operator, runtime_sources)
                        break
                    except (ValueError, OSError) as error:
                        log = root / "logs" / f"generation-{key}-{attempt + 1}.log"
                        log.parent.mkdir(parents=True, exist_ok=True)
                        log.write_text(str(error) + "\n" + (json.dumps(bundle, ensure_ascii=False, indent=2) if bundle else ""))
                        if attempt == 1:
                            generated = []
                            bundle = {"status": "blocked", "reason": str(error), "analysis": ""}
                        else:
                            task += "\nPrevious generation failed validation; correct this error: " + str(error)
                state["generation"][operator["id"]] = {
                    **{k: bundle[k] for k in ("status", "reason", "analysis")},
                    "case_policy_version": CASE_POLICY_VERSION,
                }
                if not generated:
                    incomplete = True
                    persist()
                    continue
                for case in generated:
                    source = {"operator_id": operator["id"],
                              "source_lock": source_lock["fingerprint"]}
                    if commit:
                        source["target_commit"] = commit
                    else:
                        package = operator["definition"].split("/", 1)[0]
                        revision = source_lock["packages"].get(package, {})
                        source.update(package=package, revision=revision.get("commit"),
                                      dirty=revision.get("dirty"))
                    case["source"] = source
                save_json(cases_dir / (key + ".json"), generated)
                cases.extend(generated)
                persist()
            if prepare_only or not cases:
                state["status"] = "needs_review" if incomplete or not cases else "prepared"
                persist()
                return 1 if incomplete or not cases else 0
            validate_cases(cases)
            stage(f"Verify locked sources and execute {len(cases)} generated cases on {npu}", "5/6")
            execution = workspace / "execution"
            code = run_remote(target_config, cases, execution, device=device, warmup=warmup,
                              rounds=rounds, timeout=timeout,
                              source_lock=source_lock)
            execution_report = (execution / "report.md").read_text()
            for name in ("results", "logs"):
                if (execution / name).exists():
                    if name == "results" and (root / name).exists():
                        shutil.rmtree(root / name)
                    shutil.copytree(execution / name, root / name, dirs_exist_ok=True)
            # Keep the original generated cases; execution retains the same inputs remotely.
            results = [json.loads(p.read_text()) for p in (root / "results").glob("*.json")]
            identities = [(d["kernel"], r["name"]) for d in results for r in d["scenarios"]]
            expected_cases = {(c["kernel"], c["name"]) for c in cases}
            if set(identities) != expected_cases or len(identities) != len(expected_cases):
                raise ValueError("Downloaded results do not cover exactly the generated cases")
            failures = [r for d in results for r in d["scenarios"] if r["status"] != "success"]
            state["correctness_not_checked"] = sum(
                r.get("correctness") != "passed" for d in results for r in d["scenarios"]
            )
            if code or failures:
                stage("AI failure analysis with vllm-triton-remote-benchmark", "6/6")
                shutil.copytree(root / "results", workspace / "results")
                if (root / "logs").exists():
                    shutil.copytree(root / "logs", workspace / "logs")
                shutil.copytree(cases_dir, workspace / "cases")
                try:
                    diagnosis = ai.ask(instructions("vllm-triton-remote-benchmark",
                        "Analyze failed/blocked cases from results/, full logs/, cases/, target/ and source/. "
                        "Do not run tests or modify files. Report observed errors separately from inferred cause, "
                        "cite concrete log/source paths and evidence, state uncertainty, and distinguish input, "
                        "reference, import, compile, runtime, correctness and environment failures. Do not change "
                        "measured statuses or claim a kernel bug solely from an AI-generated reference mismatch. "
                        "Give a concise Chinese failure explanation for the final report."), DIAGNOSIS_SCHEMA,
                        workspace=workspace, failure_log=root / "logs/ai-diagnosis.log",
                        label="failure analysis (vllm-triton-remote-benchmark)")
                    state["diagnosis"] = diagnosis["analysis"]
                except ValueError as error:
                    state["diagnosis"] = "AI 分析未完成：" + str(error)
                    incomplete = True
            state["status"] = "failed" if code or failures else "needs_review" if incomplete else "completed"
            stage("Finished", "6/6")
            return 1 if code or failures or incomplete else 0
    except (Exception, KeyboardInterrupt) as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "blocked"
        state["error"] = f"{type(error).__name__}: {error}"
        (root / "logs").mkdir(exist_ok=True)
        (root / "logs/pipeline.log").write_text(traceback.format_exc())
        persist()
        print(f"{operation.capitalize()} stopped: {error}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        print(f"Report: {root / 'report.md'}", flush=True)


def pipeline(repo, base, target, npu, config, output, **kwargs):
    """Run scan, case generation and NPU execution as one workflow."""
    return _execute_workflow(repo, base, target, npu, config, output,
                             operation="pipeline", **kwargs)


def generate_from_scan(scan_input, repo, output, *, sources=(), config=None, **kwargs):
    """Generate reusable direct-Triton cases from a completed scan artifact."""
    _, document = load_scan_result(scan_input)
    return _execute_workflow(repo, document["base"]["tag"], document["target"]["tag"],
                             None, config, output, scope=document["scope"], source_paths=sources,
                             prepare_only=True, scan_input=scan_input,
                             operation="generate", **kwargs)


def generate_from_kernels(kernels, sources, output, *, ref=None, config=None, **kwargs):
    """Generate cases for explicitly selected kernels from bounded local sources."""
    if not kernels or not all(isinstance(name, str) and name.strip() for name in kernels):
        raise ValueError("At least one nonempty --kernel is required")
    return _execute_workflow(None, None, None, None, config, output,
                             prepare_only=True, selected_kernels=kernels,
                             source_paths=sources, source_ref=ref,
                             operation="generate", **kwargs)
