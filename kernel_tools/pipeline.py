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
from pathlib import Path

from .ai import CASE_SCHEMA, DIAGNOSIS_SCHEMA, Codex, resolve_codex
from .cases import validate_cases
from .common import file_key, save_json
from .remote import fetch_snapshot, load_target, run_remote
from .scan import Module, write_scan
from .review import instructions, resource_root, write_sources, review_sources


def validate_bundle(bundle, operator, module, runtime_sources):
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
    tree = ast.parse(bundle["adapter_source"], filename=module + ".py")
    # Compile only: do not import or execute AI-generated code on the controller.
    compile(tree, module + ".py", "exec")
    definitions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for case in cases:
        if case["kernel"] != operator["kernel"]:
            raise ValueError("Generated case belongs to a different kernel")
        if not isinstance(case.get("scenario"), str) or not case["scenario"].strip():
            raise ValueError("Each generated case needs a scenario description")
        check = case.get("check", "")
        if not check.startswith(module + ":") or check.split(":", 1)[1] not in definitions:
            raise ValueError("Every automatic case requires a generated independent checker")
        wrapper_module, symbol = case["wrapper"].split(":", 1)
        if wrapper_module == module:
            if case["mode"] != "wrapper" or symbol not in definitions:
                raise ValueError("Generated adapters must expose a normal wrapper callable")
        else:
            path = wrapper_module.replace(".", "/") + ".py"
            if path not in runtime_sources:
                path = wrapper_module.replace(".", "/") + "/__init__.py"
            if path not in runtime_sources:
                raise ValueError(f"Case wrapper absent from actual NPU source snapshot: {case['wrapper']}")
            definition = Module(path, runtime_sources[path]).functions.get(wrapper_module + "." + symbol)
            if not definition or (case["mode"] == "triton" and not definition["jit"]):
                raise ValueError(f"Case wrapper cannot be resolved in NPU source: {case['wrapper']}")
    return cases


def render_pipeline(root, state, execution_report=""):
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# 新增 Triton 算子测试报告", "",
             f"版本：`{state['base']}` → `{state['target']}`；NPU 目标：`{state['npu']}`。", "",
             f"流程状态：**{state['status']}**；当前/最后阶段：`{state['stage']}`。", "",
             f"生成用例目录：`{state['cases_path']}`。", "",
             "静态扫描和 AI 复核仍可能遗漏动态路径；数值检查通过仅代表所列输入。", ""]
    if state.get("error"):
        lines += ["失败原因：" + state["error"], ""]
    if state.get("source_drift"):
        lines += ["运行源码与目标 tag 存在差异：" + state["source_drift"], ""]
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
    logs = sorted((root / "logs").rglob("*.log")) if (root / "logs").exists() else []
    if logs:
        lines += ["## 完整失败日志", "", *[f"- [{p.name}]({p.relative_to(root).as_posix()})" for p in logs], ""]
    (root / "report.md").write_text("\n".join(lines))
    save_json(root / "workflow.json", state)


def pipeline(repo, base, target, npu, config, output, *, scope="vllm/v1/worker/gpu",
             codex=None, model=None, ai_timeout=1800, device=None,
             warmup=10, rounds=100, timeout=600, prepare_only=False,
             allow_source_drift=False, dry_run=False, ai_client=None,
             cases_output=None):
    target_config = load_target(config, npu)
    configured_codex = resolve_codex(config, codex)
    device = device or target_config.get("device", "npu:0")
    if warmup < 0 or rounds < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Require warmup >= 0, rounds > 0 and a finite positive timeout")
    if not device.startswith("npu:") or not device[4:].isdigit():
        raise ValueError("device must be npu:<logical index>")
    root = Path(output).resolve()
    cases_dir = Path(cases_output).resolve() if cases_output is not None else root / "cases"
    if cases_dir == root or (cases_dir.is_relative_to(root) and
                             cases_dir.relative_to(root).parts[0] in {"adapters", "results", "logs"}):
        raise ValueError("--cases-output must be a separate case directory")
    if dry_run:
        print(json.dumps({"base": base, "target": target, "npu": npu, "device": device,
            "steps": ["scan", "AI review with release skill", "read actual NPU sources",
                      "AI cases/reference with case skill", "validate",
                      "stop after generation" if prepare_only else "NPU run and failure analysis"],
            "AI": {"command": [configured_codex, "exec"], "model": model or "Codex configured default"},
            "output": str(root), "cases_output": str(cases_dir),
            "note": "No AI request, SSH connection or NPU job started"}, ensure_ascii=False, indent=2))
        return 0
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Pipeline output already exists: {root}; use a new output directory")
    if cases_dir.exists() and (not cases_dir.is_dir() or any(cases_dir.iterdir())):
        raise ValueError(f"Case output already exists: {cases_dir}; use a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": 1, "base": base, "target": target, "npu": npu,
             "status": "running", "stage": "scan", "generation": {},
             "cases_path": str(cases_dir), "model": model or "Codex configured default"}
    execution_report = ""
    def persist():
        render_pipeline(root, state, execution_report)
    def stage(name):
        state["stage"] = name
        print(f"[pipeline] {name}", flush=True)
        persist()
    persist()
    try:
        ai = ai_client or Codex(configured_codex, model, ai_timeout)
        with tempfile.TemporaryDirectory(prefix="kernel-tools-pipeline-") as tmp:
            workspace = Path(tmp)
            delta = write_scan(repo, base, target, workspace / "scan", scope)
            state["delta"] = delta
            before = json.loads((workspace / "scan/base.json").read_text())
            after = json.loads((workspace / "scan/target.json").read_text())
            write_sources(repo, base, workspace / "base")
            commit, target_sources = write_sources(repo, target, workspace / "target")
            stage("AI review")
            review = review_sources(ai, delta, target_sources, before, workspace,
                                    root / "logs/ai-review.log")
            state["review"] = review
            operators = [r for r in review["operators"] if r["classification"] == "new"]
            incomplete = bool(review["unresolved"] or any(r["classification"] == "needs_review" for r in review["operators"]))
            if not operators:
                state["status"] = "needs_review" if incomplete else "no_new_operators"
                persist()
                return 1 if incomplete else 0
            stage("NPU source snapshot")
            snapshot = fetch_snapshot(target_config, workspace / "runtime", device=device,
                                      failure_log=root / "logs/snapshot.log")
            state["runtime"] = snapshot
            expected = {p: hashlib.sha256(s.encode()).hexdigest() for p, s in target_sources.items()}
            actual = {p: h for p, h in snapshot["identity"]["files"].items() if p.startswith("vllm/")}
            if expected != actual:
                drift = sorted(p for p in expected.keys() | actual.keys() if expected.get(p) != actual.get(p))
                state["source_drift"] = f"{len(drift)} Python files differ; examples: {', '.join(drift[:5])}"
                if not allow_source_drift:
                    raise ValueError("Remote vLLM differs from requested target tag. Select the matching environment, "
                                     "or explicitly use --allow-source-drift to test actual remote sources")
            if not snapshot["environment"].get("npu_available") and not prepare_only:
                raise ValueError("NPU unavailable: " + snapshot["environment"].get("npu_error", "environment probe failed"))
            runtime_sources = {p: (workspace / "runtime" / p).read_text()
                               for p in snapshot["identity"]["files"]}
            cases = []
            for operator in operators:
                key = file_key(operator["id"])
                module = "kt_generated_" + hashlib.sha256(operator["id"].encode()).hexdigest()[:16]
                stage("AI cases: " + operator["kernel"])
                task = ("Generate multi-scenario cases for this operator:\n" + json.dumps(operator, ensure_ascii=False)
                    + "\nRead target/ for intended upstream behavior and runtime/ for ACTUAL imported NPU sources. "
                    "runtime/snapshot.json records versions. Resolve Ascend patches and imports from runtime; "
                    "do not assume the upstream implementation is the effective replacement. Explain the selected "
                    "binding, shape/index/stride/pointer/alias/reset contract, coverage and measurement scope in analysis. "
                    "Return cases_json as a JSON array string and adapter_source as one Python module's source. "
                    f"The generated module name is {module}. Every case must have kernel={operator['kernel']!r}, "
                    f"a scenario description, and an independent check={module}:FUNCTION defined in adapter_source. "
                    "Use mode=triton to launch real upstream/Ascend kernels directly when possible. For complex "
                    "views/state expose a normal adapter wrapper in that generated module which calls the real "
                    "kernel; do not copy/reimplement the kernel under test or time a reference instead. Adapter "
                    "may import torch/triton/runtime packages; no shell, network, filesystem writes or production "
                    "changes. Checks run once after first invocation and before warmup; use independent CPU/Torch "
                    "reference and assert untouched regions where meaningful. If an independent reference or "
                    "legal inputs cannot be established, return blocked with reason instead of a dummy pass. "
                    "Include smoke, typical and relevant boundaries based on actual branches, not arbitrary "
                    "Cartesian products. Avoid stress/OOM sizes; all cases should fit the selected NPU. "
                    "No NPU execution at this stage.\nCase format:\n" + (resource_root() / "docs/cases.md").read_text())
                bundle = None
                for attempt in range(2):
                    try:
                        bundle = ai.ask(instructions("vllm-ascend-kernel-cases", task), CASE_SCHEMA,
                            workspace=workspace, failure_log=root / "logs" / f"ai-cases-{key}-{attempt + 1}.log")
                        generated = validate_bundle(bundle, operator, module, runtime_sources)
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
                state["generation"][operator["id"]] = {k: bundle[k] for k in ("status", "reason", "analysis")}
                if not generated:
                    incomplete = True
                    persist()
                    continue
                for case in generated:
                    case["source"] = {"target_commit": commit, "operator_id": operator["id"],
                                      "runtime_fingerprint": snapshot["identity"]["fingerprint"]}
                save_json(cases_dir / (key + ".json"), generated)
                path = root / "adapters" / (module + ".py")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(bundle["adapter_source"])
                cases.extend(generated)
                persist()
            if prepare_only or not cases:
                state["status"] = "needs_review" if incomplete or not cases else "prepared"
                persist()
                return 1 if incomplete or not cases else 0
            validate_cases(cases)
            stage("NPU execution")
            execution = workspace / "execution"
            code = run_remote(target_config, cases, execution, device=device, warmup=warmup,
                              rounds=rounds, timeout=timeout, assets=root / "adapters",
                              expected_identity=snapshot["identity"]["fingerprint"])
            execution_report = (execution / "report.md").read_text()
            for name in ("results", "logs"):
                if (execution / name).exists():
                    shutil.copytree(execution / name, root / name, dirs_exist_ok=True)
            # Keep the original generated cases and adapters; execution retains the same inputs remotely.
            results = [json.loads(p.read_text()) for p in (root / "results").glob("*.json")]
            identities = [(d["kernel"], r["name"]) for d in results for r in d["scenarios"]]
            expected_cases = {(c["kernel"], c["name"]) for c in cases}
            if set(identities) != expected_cases or len(identities) != len(expected_cases):
                raise ValueError("Downloaded results do not cover exactly the generated cases")
            failures = [r for d in results for r in d["scenarios"]
                        if r["status"] != "success" or r.get("correctness") != "passed"]
            if code or failures:
                stage("AI failure analysis")
                shutil.copytree(root / "results", workspace / "results")
                if (root / "logs").exists():
                    shutil.copytree(root / "logs", workspace / "logs")
                shutil.copytree(cases_dir, workspace / "cases")
                shutil.copytree(root / "adapters", workspace / "adapters")
                try:
                    diagnosis = ai.ask(instructions("vllm-triton-remote-benchmark",
                        "Analyze failed/blocked cases from results/, full logs/, cases/, adapters/ and runtime/. "
                        "Do not run tests or modify files. Report observed errors separately from inferred cause, "
                        "cite concrete log/source paths and evidence, state uncertainty, and distinguish input, "
                        "reference, import, compile, runtime, correctness and environment failures. Do not change "
                        "measured statuses or claim a kernel bug solely from an AI-generated reference mismatch. "
                        "Give a concise Chinese failure explanation for the final report."), DIAGNOSIS_SCHEMA,
                        workspace=workspace, failure_log=root / "logs/ai-diagnosis.log")
                    state["diagnosis"] = diagnosis["analysis"]
                except ValueError as error:
                    state["diagnosis"] = "AI 分析未完成：" + str(error)
                    incomplete = True
            state["status"] = "failed" if code or failures else "needs_review" if incomplete else "completed"
            stage("finished")
            return 1 if code or failures or incomplete else 0
    except (Exception, KeyboardInterrupt) as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "blocked"
        state["error"] = f"{type(error).__name__}: {error}"
        (root / "logs").mkdir(exist_ok=True)
        (root / "logs/pipeline.log").write_text(traceback.format_exc())
        persist()
        print(f"Pipeline stopped: {error}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        print(f"Report: {root / 'report.md'}", flush=True)
