"""Scan → AI review/cases → deterministic remote run → evidence-based report."""
from __future__ import annotations

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
from .remote import fetch_snapshot, load_target, run_remote
from .scan import Module, write_scan
from .review import (REVIEW_POLICY_VERSION, instructions, load_scan_result,
                     resource_root, review_sources, write_sources)


CASE_POLICY_VERSION = 3


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
                      cases_output=None, resume=False, scan_input=None,
                      operation="pipeline"):
    target_config = load_target(config, npu)
    configured_codex = resolve_codex(config, codex)
    device = device or target_config.get("device", "npu:0")
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
        expected_run = {"base": base, "target": target, "npu": npu}
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
        print(json.dumps({"base": base, "target": target, "npu": npu, "device": device,
            "steps": ["verify reusable scan result" if scan_input else "scan",
                      "reuse AI review" if scan_input or (resume and prior.get("review")) else "AI review with release skill", "read actual NPU sources",
                      "AI JSON cases with case skill", "validate",
                      "stop after generation" if prepare_only else "NPU run and failure analysis"],
            "AI": {**ai_plan, "command": [configured_codex, "exec"]},
            "output": str(root), "cases_output": str(cases_dir),
            "resume": resume,
            "note": "No AI request, SSH connection or NPU job started"}, ensure_ascii=False, indent=2))
        return 0
    if not resume and root.exists() and any(root.iterdir()):
        raise ValueError(f"Pipeline output already exists: {root}; use a new output directory")
    if not resume and cases_dir.exists() and (not cases_dir.is_dir() or any(cases_dir.iterdir())):
        raise ValueError(f"Case output already exists: {cases_dir}; use a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    state = prior or {"schema_version": 2, "base": base, "target": target, "npu": npu,
                      "status": "running", "stage": operation, "operation": operation, "generation": {},
                      "cases_path": str(cases_dir), "scope": scope, "stage_history": []}
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
            stage("Verify scan result against exact tag sources" if operation == "generate"
                  else "Static scan and exact tag source extraction",
                  "1/3" if operation == "generate" else "1/6")
            delta = write_scan(repo, base, target, workspace / "scan", scope, announce=False)
            state["delta"] = delta
            before = json.loads((workspace / "scan/base.json").read_text())
            after = json.loads((workspace / "scan/target.json").read_text())
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
            incomplete = bool(review["unresolved"] or any(r["classification"] == "needs_review" for r in review["operators"]))
            if not operators:
                state["status"] = "needs_review" if incomplete else "no_new_operators"
                persist()
                return 1 if incomplete else 0
            stage(f"Read and verify actual sources on {npu}",
                  "2/3" if operation == "generate" else "3/6")
            snapshot = fetch_snapshot(target_config, workspace / "runtime", device=device,
                                      failure_log=root / "logs/snapshot.log")
            state["runtime"] = snapshot
            expected = {p: hashlib.sha256(s.encode()).hexdigest() for p, s in target_sources.items()}
            actual = {p: h for p, h in snapshot["identity"]["files"].items() if p.startswith("vllm/")}
            if expected != actual:
                changed = sorted(p for p in expected.keys() & actual.keys() if expected[p] != actual[p])
                missing = sorted(expected.keys() - actual.keys())
                extra = sorted(actual.keys() - expected.keys())
                drift = changed + missing + extra
                state["source_drift"] = {
                    "changed_count": len(changed), "missing_count": len(missing),
                    "extra_count": len(extra), "examples": drift[:20],
                    "target_commit": commit,
                    "remote_package_version": snapshot["environment"].get("packages", {}).get("vllm"),
                    "remote_head": snapshot["environment"].get("source_revisions", {}).get("vllm", {}).get("head"),
                    "remote_fingerprint": snapshot["identity"]["fingerprint"],
                }
                raise ValueError(f"Remote vLLM differs from {target}: {len(changed)} changed, "
                                 f"{len(missing)} missing and {len(extra)} extra Python files. "
                                 "The remote imported source must match the target exactly; "
                                 "align the remote environment, then resume this output with --resume")
            else:
                state.pop("source_drift", None)
            if not snapshot["environment"].get("npu_available") and not prepare_only:
                raise ValueError("NPU unavailable: " + snapshot["environment"].get("npu_error", "environment probe failed"))
            runtime_sources = {p: (workspace / "runtime" / p).read_text()
                               for p in snapshot["identity"]["files"]}
            cases = []
            for index, operator in enumerate(operators, 1):
                key = file_key(operator["id"])
                saved = state.get("generation", {}).get(operator["id"], {})
                case_file = cases_dir / (key + ".json")
                if (resume and saved.get("status") == "ready" and
                        saved.get("case_policy_version") == CASE_POLICY_VERSION and case_file.is_file()):
                    generated = json.loads(case_file.read_text())
                    fingerprints = {c.get("source", {}).get("runtime_fingerprint") for c in generated}
                    if fingerprints != {snapshot["identity"]["fingerprint"]}:
                        raise ValueError(f"Saved cases for {operator['kernel']} use a different remote source "
                                         "fingerprint; start a new output directory")
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
                task = ("Generate multi-scenario cases for this operator:\n" + json.dumps(operator, ensure_ascii=False)
                    + "\nRead target/ for intended upstream behavior and runtime/ for ACTUAL imported NPU sources. "
                    "runtime/snapshot.json records versions. Resolve Ascend patches and imports from runtime; "
                    "do not assume the upstream implementation is the effective replacement. Explain the selected "
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
                    case["source"] = {"target_commit": commit, "operator_id": operator["id"],
                                      "runtime_fingerprint": snapshot["identity"]["fingerprint"]}
                save_json(cases_dir / (key + ".json"), generated)
                cases.extend(generated)
                persist()
            if prepare_only or not cases:
                state["status"] = "needs_review" if incomplete or not cases else "prepared"
                persist()
                return 1 if incomplete or not cases else 0
            validate_cases(cases)
            stage(f"Execute {len(cases)} generated cases on {npu}", "5/6")
            execution = workspace / "execution"
            code = run_remote(target_config, cases, execution, device=device, warmup=warmup,
                              rounds=rounds, timeout=timeout,
                              expected_identity=snapshot["identity"]["fingerprint"])
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
                        "Analyze failed/blocked cases from results/, full logs/, cases/ and runtime/. "
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


def generate_from_scan(scan_input, repo, npu, config, output, **kwargs):
    """Generate reusable direct-Triton cases from a completed scan artifact."""
    _, document = load_scan_result(scan_input)
    return _execute_workflow(repo, document["base"]["tag"], document["target"]["tag"],
                             npu, config, output, scope=document["scope"],
                             prepare_only=True, scan_input=scan_input,
                             operation="generate", **kwargs)
