from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import __version__
from .cases import load_cases, select_cases
from .common import save_json


def template():
    return {"ai": {"codex": "codex"}, "targets": {
        "npu160": {"host": "root@192.168.13.160", "container": "vllm_liam",
                   "python": "python3", "cwd": "/vllm-workspace/vllm-ascend",
                   "pythonpath": ["/vllm-workspace/vllm", "/vllm-workspace/vllm-ascend"],
                   "setup": "/usr/local/Ascend/ascend-toolkit/set_env.sh",
                   "device": "npu:0", "result_root": "/home/lingmutian/triton_kernel"},
        "npu165": {"host": "root@192.168.13.165", "container": "vllm_lmt",
                   "python": "python3", "cwd": "/home/lingmutian/code/vllm-ascend",
                   "pythonpath": ["/home/lingmutian/code/vllm", "/home/lingmutian/code/vllm-ascend"],
                   "setup": "/usr/local/Ascend/ascend-toolkit/set_env.sh",
                   "device": "npu:0", "result_root": "/home/lingmutian/triton_kernel"}}}


def parser():
    root = argparse.ArgumentParser(prog="kernel-tools", description="vLLM Triton 版本检查与 Ascend 单算子工具")
    root.add_argument("--version", action="version", version=__version__)
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="生成可编辑的远端配置，不修改已有文件")
    init.add_argument("--output", type=Path, default=Path("kernel-tools.json"))
    doctor = sub.add_parser("doctor", help="检查本地或远端 NPU 环境及 import 路径")
    doctor.add_argument("--npu", "--target", dest="npu",
                        help="kernel-tools.json 中的 SSH/Docker 目标")
    doctor.add_argument("--config", type=Path, default=Path("kernel-tools.json"))
    doctor.add_argument("--device")
    doctor.add_argument("--dry-run", action="store_true")
    cases = sub.add_parser("cases", help="查看或静态校验 JSON/JSONL case")
    cases.add_argument("action", choices=["list", "validate"])
    cases.add_argument("path", type=Path)
    cases.add_argument("--kernel")
    cases.add_argument("--case-name")
    run = sub.add_parser("run", help="逐 case 运行；仅失败保留完整日志")
    run.add_argument("path", type=Path, help="case 文件、cases 目录或 generate 输出目录")
    run.add_argument("--kernel")
    run.add_argument("--case-name")
    run.add_argument("--npu", "--target", dest="npu",
                     help="kernel-tools.json 中的 SSH/Docker 目标")
    run.add_argument("--config", type=Path, default=Path("kernel-tools.json"))
    run.add_argument("--cwd", type=Path, default=Path.cwd())
    run.add_argument("--pythonpath", type=Path, action="append", default=[])
    run.add_argument("--device")
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--rounds", type=int, default=100)
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--output", type=Path)
    run.add_argument("--resume", action="store_true", help="从原 --output 继续；远端会复用已记录的远端结果目录")
    run.add_argument("--expected-source-fingerprint", help=argparse.SUPPRESS)
    run.add_argument("--source-lock", type=Path, help=argparse.SUPPRESS)
    run.add_argument("--dry-run", action="store_true", help="只显示计划，不连接设备")
    verify = sub.add_parser("verify", help="核对生成时的本地源码与 NPU 实际 import 源码")
    verify.add_argument("path", type=Path, help="生成目录、cases 目录或 source-lock.json")
    verify.add_argument("--npu", help="kernel-tools.json 中的 SSH/Docker 目标；省略时检查当前环境")
    verify.add_argument("--config", type=Path, default=Path("kernel-tools.json"))
    verify.add_argument("--device")
    verify.add_argument("--dry-run", action="store_true")
    scan = sub.add_parser("scan", help="调用 AI 和 release-scan skill，核实两个 tag 之间的新增算子")
    scan.add_argument("--repo", type=Path, required=True)
    scan.add_argument("--base", required=True)
    scan.add_argument("--target", required=True)
    scan.add_argument("--scope", default="vllm/v1/worker/gpu")
    scan.add_argument("--output", type=Path, help="默认写入用户数据目录，不写入工具仓库")
    scan.add_argument("--config", type=Path, default=Path("kernel-tools.json"), help="可选；仅读取 ai 配置，不需要 NPU 配置")
    scan.add_argument("--codex", help="Codex CLI 路径；默认读配置 ai.codex 或 PATH")
    scan.add_argument("--model", help="默认使用 Codex 配置的模型")
    scan.add_argument("--reason", "--reasoning-effort", dest="reasoning_effort",
                      help="单次覆盖 Codex model_reasoning_effort，例如 low/medium/high/xhigh")
    scan.add_argument("--ai-timeout", type=float, default=1800)
    scan.add_argument("--resume", action="store_true", help="从原 --output 重新继续未完成的 AI 复核")
    generate = sub.add_parser("generate", help="从 scan 结果或手动指定 kernel 生成单算子 case")
    generate_source = generate.add_mutually_exclusive_group(required=True)
    generate_source.add_argument("--scan", type=Path, help="scan 输出目录或 review.json")
    generate_source.add_argument("--kernel", action="append", help="直接指定 kernel 名称或完整 Python ID；可重复")
    generate.add_argument("--repo", type=Path, help="--scan 模式要求的本地 vLLM 仓库")
    generate.add_argument("--source", type=Path, action="append", default=[],
                          help="本地源码文件或目录；可重复。--kernel 模式至少需要一个")
    generate.add_argument("--ref", help="--kernel 模式读取的 Git revision；默认读取当前 worktree")
    generate.add_argument("--config", type=Path, default=Path("kernel-tools.json"))
    generate.add_argument("--output", type=Path)
    generate.add_argument("--cases-output", type=Path, help="默认 <本次生成目录>/cases")
    generate.add_argument("--codex", help="Codex CLI 可执行文件；默认读配置 ai.codex 或 PATH")
    generate.add_argument("--model", help="AI 模型；默认使用本机 Codex 配置")
    generate.add_argument("--reason", "--reasoning-effort", dest="reasoning_effort",
                          help="单次覆盖 Codex model_reasoning_effort")
    generate.add_argument("--ai-timeout", type=float, default=1800)
    generate.add_argument("--resume", action="store_true", help="从原 --output 继续未完成的 case 生成")
    generate.add_argument("--dry-run", action="store_true", help="只显示计划，不调用 AI、不连接远端")
    flow = sub.add_parser("pipeline", help="扫描新增算子 → 调用 AI 生成用例 → NPU 执行与报告")
    flow.add_argument("--repo", type=Path, required=True)
    flow.add_argument("--base", required=True, help="基线 vLLM tag")
    flow.add_argument("--target", required=True, help="待测 vLLM tag")
    flow.add_argument("--npu", required=True, help="kernel-tools.json 中的远端目标，如 npu165")
    flow.add_argument("--source", type=Path, action="append", default=[],
                      help="生成 case 时额外读取的本地源码仓库或目录；可重复")
    flow.add_argument("--config", type=Path, default=Path("kernel-tools.json"))
    flow.add_argument("--output", type=Path)
    flow.add_argument("--cases-output", type=Path, help="生成用例的目录；默认 <本次运行目录>/cases")
    flow.add_argument("--scope", default="vllm/v1/worker/gpu")
    flow.add_argument("--codex", help="Codex CLI 可执行文件；默认读配置 ai.codex 或 PATH")
    flow.add_argument("--model", help="AI 模型；默认使用本机 Codex 配置")
    flow.add_argument("--reason", "--reasoning-effort", dest="reasoning_effort",
                      help="单次覆盖 Codex model_reasoning_effort，例如 low/medium/high/xhigh")
    flow.add_argument("--ai-timeout", type=float, default=1800, help="每次 AI 调用的秒数上限")
    flow.add_argument("--device")
    flow.add_argument("--warmup", type=int, default=10)
    flow.add_argument("--rounds", type=int, default=100)
    flow.add_argument("--timeout", type=float, default=600, help="每个 NPU case 的秒数上限")
    flow.add_argument("--prepare-only", action="store_true", help="从本地源码生成 case，暂不校验或运行 NPU case")
    flow.add_argument("--resume", action="store_true", help="从同一 --output 的 workflow.json 继续，复用已完成的 AI 复核和用例")
    flow.add_argument("--dry-run", action="store_true", help="仅显示流程，不调用 AI、不连接远端")
    report = sub.add_parser("report", help="从 results/*.json 重建报告")
    report.add_argument("path", type=Path)
    return root


def _source_lock_path(path):
    path = Path(path).expanduser().resolve()
    candidates = ([path] if path.is_file() and path.name == "source-lock.json" else [])
    if path.is_dir():
        candidates += [path / "source-lock.json", path.parent / "source-lock.json"]
    else:
        candidates += [path.parent / "source-lock.json", path.parent.parent / "source-lock.json"]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _case_path(path):
    path = Path(path).expanduser().resolve()
    return path / "cases" if path.is_dir() and (path / "cases").is_dir() else path


def _load_source_lock(path, *, required=False):
    lock_path = _source_lock_path(path)
    if not lock_path:
        if required:
            raise ValueError(f"source-lock.json not found for {Path(path).expanduser().resolve()}")
        return None
    from .source_lock import validate_source_lock
    lock = json.loads(lock_path.read_text())
    validate_source_lock(lock)
    return lock


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            if args.output.exists():
                raise ValueError(f"Already exists: {args.output}; edit this config directly")
            save_json(args.output, template())
            print(f"Created {args.output}. Verify host/container/source paths, then run doctor --npu npu160.")
        elif args.command == "cases":
            cases = select_cases(load_cases(args.path), args.kernel, args.case_name)
            if args.action == "list":
                for case in cases:
                    print(f"{case['kernel']} / {case['name']} {case['target']}")
            else:
                print(f"PASS: {len(cases)} cases (structure only; semantic/NPU checks not performed)")
        elif args.command == "doctor":
            if args.npu:
                from .remote import load_target, run_remote
                target = load_target(args.config, args.npu)
                return run_remote(target, [], Path("artifacts/doctor"), doctor=True,
                                  device=args.device or target.get("device", "npu:0"), dry_run=args.dry_run)
            if args.dry_run:
                print("Local environment probe; no kernel launch. Remove --dry-run to execute.")
            else:
                from .probe import environment
                value = environment(args.device or "npu:0")
                print(json.dumps(value, ensure_ascii=False, indent=2))
                return 0 if value.get("npu_available") else 1
        elif args.command == "run":
            from .runner import new_run_path, run_suite
            case_path = _case_path(args.path)
            cases = select_cases(load_cases(case_path), args.kernel, args.case_name)
            source_lock = _load_source_lock(args.source_lock or args.path)
            locked = {case.get("source", {}).get("source_lock") for case in cases
                      if case.get("source", {}).get("source_lock")}
            if len(locked) > 1:
                raise ValueError("Selected cases were generated from different local source locks")
            expected_lock = next(iter(locked), None)
            if expected_lock and not source_lock:
                raise ValueError("Generated cases require source-lock.json; run from their generation/cases directory")
            if source_lock and expected_lock and source_lock["fingerprint"] != expected_lock:
                raise ValueError("source-lock.json differs from the selected generated cases")
            fingerprints = {case.get("source", {}).get("runtime_fingerprint") for case in cases
                            if case.get("source", {}).get("runtime_fingerprint")}
            if len(fingerprints) > 1:
                raise ValueError("Selected cases were generated from different runtime source snapshots")
            generated_fingerprint = next(iter(fingerprints), None)
            if (args.expected_source_fingerprint and generated_fingerprint and
                    args.expected_source_fingerprint != generated_fingerprint):
                raise ValueError("Explicit source fingerprint differs from generated cases")
            expected_fingerprint = args.expected_source_fingerprint or generated_fingerprint
            if args.resume and not args.output:
                raise ValueError("--resume requires the original --output directory")
            output = args.output or new_run_path()
            if args.npu:
                from .remote import load_target, run_remote
                target = load_target(args.config, args.npu)
                return run_remote(target, cases, output, device=args.device or target.get("device", "npu:0"),
                                  warmup=args.warmup, rounds=args.rounds, timeout=args.timeout, dry_run=args.dry_run,
                                  expected_identity=expected_fingerprint, source_lock=source_lock,
                                  resume=args.resume)
            if args.dry_run:
                print(json.dumps({"cases": [f"{c['kernel']}/{c['name']}" for c in cases],
                                  "cwd": str(args.cwd.resolve()), "device": args.device or "npu:0",
                                  "output": str(output), "warmup": args.warmup, "rounds": args.rounds,
                                  "timeout_seconds": args.timeout}, ensure_ascii=False, indent=2))
            else:
                return run_suite(cases, cwd=args.cwd, output=output, device=args.device or "npu:0",
                                 warmup=args.warmup, rounds=args.rounds, timeout=args.timeout,
                                 pythonpath=[p.resolve() for p in args.pythonpath], resume=args.resume,
                                 expected_identity=expected_fingerprint, source_lock=source_lock)
        elif args.command == "verify":
            source_lock = _load_source_lock(args.path, required=True)
            if args.npu:
                from .remote import load_target, verify_remote
                target = load_target(args.config, args.npu)
                result = verify_remote(target, source_lock,
                                       device=args.device or target.get("device", "npu:0"),
                                       dry_run=args.dry_run)
            elif args.dry_run:
                result = {"status": "planned", "fingerprint": source_lock["fingerprint"],
                          "note": "Current Python imports were not inspected"}
            else:
                from .probe import environment
                from .source_lock import verify_source_lock
                result = verify_source_lock(source_lock, environment(args.device or "npu:0"))
            print(json.dumps(result, ensure_ascii=False))
        elif args.command == "scan":
            from .review import scan_with_ai
            from .runner import new_run_path
            if args.resume and not args.output:
                raise ValueError("scan --resume requires the original --output directory")
            return scan_with_ai(args.repo, args.base, args.target, args.output or new_run_path("scans"), scope=args.scope,
                                config=args.config, codex=args.codex, model=args.model,
                                reasoning_effort=args.reasoning_effort, ai_timeout=args.ai_timeout,
                                resume=args.resume)
        elif args.command == "generate":
            from .pipeline import generate_from_kernels, generate_from_scan
            from .runner import new_run_path
            if args.resume and not args.output:
                raise ValueError("generate --resume requires the original --output directory")
            common = dict(codex=args.codex, model=args.model,
                          reasoning_effort=args.reasoning_effort,
                          ai_timeout=args.ai_timeout,
                          cases_output=args.cases_output, resume=args.resume,
                          dry_run=args.dry_run)
            output = args.output or new_run_path("generations")
            if args.scan:
                if not args.repo:
                    raise ValueError("generate --scan requires --repo")
                if args.ref:
                    raise ValueError("generate --scan already uses its target tag; --ref is only for --kernel")
                return generate_from_scan(args.scan, args.repo, output, sources=args.source,
                                          config=args.config, **common)
            if args.repo:
                raise ValueError("generate --kernel uses --source; do not specify --repo")
            if not args.source:
                raise ValueError("generate --kernel requires at least one local --source file or directory")
            return generate_from_kernels(args.kernel, args.source, output, ref=args.ref,
                                         config=args.config, **common)
        elif args.command == "pipeline":
            from .pipeline import pipeline
            from .runner import new_run_path
            if args.resume and not args.output:
                raise ValueError("pipeline --resume requires the original --output directory")
            return pipeline(args.repo, args.base, args.target, args.npu, args.config,
                            args.output or new_run_path(), scope=args.scope, codex=args.codex,
                            model=args.model, reasoning_effort=args.reasoning_effort,
                            ai_timeout=args.ai_timeout, device=args.device,
                            warmup=args.warmup, rounds=args.rounds, timeout=args.timeout,
                            prepare_only=args.prepare_only, dry_run=args.dry_run,
                            cases_output=args.cases_output, resume=args.resume,
                            source_paths=args.source)
        elif args.command == "report":
            from .runner import render_report
            if not list((args.path / "results").glob("*.json")):
                raise ValueError("No result JSON files found")
            render_report(args.path)
            print(args.path / "report.md")
        return 0
    except (ValueError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Inspect existing results/processes before resuming.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
