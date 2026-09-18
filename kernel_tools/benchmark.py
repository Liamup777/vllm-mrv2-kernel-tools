#!/usr/bin/env python3
"""Benchmark one wrapper or raw Triton kernel with captured tensor inputs.

The input may be a JSON object, a JSON array, or JSON Lines. The callable is
provided separately as ``file.py:function``. NPU events measure every profiling
round independently.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import statistics
import sys
import traceback
import hashlib
import inspect
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# Derived from vllm-ascend kernel_test_frame; see NOTICE.
PHASE = "environment"
torch = None


def load_backend():
    global torch
    import torch as backend
    import torch_npu  # noqa: F401
    torch = backend


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load benchmark cases from JSON or JSONL."""
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Input file is empty: {path}")

    try:
        document = json.loads(content)
    except json.JSONDecodeError:
        document = [json.loads(line) for line in content.splitlines() if line.strip()]

    if isinstance(document, dict):
        cases = [document]
    elif isinstance(document, list) and all(isinstance(case, dict) for case in document):
        cases = document
    else:
        raise ValueError("Input must be a JSON object, an array of objects, or JSONL objects")

    if not cases:
        raise ValueError("Input does not contain any benchmark cases")
    return cases


def select_cases(
    cases: list[dict[str, Any]],
    kernel: str | None,
    max_cases: int | None = None,
    case_name: str | None = None,
) -> list[dict[str, Any]]:
    """Select kernel records and, optionally, one named case from a capture."""
    if max_cases is not None and max_cases <= 0:
        raise ValueError("max_cases must be a positive integer")
    selected = cases if kernel is None else [case for case in cases if case.get("kernel") == kernel]
    if not selected:
        raise ValueError(f"No input records found for kernel: {kernel}")
    if case_name is not None:
        selected = [case for case in selected if case.get("name") == case_name]
        if not selected:
            raise ValueError(f"No input record found for case name: {case_name}")
    return selected[:max_cases]


def resolve_wrapper(path: str) -> Callable[..., Any]:
    """Resolve ``file.py:function`` or ``module.path:function`` to a callable."""
    if ":" not in path:
        raise ValueError(f"Wrapper must use 'file.py:function' syntax, got: {path!r}")
    source, attribute_path = path.rsplit(":", 1)
    source_path = Path(source)
    if source_path.suffix == ".py" or source_path.exists():
        source_path = source_path.expanduser().resolve()
        spec = importlib.util.spec_from_file_location(f"_operator_benchmark_{hashlib.sha256(str(source_path).encode()).hexdigest()[:16]}", source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import wrapper file: {source_path}")
        value: Any = importlib.util.module_from_spec(spec)
        # Dataclass decorators and recursive imports need this registration.
        sys.modules[spec.name] = value
        spec.loader.exec_module(value)
    else:
        value = importlib.import_module(source)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    if not callable(value):
        raise TypeError(f"Resolved wrapper is not callable: {path}")
    return value


def _resolve_dtype(name: str) -> torch.dtype:
    attribute = name.removeprefix("torch.")
    dtype = getattr(torch, attribute, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return dtype


def _make_tensor(
    spec: dict[str, Any],
    default_device: str,
    keepalive: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    shape = spec.get("shape")
    if not isinstance(shape, list) or not all(isinstance(size, int) and size >= 0 for size in shape):
        raise ValueError(f"Tensor shape must be a list of non-negative integers: {shape!r}")

    dtype = _resolve_dtype(spec.get("dtype", "float32"))
    device = spec.get("device", default_device)
    initializer = spec.get("initializer", "zeros")

    if initializer == "data_ptrs":
        pointee_specs = spec.get("pointees")
        if not isinstance(pointee_specs, list) or not pointee_specs or not all(
            isinstance(pointee_spec, dict) for pointee_spec in pointee_specs
        ):
            raise ValueError("data_ptrs initializer requires a non-empty 'pointees' list")
        if len(shape) != 1 or shape[0] != len(pointee_specs):
            raise ValueError("data_ptrs tensor shape must match the number of pointees")
        if dtype != torch.uint64:
            raise ValueError("data_ptrs initializer requires dtype torch.uint64")

        pointees = [_make_tensor(pointee_spec, default_device, keepalive) for pointee_spec in pointee_specs]
        if keepalive is not None:
            keepalive.extend(pointees)
        return torch.tensor([tensor.data_ptr() for tensor in pointees], dtype=dtype, device=device)

    if initializer == "values":
        def flatten(value):
            return [x for item in value for x in flatten(item)] if isinstance(value, list) else [value]
        return torch.tensor(flatten(spec["values"]), dtype=dtype, device=device).reshape(shape)
    if initializer == "zeros":
        return torch.zeros(shape, dtype=dtype, device=device)
    if initializer == "ones":
        return torch.ones(shape, dtype=dtype, device=device)
    if initializer == "full":
        return torch.full(shape, spec["value"], dtype=dtype, device=device)
    if initializer == "rand":
        return torch.rand(shape, dtype=dtype, device=device)
    if initializer == "randn":
        return torch.randn(shape, dtype=dtype, device=device)
    if initializer == "randint":
        return torch.randint(spec.get("low", 0), spec["high"], shape, dtype=dtype, device=device)
    if initializer == "arange":
        start = spec.get("start", 0)
        step = spec.get("step", 1)
        if not isinstance(start, (int, float)) or not isinstance(step, (int, float)) or step == 0:
            raise ValueError("arange initializer requires numeric 'start' and non-zero 'step'")
        stop = start + math.prod(shape) * step
        tensor = torch.arange(start, stop, step, dtype=dtype, device=device)
        return tensor.reshape(shape)
    raise ValueError(f"Unsupported tensor initializer: {initializer!r}")


def materialize(
    value: Any,
    default_device: str,
    keepalive: list[torch.Tensor] | None = None,
) -> Any:
    """Recursively turn tensor specifications into tensors."""
    if isinstance(value, dict) and "shape" in value:
        tensor = _make_tensor(value, default_device, keepalive)
        if keepalive is not None:
            keepalive.append(tensor)
        return tensor
    if isinstance(value, dict):
        return {key: materialize(item, default_device, keepalive) for key, item in value.items()}
    if isinstance(value, list):
        return [materialize(item, default_device, keepalive) for item in value]
    return value


def percentile(values: Sequence[float], percent: float) -> float:
    """Return an interpolated percentile without requiring NumPy."""
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(latencies_ms: Sequence[float]) -> dict[str, float]:
    return {
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p90_ms": percentile(latencies_ms, 90),
        "p99_ms": percentile(latencies_ms, 99),
    }


def build_invoker(
    target: Any,
    mode: str,
    grid: Any,
) -> Callable[[list[Any], dict[str, Any]], Any]:
    """Build a regular wrapper call or a captured Triton grid launch."""
    if mode == "wrapper":
        return lambda args, kwargs: target(*args, **kwargs)
    if mode != "triton":
        raise ValueError(f"Unsupported benchmark mode: {mode}")
    if not isinstance(grid, list) or not grid or not all(isinstance(size, int) and size > 0 for size in grid):
        raise ValueError(f"Triton mode requires a non-empty grid of positive integers, got: {grid!r}")
    try:
        launcher = target[tuple(grid)]
    except TypeError as error:
        raise TypeError("Triton mode target must support kernel[grid](...) launches") from error
    return lambda args, kwargs: launcher(*args, **_normalize_triton_arguments(target, kwargs))


def _normalize_triton_arguments(target: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    """Map captured tensor names such as ``logits`` to kernel ``logits_ptr``."""
    arg_names = getattr(target, "arg_names", None)
    if not isinstance(arg_names, (list, tuple)):
        return arguments

    normalized = dict(arguments)
    for arg_name in arg_names:
        if arg_name.endswith("_ptr") and arg_name not in normalized:
            captured_name = arg_name.removesuffix("_ptr")
            if captured_name in normalized:
                normalized[arg_name] = normalized.pop(captured_name)
    return normalized


def benchmark_case(case: dict[str, Any]) -> dict[str, Any]:
    global PHASE
    PHASE = "environment"
    load_backend()
    device = case.get("device", "npu:0")
    torch.npu.set_device(device)
    torch.manual_seed(case.get("seed", 0))
    torch.npu.manual_seed_all(case.get("seed", 0))
    PHASE = "import"
    wrapper_path = case["wrapper"]
    wrapper = resolve_wrapper(wrapper_path)
    checker = resolve_wrapper(case["check"]) if case.get("check") else None
    function = getattr(wrapper, "fn", wrapper)
    try:
        actual_file = str(Path(inspect.getfile(function)).resolve())
    except (TypeError, OSError):
        actual_file = None
    binding = {"target": wrapper_path, "module": getattr(function, "__module__", None),
               "file": actual_file}
    if actual_file and Path(actual_file).is_file():
        binding["file_sha256"] = hashlib.sha256(Path(actual_file).read_bytes()).hexdigest()
    PHASE = "input"
    keepalive: list[torch.Tensor] = []
    args = materialize(case.get("args", []), device, keepalive)
    kwargs = materialize(case.get("arguments", case.get("kwargs", {})), device, keepalive)
    invoke = build_invoker(wrapper, case["mode"], case.get("grid"))
    # Preserve pointer-table pointees too. Never replace allocations or addresses.
    backups = [(tensor, tensor.clone()) for tensor in keepalive] if case.get("reset_inputs") else []

    def reset():
        for tensor, initial in backups:
            tensor.copy_(initial)
        if backups:
            torch.npu.synchronize()

    PHASE = "runtime"
    output = invoke(args, kwargs)
    torch.npu.synchronize()
    correctness = "not_checked"
    if checker:
        PHASE = "correctness"
        result = checker(args, kwargs, output)
        if result is not None and result is not True:
            raise AssertionError("correctness checker must return None/True or raise on mismatch")
        torch.npu.synchronize()
        correctness = "passed"
    PHASE = "runtime"
    warmup = case.get("warmup", 10)
    rounds = case.get("profiling_rounds", 100)
    for _ in range(warmup):
        reset()
        invoke(args, kwargs)
    torch.npu.synchronize()
    events = []
    for _ in range(rounds):
        reset()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        invoke(args, kwargs)
        end.record()
        events.append((start, end))
    torch.npu.synchronize()
    latencies_ms = [start.elapsed_time(end) for start, end in events]
    return {"name": case["name"], "wrapper": wrapper_path, "mode": case["mode"],
            "kernel": case["kernel"], "grid": case.get("grid"), "device": device,
            "warmup": warmup, "profiling_rounds": rounds, "binding": binding,
            "correctness": correctness, "seed": case.get("seed", 0),
            "benchmark_scope": case.get("benchmark_scope", (
                "launch only; input reset and check excluded" if case["mode"] == "triton" else
                "wrapper callable; internal allocation/construction included; external reset/check excluded")),
            "latencies_ms": latencies_ms, "summary": summarize(latencies_ms)}


def _override(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    case = dict(case)
    for key in ("wrapper", "mode", "device", "warmup", "profiling_rounds"):
        value = getattr(args, key)
        if value is not None:
            case[key] = value
    if args.device is not None:
        case["args"] = _replace_tensor_devices(case.get("args", []), args.device)
        if "kwargs" in case:
            case["kwargs"] = _replace_tensor_devices(case["kwargs"], args.device)
        if "arguments" in case:
            case["arguments"] = _replace_tensor_devices(case["arguments"], args.device)
    return case


def _replace_tensor_devices(value: Any, device: str) -> Any:
    """Apply a command-line device override to all nested tensor specs."""
    if isinstance(value, dict) and "shape" in value:
        return {**value, "device": device}
    if isinstance(value, dict):
        return {key: _replace_tensor_devices(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_tensor_devices(item, device) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--wrapper", required=True, help="Wrapper or Triton kernel (file.py:function)")
    parser.add_argument("--mode", choices=("wrapper", "triton"), default="wrapper")
    parser.add_argument("--kernel", help="Select this kernel from a mixed captured input file")
    parser.add_argument("--case-name", help="Select one named case after kernel filtering")
    parser.add_argument("--max-cases", type=int, help="Benchmark only the first N selected input records")
    parser.add_argument("--device", default="npu:0", help="NPU device")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup rounds")
    parser.add_argument("--profiling-rounds", type=int, default=100, help="Measured rounds")
    parser.add_argument("--output", type=Path, help="Write full JSON results to this path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from .cases import load_cases as checked_load, select_cases as checked_select
        cases = checked_select(checked_load(args.input_file), args.kernel, args.case_name)
        if args.max_cases:
            cases = cases[:args.max_cases]
        results = [benchmark_case(_override(case, args)) for case in cases]
        encoded = json.dumps(results, indent=2)
        if args.output:
            args.output.write_text(encoded + "\n")
        print(encoded)
        return 0
    except Exception as error:
        if args.output:
            args.output.write_text(json.dumps({"error": {"phase": PHASE,
                "reason": f"{type(error).__name__}: {error}"}}, ensure_ascii=False) + "\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
