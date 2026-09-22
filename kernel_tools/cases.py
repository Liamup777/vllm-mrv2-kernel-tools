from __future__ import annotations

import json
import math
from pathlib import Path

DTYPES = {"bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16",
          "int32", "int64", "float16", "bfloat16", "float32", "float64"}
INITIALIZERS = {"zeros", "ones", "full", "rand", "randn", "randint", "arange",
                "data_ptrs", "values"}
TENSOR_KEYS = {"shape", "dtype", "device", "initializer", "value", "low", "high",
               "start", "step", "pointees", "values"}


def load_cases(path):
    path = Path(path)
    paths = (sorted([p for p in [*path.glob("*.json"), *path.glob("*.jsonl")]
                     if p.name != "source-lock.json"]) if path.is_dir() else [path])
    cases = []
    for file in paths:
        text = file.read_text(encoding="utf-8").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list) or not all(isinstance(c, dict) for c in data):
            raise ValueError(f"{file}: expected JSON object, array, or JSONL")
        cases.extend(data)
    if not cases:
        raise ValueError(f"No cases found in {path}")
    validate_cases(cases)
    return cases


def validate_cases(cases):
    seen = set()
    for case in cases:
        label = f"{case.get('kernel', '?')}/{case.get('name', '?')}"
        if "target" not in case and "wrapper" in case:
            if case.get("mode") == "triton":
                raise ValueError(
                    f"{label}: legacy Triton case; rename 'wrapper' to 'target' and remove 'mode'"
                )
            raise ValueError(f"{label}: legacy wrapper case cannot be migrated; regenerate a direct Triton case")
        for key in ("name", "kernel", "target"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"{label}: requires nonempty '{key}'")
        identity = (case["kernel"], case["name"])
        if identity in seen:
            raise ValueError(f"Duplicate case: {label}")
        seen.add(identity)
        if ":" not in case["target"]:
            raise ValueError(f"{label}: target must be module:symbol or file.py:symbol")
        if "mode" in case or "wrapper" in case:
            raise ValueError(f"{label}: legacy mode/wrapper fields are unsupported; use a direct Triton target")
        grid = case.get("grid")
        if not isinstance(grid, list) or not grid or any(type(x) is not int or x <= 0 for x in grid):
            raise ValueError(f"{label}: grid must contain positive integers")
        if "kwargs" in case and "arguments" in case:
            raise ValueError(f"{label}: use arguments or legacy kwargs, not both")
        if not isinstance(case.get("arguments", case.get("kwargs", {})), dict):
            raise ValueError(f"{label}: arguments must be an object")
        if not isinstance(case.get("args", []), list):
            raise ValueError(f"{label}: args must be a list")
        for key, minimum in (("seed", 0), ("warmup", 0), ("profiling_rounds", 1)):
            if key in case and (type(case[key]) is not int or case[key] < minimum):
                raise ValueError(f"{label}: {key} must be an integer >= {minimum}")
        if "reset_inputs" in case and not isinstance(case["reset_inputs"], bool):
            raise ValueError(f"{label}: reset_inputs must be boolean")
        if "check" in case:
            raise ValueError(
                f"{label}: operator-specific check callbacks are unsupported; "
                "remove 'check' and report correctness=not_checked"
            )
        validate_value(case.get("arguments", case.get("kwargs", {})), label)
        validate_value(case.get("args", []), label)


def validate_value(value, label):
    if isinstance(value, dict) and "shape" in value:
        unknown = value.keys() - TENSOR_KEYS
        if unknown:
            raise ValueError(f"{label}: unsupported tensor fields {sorted(unknown)}; extend the framework materializer")
        shape = value["shape"]
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"{label}: invalid tensor shape")
        dtype = value.get("dtype", "float32").removeprefix("torch.")
        if dtype not in DTYPES:
            raise ValueError(f"{label}: unsupported dtype {dtype}")
        kind = value.get("initializer", "zeros")
        if kind not in INITIALIZERS:
            raise ValueError(f"{label}: unsupported initializer {kind}")
        if kind == "data_ptrs":
            pointees = value.get("pointees")
            if dtype != "uint64" or not isinstance(pointees, list) or not pointees or shape != [len(pointees)]:
                raise ValueError(f"{label}: data_ptrs requires uint64, shape [len(pointees)], and nonempty pointees")
            if not all(isinstance(p, dict) and "shape" in p for p in pointees):
                raise ValueError(f"{label}: invalid pointee specification")
            for p in pointees:
                validate_value(p, label)
        if kind == "full" and "value" not in value:
            raise ValueError(f"{label}: full requires value")
        if kind == "randint":
            low, high = value.get("low", 0), value.get("high")
            if type(low) is not int or type(high) is not int or high <= low:
                raise ValueError(f"{label}: randint requires integer high > low")
        if kind == "arange" and value.get("step", 1) == 0:
            raise ValueError(f"{label}: arange step cannot be zero")
        if kind == "values":
            def flatten(x):
                return [y for v in x for y in flatten(v)] if isinstance(x, list) else [x]
            if "values" not in value or len(flatten(value["values"])) != math.prod(shape):
                raise ValueError(f"{label}: values count must match shape")
    elif isinstance(value, dict):
        for child in value.values():
            validate_value(child, label)
    elif isinstance(value, list):
        for child in value:
            validate_value(child, label)


def select_cases(cases, kernel=None, case_name=None):
    selected = [c for c in cases if (not kernel or c["kernel"] == kernel)
                and (not case_name or c["name"] == case_name)]
    if not selected:
        raise ValueError("No matching cases; use 'cases list' to inspect names")
    if case_name and len(selected) != 1:
        raise ValueError("Case name matches multiple kernels; also specify --kernel")
    return selected
