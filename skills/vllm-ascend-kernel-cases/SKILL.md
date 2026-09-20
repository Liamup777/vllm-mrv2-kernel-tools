---
name: vllm-ascend-kernel-cases
description: Generate source-grounded named JSON cases for kernel-tools by inspecting a Triton kernel, its real launcher, and effective Ascend binding. Includes shape, index, stride, pointer and reset contracts.
---

# Generate cases

Confirm selected vLLM/Ascend HEADs, dirty state and actual target binding. Do not switch an active user checkout to obtain a runner: this repository owns its standalone framework. Distinguish testing upstream code from an Ascend replacement.

Trace the definition, direct launch and production wrapper. The wrapper is source evidence only: every generated case must target the Triton JIT object itself and execute `target[grid](...)`. Derive every parameter, dtype, shape, stride, index bound, mask, sentinel, alias, grid, constexpr and launch option from that source. Pointer tables require real kept-alive pointees. Related indices and lengths cannot be independently randomized. Production-impossible layouts must not be represented as production coverage.

Read [case format](../../docs/cases.md). Generate JSON arrays with unique `(kernel, name)`, `target`, `arguments`, and `grid`. `target` must resolve to the real Triton JIT kernel. Never emit `mode`, `wrapper`, or a Python adapter. Include source commit, intended scenario and uncovered conditions. Use only existing framework materializers. If noncontiguous views, shared storage, correlated tensors or custom state cannot be represented, mark that coverage blocked and name the missing framework capability.

Choose smoke, typical, meaningful tile boundaries and stress cases based on real branches, not a fixed count or Cartesian product. For in-place kernels reason about the pre-check launch and repeated warmup/measurement. Set reset_inputs when appropriate, estimate its memory cost and state what is timed. Do not generate or select an operator-specific checker; all cases report correctness=not_checked until the framework has a generic JSON comparison protocol.

Validate with `python -m kernel_tools cases validate CASES.json`, then list/select the exact case. Static validation does not establish index legality, compilation, precision or runtime binding. Default to generating cases; run only within the requested execution scope. Deliver case files and commands, not intermediate debug directories.

In `kernel-tools pipeline`, inspect the downloaded runtime source snapshot and return only the requested case JSON string. The controller writes and validates it; do not modify files or run tests in this read-only generation stage. Never emit `check`. If legal inputs or the effective binding cannot be established with the current JSON protocol, return blocked with a specific missing capability. Never simplify a failing kernel into a different test to claim success.
