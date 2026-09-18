---
name: vllm-ascend-kernel-cases
description: Generate source-grounded named JSON cases for kernel-tools by inspecting a Triton kernel, its real launcher, and effective Ascend binding. Includes shape, index, stride, pointer and reset contracts.
---

# Generate cases

Confirm selected vLLM/Ascend HEADs, dirty state and actual target binding. Do not switch an active user checkout to obtain a runner: this repository owns its standalone framework. Distinguish testing upstream code from an Ascend replacement.

Trace the definition, direct launch and wrapper. Derive every parameter, dtype, shape, stride, index bound, mask, sentinel, alias, grid, constexpr and launch option from that source. Pointer tables require real kept-alive pointees. Related indices and lengths cannot be independently randomized. Production-impossible layouts must not be represented as production coverage.

Read [case format](../../docs/cases.md). Generate JSON arrays with unique `(kernel, name)`, mode, wrapper, arguments, and grid for raw Triton. Include source commit, intended scenario and uncovered conditions. Use the existing framework capabilities; for noncontiguous views, shared storage or custom state use an explicit adapter instead of inventing JSON fields.

Choose smoke, typical, meaningful tile boundaries and stress cases based on real branches, not a fixed count or Cartesian product. For in-place kernels reason about the pre-check launch and repeated warmup/measurement. Set reset_inputs when appropriate, estimate its memory cost and state what is timed. Add an independent check callable when available; absent reference means correctness=not_checked.

Validate with `python -m kernel_tools cases validate CASES.json`, then list/select the exact case. Static validation does not establish index legality, compilation, precision or runtime binding. Default to generating cases; run only within the requested execution scope. Deliver case files and commands, not intermediate debug directories.

In `kernel-tools pipeline`, inspect the downloaded runtime source snapshot and return the requested case JSON string plus adapter/reference Python source. The controller writes and validates them; do not modify files or run tests in this read-only generation stage. Every ready case needs an independent checker. If legal inputs, effective binding or a reference cannot be established, return blocked with a specific reason. Generated adapters may construct inputs and call the actual kernel, but must not replace it with copied code or time the reference. Keep filesystem/network/process operations out of adapters. Never simplify a failing kernel into a different test to claim success.
