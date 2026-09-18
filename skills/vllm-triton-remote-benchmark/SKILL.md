---
name: vllm-triton-remote-benchmark
description: Run source-grounded single-operator cases on a configured Ascend SSH or Docker target using kernel-tools, retain per-case results and full failure logs, and diagnose failures without changing production kernels.
---

# Remote execution

Use [README](../../README.md) for init/doctor/run commands. Targets come from the local kernel-tools.json; no hardcoded password or automatic host-key bypass. Inspect doctor output, source HEAD/dirty state, Python imports, package versions including triton-ascend, visible-device mapping and NPU occupancy before selecting an idle device. Do not terminate unrelated processes.

For case generation apply [kernel cases](../vllm-ascend-kernel-cases/SKILL.md). Verify upstream target versus Ascend replacement explicitly. Uploading this versioned standalone tool is intentional; no switch to kernel_test_frame or replacement of the user's production runner is needed.

Run one smoke case, then all planned cases using `python -m kernel_tools run CASES --target NAME`. The tool sequentially isolates cases in processes, retains failures and continues. Source/input/environment changes require a new run. Resume is only safe with matching source, cases, environment and tool. After an SSH interruption inspect the old process and remote run path before restarting.

Keep cases/, results/, report.md and only failed logs/. Report original execution facts and distinguish input/import/compile/runtime/correctness/timeout/blocked. Failed logs remain complete; summaries stay short. Never silently modify kernel source or substitute simpler input to present the original failure as a pass.

success means the execution/measurement protocol passed. Only an explicit check callback passing can establish case-level correctness. A standalone case proves neither E2E reachability nor entire-model correctness; those require separate runtime evidence.

`kernel-tools pipeline` delegates review/generation/diagnosis to Codex and owns SSH, source checks, case isolation and report persistence in fixed code. In its read-only diagnosis stage, analyze the supplied results and full failure logs without executing remote commands or changing measured statuses. Distinguish observed evidence from inferred causes, including faulty generated inputs or reference code. If the user asks only to install or prepare the tool, do not choose production kernels and start remote jobs on their behalf.
