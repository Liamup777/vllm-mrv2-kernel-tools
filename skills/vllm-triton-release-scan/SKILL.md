---
name: vllm-triton-release-scan
description: Review vLLM MRV2 Triton operators at exact tags, verify direct launch and wrapper evidence, and explain release differences using kernel-tools scan output. Static candidates need review before a complete inventory is claimed.
---

# Release review

The `scan` CLI invokes Codex with this skill; `pipeline` invokes the same review stage before case generation. Read the supplied exact source snapshots and internal static candidate files. Do not recursively invoke scan or pipeline from this stage. Static candidates are hints, not a complete inventory or proof of runtime reachability.

Inspect the exact tag commits recorded in the output. Include directly launched `kernel[grid](...)` functions reached from `vllm/v1/worker/gpu`, including imported wrappers outside the directory. Exclude JIT-only helpers from the operator count. Verify unresolved launches, dynamic aliases, inheritance, conditional backend choices and monkey patches against source. A move/rename candidate is not a new operator merely because its path changed.

Scope means MRV2 framework operations: input/block-table preparation, sampling, speculative decoding and model-state management, including explicitly called external operator wrappers and context methods. Stop at generic model loading/registries, model.forward/__call__, attention-backend execution and arbitrary PyTorch dispatch. Do not expand those generic entry points into an inventory of every architecture, attention, MoE or quantization kernel. External inclusion needs a concrete MRV2 framework operation path, such as the Mamba state's context copy methods, rather than only model execution reachability.

For each verified operator retain definition, direct launch site, wrapper, worker entry path, activation condition and exact source version. Inspect changed wrappers and transitive helpers even when the kernel body is unchanged. Unverified paths remain unresolved.

Return the requested structured review with confirmed additions, non-additions, evidence and unresolved items; keep validation status explicit. When delivering a complete inventory to the user, create a verified XLSX with Summary, Launchable Kernels and Excluded Helpers using the available spreadsheet skill. Automated scan/pipeline stages return JSON for the controller to render review.md; the interactive XLSX requirement does not apply to these stages.

See [scanning](../../docs/scanning.md) for the manual workflow and the boundary between script execution and AI review. Do not launch NPU work just because a tag was found unless that execution is part of the user's requested workflow.

When invoked by `kernel-tools scan` or `kernel-tools pipeline`, use the supplied exact base/target source snapshots and return the requested structured review. Account for every added/moved candidate, investigate unresolved launches, and explicitly record remaining uncertainty. The pipeline schema replaces the interactive XLSX deliverable for this stage; do not start case generation or NPU jobs from the review stage.
