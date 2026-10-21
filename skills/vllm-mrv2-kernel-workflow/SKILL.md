---
name: vllm-mrv2-kernel-workflow
description: Scan vLLM MRV2 Triton changes between tags or commits, generate source-grounded kernel cases, and optionally run them on a configured Ascend target using kernel-tools. Use for release or commit migration checks and named-kernel test preparation.
---

# MRV2 kernel workflow

This is the user-facing entrypoint for the versioned `vllm-kernel-tools` package.
Keep the complete repository or installed package resources: the stage skills,
Python CLI and documentation are required dependencies.
If this entry is loaded through a symlink, resolve its real filesystem location
before following relative references or sibling skill links. CLI installation
and skill discovery are separate steps; see setup for both.

## Choose the requested outcome

| User request | Entry point | Inputs |
|---|---|---|
| Compare MRV2 kernels | `scan` | Local vLLM repo, base and target tag/commit, output |
| Generate cases from a review | `generate --scan` | Review output, local vLLM repo, optional Ascend sources |
| Generate cases for known kernels | `generate --kernel` | Kernel names and local source paths; optional `--ref` |
| Scan and prepare without an NPU | `pipeline --prepare-only` | Local repo and base/target; no remote target needed |
| Execute existing cases | `doctor`, `verify` when a lock exists, then `run` | Cases, private target config, selected device |
| Scan through remote execution | `pipeline --npu` | Local sources, base/target, private target config, device |

The scan CLI compares two source states and focuses on additions. It is not a
single-version complete inventory or an XLSX exporter. If that is the requested
deliverable, explain this boundary before choosing a different workflow.

## Execution

1. Establish the requested stage, local source paths, output directory and source
   identities. Use explicit tags or 7–40 hexadecimal-character commit SHAs for scan/pipeline;
   preserve the user's checkout. Record whether Ascend replacement sources were supplied.
2. Read [setup](references/setup.md) for first use or a new target. Only device
   execution needs SSH/Docker/NPU configuration. Missing configuration is a setup
   task, not a failed kernel test.
3. Read [workflow](references/workflow.md) for the selected command, source locks,
   outputs and recovery. Invoke the CLI from this repository or an installed
   environment. The controller owns its internal AI calls; do not manually launch
   child stages concurrently or recursively call pipeline from a child AI stage.
4. If an operation fails, read [troubleshooting](references/troubleshooting.md)
   and diagnose the current attempt before retrying. Keep the original failure evidence.
5. Read [acceptance](references/acceptance.md) before delivery. Return source
   commits, actual stage/status, artifact paths and unresolved coverage.

## Stage ownership

- [Release review](../vllm-triton-release-scan/SKILL.md): the controller supplies
  snapshots and a schema; the child reviews direct-launch evidence.
- [Case generation](../vllm-ascend-kernel-cases/SKILL.md): the child derives input
  contracts from supplied local sources and returns schema-conforming cases.
- [Remote benchmark](../vllm-triton-remote-benchmark/SKILL.md): execution guidance;
  inside pipeline's AI diagnosis call it only analyzes supplied logs and results.

Calling these stages does not prove numerical correctness. Current execution
reports `correctness=not_checked`; distinguish compile/run/measurement success
from accuracy and model-level coverage. Keep secrets and private target settings
outside published examples and reports.
