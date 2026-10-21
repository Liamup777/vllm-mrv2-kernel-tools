# Commands, outputs and recovery

Run these examples from the tool repository or its installed Python environment.
Replace source refs, source paths and outputs with the user's values. Output
directories must be new/empty except when explicitly resuming.

## Local stages

```bash
python3 -m kernel_tools scan --repo /path/to/vllm \
  --base BASE_TAG_OR_SHA --target TARGET_TAG_OR_SHA \
  --output /path/to/results/scan

python3 -m kernel_tools generate --scan /path/to/results/scan \
  --repo /path/to/vllm --source /path/to/vllm-ascend \
  --output /path/to/results/generated
```

For named kernels, skip scan:

```bash
python3 -m kernel_tools generate --kernel KERNEL_NAME \
  --source /path/to/vllm-ascend/path/to/kernel.py \
  --output /path/to/results/generated
```

Scan and generation use Codex but no SSH connection. Supplying Ascend sources
allows local analysis of patches; generation alone does not verify the remote
binding. Only AI-confirmed additions are selected automatically after a scan.
Use named-kernel generation for existing kernels requiring tests.

For combined local preparation, omit `--npu`:

```bash
python3 -m kernel_tools pipeline --repo /path/to/vllm \
  --source /path/to/vllm-ascend \
  --base BASE_TAG_OR_SHA --target TARGET_TAG_OR_SHA \
  --prepare-only --output /path/to/results/prepared
```

Add `--dry-run` to generate/pipeline to inspect the plan without AI or SSH.
`scan` has no dry-run option. Neither a dry run nor JSON structure validation
proves semantic correctness or device execution.

## Remote execution

After setup, validate and inspect the generated cases:

```bash
python3 -m kernel_tools cases validate /path/to/results/generated/cases
python3 -m kernel_tools cases list /path/to/results/generated/cases
python3 -m kernel_tools verify /path/to/results/generated \
  --config /path/to/kernel-tools.json --npu my-npu
```

If verify succeeds, execute a listed smoke case, then the selected case set in a
separate output directory:

```bash
python3 -m kernel_tools run /path/to/results/generated \
  --kernel KERNEL_NAME --case-name SMOKE_CASE_NAME \
  --config /path/to/kernel-tools.json --npu my-npu \
  --device npu:0 --output /path/to/results/smoke

python3 -m kernel_tools run /path/to/results/generated \
  --config /path/to/kernel-tools.json --npu my-npu \
  --device npu:0 --output /path/to/results/run
```

Use actual names from `cases list`. Keep source state unchanged between verify
and run. Standalone run does not inspect source-lock.json. For hand-written cases
without a lock, explicitly record the source identity and absence of lock verification.

For a requested end-to-end job:

```bash
python3 -m kernel_tools pipeline --repo /path/to/vllm \
  --source /path/to/vllm-ascend \
  --base BASE_TAG_OR_SHA --target TARGET_TAG_OR_SHA \
  --config /path/to/kernel-tools.json --npu my-npu \
  --device npu:0 --output /path/to/results/pipeline
```

Pipeline verifies the lock before executing generated cases. It does not run a
separate smoke phase. Use the staged commands when smoke-first execution is needed.
The lock checks imported repository HEADs, not dirty-file equality or numerical
correctness. Base source only needs to exist locally; remote vLLM must match the
target commit, and selected Ascend sources must also match their locked commit.

## Outputs

| Stage | Persistent artifacts | Interpretation |
|---|---|---|
| Scan | `review.json`, `review.md`, failed `logs/` | Source review; unresolved items remain visible |
| Generate / prepare | `cases/`, `source-lock.json`, `workflow.json`, `report.md` | Case preparation, not execution |
| Standalone run | `cases/`, `results/`, `report.md`, failed `logs/`; SSH runs also retain local `remote-run.json` | Per-case execution/measurement status |
| Pipeline | Cases, lock, workflow, results/report and failed logs when execution is reached | Inspect actual stage and status |

No new operators can legitimately yield `no_new_operators` without cases or a
device run. `prepared` means generation finished; `needs_review`/`blocked`/`failed`
require reporting their remaining work. Detailed formats live in
[pipeline documentation](../../../docs/pipeline.md) and [case format](../../../docs/cases.md).

## Recovery

Use the same inputs and output directory with `--resume`. Scan checks ref/commit
identity; pipeline also checks the stored base/target commits before review reuse.
Changed sources or case contracts require a new run. Completed pipeline jobs are
not resumed. A host-only prepared result can later be consumed by verify/run;
adding a different NPU target to that workflow's resume changes its recorded inputs.

Standalone remote run retains `remote-run.json`, checks for an active runner and
can reuse its remote output directory. Pipeline resumes reuse review/case generation
but may start a fresh execution attempt. After interrupted SSH/pipeline execution,
inspect the last remote job and logs before retrying; do not assume it stopped.
