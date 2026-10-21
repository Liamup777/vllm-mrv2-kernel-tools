# Acceptance and evidence

## Source review

- Record base/target inputs and resolved commits, scope and review status.
- Count locally defined directly launchable kernels and explicitly imported
  Triton symbols directly launched within the GPU directory. Keep helpers,
  moves/renames, wrapper-only imports and unresolved paths distinct.
- Do not present static candidates as AI-reviewed inventory. CLI output is a
  delta report focused on new operators, not a single-version XLSX catalog.

## Generated cases

- Case validation passes; each selected kernel has ready cases or an explicit
  blocked reason. Review parameter relationships and production launch contracts.
- Targets are real JIT symbols with grid and supported JSON arguments. No Python
  adapter, wrapper execution or invented reference checker is accepted.
- Identify the local upstream or Ascend binding used and retained source lock.
  Preparation proves neither compilation nor actual remote binding.

## Execution, only when requested

- Record target/device, runtime environment and source verification result. A
  lock compares commits only; disclose relevant dirty-source differences.
- Every selected case has a result, including failed/blocked/timed-out cases.
- Keep full failure logs and report the stage actually reached. Check current-run
  results before interpreting older logs retained after resume.
- `success` describes the runner/measurement protocol; preserve
  `correctness=not_checked`. Neither latency nor exit code establishes precision,
  model-level reachability or service throughput.

## Handoff

- Link actual reports/cases/results; distinguish generated plans, test fixtures,
  dry runs and real NPU evidence.
- Report missing prerequisites and untested scope. Do not mark an entire workflow
  reproduced based on schema tests or local fake-transport tests.
- Share a sanitized configuration example and reproducible commands, keeping
  private host settings and credentials outside published artifacts.
