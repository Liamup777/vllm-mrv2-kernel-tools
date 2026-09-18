# Working on this tool

- Control-plane commands must work with Python standard library only. Import
  torch/torch_npu/triton only in the worker or explicit environment probe.
- Preserve JSON/JSONL compatibility with the benchmark_wrapper contract.
- Never silently turn unverified AST candidates into a complete inventory or
  benchmark success into correctness/production coverage.
- Keep generated run artifacts in cases/, results/, report.md and failed logs/.
- Preserve user source checkouts; remote execution transfers this tool and case
  data, not an edited production checkout. Do not hardcode secrets.
- Run `python -m unittest discover -s tests -v` for behavioral changes. NPU runs
  require a configured target; report when only local checks were possible.
- For new kernel cases, use skills/vllm-ascend-kernel-cases/SKILL.md.
- For release review, use skills/vllm-triton-release-scan/SKILL.md.
- For NPU execution, use skills/vllm-triton-remote-benchmark/SKILL.md.
