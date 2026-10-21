# Diagnose the first failing stage

| Evidence | Check / next action |
|---|---|
| Missing Codex executable or authentication | Check `ai.codex`/`--codex` in the host environment and authenticate there; a dry run does not test credentials |
| Unknown tag/commit | Check the requested Git object exists locally; use a literal tag or unambiguous 7–40 character SHA, not a branch or revision expression |
| Missing config / unknown target | Initialize an example, edit it, or select an existing private file and target name |
| SSH timeout / authentication failure | Test the configured alias with BatchMode and bounded ConnectTimeout; inspect routing, agent and known-host state |
| Docker container not running | Confirm target container identity and running state; this is distinct from SSH failure |
| Source verification mismatch | Compare actual import path and HEAD with source-lock.json; align the intended environment deliberately, never silently checkout/reset |
| ImportError before launch | Check actual module locations and package versions; mixed Triton distributions can fail before any kernel compiles |
| Case schema or input failure | Check supported materializers and launcher contracts; preserve the rejected case and identify missing representational capability |
| Compiler/runtime error | Read the complete case log and final diagnostic; separate generated-input faults from kernel or compiler hypotheses |
| Interrupted remote job | Inspect remote process/output identity before retrying; a disconnected SSH session does not prove the worker exited |

After an environment correction, resume only if the saved inputs still match.
If source commits or cases changed, choose a new output. Report raw observations
separately from inferred causes. Stop repeating the same failed operation when
the missing prerequisite requires user or administrator action.
