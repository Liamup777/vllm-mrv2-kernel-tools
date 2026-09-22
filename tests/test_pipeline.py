import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kernel_tools.ai import CASE_SCHEMA, DIAGNOSIS_SCHEMA, REVIEW_SCHEMA, Codex
from kernel_tools.common import save_json
from kernel_tools.pipeline import pipeline, validate_bundle
from kernel_tools.remote import fetch_snapshot
from kernel_tools.snapshot import source_files, source_identity


SOURCE = """import triton
@triton.jit
def added(x): return x
def wrapper(x): added[(1,)](x)
"""
PATH = "vllm/v1/worker/gpu/new.py"
IDENTITY = "vllm.v1.worker.gpu.new.added"
OPERATOR = {"id": IDENTITY, "kernel": "added", "definition": PATH, "classification": "new",
            "reason": "Only in target tag", "evidence": PATH + ":4 wrapper directly launches added"}
REVIEW_OPERATOR = {key: OPERATOR[key] for key in ("id", "classification", "reason", "evidence")}
MODULE = "kt_generated_" + hashlib.sha256(IDENTITY.encode()).hexdigest()[:16]


def generated():
    # Protocol fixture, not a real kernel correctness test.
    return {"status": "ready", "reason": "", "analysis": "Protocol fixture, x is a scalar",
            "cases_json": json.dumps([{"name": "smoke", "kernel": "added", "scenario": "protocol",
                "target": "vllm.v1.worker.gpu.new:added", "grid": [1],
                "arguments": {"x": 1}}])}


class FakeAI:
    def __init__(self, fail_cases=False, omit=False):
        self.calls = []
        self.fail_cases, self.omit = fail_cases, omit

    def ask(self, prompt, schema, **kwargs):
        self.calls.append((prompt, schema))
        if schema == REVIEW_SCHEMA:
            operators = [] if self.omit else [dict(REVIEW_OPERATOR)]
            return {"operators": operators, "unresolved": [], "summary": "one new kernel"}
        if schema == CASE_SCHEMA:
            result = generated()
            if self.fail_cases:
                result["cases_json"] = "invalid JSON"
            return result
        if schema == DIAGNOSIS_SCHEMA:
            return {"analysis": "Observed CompilationError in logs/failure.log; compiler cause is a hypothesis."}
        raise AssertionError("Unexpected AI stage")


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        for args in (("init", "-q"), ("config", "user.name", "Test"),
                     ("config", "user.email", "test@example.invalid")):
            self.git(*args)
        source = self.repo / PATH
        source.parent.mkdir(parents=True)
        source.write_text("pass\n")
        self.commit("v1.0.0")
        source.write_text(SOURCE)
        self.commit("v1.1.0")
        self.config = self.root / "config.json"
        save_json(self.config, {"targets": {"test": {"host": "fixture", "cwd": str(self.repo),
            "python": sys.executable, "result_root": str(self.root / "remote")}}})
        self.output = self.root / "out"
        self.ai = FakeAI()

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True)

    def commit(self, tag):
        self.git("add", ".")
        self.git("commit", "-qm", tag)
        self.git("tag", tag)

    def snapshot(self, target, destination, **kwargs):
        file = destination / PATH
        file.parent.mkdir(parents=True)
        file.write_text(SOURCE)
        info = {"npu_available": True, "packages": {"triton-ascend": "test"}, "python": "test"}
        value = {"environment": info, "identity": source_identity(info, {PATH: SOURCE.encode()})}
        save_json(destination / "snapshot.json", value)
        return value

    def execute(self, target, cases, output, **kwargs):
        self.assertTrue(kwargs["expected_identity"])
        output.mkdir()
        save_json(output / "results/added.json", {"kernel": "added", "scenarios": [
            {"name": c["name"], "status": "success", "correctness": "not_checked"} for c in cases]})
        (output / "report.md").write_text("# Measured fixture report\n")
        return 0

    def run_flow(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return pipeline(self.repo, "v1.0.0", "v1.1.0", "test", self.config, self.output,
                            ai_client=self.ai, **kwargs)

    def state(self):
        return json.loads((self.output / "workflow.json").read_text())

    def test_complete_pipeline_and_minimal_artifacts(self):
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=self.snapshot), \
             patch("kernel_tools.pipeline.run_remote", side_effect=self.execute):
            self.assertEqual(self.run_flow(), 0)
        self.assertEqual(self.state()["status"], "completed")
        self.assertEqual(len(self.ai.calls), 2)
        self.assertIn("Apply this skill", self.ai.calls[0][0])
        self.assertEqual({p.name for p in self.output.iterdir()},
                         {"cases", "results", "workflow.json", "report.md"})

    def test_missing_candidate_blocks_before_ssh(self):
        self.ai.omit = True
        with patch("kernel_tools.pipeline.fetch_snapshot") as snapshot:
            self.assertEqual(self.run_flow(), 1)
            snapshot.assert_not_called()
        self.assertIn("omitted candidates", self.state()["error"])

    def test_invalid_case_retries_once_and_never_executes(self):
        self.ai.fail_cases = True
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=self.snapshot), \
             patch("kernel_tools.pipeline.run_remote") as remote:
            self.assertEqual(self.run_flow(), 1)
            remote.assert_not_called()
        self.assertEqual(len(self.ai.calls), 3)
        self.assertEqual(self.state()["generation"][IDENTITY]["status"], "blocked")
        self.assertEqual(len(list((self.output / "logs").glob("generation*.log"))), 2)

    def test_prepare_only_does_not_execute(self):
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=self.snapshot), \
             patch("kernel_tools.pipeline.run_remote") as remote:
            self.assertEqual(self.run_flow(prepare_only=True), 0)
            remote.assert_not_called()
        self.assertEqual(self.state()["status"], "prepared")

    def test_resume_reuses_review_and_generated_cases(self):
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=self.snapshot), \
             patch("kernel_tools.pipeline.run_remote") as remote:
            self.assertEqual(self.run_flow(prepare_only=True), 0)
            remote.assert_not_called()
        self.assertEqual(len(self.ai.calls), 2)
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=self.snapshot), \
             patch("kernel_tools.pipeline.run_remote", side_effect=self.execute):
            self.assertEqual(self.run_flow(resume=True), 0)
        self.assertEqual(len(self.ai.calls), 2)
        self.assertEqual(self.state()["status"], "completed")
        self.assertTrue(self.state()["resumes"])

    def test_resume_rechecks_review_when_scope_policy_changed(self):
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=self.snapshot), \
             patch("kernel_tools.pipeline.run_remote") as remote:
            self.assertEqual(self.run_flow(prepare_only=True), 0)
            remote.assert_not_called()
        state = self.state()
        state.pop("review_policy_version")
        save_json(self.output / "workflow.json", state)
        calls = len(self.ai.calls)
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=self.snapshot), \
             patch("kernel_tools.pipeline.run_remote", side_effect=self.execute):
            self.assertEqual(self.run_flow(resume=True), 0)
        self.assertEqual(len(self.ai.calls), calls + 1)

    def test_resume_after_remote_source_is_aligned_reuses_review(self):
        def drift(*args, **kwargs):
            value = self.snapshot(*args, **kwargs)
            value["identity"]["files"][PATH] = "different"
            return value
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=drift):
            self.assertEqual(self.run_flow(), 1)
        self.assertEqual(len(self.ai.calls), 1)
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=self.snapshot), \
             patch("kernel_tools.pipeline.run_remote", side_effect=self.execute):
            self.assertEqual(self.run_flow(resume=True), 0)
        self.assertEqual(len(self.ai.calls), 2)

    def test_custom_case_directory_is_saved_and_reported(self):
        case_dir = self.root / "generated-cases"
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=self.snapshot), \
             patch("kernel_tools.pipeline.run_remote", side_effect=self.execute):
            self.assertEqual(self.run_flow(cases_output=case_dir), 0)
        self.assertEqual(self.state()["cases_path"], str(case_dir.resolve()))
        self.assertEqual(len(list(case_dir.glob("*.json"))), 1)
        self.assertFalse((self.output / "cases").exists())
        self.assertIn(str(case_dir.resolve()), (self.output / "report.md").read_text())

    def test_custom_case_directory_refuses_overwrite(self):
        case_dir = self.root / "generated-cases"
        case_dir.mkdir()
        (case_dir / "existing.json").write_text("original")
        with self.assertRaisesRegex(ValueError, "Case output already exists"):
            self.run_flow(cases_output=case_dir)
        self.assertEqual((case_dir / "existing.json").read_text(), "original")

    def test_failure_analysis_reads_custom_case_directory(self):
        case_dir = self.root / "generated-cases"

        def failed_run(*args, **kwargs):
            self.execute(*args, **kwargs)
            output = args[2]
            save_json(output / "results/added.json", {"kernel": "added", "scenarios": [
                {"name": "smoke", "status": "failed", "correctness": "not_checked"}]})
            return 1

        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=self.snapshot), \
             patch("kernel_tools.pipeline.run_remote", side_effect=failed_run):
            self.assertEqual(self.run_flow(cases_output=case_dir), 1)
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(len(list(case_dir.glob("*.json"))), 1)
        self.assertEqual(len(self.ai.calls), 3)

    def test_source_mismatch_blocks_generation_and_execution(self):
        def drift(*args, **kwargs):
            value = self.snapshot(*args, **kwargs)
            value["identity"]["files"][PATH] = "different"
            return value
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=drift), \
             patch("kernel_tools.pipeline.run_remote") as remote:
            self.assertEqual(self.run_flow(), 1)
            remote.assert_not_called()
        self.assertEqual(len(self.ai.calls), 1)
        self.assertIn("differs", self.state()["error"])

    def test_failed_run_keeps_full_log_and_calls_diagnosis(self):
        def fail(*args, **kwargs):
            self.execute(*args, **kwargs)
            output = args[2]
            save_json(output / "results/added.json", {"kernel": "added", "scenarios": [
                {"name": "smoke", "status": "failed", "correctness": "not_checked"}]})
            (output / "logs").mkdir()
            (output / "logs/failure.log").write_text("CompilationError: full diagnostic\n")
            return 1
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=self.snapshot), \
             patch("kernel_tools.pipeline.run_remote", side_effect=fail):
            self.assertEqual(self.run_flow(), 1)
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual((self.output / "logs/failure.log").read_text(), "CompilationError: full diagnostic\n")
        self.assertIn("hypothesis", (self.output / "report.md").read_text())

    def test_missing_result_cannot_be_completed(self):
        def incomplete(*args, **kwargs):
            self.execute(*args, **kwargs)
            save_json(args[2] / "results/added.json", {"kernel": "added", "scenarios": []})
            return 0
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=self.snapshot), \
             patch("kernel_tools.pipeline.run_remote", side_effect=incomplete):
            self.assertEqual(self.run_flow(), 1)
        self.assertIn("exactly the generated cases", self.state()["error"])

    def test_target_must_exist_and_operator_check_is_rejected(self):
        bundle = generated()
        cases = json.loads(bundle["cases_json"])
        cases[0]["check"] = "generated_reference:check"
        bundle["cases_json"] = json.dumps(cases)
        with self.assertRaisesRegex(ValueError, "operator-specific check callbacks are unsupported"):
            validate_bundle(bundle, OPERATOR, {PATH: SOURCE})
        with self.assertRaisesRegex(ValueError, "absent"):
            validate_bundle(generated(), OPERATOR, {})


class CodexProtocolTest(unittest.TestCase):
    def test_snapshot_excludes_vllm_generated_version_file_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "vllm"
            package.mkdir()
            (package / "__init__.py").write_text("# tracked\n")
            (package / "_version.py").write_text("# generated by vcs-versioning\n")
            (package / "extra.py").write_text("# must remain visible\n")
            files = source_files({"imports": {"vllm": str(package / "__init__.py")}})
        self.assertEqual(set(files), {"vllm/__init__.py", "vllm/extra.py"})

    def test_real_subprocess_protocol_and_failure_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "codex"
            fake.write_text(f"#!{sys.executable}\n" + '''import sys,json
from pathlib import Path
args=sys.argv[1:]
assert args[0]=='exec'
assert args[args.index('--sandbox')+1]=='read-only'
assert sys.stdin.read()=='test prompt'
schema=json.loads(Path(args[args.index('--output-schema')+1]).read_text())
assert 'analysis' in schema['properties']
Path(args[args.index('--output-last-message')+1]).write_text('{"analysis":"ok"}')
print('verbose CLI chatter')
''')
            fake.chmod(0o755)
            ai = Codex(str(fake), timeout=10)
            log = root / "failed.log"
            self.assertEqual(ai.ask("test prompt", DIAGNOSIS_SCHEMA, workspace=root, failure_log=log),
                             {"analysis": "ok"})
            self.assertFalse(log.exists())
            fake.write_text(f"#!{sys.executable}\nprint('ERROR: authentication failed')\nraise SystemExit(1)\n")
            with self.assertRaisesRegex(ValueError, "authentication failed"):
                ai.ask("test prompt", DIAGNOSIS_SCHEMA, workspace=root, failure_log=log)
            self.assertIn("authentication failed", log.read_text())

    def test_remote_snapshot_archive_survives_setup_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup = root / "setup.sh"
            setup.write_text("echo setup-banner\n")
            target = {"host": "local-fixture", "cwd": str(root), "python": sys.executable, "setup": str(setup)}
            with patch("kernel_tools.remote.ssh_command", side_effect=lambda t, c: ["bash", "-c", c]):
                snapshot = fetch_snapshot(target, root / "snapshot")
            self.assertIn("fingerprint", snapshot["identity"])
            self.assertTrue((root / "snapshot/snapshot.json").is_file())


class ReviewCommandTest(unittest.TestCase):
    def setUp(self):
        self.fixture = PipelineTest()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_ai_scan_without_npu_config(self):
        from kernel_tools.review import scan_with_ai
        f = self.fixture
        with contextlib.redirect_stdout(io.StringIO()):
            code = scan_with_ai(f.repo, "v1.0.0", "v1.1.0", f.output, ai_client=f.ai)
        self.assertEqual(code, 0)
        self.assertEqual(len(f.ai.calls), 1)
        self.assertEqual({p.name for p in f.output.iterdir()}, {"review.md", "review.json"})
        report = json.loads((f.output / "review.json").read_text())
        self.assertEqual(report["status"], "reviewed")
        self.assertFalse(report["complete_inventory"])
        operator = report["review"]["operators"][0]
        self.assertEqual(operator["classification"], "new")
        self.assertEqual(operator["kernel"], "added")
        self.assertEqual(operator["definition"], PATH)

    def test_ai_failure_never_reports_static_scan_as_reviewed(self):
        from kernel_tools.review import scan_with_ai
        f = self.fixture
        f.ai.omit = True
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(scan_with_ai(f.repo, "v1.0.0", "v1.1.0", f.output, ai_client=f.ai), 1)
        report = json.loads((f.output / "review.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("omitted candidates", report["error"])
        self.assertTrue((f.output / "logs/scan.log").is_file())

    def test_scan_resume_retries_failed_review_and_reuses_completed_review(self):
        from kernel_tools.review import scan_with_ai
        f = self.fixture
        f.ai.omit = True
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(scan_with_ai(f.repo, "v1.0.0", "v1.1.0", f.output,
                                          ai_client=f.ai), 1)
        f.ai.omit = False
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(scan_with_ai(f.repo, "v1.0.0", "v1.1.0", f.output,
                                          ai_client=f.ai, resume=True), 0)
        calls = len(f.ai.calls)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(scan_with_ai(f.repo, "v1.0.0", "v1.1.0", f.output,
                                          ai_client=f.ai, resume=True), 0)
        self.assertEqual(len(f.ai.calls), calls)
        report = json.loads((f.output / "review.json").read_text())
        self.assertEqual(report["status"], "reviewed")
        self.assertEqual(len(report["resumes"]), 1)

    def test_cli_scan_routes_to_ai(self):
        from kernel_tools.cli import main
        with patch("kernel_tools.review.scan_with_ai", return_value=0) as review:
            self.assertEqual(main(["scan", "--repo", ".", "--base", "v1.0.0", "--target", "v1.1.0",
                                   "--output", "unused"]), 0)
            review.assert_called_once()

    def test_generate_reuses_scan_result_without_second_release_review(self):
        from kernel_tools.pipeline import generate_from_scan
        from kernel_tools.review import scan_with_ai
        f = self.fixture
        scan_output = f.root / "scan"
        generation_output = f.root / "generation"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(scan_with_ai(f.repo, "v1.0.0", "v1.1.0", scan_output,
                                          ai_client=f.ai), 0)
        self.assertEqual(len(f.ai.calls), 1)
        output = io.StringIO()
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=f.snapshot), \
             patch("kernel_tools.pipeline.run_remote") as remote, \
             contextlib.redirect_stdout(output):
            self.assertEqual(generate_from_scan(scan_output, f.repo, "test", f.config,
                                                generation_output, ai_client=f.ai), 0)
            remote.assert_not_called()
        self.assertIn("[generate 1/3] Verify scan result", output.getvalue())
        self.assertIn("[generate 2/3] Read and verify actual sources", output.getvalue())
        self.assertIn("[generate 3/3] Generate cases", output.getvalue())
        self.assertNotIn("[pipeline", output.getvalue())
        self.assertEqual(len(f.ai.calls), 2)
        self.assertEqual(f.ai.calls[1][1], CASE_SCHEMA)
        state = json.loads((generation_output / "workflow.json").read_text())
        self.assertEqual(state["status"], "prepared")
        self.assertEqual(state["operation"], "generate")
        self.assertEqual(Path(state["scan_source"]), (scan_output / "review.json").resolve())
        self.assertEqual(len(list((generation_output / "cases").glob("*.json"))), 1)

    def test_generate_explicit_kernel_skips_release_scan(self):
        from kernel_tools.pipeline import generate_from_kernels
        f = self.fixture
        generation_output = f.root / "manual-generation"
        output = io.StringIO()
        with patch("kernel_tools.pipeline.fetch_snapshot", side_effect=f.snapshot), \
             patch("kernel_tools.pipeline.run_remote") as remote, \
             contextlib.redirect_stdout(output):
            self.assertEqual(generate_from_kernels(["added"], f.repo, "v1.1.0", "test",
                                                   f.config, generation_output,
                                                   ai_client=f.ai), 0)
            remote.assert_not_called()
        self.assertEqual(len(f.ai.calls), 1)
        self.assertEqual(f.ai.calls[0][1], CASE_SCHEMA)
        self.assertIn("[generate 1/3] Resolve explicitly selected kernels", output.getvalue())
        self.assertNotIn("release review", output.getvalue())
        state = json.loads((generation_output / "workflow.json").read_text())
        self.assertEqual(state["selection_mode"], "manual")
        self.assertEqual(state["selected_kernels"], ["added"])
        self.assertEqual(state["review"]["operators"][0]["id"], IDENTITY)

    def test_explicit_kernel_requires_unique_target_tag_jit(self):
        from kernel_tools.pipeline import generate_from_kernels
        f = self.fixture
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(generate_from_kernels(["missing"], f.repo, "v1.1.0", "test",
                                                   f.config, f.root / "missing",
                                                   ai_client=f.ai), 1)
        state = json.loads((f.root / "missing/workflow.json").read_text())
        self.assertIn("Triton JIT kernel not found", state["error"])
        self.assertEqual(len(f.ai.calls), 0)

    def test_cli_generate_routes_completed_scan(self):
        from kernel_tools.cli import main
        with patch("kernel_tools.pipeline.generate_from_scan", return_value=0) as generate:
            self.assertEqual(main(["generate", "--scan", "scan-output", "--repo", ".",
                                   "--npu", "test", "--output", "generation-output"]), 0)
            generate.assert_called_once()

    def test_cli_generate_routes_explicit_kernel(self):
        from kernel_tools.cli import main
        with patch("kernel_tools.pipeline.generate_from_kernels", return_value=0) as generate:
            self.assertEqual(main(["generate", "--kernel", "added", "--target", "v1.1.0",
                                   "--repo", ".", "--npu", "test",
                                   "--output", "generation-output"]), 0)
            generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
