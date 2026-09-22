import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kernel_tools.benchmark import resolve_symbol, resolve_target
from kernel_tools.cases import load_cases, select_cases, validate_cases
from kernel_tools.common import save_json
from kernel_tools.remote import safe_extract, target_command, run_remote, verify_remote
from kernel_tools.runner import device_lock, run_suite
from kernel_tools.runner import new_run_path
from kernel_tools.scan import compare, scan
from kernel_tools.source_lock import local_source, make_source_lock, verify_source_lock


def case(name="ok", kernel="test_kernel"):
    return {"name": name, "kernel": kernel, "target": "fixture:kernel",
            "grid": [1], "arguments": {"x": {"shape": [3], "dtype": "int32", "initializer": "arange"}}}


class CasesTest(unittest.TestCase):
    def test_default_output_is_outside_current_repository(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"VLLM_KERNEL_TOOLS_HOME": str(Path(tmp) / "data")}):
            output = new_run_path()
            scan = new_run_path("scans")
            self.assertTrue(output.is_relative_to((Path(tmp) / "data/runs").resolve()))
            self.assertTrue(scan.is_relative_to((Path(tmp) / "data/scans").resolve()))
            self.assertFalse(output.exists())

    def test_legacy_kwargs_jsonl_and_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            old = case()
            old["kwargs"] = old.pop("arguments")
            path.write_text(json.dumps(old) + "\n" + json.dumps(case("second")))
            self.assertEqual(len(load_cases(path)), 2)
            self.assertEqual(select_cases(load_cases(path), case_name="second")[0]["name"], "second")

    def test_duplicate_ambiguous_and_unsupported_tensor(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_cases([case(), case()])
        with self.assertRaisesRegex(ValueError, "multiple kernels"):
            select_cases([case(), case(kernel="other")], case_name="ok")
        invalid = case()
        invalid["arguments"]["x"]["stride"] = [2]
        with self.assertRaisesRegex(ValueError, "materializer"):
            validate_cases([invalid])

    def test_pointer_contract_and_explicit_values(self):
        c = case()
        c["arguments"]["x"] = {"shape": [2], "dtype": "uint64", "initializer": "data_ptrs",
                                  "pointees": [{"shape": [4]}, {"shape": [3]}]}
        validate_cases([c])
        c["arguments"]["x"]["shape"] = [1]
        with self.assertRaisesRegex(ValueError, "data_ptrs"):
            validate_cases([c])
        c["arguments"]["x"] = {"shape": [2, 2], "initializer": "values", "values": [[1, 2], [3, 4]]}
        validate_cases([c])
        c["arguments"]["x"]["values"] = [1]
        with self.assertRaisesRegex(ValueError, "count"):
            validate_cases([c])

    def test_file_loader_registers_dataclasses(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "module with spaces.py"
            file.write_text("from __future__ import annotations\nfrom dataclasses import dataclass\n"
                            "@dataclass\nclass Value:\n    n: int\ndef launch():\n    return Value(7)\n")
            self.assertEqual(resolve_symbol(str(file) + ":launch")().n, 7)

    def test_triton_target_may_be_launchable_without_being_callable(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "launchable.py"
            file.write_text("class Launchable:\n    def __getitem__(self, grid): return grid\nkernel = Launchable()\n")
            target = resolve_target(str(file) + ":kernel")
            self.assertEqual(target[(2,)], (2,))

    def test_remote_run_ignores_fingerprint_stored_in_generated_case(self):
        from kernel_tools.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "case.json"
            generated = case()
            generated["source"] = {"runtime_fingerprint": "abc123"}
            input_file.write_text(json.dumps([generated]))
            with patch("kernel_tools.remote.load_target", return_value={"device": "npu:0"}), \
                 patch("kernel_tools.remote.run_remote", return_value=0) as remote:
                self.assertEqual(main(["run", str(input_file), "--target", "test"]), 0)
            self.assertNotIn("expected_identity", remote.call_args.kwargs)

    def test_remote_run_does_not_require_generation_source_lock(self):
        from kernel_tools.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "generation"
            generated = case()
            generated["source"] = {"source_lock": "generated-lock-fingerprint"}
            save_json(root / "cases/case.json", [generated])
            with patch("kernel_tools.remote.load_target", return_value={"device": "npu:0"}), \
                 patch("kernel_tools.remote.run_remote", return_value=0) as remote:
                self.assertEqual(main(["run", str(root), "--npu", "test"]), 0)
            self.assertNotIn("source_lock", remote.call_args.kwargs)

    def test_verify_routes_source_lock_to_npu(self):
        from kernel_tools.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = make_source_lock({"vllm": {"commit": "a" * 40, "dirty": False,
                "files": {"vllm/kernel.py": "b" * 64}}})
            save_json(root / "source-lock.json", lock)
            verified = {"status": "verified", "fingerprint": lock["fingerprint"], "packages": {}}
            with patch("kernel_tools.remote.load_target", return_value={"device": "npu:0"}), \
                 patch("kernel_tools.remote.verify_remote", return_value=verified) as remote, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["verify", str(root), "--npu", "test"]), 0)
            remote.assert_called_once()


class SourceLockTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        subprocess.run(["git", "-C", str(self.root), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(self.root), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.root), "config", "user.name", "Test"], check=True)
        package = self.root / "vllm_ascend"
        package.mkdir()
        (package / "__init__.py").write_text("# package\n")
        (package / "kernel.py").write_text(
            "import triton\n@triton.jit\ndef selected_kernel(x): return x\n")
        (package / "caller.py").write_text(
            "from .kernel import selected_kernel\ndef launch(x): selected_kernel[(1,)](x)\n")
        (package / "unrelated.py").write_text("VALUE = 1\n")
        (self.root / ".gitignore").write_text("/vllm_ascend/_build_info.py\n/vllm_ascend/generated/\n")
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.root), "commit", "-qm", "fixture"], check=True)
        self.head = subprocess.check_output(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True).strip()
        (package / "_build_info.py").write_text("BUILD = 'generated'\n")
        (package / "generated").mkdir()
        (package / "generated/op.py").write_text("# generated op\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_targeted_context_but_complete_package_lock(self):
        (self.root / "vllm_ascend/untracked.py").write_text("# intentional local source\n")
        context, packages = local_source(
            self.root / "vllm_ascend/kernel.py", selectors=["selected_kernel"])
        self.assertEqual(set(context), {
            "vllm_ascend/kernel.py", "vllm_ascend/caller.py"})
        self.assertIn("vllm_ascend/unrelated.py", packages["vllm_ascend"]["files"])
        self.assertIn("vllm_ascend/untracked.py", packages["vllm_ascend"]["files"])
        self.assertNotIn("vllm_ascend/_build_info.py", packages["vllm_ascend"]["files"])
        self.assertNotIn("vllm_ascend/generated/op.py", packages["vllm_ascend"]["files"])
        lock = make_source_lock(packages)
        info = {"imports": {"vllm_ascend": str(self.root / "vllm_ascend/__init__.py")},
                "source_revisions": {"vllm_ascend": {"head": self.head, "dirty": False}}}
        self.assertEqual(verify_source_lock(lock, info)["status"], "verified")
        (self.root / "vllm_ascend/unrelated.py").write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "1 changed"):
            verify_source_lock(lock, info)

    def test_ref_reads_committed_source_instead_of_dirty_worktree(self):
        kernel = self.root / "vllm_ascend/kernel.py"
        kernel.write_text("BROKEN WORKTREE\n")
        context, packages = local_source(kernel, ref=self.head, selectors=["selected_kernel"])
        self.assertIn("@triton.jit", context["vllm_ascend/kernel.py"])
        self.assertFalse(packages["vllm_ascend"]["dirty"])
        self.assertEqual(packages["vllm_ascend"]["commit"], self.head)


FAKE_WORKER = r'''
import argparse, json, time
from pathlib import Path
p=argparse.ArgumentParser()
for key in ('input-file','kernel','case-name','target','device','warmup','profiling-rounds','output'):
    p.add_argument('--'+key,required=True)
a=p.parse_args()
case=json.loads(Path(a.input_file).read_text())[0]
if a.case_name=='timeout':
    print('before timeout',flush=True)
    time.sleep(10)
if a.case_name=='fail':
    print('CompilationError: UB overflow, full diagnostic line')
    Path(a.output).write_text(json.dumps({'error':{'phase':'runtime','reason':'CompilationError: UB overflow'}}))
    raise SystemExit(1)
if a.case_name=='wrong':
    Path(a.output).write_text(json.dumps({'error':{'phase':'correctness','reason':'AssertionError: output differs'}}))
    raise SystemExit(1)
row={k:case[k] for k in ('name','kernel','target')}
row.update(device=a.device,warmup=int(a.warmup),profiling_rounds=int(a.profiling_rounds),
           latencies_ms=[.01]*int(a.profiling_rounds),
           summary={k+'_ms':.01 for k in ('mean','p50','p90','p99','min','max')},
           correctness='not_checked')
if a.case_name=='bad_identity': row['kernel']='wrong_kernel'
if a.case_name=='bad_samples': row['latencies_ms']=[]
Path(a.output).write_text(json.dumps([row]))
'''


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.worker = self.root / "fake worker.py"
        self.worker.write_text(FAKE_WORKER)
        self.output = self.root / "run"
        self.info = {"npu_available": True, "python": "fake-for-protocol-test"}

    def tearDown(self):
        self.temp.cleanup()

    def run_cases(self, cases, **kwargs):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = run_suite(cases, cwd=self.root, output=self.output, rounds=2,
                               timeout=.5, probe_info=self.info,
                               worker_command=[sys.executable, str(self.worker)], **kwargs)
        self.stdout = stdout.getvalue()
        return status

    def rows(self):
        return [r for p in (self.output / "results").glob("*.json")
                for r in json.loads(p.read_text())["scenarios"]]

    def test_failure_timeout_continues_and_only_failed_logs(self):
        self.assertEqual(self.run_cases([case("fail"), case("timeout"), case("ok")]), 1)
        rows = self.rows()
        self.assertEqual([r["status"] for r in rows], ["failed", "failed", "success"])
        self.assertEqual(rows[0]["failure"]["phase"], "compile")
        self.assertEqual(rows[1]["failure"]["phase"], "timeout")
        self.assertEqual(rows[2]["correctness"], "not_checked")
        self.assertAlmostEqual(rows[2]["latency_us"]["mean"], 10.)
        self.assertEqual(len(list((self.output / "logs").rglob("*.log"))), 2)
        self.assertIn("full diagnostic line", (self.output / rows[0]["log"]).read_text())
        self.assertNotIn("latencies_ms", next((self.output / "results").glob("*.json")).read_text())
        self.assertIn("CompilationError: UB overflow, full diagnostic line", self.stdout)
        self.assertIn("kernel/fail: failed — compile", self.stdout)
        self.assertIn("kernel/ok: success — mean=10.000 us", self.stdout)

    def test_reject_wrong_identity_and_missing_samples(self):
        self.run_cases([case("bad_identity"), case("bad_samples")])
        self.assertEqual([r["failure"]["phase"] for r in self.rows()], ["result", "result"])

    def test_correctness_failure_is_separate(self):
        self.run_cases([case("wrong")])
        self.assertEqual(self.rows()[0]["correctness"], "failed")
        self.assertNotIn("latency_us", self.rows()[0])

    def test_blocked_environment_still_writes_every_case(self):
        self.info = {"npu_available": False, "npu_error": "No torch_npu"}
        self.assertEqual(self.run_cases([case(), case("second")]), 1)
        self.assertEqual([r["status"] for r in self.rows()], ["blocked", "blocked"])
        self.assertIn("[environment] blocked: No torch_npu", self.stdout)
        self.assertIn("test_kernel/ok: blocked — No torch_npu", self.stdout)

    def test_source_changed_after_generation_blocks_worker(self):
        self.assertEqual(self.run_cases([case()], expected_identity="mismatched-fingerprint"), 1)
        row = self.rows()[0]
        self.assertEqual(row["status"], "blocked")
        self.assertEqual(row["attempts"], 0)
        self.assertIn("changed since AI", row["failure"]["reason"])

    def test_no_overwrite_and_strict_resume(self):
        snapshot = {"path": str(self.root), "head": "a" * 40, "dirty": False}
        with patch("kernel_tools.runner.source_snapshot", return_value=snapshot):
            self.assertEqual(self.run_cases([case()]), 0)
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.run_cases([case()])
            self.assertEqual(self.run_cases([case()], resume=True), 0)
            self.assertEqual(self.rows()[0]["attempts"], 1)
            with self.assertRaisesRegex(ValueError, "changed"):
                self.run_cases([case("changed")], resume=True)

    def test_device_lock_prevents_overlap(self):
        with device_lock("npu:999"):
            with self.assertRaisesRegex(ValueError, "Another"):
                with device_lock("npu:999"):
                    pass

    def test_busy_device_leaves_blocked_results(self):
        with device_lock("npu:0"):
            self.assertEqual(self.run_cases([case(), case("second")]), 1)
        self.assertEqual([r["status"] for r in self.rows()], ["blocked", "blocked"])


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        self.write("vllm/v1/worker/gpu/sample.py", """import triton
from vllm.ops.outer import launch as imported_launch
@triton.jit
def helper(x): return x + 1
@triton.jit
def local(x): return helper(x)
def wrapper(x): local[(1,)](x)
def caller(x): imported_launch(x)
def dynamic(x): factory()[1](x)
""")
        self.write("vllm/ops/outer.py", """import triton
@triton.jit
def external(x): return x
def launch(x): external[(1,)](x)
""")
        self.commit("v1.0.0")

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], stderr=subprocess.DEVNULL, text=True).strip()

    def write(self, path, value):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(value)

    def commit(self, tag):
        self.git("add", ".")
        self.git("commit", "-qm", tag)
        self.git("tag", tag)

    def test_imported_launch_helper_and_dynamic(self):
        result = scan(self.root, "v1.0.0")
        self.assertEqual({k["kernel"] for k in result["kernels"]}, {"local", "external"})
        self.assertEqual(len(result["unresolved_launches"]), 1)
        self.assertEqual(len(result["unlaunched_jit"]), 1)
        self.assertFalse(result["complete_inventory"])
        external = next(k for k in result["kernels"] if k["kernel"] == "external")
        self.assertEqual(external["source_type"], "imported")
        self.assertEqual(external["launches"][0]["call_path"],
                         ["vllm.v1.worker.gpu.sample.caller", "vllm.ops.outer.launch"])

    def test_helper_change_retests_callers_and_uses_tag_not_branch(self):
        base = scan(self.root, "v1.0.0")
        file = self.root / "vllm/v1/worker/gpu/sample.py"
        file.write_text(file.read_text().replace("x + 1", "x + 2"))
        self.commit("v1.1.0")
        self.git("branch", "v1.0.0")
        target = scan(self.root, "v1.1.0")
        self.assertEqual(scan(self.root, "v1.0.0")["commit"], base["commit"])
        delta = compare(base, target)
        changed = next(x for x in delta["changed"] if x["id"].endswith(".local"))
        self.assertIn("dependency", changed["changes"])
        self.assertNotIn("body", changed["changes"])

class TransportTest(unittest.TestCase):
    def test_remote_run_stops_when_source_verification_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "download"
            lock = make_source_lock({"vllm": {"commit": "a" * 40, "dirty": False,
                "files": {"vllm/kernel.py": "b" * 64}}})
            target = {"host": "simulated", "cwd": str(Path(tmp)), "python": sys.executable,
                      "result_root": str(Path(tmp) / "remote")}
            with patch("kernel_tools.remote.verify_remote",
                       side_effect=ValueError("Source verification failed: vllm commit differs")):
                with self.assertRaisesRegex(ValueError, "Source verification failed"):
                    run_remote(target, [case()], output, source_lock=lock)
            self.assertFalse(output.exists())

    def test_remote_source_verification_with_local_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            package = source / "vllm"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("# fixture\n")
            (package / "kernel.py").write_text("VALUE = 1\n")
            (source / ".gitignore").write_text("/vllm/_build_info.py\n")
            for args in (("init", "-q"), ("config", "user.email", "test@example.invalid"),
                         ("config", "user.name", "Test"), ("add", "."),
                         ("commit", "-qm", "fixture")):
                subprocess.run(["git", "-C", str(source), *args], check=True)
            (package / "_build_info.py").write_text("BUILD = 'generated'\n")
            _, packages = local_source(package)
            lock = make_source_lock(packages)
            target = {"host": "simulated", "cwd": str(source), "python": sys.executable,
                      "pythonpath": [str(source)], "result_root": str(root / "remote")}
            with patch("kernel_tools.remote.ssh_command", side_effect=lambda t, c: ["bash", "-c", c]):
                result = verify_remote(target, lock)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["packages"]["vllm"]["commit"], packages["vllm"]["commit"])

    def test_upload_execute_download_with_local_ssh_transport(self):
        # Exercise real tar transfer / subprocess execution / result download,
        # replacing only the SSH wire with a local shell. No NPU or network.
        import venv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv.EnvBuilder(with_pip=False).create(root / "python")
            target = {"host": "simulated", "cwd": str(root),
                      "python": str(root / "python/bin/python"), "result_root": str(root / "remote")}
            def local_wire(target, command):
                return ["bash", "-c", command]
            with patch("kernel_tools.remote.ssh_command", side_effect=local_wire):
                with contextlib.redirect_stdout(io.StringIO()):
                    status = run_remote(target, [case()], root / "download")
            self.assertEqual(status, 1)
            self.assertTrue((root / "download/report.md").is_file())
            result = json.loads(next((root / "download/results").glob("*.json")).read_text())
            self.assertEqual(result["scenarios"][0]["status"], "blocked")

    def test_remote_resume_reuses_recorded_remote_output(self):
        import venv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "-C", str(source), "init", "-q"], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
            (source / "tracked").write_text("fixture\n")
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
            venv.EnvBuilder(with_pip=False).create(root / "python")
            target = {"host": "simulated", "cwd": str(source),
                      "python": str(root / "python/bin/python"), "result_root": str(root / "remote")}
            output = root / "download"
            with patch("kernel_tools.remote.ssh_command", side_effect=lambda t, c: ["bash", "-c", c]):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run_remote(target, [case()], output), 1)
                    remote_output = json.loads((output / "remote-run.json").read_text())["remote_output"]
                    self.assertEqual(run_remote(target, [case()], output, resume=True), 1)
            manifest = json.loads((output / "remote-run.json").read_text())
            self.assertEqual(manifest["remote_output"], remote_output)
            self.assertEqual(manifest["attempts"], 2)
            self.assertEqual(manifest["status"], "downloaded")

    def test_remote_resume_refuses_changed_input(self):
        import venv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv.EnvBuilder(with_pip=False).create(root / "python")
            target = {"host": "simulated", "cwd": str(root),
                      "python": str(root / "python/bin/python"), "result_root": str(root / "remote")}
            output = root / "download"
            with patch("kernel_tools.remote.ssh_command", side_effect=lambda t, c: ["bash", "-c", c]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_remote(target, [case()], output), 1)
                with self.assertRaisesRegex(ValueError, "cases, target, tool"):
                    run_remote(target, [case("changed")], output, resume=True)

    def test_remote_resume_refuses_while_original_process_is_active(self):
        import venv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv.EnvBuilder(with_pip=False).create(root / "python")
            target = {"host": "simulated", "cwd": str(root),
                      "python": str(root / "python/bin/python"), "result_root": str(root / "remote")}
            output = root / "download"
            with patch("kernel_tools.remote.ssh_command", side_effect=lambda t, c: ["bash", "-c", c]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_remote(target, [case()], output), 1)
            with patch("kernel_tools.remote.remote_run_active", return_value=True):
                with self.assertRaisesRegex(ValueError, "still active"):
                    run_remote(target, [case()], output, resume=True)

    def test_command_quotes_paths_and_values(self):
        target = {"cwd": "/source/a b", "container": "vllm", "pythonpath": ["/source/a b"],
                  "setup": "/opt/a b/setup.sh", "env": {"VALUE": "$(false); x"}}
        command = target_command(target, ["python", "-m", "kernel_tools", "doctor"])
        import shlex
        argv = shlex.split(command)
        self.assertEqual(argv[:5], ["docker", "exec", "vllm", "bash", "-lc"])
        self.assertIn("'VALUE=$(false); x'", argv[-1])

    def test_archive_rejects_traversal_and_symlinks(self):
        for name, kind in [("../../outside", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]:
            with tempfile.TemporaryDirectory() as tmp:
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode="w") as tar:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    tar.addfile(member)
                stream.seek(0)
                with tarfile.open(fileobj=stream) as tar:
                    with self.assertRaisesRegex(ValueError, "Unsafe"):
                        safe_extract(tar, tmp)


if __name__ == "__main__":
    unittest.main()
