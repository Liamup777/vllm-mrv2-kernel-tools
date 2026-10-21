"""Offline first-use checks; these are not AI-quality or NPU execution tests."""
import contextlib
import io
import json
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kernel_tools.cli import main, parser, template
from kernel_tools.remote import load_target, ssh_command, target_command
from kernel_tools.review import resource_root


class OnboardingTest(unittest.TestCase):
    def test_init_uses_public_template_and_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "kernel-tools.json"
            with contextlib.redirect_stdout(io.StringIO()), patch("subprocess.run") as run:
                self.assertEqual(main(["init", "--output", str(config)]), 0)
            run.assert_not_called()
            self.assertEqual(json.loads(config.read_text()), template())
            target = load_target(config, "my-npu")
            self.assertEqual(target["host"], "npu-lab")
            self.assertEqual(target["cwd"], "/workspace/vllm-ascend")
            original = config.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["init", "--output", str(config)]), 2)
            self.assertEqual(config.read_bytes(), original)

    def test_doctor_dry_run_uses_batchmode_without_connecting(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps(template()))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), patch("subprocess.run") as run:
                self.assertEqual(main(["doctor", "--config", str(config),
                                       "--npu", "my-npu", "--dry-run"]), 0)
            run.assert_not_called()
            self.assertIn("docker", stdout.getvalue())
            target = load_target(config, "my-npu")
            command = target_command(target, ["python3", "--version"])
            self.assertIn("docker exec vllm-container", command)
            self.assertIn("BatchMode=yes", ssh_command(target, command))
            target.pop("container")
            self.assertNotIn("docker", target_command(target, ["python3", "--version"]))

    def test_prepare_only_dry_run_no_config_or_output_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), patch("subprocess.run") as run:
                code = main(["pipeline", "--repo", tmp, "--base", "v1", "--target", "v2",
                             "--config", str(Path(tmp) / "missing.json"), "--output", str(output),
                             "--prepare-only", "--dry-run"])
            self.assertEqual(code, 0)
            run.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(json.loads(stdout.getvalue())["steps"][-1], "stop after generation")

    def test_skill_and_reference_links_exist(self):
        root = resource_root()
        files = [root / "README.md", *list((root / "skills").rglob("*.md"))]
        self.assertGreaterEqual(len(files), 8)
        for file in files:
            prose = re.sub(r"`[^`]*`", "", file.read_text())
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", prose):
                if "://" in target or target.startswith("#"):
                    continue
                self.assertTrue((file.parent / target.split("#", 1)[0]).exists(),
                                f"Broken link in {file}: {target}")

    def test_readme_config_matches_template(self):
        readme = (resource_root() / "README.md").read_text()
        examples = re.findall(r"```json\n(.*?)\n```", readme, re.S)
        configs = [json.loads(example) for example in examples]
        self.assertEqual([config for config in configs if "targets" in config], [template()])

    def test_readme_shell_syntax_and_cli_arguments(self):
        readme = (resource_root() / "README.md").read_text()
        blocks = re.findall(r"```bash\n(.*?)\n```", readme, re.S)
        checked = 0
        for block in blocks:
            subprocess.run(["bash", "-n"], input=block, text=True,
                           check=True, capture_output=True)
            for line in block.replace("\\\n", " ").splitlines():
                if line.startswith("kernel-tools ") and "--help" not in line:
                    parser().parse_args(shlex.split(line)[1:])
                    checked += 1
        self.assertGreaterEqual(checked, 10)

    def test_readme_project_registration_preserves_existing_entries(self):
        root = resource_root()
        readme = (root / "README.md").read_text()
        blocks = re.findall(r"```bash\n(.*?)\n```", readme, re.S)
        recipe, = [block for block in blocks if 'skill_link=' in block]
        with tempfile.TemporaryDirectory(prefix="skill registration ") as tmp:
            project = Path(tmp).resolve() / "project with spaces"
            recipe = recipe.replace('project_dir="/absolute/path/to/your/project"',
                                    "project_dir=" + shlex.quote(str(project)))
            link = project / ".agents/skills/vllm-mrv2-kernel-workflow"

            def run():
                return subprocess.run(["bash", "-eu", "-c", recipe], cwd=root,
                                      check=True, capture_output=True, text=True)

            run()
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), root / "skills/vllm-mrv2-kernel-workflow")
            self.assertTrue((link.resolve() / "references/setup.md").is_file())
            self.assertTrue((link.resolve().parent / "vllm-ascend-kernel-cases/SKILL.md").is_file())
            run()
            self.assertEqual(link.resolve(), root / "skills/vllm-mrv2-kernel-workflow")
            link.unlink()
            link.write_text("existing user file")
            run()
            self.assertEqual(link.read_text(), "existing user file")
            link.unlink()
            link.mkdir()
            (link / "keep").write_text("existing directory")
            run()
            self.assertEqual(list(link.iterdir()), [link / "keep"])
            (link / "keep").unlink()
            link.rmdir()
            missing = project / "missing skill"
            link.symlink_to(missing)
            run()
            self.assertEqual(link.readlink(), missing)

    def test_control_plane_has_no_accelerator_imports(self):
        code = ("import sys; from kernel_tools.cli import template; "
                "assert template()['targets']; "
                "assert not {'torch', 'torch_npu', 'triton'} & sys.modules.keys()")
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
