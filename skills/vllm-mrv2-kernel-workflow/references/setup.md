# Setup and target configuration

## Local host

Use Python 3.11+, Git and the complete repository. The control-plane Python
package has no third-party runtime dependencies. Install its CLI with
`python3 -m pip install -e . --no-deps`, or run `python3 -m kernel_tools` from
the repository root. Build/install tooling may download setuptools.

Scan and generation invoke the separately installed, authenticated Codex CLI.
Use `codex --version` to check executable discovery; a dry run does not verify
authentication. Use the user's configured model unless they select an override.
`--codex /path/to/codex` or private `ai.codex` can select the executable.

Installing the Python CLI does not register the skill. Either give the agent the
absolute path to `skills/vllm-mrv2-kernel-workflow/SKILL.md`, or link that entry
directory into a supported project's `.agents/skills/` location. The tested,
non-overwriting shell recipe and user prompts are in [README](../../../README.md).
Keep the complete repository; resolve a symlink to its real location before
following references. Copying only this skill directory loses sibling resources.
Do not register all internal stages merely to make the entry discoverable.

Give the agent the CLI/Python executable from the installed environment; activating
a venv in a separate terminal does not select the agent's interpreter. An installed
wheel preserves resources under `<python-prefix>/share/vllm-kernel-tools/`; locate
them with `python -c 'from kernel_tools.review import resource_root; print(resource_root())'`
using that environment's Python.

## Private remote configuration

Only remote execution requires this step. Run:

```bash
python3 -m kernel_tools init --output kernel-tools.json
```

This copies the [public example](../../../kernel-tools.example.json) and refuses
to overwrite an existing file. Edit `my-npu` for the user's environment. An
existing private configuration should be reused, not regenerated. Pass it through
`--config /absolute/path/to/kernel-tools.json` rather than assuming the agent's cwd.
`kernel-tools.json` is ignored in this repository; keep copies elsewhere private too.

| Field | Meaning |
|---|---|
| `targets.my-npu` | Target name selected by `--npu my-npu`, not a host address or card number |
| `host` | Existing SSH alias or user@host; authentication belongs to OpenSSH |
| `container` | Optional running Docker container; omit for a remote bare-metal environment |
| `python` | Python executable in the selected remote environment |
| `cwd` | Absolute remote working directory |
| `pythonpath` | Absolute source roots, in import precedence order |
| `setup` | Optional environment script sourced before Python |
| `device` | Logical NPU index, e.g. `npu:0`; select an available device |
| `result_root` | Absolute persistent output directory in the selected environment |
| `env` | Optional non-secret environment variables, including physical-to-logical device mapping |

In Docker mode all paths above are container paths. Local `--repo` and `--source`
are host-side paths and need not match them. `--config` and `--output` are also
local; remote `result_root` is a separate retained copy. Remove `container` for
bare-metal execution. The tool uploads its runner and cases, not production source.
It does not install dependencies, start containers or switch source branches;
the user supplies the remote vLLM/Ascend source and compatible stack.

Device selection refers to process-local logical indices. Check container device
mounts and visibility environment variables before relating them to physical cards.
An SSH alias can differ from the JSON target name: the example selects `my-npu`
but connects to `npu-lab`.

Example private SSH entry (replace the example hostname and account):

```sshconfig
Host npu-lab
    HostName npu.example.com
    User developer
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

Inspect the planned command, then check the configured environment:

```bash
python3 -m kernel_tools doctor --config kernel-tools.json --npu my-npu --dry-run
ssh -o BatchMode=yes -o ConnectTimeout=10 npu-lab true
python3 -m kernel_tools doctor --config kernel-tools.json --npu my-npu
```

The SSH alias must already have working key authentication and trusted host-key
configuration. Do not put passwords/private keys in JSON or change host-key policy
to hide a connection failure. For setup-only requests, stop after the requested
checks; do not start kernel workloads. For execution, inspect imports, package
versions, device availability and container state before choosing a device.

The target needs compatible PyTorch, torch_npu, Triton-Ascend, CANN and source
dependencies. A usable tool installation or SSH connection alone does not establish
that stack's compatibility.
