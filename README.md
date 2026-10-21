# vLLM Kernel Tools

面向 vLLM MRV2 Triton 变更检查与 Ascend 单算子测试的 **Python 工具 + Agent Skill**。

- 比较两个 Git tag 或 commit，复核直接 launch 的 Triton 算子变化。
- 根据本地 vLLM / vLLM-Ascend 源码，为确认新增的算子或指定 kernel 生成 JSON 用例。
- 可选连接 SSH / Docker NPU 环境，校验源码 commit，逐 case 执行并保存结果和失败日志。
- 用户从总入口 Skill 提出任务，固定 Python 程序负责各阶段编排；无需手动依次调用内部 Skill。

边界：CLI `scan` 是两版本差异复核，不是单版本全量 XLSX 清单工具。用例直接执行 `kernel[grid](...)`；运行成功只说明执行/测量完成，当前数值正确性统一为 `correctness=not_checked`。

## 1. 获取仓库与安装工具

本机需要 Python 3.11+、Git。扫描和用例生成还需要单独安装、认证可用的 Codex CLI。仅查看帮助、校验 case、dry-run 不需要 AI 或 NPU。远端需用户自行准备相互兼容的 Torch、torch_npu、Triton-Ascend、CANN、vLLM 和 vLLM-Ascend 环境。

```bash
git clone https://github.com/Liamup777/vllm-mrv2-kernel-tools.git
cd vllm-mrv2-kernel-tools

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e . --no-deps
kernel-tools --help
```

已有仓库和可用 Python 环境时，无需重新 clone 或创建环境，在仓库根目录安装即可。控制端没有第三方 Python 运行依赖；构建安装可能下载 setuptools，不会安装或升级 NPU 软件栈。

`kernel-tools` 与同一 Python 环境中的 `python -m kernel_tools` 等价。若不安装，也可从仓库根目录直接使用后者。让 Agent 执行时，应提供该虚拟环境的 Python/CLI 路径，不能假定另一个终端的 `activate` 会影响 Agent。

```bash
codex --version
```

这只检查 Codex CLI 可执行文件，不证明认证可用。沿用你自己的 Codex 安装与认证方式；非 PATH 中的可执行文件可通过 `--codex /absolute/path/to/codex` 或配置文件的 `ai.codex` 指定。默认沿用 Codex 模型配置，也可用 `--model`、`--reason` 单次覆盖。

## 2. 加载 Skill

**安装 Python 工具不等于注册 Skill。** 总入口为 [vllm-mrv2-kernel-workflow](skills/vllm-mrv2-kernel-workflow/SKILL.md)，以下两种加载方式任选一种。

### 方式 A：直接提供入口路径

让编程助手读取完整仓库中的入口，例如：

```text
请读取 /absolute/path/to/vllm-mrv2-kernel-tools/skills/vllm-mrv2-kernel-workflow/SKILL.md，
按照这个 Skill 完成下面的任务。
工具使用 /absolute/path/to/vllm-mrv2-kernel-tools/.venv/bin/python。
```

### 方式 B：接入项目的技能目录

对于支持 `.agents/skills/` 和软链接发现的客户端，在 Linux/macOS 上从本工具仓库根目录执行。将 `project_dir` 替换为实际项目的绝对路径：

```bash
skill_repo="$(pwd -P)"
project_dir="/absolute/path/to/your/project"
skill_link="$project_dir/.agents/skills/vllm-mrv2-kernel-workflow"
mkdir -p "$project_dir/.agents/skills"
if [ -e "$skill_link" ] || [ -L "$skill_link" ]; then
  echo "已有同名条目，请先检查：$skill_link"
else
  ln -s "$skill_repo/skills/vllm-mrv2-kernel-workflow" "$skill_link"
fi
```

此命令只链接总入口，不覆盖已有文件、目录或软链接，也不修改 SSH/NPU 配置。保留整个工具仓库，不要移动或删除链接目标。入口要求按软链接的真实位置解析相邻 references、内部 Skill 和文档；不支持这种加载方式的客户端请使用方式 A。

在目标项目中确认入口已被识别。Codex CLI/IDE 可通过 `/skills` 查看或用 `$vllm-mrv2-kernel-workflow` 显式调用；未刷新时重启客户端。机制见 [Codex Skill 官方说明](https://learn.chatgpt.com/docs/build-skills)。其他客户端按自己的技能发现规则加载，不假定调用语法相同。

不能只孤立复制一个 `SKILL.md` 或单个 Skill 目录：本工作流依赖 Python 工具、内部 Skill 和相邻文档。普通 wheel 也包含完整资源，可在已安装的 Python 环境中定位：

```bash
python -c 'from kernel_tools.review import resource_root; print(resource_root())'
```

## 3. 本地使用：无需 NPU 配置

先在工具仓库根目录检查本机工具，不会调用 AI 或连接远端：

```bash
kernel-tools cases list examples/fill_num_accepted.json
kernel-tools cases validate examples/fill_num_accepted.json
kernel-tools run examples/fill_num_accepted.json --dry-run
```

下面将路径、版本和输出目录替换为实际值。`--base` / `--target` 支持本地 tag 或 7–40 位十六进制 commit SHA，推荐完整 SHA；必须已存在于本地仓库，不接受分支名或 `HEAD~1` 等表达式。工具通过 Git object 读取它们，不切换源码分支。输出目录使用新的目录，恢复旧任务才加 `--resume`。

### 只扫描变化

给助手的任务：

```text
使用 vllm-mrv2-kernel-workflow。
vLLM 仓库：/path/to/vllm
基线：v0.28.0；目标：v0.29.0
只扫描 MRV2 Triton 算子变化，不生成用例、不连接 NPU。
输出到新的目录 /path/to/results/scan-029，交付复核报告和未解决项。
```

也可直接运行 CLI：

```bash
kernel-tools scan --repo /path/to/vllm \
  --base v0.28.0 --target v0.29.0 \
  --output /path/to/results/scan-029
```

此阶段会调用 AI，输出 `review.json` 和 `review.md`；`scan` 没有 `--dry-run`。静态 AST 候选只是辅助，AI 失败不会被包装成“复核通过”。

### 扫描并生成用例，暂不执行

给助手的任务：

```text
使用 vllm-mrv2-kernel-workflow。
比较 /path/to/vllm 的 v0.28.0 与 v0.29.0，
结合 /path/to/vllm-ascend 的本地源码，为确认新增的算子生成用例。
只准备，不连接 NPU。输出到新的目录 /path/to/results/prepared-029。
```

```bash
kernel-tools pipeline --repo /path/to/vllm \
  --source /path/to/vllm-ascend \
  --base v0.28.0 --target v0.29.0 \
  --prepare-only --output /path/to/results/prepared-029
```

`--prepare-only` 仍会调用 AI；再加 `--dry-run` 才只显示计划。已有 scan 结果时用 `generate --scan` 复用复核，不必重复扫描，详见 [分阶段命令](docs/pipeline.md)。

### 只为指定 kernel 生成用例

无需比较版本，给助手 kernel 名和本地源码范围即可：

```bash
kernel-tools generate --kernel _fill_num_accepted_kernel \
  --source /path/to/vllm/vllm/v1/worker/gpu/model_states/mamba_hybrid.py \
  --output /path/to/results/fill-num-accepted
```

按你的源码选择实际存在的 kernel/文件。`--kernel` 和 `--source` 可重复；有重名时使用完整 Python ID。默认读取当前 worktree，`--ref COMMIT` 可选历史源码。此模式跳过 release review，只调用用例生成。

## 4. 配置远端 NPU（仅远端执行需要）

已有可用配置可直接沿用，不需要重新生成密钥或重建 JSON。以下是新用户的配置步骤。

### 4.1 准备 SSH 免密访问

`kernel-tools` 通过 OpenSSH 的 `BatchMode=yes` 连接，不弹出密码提示，也不保存密码。私钥、端口、跳板机配置属于 SSH，而不是 JSON。

在本机 `~/.ssh/config` 中按实际环境添加或复用别名，例如：

```sshconfig
Host npu-lab
    HostName npu.example.com
    User developer
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

`HostName`、用户和私钥路径都需替换。优先复用已有密钥；尚未安装公钥时，可按管理员流程或 `ssh-copy-id -i /path/to/key.pub user@host` 配置。首次连接应核对服务器指纹，不关闭主机密钥校验。随后检查非交互认证：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 npu-lab true
```

这只验证 SSH，不证明 Docker、Python 或 NPU 栈可用。

### 4.2 配置执行目标

在工具仓库根目录执行：

```bash
kernel-tools init --output kernel-tools.json
```

命令从 [通用模板](kernel-tools.example.json) 创建文件，不连接远端，不覆盖已有配置。将示例值改为实际环境：

```json
{
  "ai": {
    "codex": "codex"
  },
  "targets": {
    "my-npu": {
      "host": "npu-lab",
      "container": "vllm-container",
      "python": "python3",
      "cwd": "/workspace/vllm-ascend",
      "pythonpath": ["/workspace/vllm", "/workspace/vllm-ascend"],
      "setup": "/usr/local/Ascend/ascend-toolkit/set_env.sh",
      "device": "npu:0",
      "result_root": "/workspace/kernel-results"
    }
  }
}
```

| 项目 | 配置含义及注意点 |
|---|---|
| `my-npu` | 工具的目标名称，由 `--npu my-npu` 选择；不是 IP 或卡号 |
| `host` | SSH 别名或 `user@host`；可以与目标名称不同 |
| `container` | 已启动的 Docker 容器名；裸机运行时删除这一项 |
| `python` | 远端执行环境中的 Python，必须能导入被测软件栈 |
| `cwd` | 远端工作目录，必须为绝对路径 |
| `pythonpath` | 远端 vLLM / vLLM-Ascend 源码根目录，按列表顺序加入 `PYTHONPATH` |
| `setup` | 可选的 Ascend 环境脚本，执行 Python 前加载 |
| `device` | 进程内逻辑设备，CLI `--device` 可覆盖；需核对容器和可见设备映射 |
| `result_root` | 远端结果保留目录；Docker 模式建议位于持久化挂载路径 |
| `env` | 可选环境变量字典，例如 `ASCEND_RT_VISIBLE_DEVICES`；不要放密码或 token |

路径分两侧：CLI 的 `--repo`、`--source`、`--config`、`--output` 是本地路径；JSON 中 `cwd`、`pythonpath`、`setup`、`result_root` 是远端路径，设置 `container` 后均按容器内路径解释。`--output` 和 `result_root` 不是同一个目录。

`device` 不是始终等于宿主机物理卡号。设置可见设备变量或使用部分设备挂载后，应在实际容器/Python 环境中确认逻辑编号与物理卡的对应关系，再选空闲设备；不能仅凭 `npu:0` 判断用了哪张卡。

工具会上传运行器和用例，不要求远端预装本工具，但不会自动安装 NPU 依赖、启动容器、同步生产源码或切换远端分支。Docker 模式需当前 SSH 用户有 `docker exec` / `docker cp` 权限。

配置文件在本仓库内已被 Git 忽略；若保存到其他项目也需自行排除。不要提交真实机器配置、密码、token 或私钥。配置可放任意私有位置，执行时显式传 `--config /absolute/path/to/kernel-tools.json`，避免 Agent 工作目录不同而读错文件。

### 4.3 检查配置与实际环境

```bash
# 只看计划，不连接远端
kernel-tools doctor --config /path/to/kernel-tools.json --npu my-npu --dry-run

# 实际检查 SSH、容器、Python import 和 NPU 环境
kernel-tools doctor --config /path/to/kernel-tools.json --npu my-npu
```

检查实际 import 路径、包版本、设备可用性和占用状态，不只看 SSH 是否成功。`doctor` 通过也不代表待测 kernel 可编译或数值正确。

## 5. 执行远端测试

### 使用已生成用例

先列出用例、检查格式，并校验远端实际导入仓库的 commit：

```bash
kernel-tools cases list /path/to/results/prepared-029/cases
kernel-tools cases validate /path/to/results/prepared-029/cases
kernel-tools verify /path/to/results/prepared-029 \
  --config /path/to/kernel-tools.json --npu my-npu
```

`verify` 成功且源码保持不变后，选择实际列出的一个 smoke case，再运行完整用例。下面的 `KERNEL_NAME` 和 `SMOKE_CASE_NAME` 必须替换：

```bash
kernel-tools run /path/to/results/prepared-029 \
  --kernel KERNEL_NAME --case-name SMOKE_CASE_NAME \
  --config /path/to/kernel-tools.json --npu my-npu \
  --output /path/to/results/smoke-029

kernel-tools run /path/to/results/prepared-029 \
  --config /path/to/kernel-tools.json --npu my-npu \
  --output /path/to/results/run-029
```

上述执行沿用配置中的设备。独立 `run` **不会自动校验 source lock**；有 lock 时先 `verify`，失败就停止。手写无 lock 的 case 仍能运行，但应另行记录实际源码身份。

### 从扫描到执行一条命令完成

给助手的任务示例：

```text
使用 vllm-mrv2-kernel-workflow 执行完整流程。
vLLM：/path/to/vllm；Ascend：/path/to/vllm-ascend
基线：v0.28.0；目标：v0.29.0
配置：/path/to/kernel-tools.json；目标名称：my-npu；设备沿用配置。
输出：新的目录 /path/to/results/pipeline-029。
执行前检查环境和设备占用；源码 commit 不一致时停止，不自动切换远端分支。
交付每个 case 的执行状态、报告路径及未验证的覆盖范围。
```

```bash
kernel-tools pipeline --repo /path/to/vllm \
  --source /path/to/vllm-ascend \
  --base v0.28.0 --target v0.29.0 \
  --config /path/to/kernel-tools.json --npu my-npu \
  --output /path/to/results/pipeline-029
```

完整 `pipeline` 在执行前自动校验 source lock；远端 vLLM 必须匹配 target commit，所选 Ascend 源码也需匹配锁定的 commit。base 只需本地存在。校验只比对实际 import 对应的 Git HEAD，不检查工作区修改或逐文件哈希。

pipeline 不包含单独的 smoke-first 阶段，需要先试一个 case 时请使用前面的分阶段方式。完整命令加 `--dry-run` 不调用 AI、不连接远端，但需可解析的目标配置。

### 在 NPU 环境内直接运行

若本工具已在 NPU 环境的 Python 中可用，也可不经过 SSH：

```bash
kernel-tools doctor
kernel-tools run /path/to/cases.json \
  --cwd /workspace/vllm-ascend \
  --pythonpath /workspace/vllm --pythonpath /workspace/vllm-ascend \
  --device npu:0 --warmup 10 --rounds 100 --timeout 600 \
  --output /workspace/kernel-results/local-run
```

先初始化该环境的 CANN/设备设置。有 source lock 时也应先在同一 import 环境显式 `verify`。设备锁只约束本工具的作业，不证明 NPU 没有被其他程序占用。

## 6. 结果、恢复与排错

| 阶段 | 主要产物 | 能说明什么 |
|---|---|---|
| scan | `review.json`、`review.md` | AI 源码复核及未解决项，不是运行覆盖证明 |
| generate / prepare | `cases/`、`source-lock.json`、`workflow.json`、`report.md` | 用例已准备，不代表已执行 |
| run | `cases/`、`results/`、`report.md`、失败 `logs/` | 每个 case 的实际执行/测量状态；SSH 模式另保留本地 `remote-run.json` |
| pipeline | 上述对应阶段的产物 | 按最后到达的阶段解释，不能把准备完成当成测试通过 |

运行过程显示阶段、逐算子进度和 AI 调用心跳；失败保留完整日志。当前不提供通用数值比较协议，`success` 不能解释为精度正确、模型功能已覆盖或端到端性能提升。

`scan`、`generate`、`run`、`pipeline` 支持原 `--output` 加 `--resume`，但须保持对应输入身份不变。SSH 中断后先确认旧作业状态；独立远端 run 会检查活动进程，pipeline 恢复可能新建执行目录，不要盲目重复启动。源码变更后重新核对 lock 并使用新输出目录。细节见 [恢复规则](docs/pipeline.md)。

可用 `--cases-output` 指定独立用例目录。默认运行产物位于系统用户数据目录，不写工具仓库；macOS 使用 `~/Library/Application Support/vllm-kernel-tools/`，可用 `VLLM_KERNEL_TOOLS_HOME` 修改根目录。

常见问题：

- 找不到 `kernel-tools`：确认 Agent/终端使用安装它的 Python 环境，或使用 `.venv/bin/kernel-tools` 的绝对路径。
- Skill 未发现：检查客户端技能目录和软链接目标，或直接提供完整 `SKILL.md` 路径。
- SSH 超时/认证失败：先检查 `ssh -o BatchMode=yes ... true`，不要把密码写进 JSON。
- 容器未运行：核对 `container` 和 Docker 权限，不能当作 kernel 失败。
- commit 不一致：停止测试，核对 `pythonpath`、实际 import 和目标版本，不绕过 source lock。
- Triton 导入/编译失败：按日志区分环境兼容性和 kernel 输入/实现问题，不能直接归因于算子。

更多细节见 [排错参考](skills/vllm-mrv2-kernel-workflow/references/troubleshooting.md)、[case 格式](docs/cases.md) 和 [扫描原理](docs/scanning.md)。

## Skill 目录与开发验证

| Skill | 职责 |
|---|---|
| [vllm-mrv2-kernel-workflow](skills/vllm-mrv2-kernel-workflow/SKILL.md) | 用户总入口；按需读取配置、流程、验收及排错 references |
| [vllm-triton-release-scan](skills/vllm-triton-release-scan/SKILL.md) | 控制器内部的差异复核 |
| [vllm-ascend-kernel-cases](skills/vllm-ascend-kernel-cases/SKILL.md) | 控制器内部的输入契约分析与用例生成 |
| [vllm-triton-remote-benchmark](skills/vllm-triton-remote-benchmark/SKILL.md) | 远端执行指导；控制器调用时只做只读失败分析 |

在工具仓库根目录验证：

```bash
python -m unittest discover -s tests -v
python scripts/check_install.py
```

第一项使用模拟 AI/worker/SSH 验证协议和异常路径；第二项构建 wheel，在临时虚拟环境、仓库外检查 CLI 和打包资源，可能下载构建依赖。它们不代表真实 AI 生成质量或 NPU 编译、精度、性能通过。

框架来源与改动见 [NOTICE](NOTICE)。
