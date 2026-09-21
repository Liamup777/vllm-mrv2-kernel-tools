# vLLM Kernel Tools

一个用于 vLLM MRV2 Triton 版本检查和 Ascend 单算子测试的小工具。

- 一条 `pipeline` 命令串联版本扫描、Codex/skill 用例生成、NPU 执行和失败分析。
- `scan`、`generate`、`run` 也可分别执行；每一步直接消费上一步的持久化产物。
- 使用 JSON/JSONL case 直接执行 `kernel[grid](...)`，只测 Triton kernel 本体。
- 一条命令按算子或 case 筛选、运行、汇总；失败保存完整日志。
- 从指定 Git tag 扫描静态候选并比较版本，不切换被测仓库的分支。
- 控制端只需 Python 3.10+；NPU 执行端使用已有 Torch / torch_npu / Triton / CANN 环境。

## 完整流程（推荐入口）

```bash
python3 -m kernel_tools pipeline --repo ../vllm \
  --base v0.28.0 --target v0.29.0 --npu npu165
```

首次使用需先登录 Codex CLI、核对 `kernel-tools.json` 并配置 SSH 密钥登录。加 `--dry-run` 只看计划。默认使用本机 Codex 模型和推理强度配置；`--model` 与 `--reason` 可为单次运行覆盖。详见 [完整流程说明](docs/pipeline.md)。

运行时会显示 1/6 至 6/6 的阶段、逐算子生成进度、实际 Codex CLI/模型，以及长 AI 调用的 30 秒心跳。中断或前置检查阻塞后，用相同命令和输出目录追加 `--resume`；已完成的 AI 复核与源码指纹一致的 case 会被复用。

生成的用例默认保存在本次运行目录的 `cases/` 中；使用 `--cases-output ~/vllm-kernel-cases/v0.29.0` 可指定独立目录。`--output` 仍指定整次运行的报告目录。未设置输出目录时，macOS 默认写入 `~/Library/Application Support/vllm-kernel-tools/`，不会在工具仓库生成运行文件；其他系统使用各自的用户数据目录。可通过 `VLLM_KERNEL_TOOLS_HOME` 改写根目录。

三个阶段也可以独立执行：

```bash
python3 -m kernel_tools scan --repo ../vllm \
  --base v0.28.0 --target v0.29.0 --output ~/kernel-results/scan-029

python3 -m kernel_tools generate --scan ~/kernel-results/scan-029 \
  --repo ../vllm --npu npu162 --output ~/kernel-results/generate-029

python3 -m kernel_tools run ~/kernel-results/generate-029/cases \
  --target npu162 --output ~/kernel-results/run-029
```

`generate` 复用 `scan` 的 AI 复核结果，不重复调用 release review；它仍会用固定代码核对 tag、commit、范围和候选，并读取远端实际源码。`run` 会自动读取生成 case 中的远端源码指纹，环境发生变化时拒绝执行。`cases list/validate` 只用于查看和静态校验已有 case，不负责生成。

## 最快开始

在这个仓库根目录运行，无需安装依赖：

```bash
python -m kernel_tools --help
python -m kernel_tools cases list examples/fill_num_accepted.json
python -m kernel_tools cases validate examples/fill_num_accepted.json
python -m kernel_tools run examples/fill_num_accepted.json --dry-run
```

也可以安装命令行入口（不安装/升级 NPU 依赖）：

```bash
python -m pip install -e . --no-deps
kernel-tools --help
```

`python -m kernel_tools` 与 `kernel-tools` 完全等价。示例有 3 个 `_fill_num_accepted_kernel` case；示例 JSON 的 `source` 记录生成依据，换源码版本后应重新核对。当前框架不内置单算子 reference，运行成功仍会明确记录 `correctness=not_checked`。

## 从本机跑远端 NPU

先生成配置文件：

```bash
python -m kernel_tools init
```

这会在当前目录创建 `kernel-tools.json`。文件已存在时不会覆盖，直接编辑即可。每个 `targets` 成员是一套 NPU 环境，名称由 `--target` 或 `--npu` 引用。例如 Docker 环境：

```json
{
  "ai": {
    "codex": "codex"
  },
  "targets": {
    "npu162": {
      "host": "root@192.168.13.162",
      "container": "vllm_lmt",
      "python": "python3",
      "cwd": "/home/lingmutian/code/vllm-ascend",
      "pythonpath": [
        "/home/lingmutian/code/vllm",
        "/home/lingmutian/code/vllm-ascend"
      ],
      "setup": "/usr/local/Ascend/ascend-toolkit/set_env.sh",
      "device": "npu:0",
      "result_root": "/home/lingmutian/triton_kernel",
      "env": {
        "ASCEND_RT_VISIBLE_DEVICES": "0"
      }
    }
  }
}
```

字段含义：

| 字段 | 是否必需 | 含义 |
|---|---|---|
| `host` | 是 | SSH alias 或 `user@host`。工具使用 `BatchMode=yes`，需提前配置 SSH key/agent 和主机指纹。 |
| `container` | 否 | Docker 容器名。设置后，后续路径和 Python 都按容器内环境解释；省略则直接在远端宿主机运行。 |
| `python` | 是 | 远端或容器内 Python 命令，例如 `python3` 或虚拟环境绝对路径。该环境需已安装 Torch、torch_npu、Triton 和 vLLM 运行依赖。 |
| `cwd` | 是 | vLLM-Ascend 工作目录，必须是远端运行环境中的绝对路径。 |
| `pythonpath` | 否 | 实际使用的 vLLM、vLLM-Ascend 等源码路径，必须使用绝对路径。顺序就是 `PYTHONPATH` 顺序。 |
| `setup` | 否 | 执行 Python 前通过 `source` 加载的 Ascend 环境脚本。Docker 模式下必须是容器内路径。 |
| `device` | 否 | 进程内逻辑设备，默认 `npu:0`。 |
| `result_root` | 是 | 远端结果保留目录，必须是绝对路径；Docker 模式下是容器内路径，建议对应持久化挂载目录。 |
| `env` | 否 | 额外环境变量，例如用 `ASCEND_RT_VISIBLE_DEVICES` 把物理卡映射为进程内的 `npu:0`。不要在这里保存密码或 token。 |

裸机环境删除 `container` 即可，例如：

```json
{
  "host": "npu-host-alias",
  "python": "/opt/venvs/vllm/bin/python",
  "cwd": "/workspace/vllm-ascend",
  "pythonpath": ["/workspace/vllm", "/workspace/vllm-ascend"],
  "setup": "/usr/local/Ascend/ascend-toolkit/set_env.sh",
  "device": "npu:0",
  "result_root": "/workspace/kernel-results"
}
```

### 配置 SSH 免密连接

工具执行 SSH 时设置了 `BatchMode=yes`，不会弹出密码输入框，也不读取或保存服务器密码。需要提前配置 SSH key 或可用的 ssh-agent。

本机还没有密钥时先生成一对：

```bash
ssh-keygen -t ed25519 -C "vllm-kernel-tools"
```

第一次把公钥安装到 NPU 服务器时，可以在终端中手动输入一次服务器密码：

```bash
ssh-copy-id root@192.168.13.162
```

如果系统没有 `ssh-copy-id`，可以使用：

```bash
cat ~/.ssh/id_ed25519.pub | ssh root@192.168.13.162 \
  'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys'
```

随后用与工具完全相同的非交互方式检查认证：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 root@192.168.13.162 true
```

命令无输出且退出码为 0，就可以在配置中使用 `"host": "root@192.168.13.162"`。也可以在 `~/.ssh/config` 中配置私钥、端口、跳板机或 alias：

```sshconfig
Host npu162
  HostName 192.168.13.162
  User root
  IdentityFile ~/.ssh/id_ed25519
```

此时 `kernel-tools.json` 可简化为 `"host": "npu162"`。如果服务器只允许密码登录且不能安装公钥，当前工具不能连接；不要把密码写入配置文件或命令行。

工具不要求 NPU 环境预先安装 `vllm-kernel-tools`。运行时会通过 SSH 把工具代码和 JSON case 上传到远端临时目录；目标环境只需具备被测运行栈。Docker 模式还要求远端用户能执行 `docker exec`、`docker cp`，且目标容器已经启动。

配置后先检查连接、环境和实际 import 路径：

```bash
python -m kernel_tools doctor --config kernel-tools.json --target npu162
```

`doctor` 会显示 Python、Torch、torch_npu、Triton、CANN、NPU 状态，以及 vLLM/vLLM-Ascend 的实际 import 位置。若这里显示的源码路径与 `pythonpath` 预期不一致，应先修正配置，不要直接运行 pipeline。

只检查将要使用的 SSH/Docker 命令而不连接远端：

```bash
python -m kernel_tools doctor --config kernel-tools.json --target npu162 --dry-run
```

确认环境后运行单算子 case：

```bash
python -m kernel_tools run examples/fill_num_accepted.json \
  --config kernel-tools.json --target npu162 \
  --case-name smoke_b1 --output artifacts/smoke

python -m kernel_tools run examples/fill_num_accepted.json \
  --config kernel-tools.json --target npu162 --output artifacts/fill-all
```

工具自动上传自身和 JSON case 到临时目录，在容器中使用配置的源码运行，再把报告、case、结果、失败日志取回本地。远端结果也保留在配置的 `result_root`。无需切换 vllm-ascend 到 `kernel_test_frame`，也不会重装远端 Torch/Triton。输入构造由统一测试框架提供，自动流程不生成辅助 Python 文件或单算子 reference。

查看计划而不连接远端：

```bash
python -m kernel_tools run examples/fill_num_accepted.json --target npu165 --dry-run
```

## 在 NPU 环境内直接运行

在已初始化 Ascend 环境的终端中：

```bash
python -m kernel_tools doctor
python -m kernel_tools run examples/fill_num_accepted.json \
  --cwd /home/lingmutian/code/vllm-ascend \
  --pythonpath /home/lingmutian/code/vllm \
  --pythonpath /home/lingmutian/code/vllm-ascend \
  --device npu:0 --warmup 10 --rounds 100 --timeout 600 \
  --output artifacts/fill-all
```

`npu:0` 是进程中的逻辑设备编号；需要物理卡映射时在配置的 `env` 中设置 `ASCEND_RT_VISIBLE_DEVICES`。设备锁只约束本工具的作业，不能阻止外部程序占用 NPU；不要把拿到锁当作设备独占证明。

支持输入一个目录运行其中所有 `.json` / `.jsonl` case，支持 `--kernel` 和 `--case-name` 精确筛选。失败不会中止后续 case；Ctrl-C 会记录当前中断并把剩余项标为 blocked。设备环境不可用时所有选中 case 都有 blocked 结果。

本地断点续跑：使用完全相同的命令与 `--output`，追加 `--resume`。只有已成功项被复用，失败项重跑。源码需为干净 Git checkout；case、源码 HEAD、工具代码、环境或测量参数发生变化时拒绝复用。远端续跑需进入容器，从保留的结果目录继续；SSH 断开后先确认原进程状态，避免重复启动。

## 结果只有需要的东西

```text
artifacts/fill-all/
  cases/       # 真正执行的 case，每个算子一份
  results/     # 每个算子的所有场景结果和实际绑定
  logs/        # 仅失败场景的完整日志
  report.md    # 场景结果、耗时、失败原因与日志链接
```

文件名带短哈希，防止同名/长名称覆盖。成功只保留 mean/p50/p90/p99/min/max；不保存逐轮耗时、中间输入或成功日志。当前没有通用数值比较协议，因此报告统一使用 `correctness=not_checked`，不能把编译和运行成功解释为数值正确。

## 扫描某个版本的变化

```bash
python -m kernel_tools scan --repo ../vllm \
  --base v0.28.0 --target v0.29.0 \
  --output ~/vllm-kernel-tools-output/v0.28.0--v0.29.0
```

`scan` 自动调用 Codex 和 release-scan skill，只分析源码，不连接 NPU。控制端需要可用且已登录的 Codex CLI；可读取 `kernel-tools.json` 的 `ai.codex`，无需配置 NPU 目标。

输出 `review.md` 和 `review.json`。报告分别列出 AI 确认新增、待核实和已有/非新增项，不把已有算子的变化混入新增数。静态 AST 候选只作为内部辅助；读取精确 tag，不切换源码分支。

AI 会检查别名和显式导入的外部 Triton kernel，但复核仍可能遗漏。报告不是全覆盖证明，也不是 NPU 数值正确性或性能报告。AI 失败时明确记录失败及日志，不退回静态结果冒充复核完成。

扫描的具体实现和 AI 参与的边界见 [扫描原理与手动流程](docs/scanning.md)。

## 目录与开发

| 目录 | 内容 |
|---|---|
| `kernel_tools/` | 固定程序：CLI、扫描、校验、SSH/Docker 执行、报告 |
| `examples/` | 可读、可筛选的最小 case 示例 |
| `skills/` | AI 处理调用链、输入契约和失败分析时的规范 |
| `docs/` | case 扩展、扫描原理与手动流程 |
| `tests/` | 无需 NPU 的协议、异常路径和版本扫描检查 |

```bash
python -m unittest discover -s tests -v
```

本仓库测试模拟 worker 来检查编排协议、日志、超时和重跑规则，不能代替真实 NPU 编译/精度/性能验证。框架来源与改动见 [NOTICE](NOTICE)。
