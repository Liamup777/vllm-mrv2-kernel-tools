# vLLM Kernel Tools

一个用于 vLLM MRV2 Triton 版本检查和 Ascend 单算子测试的小工具。

- 一条 `pipeline` 命令串联版本扫描、Codex/skill 用例生成、NPU 执行和失败分析。
- 使用 JSON/JSONL case 直接执行 `kernel[grid](...)`，只测 Triton kernel 本体。
- 一条命令按算子或 case 筛选、运行、汇总；失败保存完整日志。
- 从指定 Git tag 扫描静态候选并比较版本，不切换被测仓库的分支。
- 控制端只需 Python 3.10+；NPU 执行端使用已有 Torch / torch_npu / Triton / CANN 环境。

## 完整流程（推荐入口）

```bash
python3 -m kernel_tools pipeline --repo ../vllm \
  --base v0.28.0 --target v0.29.0 --npu npu165
```

首次使用需先登录 Codex CLI、核对 `kernel-tools.json` 并配置 SSH 密钥登录。加 `--dry-run` 只看计划；加 `--prepare-only` 会调用 AI 并读取远端源码，生成用例后停止。默认使用本机 Codex 模型配置；`--model` 可覆盖。详见 [完整流程说明](docs/pipeline.md)。

运行时会显示 1/6 至 6/6 的阶段、逐算子生成进度、实际 Codex CLI/模型，以及长 AI 调用的 30 秒心跳。中断或前置检查阻塞后，用相同命令和输出目录追加 `--resume`；已完成的 AI 复核与源码指纹一致的 case 会被复用。

生成的用例默认保存在本次运行目录的 `cases/` 中；使用 `--cases-output ~/vllm-kernel-cases/v0.29.0` 可指定独立目录。`--output` 仍指定整次运行的报告目录。未设置输出目录时，macOS 默认写入 `~/Library/Application Support/vllm-kernel-tools/`，不会在工具仓库生成运行文件；其他系统使用各自的用户数据目录。可通过 `VLLM_KERNEL_TOOLS_HOME` 改写根目录。

`scan` 通过 `codex exec` 调用 AI 和 release-scan skill 核实新增算子；`pipeline` 在相同扫描阶段后继续生成用例和运行测试。新增算子有无法确定的输入或 reference 时，报告 blocked 并继续其他算子。

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

`python -m kernel_tools` 与 `kernel-tools` 完全等价。示例有 3 个 `_fill_num_accepted_kernel` case 和独立的输出检查；示例 JSON 的 `source` 记录生成依据，换源码版本后应重新核对。仓库本身的 CPU 检查不代表这些 case 已在目标 NPU 运行成功。

## 从本机跑远端 NPU

先生成一次配置，修改主机、容器、Python 和源码路径：

```bash
python -m kernel_tools init
# 编辑 kernel-tools.json；内置 npu160 / npu165 示例，不含密码。
python -m kernel_tools doctor --target npu160
```

远端使用已有 SSH key/agent 和已信任的主机指纹。`doctor` 会显示软件版本、实际 import 位置与 npu-smi 信息；确认选用设备空闲。

```bash
python -m kernel_tools run examples/fill_num_accepted.json \
  --target npu160 --case-name smoke_b1 --output artifacts/smoke

python -m kernel_tools run examples/fill_num_accepted.json \
  --target npu160 --output artifacts/fill-all
```

工具自动上传自身和 JSON case 到临时目录，在容器中使用配置的源码运行，再把报告、case、结果、失败日志取回本地。远端结果也保留在配置的 `result_root`。无需切换 vllm-ascend 到 `kernel_test_frame`，也不会重装远端 Torch/Triton。输入构造和可选 reference 由统一测试框架提供，自动流程不生成辅助 Python 文件。

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

文件名带短哈希，防止同名/长名称覆盖。成功只保留 mean/p50/p90/p99/min/max；不保存逐轮耗时、中间输入或成功日志。报告中的 `correctness=not_checked` 明确表示未校验数值；有 `check` 回调且断言通过才是 `passed`。

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
