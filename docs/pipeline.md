# 一条命令完成新增算子测试

```bash
python3 -m kernel_tools pipeline --repo ../vllm \
  --base v0.28.0 --target v0.29.0 --npu npu165
```

`--target` 是待测 vLLM tag，`--npu` 是 `kernel-tools.json` 中的远端环境名。每次生成独立结果目录，结束时打印 `report.md` 路径。流程按需运行，不含周期调度。

用例保存位置由 `--cases-output` 控制，默认是 `<本次运行目录>/cases/`。本次运行目录由 `--output` 指定；不设置时自动生成 `artifacts/runs/<时间>-<随机后缀>/`，所以用例也始终有默认路径。报告和 `workflow.json` 都会记录实际的用例路径。指定 `--cases-output` 时，目录必须为空或不存在，避免覆盖已有用例。

## 首次准备

1. 控制端安装并登录 Codex CLI，确认 `codex --version` 和 `codex login status` 可用。工具复用其认证、模型和供应商配置；实际 AI 调用会使用相应账户额度。可用 `--codex /path/to/codex` 或 `--model MODEL` 指定入口或模型。
2. 本地 `--repo` 仓库中已有两个指定 tag。扫描不自动拉取 tag。
3. 配置远端 SSH 密钥登录、容器、Python、源码目录和设备，先运行 `python3 -m kernel_tools doctor --target npu165`。使用已分配且可用的设备；工具的协作锁不代表外部任务没有占卡。
4. 远端已安装适用的 Torch/torch_npu/Triton/CANN 和 vLLM/vLLM-Ascend。工具不替你安装或切换这些环境。

也可在 `kernel-tools.json` 顶层设置 `"ai": {"codex": "/path/to/codex"}`，相对路径以该配置文件所在目录为基准。命令行 `--codex` 优先。不更改系统全局 Codex；若服务端报模型要求更新客户端，更新所选 CLI 或指定已有的新入口，工具不会偷偷更换模型。

先看计划，不联网、不调用模型：

```bash
python3 -m kernel_tools pipeline --repo ../vllm \
  --base v0.28.0 --target v0.29.0 --npu npu165 --dry-run
```

只生成用例、暂不执行算子：

```bash
python3 -m kernel_tools pipeline --repo ../vllm \
  --base v0.28.0 --target v0.29.0 --npu npu165 \
  --prepare-only --output artifacts/prepared-029
```

这会调用 AI，并连接 NPU 环境读取软件信息和 Python 源码；不会启动 case。查看产物后，也可通过 `run artifacts/prepared-029/cases --assets artifacts/prepared-029/adapters --target npu165` 手动运行。普通 `run` 不自动核对生成时的源码指纹，重新运行前需确认环境未变。

想把用例单独放在固定位置，可以在上面的命令中加 `--cases-output cases/v0.29.0`；之后手动运行用 `run cases/v0.29.0 --assets artifacts/prepared-029/adapters --target npu165`。`adapters/` 继续保存在运行目录中，包含与用例配套的输入构造和检查代码。

## 哪些由代码处理，哪些交给 AI

| 阶段 | 执行者 | 实际行为 |
|---|---|---|
| 版本扫描 | Python/Git | 读取精确 tag，以 AST 识别启动点、调用关系和源码变化 |
| 新增算子复核 | Codex + release skill | 复核新增/改名、helper、动态调用和启动条件；记录未解决项 |
| 环境快照 | Python/SSH | 下载实际 import 的 vLLM/vLLM-Ascend Python 源码和环境信息 |
| 用例生成 | Codex + case skill | 分析真实输入契约和 Ascend 绑定，生成多场景 case、adapter 和独立 reference |
| 校验及执行 | Python/SSH | 校验 JSON、Python 语法和 callable 位置；上传工具及辅助代码，逐 case 运行 |
| 结果与报告 | Python，失败时再调用 Codex | 固化执行状态、耗时、数值检查和完整失败日志；AI 给出有证据的失败原因 |

程序确实调用 `codex exec --sandbox read-only --output-schema ... --output-last-message ...`。每次调用都会把对应 `SKILL.md` 的内容放进任务提示中，提供只读源码快照，并要求结构化 JSON。AI 返回的文件内容由程序校验和落盘，不把模型文字作为 shell 命令执行。具体调用见 `kernel_tools/ai.py`；非交互机制参考 [Codex 官方文档](https://developers.openai.com/codex/noninteractive)。

AI 复核、每个算子的生成、失败分析是独立调用，默认单次上限 1800 秒；使用 `--ai-timeout` 修改。生成或格式校验失败最多自动重试一次；无法处理的算子标 blocked，继续其他算子。运行失败不会自动修改生产 kernel 或反复缩小输入重跑。当前没有自动恢复中断 pipeline 的功能，重启需新结果目录；SSH 中断后先确认远端作业状态。

## 版本与正确性

- 默认要求远端 vLLM Python 源码与目标 tag 一致。不同就停止，并在报告说明差异；不会擅自 checkout/reset 生产源码。
- 如果你明确要测远端已有的定制版本，可加 `--allow-source-drift`。此时 AI 按实际远端源码生成用例，报告保留版本差异，不能把结果称为目标 tag 的原版测试结果。
- 生成后、执行前再核对实际源码与软件环境指纹；发生变化时阻塞执行。快照覆盖包内普通 `.py` 文件，原生库只记录包版本，不是整个环境镜像的字节级证明。
- 自动生成的 ready case 必须有独立 checker；无法建立合理输入或 reference 时明确 blocked。AI 生成的 checker 仍可能出错，失败分析应区分 kernel 错误与测试本身错误。
- 程序能证明产物格式、启动位置和运行结果，不能证明 AI 的语义判断绝对正确，也不能将单算子成功视为整网命中。

## 只保留最终需要的产物

```text
artifacts/runs/<本次运行>/
  cases/         默认用例目录；设置 --cases-output 时改存到指定目录
  adapters/      对应输入构造和独立数值检查 Python 文件
  results/       每个算子的实际执行结果（执行后才有）
  logs/          仅失败 case、失败 AI 调用和流程错误的完整日志
  workflow.json  版本/环境指纹、AI 判断、各算子生成状态
  report.md      场景结果、失败原因、日志链接
```

源码快照、扫描中间文件、AI 成功调用输出和事件流使用临时目录，结束后清理。生成的 case/reference 和关键版本证据保留，以便复查。远端也保留执行结果及辅助代码。

`completed` 表示已确认的新算子均生成用例，所有执行成功且没有报告未解决项；它不是完整盘点的形式化证明。`needs_review` 表示仍有未解决项或某些算子无法生成；`failed` 表示运行失败；`blocked` 表示环境或前序条件阻止流程；`prepared` 表示仅生成；`no_new_operators` 表示本次复核未确认新增算子。
