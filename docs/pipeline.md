# 三个独立阶段或一条 pipeline

完整工作流由三个可独立执行的阶段组成：

```bash
python3 -m kernel_tools scan --repo ../vllm \
  --base v0.28.0 --target v0.29.0 --output ~/kernel-results/scan-029

python3 -m kernel_tools generate --scan ~/kernel-results/scan-029 \
  --repo ../vllm --npu npu162 --output ~/kernel-results/generate-029

python3 -m kernel_tools run ~/kernel-results/generate-029/cases \
  --target npu162 --output ~/kernel-results/run-029
```

`scan` 只读本地精确 tag 并完成 AI release review。`generate` 消费其输出目录或其中的 `review.json`，不会再次调用 release review；它会重新验证 tag、commit、扫描范围和 AST 候选，读取远端真实源码，然后逐算子生成 case。`run` 消费 case，自动使用其中保存的远端源码指纹，防止生成后环境变化。三个输出目录互相独立，任一步失败都可使用该步骤自己的 `--resume` 方式继续；`scan` 失败时重新运行到一个新目录。

也可以用一条命令完成相同流程：

```bash
python3 -m kernel_tools pipeline --repo ../vllm \
  --base v0.28.0 --target v0.29.0 --npu npu165
```

`--target` 是待测 vLLM tag，`--npu` 是 `kernel-tools.json` 中的远端环境名。每次生成独立结果目录，结束时打印 `report.md` 路径。流程按需运行，不含周期调度。

用例保存位置由 `--cases-output` 控制，默认是 `<本次运行目录>/cases/`。本次运行目录由 `--output` 指定；不设置时写入系统用户数据目录，例如 macOS 为 `~/Library/Application Support/vllm-kernel-tools/runs/<时间>-<随机后缀>/`。Linux 默认使用 `${XDG_DATA_HOME:-~/.local/share}/vllm-kernel-tools/runs/`。可设置 `VLLM_KERNEL_TOOLS_HOME` 统一覆盖根目录。工具仓库不会因默认运行产生报告、case 或临时快照。报告和 `workflow.json` 都会记录实际的用例路径。指定 `--cases-output` 时，目录必须为空或不存在，避免覆盖已有用例。

## 首次准备

1. 控制端安装并登录 Codex CLI，确认 `codex --version` 和 `codex login status` 可用。工具复用其认证、模型和供应商配置；实际 AI 调用会使用相应账户额度。可用 `--codex /path/to/codex`、`--model MODEL` 和 `--reason high` 指定入口、模型与单次推理强度。`--reasoning-effort` 是 `--reason` 的完整别名；未设置时沿用 `~/.codex/config.toml` 的 `model_reasoning_effort`。启动时会打印实际 CLI、模型、推理强度及其来源；每次 AI 调用打印用途、提示词长度、超时和耗时，超过 30 秒时持续输出心跳。
2. 本地 `--repo` 仓库中已有两个指定 tag。扫描不自动拉取 tag。
3. 配置远端 SSH 密钥登录、容器、Python、源码目录和设备，先运行 `python3 -m kernel_tools doctor --target npu165`。使用已分配且可用的设备；工具的协作锁不代表外部任务没有占卡。
4. 远端已安装适用的 Torch/torch_npu/Triton/CANN 和 vLLM/vLLM-Ascend。工具不替你安装或切换这些环境。

也可在 `kernel-tools.json` 顶层设置 `"ai": {"codex": "/path/to/codex"}`，相对路径以该配置文件所在目录为基准。命令行 `--codex` 优先。不更改系统全局 Codex；若服务端报模型要求更新客户端，更新所选 CLI 或指定已有的新入口，工具不会偷偷更换模型。

先看计划，不联网、不调用模型：

```bash
python3 -m kernel_tools pipeline --repo ../vllm \
  --base v0.28.0 --target v0.29.0 --npu npu165 --dry-run
```

兼容原有用法时，也可以让 pipeline 生成用例后停止：

```bash
python3 -m kernel_tools pipeline --repo ../vllm \
  --base v0.28.0 --target v0.29.0 --npu npu165 \
  --prepare-only --output artifacts/prepared-029
```

这会调用 AI，并连接 NPU 环境读取软件信息和 Python 源码；不会启动 case。新流程更推荐显式使用 `scan` 后接 `generate`。查看产物后，也可通过 `run artifacts/prepared-029/cases --target npu165` 手动运行；`run` 会自动核对 case 中保存的源码指纹。

想把用例单独放在固定位置，可以在上面的命令中加 `--cases-output ~/vllm-kernel-cases/v0.29.0`；之后手动运行用 `run ~/vllm-kernel-cases/v0.29.0 --target npu165`。


## 本地 tag 与远端源码

`--base v0.28.0 --target v0.29.0` 的 tag 只要求存在于本机 `--repo` 指向的 vLLM Git 仓库。扫描器通过本地 tag 读取两个版本，远端服务器和容器不需要保存这些 tag，甚至不要求远端源码目录带有 `.git`。

生成用例前，工具通过 Python 的实际 import 路径下载远端 vLLM/vLLM-Ascend 源码快照。默认把远端 vLLM 的 Python 文件与本机 target tag 逐文件比较：内容吻合即可继续，与远端是否存在 tag 无关。内容不同则停止，并在报告中列出差异，防止把其他版本的运行结果写成 v0.29.0 结果。

远端实际导入的 vLLM Python 源码必须与目标 tag 完全一致。存在任何修改、缺失或新增文件时 pipeline 都会停止并报告差异；没有跳过此校验的选项。唯一排除项是 vLLM 通过 vcs-versioning/setuptools-scm 生成且由上游 `.gitignore` 明确忽略的 `vllm/_version.py`，它是安装版本元数据，不是 tag 管理的 kernel 源码。工具不会 checkout、reset 或覆盖远端源码，需要先在远端准备匹配环境。

报告会区分内容修改、远端缺少和远端新增的 Python 文件，并记录目标 commit、远端实际 Git HEAD（可检测时）、包版本和源码指纹。包 metadata 版本可能与 checkout HEAD 不一致，源码身份以实际导入路径的内容和 Git HEAD 为准。

## 哪些由代码处理，哪些交给 AI

| 阶段 | 执行者 | 实际行为 |
|---|---|---|
| 版本扫描 | Python/Git | 读取精确 tag，以 AST 识别启动点、调用关系和源码变化 |
| 新增算子复核 | Codex + release skill | 复核新增/改名、helper、动态调用和启动条件；记录未解决项 |
| 环境快照 | Python/SSH | 下载实际 import 的 vLLM/vLLM-Ascend Python 源码和环境信息 |
| 用例生成 | Codex + case skill | 分析真实输入契约和 Ascend 绑定，生成框架可直接读取的多场景 JSON case |
| 校验及执行 | Python/SSH | 校验 JSON 和 Triton target 位置；上传工具与 case，逐 case 运行 |
| 结果与报告 | Python，失败时再调用 Codex | 固化执行状态、耗时、数值检查和完整失败日志；AI 给出有证据的失败原因 |

新增算子范围只包含定义在 `vllm/v1/worker/gpu` 下的直接 launch kernel，以及被该目录模块显式 import 并由同一模块直接 launch 的外部 Triton kernel。不会通过对象方法、context、metadata、接口实现或模型专属调用链递归扩展范围。

程序确实调用 `codex exec --sandbox read-only --output-schema ... --output-last-message ...`。release review 的提示词包含 release-scan skill 和扫描范围/证据要求；逐算子生成包含 kernel 描述、目标及远端源码、case schema 和 kernel-cases skill；失败分析包含结果、失败日志及 remote-benchmark skill。每次调用都要求结构化 JSON。AI 返回的文件内容由程序校验和落盘，不把模型文字作为 shell 命令执行。具体完整模板见 `kernel_tools/review.py` 和 `kernel_tools/pipeline.py`；非交互机制参考 [Codex 官方文档](https://developers.openai.com/codex/noninteractive)。

AI 复核、每个算子的生成、失败分析是独立调用，默认单次上限 1800 秒；使用 `--ai-timeout` 修改。生成或格式校验失败最多自动重试一次；无法处理的算子标 blocked，继续其他算子。运行失败不会自动修改生产 kernel 或反复缩小输入重跑。

pipeline 被中断或阻塞后，使用完全相同的参数和 `--output`，追加 `--resume`。工具会重新核对本地 tag 和远端源码，复用完成的 release review；已生成的 case 只有在远端源码指纹一致时才复用。未生成的算子继续生成，NPU 执行阶段重新运行。若上次因源码不一致而阻塞，先对齐远端源码，再使用 `--resume`。SSH 在执行阶段断开时，先确认远端是否仍有任务，避免重复运行。

```bash
python3 -m kernel_tools pipeline --repo ../vllm \
  --base v0.28.0 --target v0.29.0 --npu npu162 \
  --output ~/kernel-results/run-029 --resume
```

## 版本与正确性

- 默认要求远端 vLLM Python 源码与目标 tag 一致。不同就停止，并在报告说明差异；不会擅自 checkout/reset 生产源码。
- 源码不同就固定停止；不存在绕过源码身份校验的参数。对齐远端实际 import 的源码后才能继续。
- 生成后、执行前再核对实际源码与软件环境指纹；发生变化时阻塞执行。快照覆盖包内普通 `.py` 文件，原生库只记录包版本，不是整个环境镜像的字节级证明。
- 自动生成的 case 只保存真实 Triton `target`、`grid` 和参数，固定执行 `target[grid](...)`；不支持 Python wrapper 模式，也不生成辅助 Python adapter/checker。生产 wrapper 仅作为推导输入与启动配置的源码依据。框架不保存单算子 reference，所有场景明确标记 `correctness=not_checked`。JSON materializer 无法表达合法输入时明确 blocked，并记录需要扩展的框架能力。
- 程序能证明产物格式、启动位置和运行结果，不能证明 AI 的语义判断绝对正确，也不能将单算子成功视为整网命中。

## 只保留最终需要的产物

```text
<用户数据目录>/runs/<本次运行>/
  cases/         默认用例目录；设置 --cases-output 时改存到指定目录
  results/       每个算子的实际执行结果（执行后才有）
  logs/          仅失败 case、失败 AI 调用和流程错误的完整日志
  workflow.json  版本/环境指纹、AI 判断、各算子生成状态
  report.md      场景结果、失败原因、日志链接
```

源码快照、扫描中间文件、AI 成功调用输出和事件流使用临时目录，结束后清理。生成的 JSON case 和关键版本证据保留，以便复查。远端也保留执行结果。

`completed` 表示已确认的新算子均生成用例，所有执行成功且没有报告未解决项；它不是完整盘点的形式化证明。`needs_review` 表示仍有未解决项或某些算子无法生成；`failed` 表示运行失败；`blocked` 表示环境或前序条件阻止流程；`prepared` 表示仅生成；`no_new_operators` 表示本次复核未确认新增算子。
