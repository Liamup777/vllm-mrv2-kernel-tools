# 本地生成、远端校验和 NPU 执行

完整工作流分为三个可独立执行的阶段：

```bash
python3 -m kernel_tools scan --repo ../vllm \
  --base v0.28.0 --target v0.29.0 \
  --output ~/kernel-results/scan-029

python3 -m kernel_tools generate --scan ~/kernel-results/scan-029 \
  --repo ../vllm --source ../vllm-ascend \
  --output ~/kernel-results/generate-029

python3 -m kernel_tools run ~/kernel-results/generate-029 \
  --npu npu162 --output ~/kernel-results/run-029
```

`scan` 和 `generate` 只读取本地源码。`generate` 不需要 NPU 配置，也不连接 SSH。`run` 读取生成目录中的 `source-lock.json`，先核对远端实际 import 的源码，再运行 case。三个输出目录互相独立，每一步都可使用原 `--output` 加 `--resume` 继续。

一条命令可以执行同样的流程：

```bash
python3 -m kernel_tools pipeline --repo ../vllm \
  --source ../vllm-ascend \
  --base v0.28.0 --target v0.29.0 \
  --npu npu162 --output ~/kernel-results/run-029
```

pipeline 的扫描和生成阶段仍在本机完成，直到源码校验和执行阶段才连接 `--npu`。流程按需运行，不包含周期调度。

## 指定单个 kernel

已知 kernel 时可以跳过 release scan：

```bash
python3 -m kernel_tools generate \
  --kernel _compact_sampling_mask_kernel_ascend \
  --source ../vllm-ascend/vllm_ascend/ops/triton/v2/sample/sampling_mask.py \
  --output ~/kernel-results/compact-sampling
```

`--source` 可以是文件或目录，也可以重复传入。工具只解析这个范围来定位指定的 `@triton.jit`，并从同一 package 中补充包含该符号的 Python 调用文件。它不会枚举本地或远端仓库中的全部 Triton kernel。短名称有重名时需要传完整 Python ID。

默认读取当前 worktree。需要从明确的历史提交生成时使用：

```bash
python3 -m kernel_tools generate \
  --kernel _compact_sampling_mask_kernel_ascend \
  --source ../vllm-ascend/vllm_ascend/ops/triton/v2/sample/sampling_mask.py \
  --ref 16bb2fb \
  --output ~/kernel-results/compact-sampling
```

`--ref` 由本地 Git 仓库解析，不会 checkout 或修改当前分支。生成阶段从该 revision 的 Git object 读取文件。

## source lock 和运行门禁

生成目录保留：

```text
<generation>/
  cases/
  source-lock.json
  workflow.json
  report.md
```

`source-lock.json` 对每个用到的 package 记录：

- 本地解析出的 commit；
- 指定的 ref；
- 生成时工作区是否有 Python 源码修改；
- package 内 Git 已跟踪 `.py` 文件以及未被忽略的未跟踪 `.py` 文件的 SHA-256。

单 kernel 的 AI 上下文仍然只有指定文件和定点找到的调用文件；完整 package 哈希只用于版本门禁，不会把全部源码发送给 AI。

可以在运行前单独检查：

```bash
python3 -m kernel_tools verify ~/kernel-results/compact-sampling \
  --npu npu160
```

校验读取远端 Python 实际 import 的 `vllm` / `vllm_ascend` 路径，并检查：

1. import 路径位于可识别的 Git checkout；
2. Git HEAD 与本地 lock 的 commit 相同；
3. Git 管理范围内的 Python 文件没有修改、缺失或新增。

Git 忽略的构建产物不参与比较，例如 vLLM-Ascend 构建生成的 `_build_info.py` 和 `_cann_ops_custom/`。未被忽略的未跟踪 Python 文件仍会进入 source lock，因此可以验证尚未提交的本地 kernel 修改。

任一项不一致都会报出 package、期望 commit、实际 commit 和文件差异，并停止执行。没有绕过源码校验的选项。`run` 内部始终再次执行相同校验，单独运行 `verify` 只是为了提前检查环境。

工具不会 checkout、reset 或修改远端仓库。包 metadata 版本仅作为环境信息，源码身份以实际 import 路径、Git HEAD 和文件哈希为准。

## 本地 tag 和额外源码

release scan 的 `--base`、`--target` 只要求存在于本机 `--repo`。扫描器通过 Git object 读取两个 tag，不切换当前分支。

`generate --scan` 会再次核对 scan 产物中的 tag、commit、范围和 AST 证据。目标 vLLM 源码固定来自 scan 对应的 target tag；额外的本地 `--source` 通常指向 vLLM-Ascend，用于分析实际 Ascend binding。两者都会进入同一个 source lock，远端必须同时匹配。

## AI 和固定代码的边界

| 阶段 | 执行者 | 行为 |
|---|---|---|
| tag 差异和静态候选 | Python/Git | 读取本地精确 tag，用 AST 找直接 launch、调用关系和源码变化 |
| 新增算子复核 | Codex + release skill | 核实新增、改名、helper、显式 import 和启动条件 |
| 定点源码收集 | Python/Git | 根据 `--source` 和 kernel 名构造受限本地上下文 |
| case 生成 | Codex + case skill | 从本地上下文分析输入契约，返回直接 Triton JSON case |
| source lock | Python/Git | 记录本地 commit 和 package Python 文件哈希 |
| 远端校验及执行 | Python/SSH | 核对实际 import/HEAD/哈希后上传工具和 case，逐 case 运行 |
| 失败分析 | Codex + remote benchmark skill | 只在运行失败后读取结果、失败日志和已提供源码证据 |

程序调用 `codex exec --sandbox read-only --output-schema ... --output-last-message ...`。release review、每个 kernel 的 case 生成和失败分析是独立调用。AI 返回 JSON 后由固定代码校验，不把模型文字当作 shell 命令执行。

生成 case 只保存真实 Triton `target`、`grid` 和参数，固定执行 `target[grid](...)`。不支持 Python wrapper、临时 adapter 或单算子 reference，执行成功仍记录 `correctness=not_checked`。

## 恢复和输出

scan、generate、run 和 pipeline 都支持 `--resume`。恢复时：

- scan 复用已完成的 release review；
- generate 复用 source lock 相同的已生成 case；
- run 每次重新核对远端源码；
- 远端已有活动进程时拒绝重复启动；
- case、source lock、目标配置、工具代码或测量参数变化时拒绝复用旧结果。

`--cases-output` 可把 case 放到独立目录。工具会同时放置对应的 `source-lock.json`，因此该目录可直接传给 `verify` 或 `run`。

临时源码上下文、扫描中间文件和 AI 成功事件流保存在系统临时目录，结束后清理，不写入工具仓库。
