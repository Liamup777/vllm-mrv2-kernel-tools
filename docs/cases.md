# Case 格式与扩展

输入支持 JSON 对象、JSON 数组、JSONL，推荐一个算子一个 JSON 数组。每条至少包含：

```json
{
  "name": "smoke_b1",
  "kernel": "_fill_num_accepted_kernel",
  "target": "vllm.v1.worker.gpu.model_states.mamba_hybrid:_fill_num_accepted_kernel",
  "grid": [1],
  "seed": 42,
  "check": "kernel_tools.references:check_fill_num_accepted",
  "arguments": {
    "idx_mapping_ptr": {"shape": [1], "dtype": "int32", "initializer": "arange"},
    "num_accepted_ptr": {"shape": [1], "dtype": "int32", "initializer": "zeros"},
    "num_sampled": 1
  }
}
```

- `target` 必须指向可用 `kernel[grid](...)` 直接启动的 Triton JIT kernel，支持 `module:symbol` 或 `file.py:symbol`。远端路径以远端源码为准。
- 框架不执行 Python wrapper，也不接受 `mode`、`wrapper` 或每次运行专用的 adapter。生产 wrapper 只用于推导参数关系、grid、constexpr 和 launch options。
- `arguments` 是 keyword arguments，`args` 是 positional arguments。兼容旧 `kwargs`，但不能与 `arguments` 同时出现。
- grid、constexpr、launch options 从真实生产 wrapper 提取；`num_warps` 等 launch 参数放在 `arguments` 中。旧 `launch_config` 仅是元数据，不影响执行。
- `(kernel, name)` 必须唯一。CLI `--warmup` / `--rounds` / `--device` 优先，统一控制当前运行。
- `source`、`scenario`、`coverage`、`benchmark_scope` 可保存分析说明，但工具不把这些说明当运行证明。

## 张量构造

`shape` 为非负整数数组，默认 `dtype=float32`、`initializer=zeros`。支持：

| initializer | 补充字段 |
|---|---|
| `zeros` / `ones` / `rand` / `randn` | 无 |
| `full` | `value` |
| `randint` | 整数 `high`，可选 `low` |
| `arange` | 可选 `start`、`step` |
| `values` | `values`，元素数必须等于 shape 的乘积 |
| `data_ptrs` | `dtype=uint64`、`shape=[指针数]`、`pointees` |

`data_ptrs` 使用真实分配的 pointee 地址并保活，不能用随机整数或全零代替指针。静态校验只能验证结构，不能证明索引、容量、dtype、边界与 kernel 语义兼容。

共享存储、非连续 view、关联随机张量和实例状态目前不能由 JSON materializer 表达。不支持的字段（例如凭空写 `stride`）会被拒绝；case 应标记相应场景未覆盖，并通过一次性的框架能力扩展解决，而不是生成每次运行专用的 adapter。

## 随机性、原地修改和计时

每 case 默认 seed=0，可显式设置。worker 在独立进程里设置 Torch/NPU 随机种子。

默认复用输入；适用于输出覆盖写、输入不变或幂等操作。会改变下一轮语义的算子可设置 `reset_inputs: true`，框架保留构造张量的初始副本，在 warmup/每次计时前恢复；额外内存约等于这些张量的总大小。恢复和检查不计入 kernel latency。框架无法恢复的复杂状态应标记未覆盖。

worker 先做一次不计时执行用于编译和可选检查，随后 warmup 和测量。对未重置的原地算法，这次预执行也会改变状态，必须在契约中考虑。

## 数值检查

可选 `check: kernel_tools.references:function`，只能使用框架中已经实现的 reference。函数签名为 `check(args, kwargs, output)`；Triton kernel 通常通过 kwargs 中的输出 tensor 检查。通过时返回 None/True，失败时抛出异常或返回 False。

检查在第一次执行并同步后进行，且在计时前。检查失败记为 `phase=correctness`，不生成耗时。没有 checker 一律 `correctness=not_checked`。检查过一次输入不能泛化为所有 shape 或模型精度正确。

示例的 `check_fill_num_accepted` 只适用于 output 初始为零的 case；它检查 sentinel 被跳过、映射位置写入指定值、未触及位置保持零。

## 状态与日志

- `success`：退出码、输出身份、样本数量、有限非负延迟均满足协议。
- `failed`：单 case 错误；阶段单列 input/import/compile/runtime/correctness/timeout/result/unknown。
- `blocked`：环境不可用或前序中断。
- `pending` / `running` / `interrupted`：可恢复的执行状态。

失败日志保留 worker 完整输出，报告只提取错误摘要；无法定位阶段时保留 unknown。执行中断后不要把 partial 数据当成功。
