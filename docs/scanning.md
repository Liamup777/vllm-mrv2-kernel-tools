# scan：通过 AI skill 核实新增算子

```bash
python3 -m kernel_tools scan --repo ../vllm \
  --base v0.28.0 --target v0.29.0 \
  --output ~/vllm-kernel-tools-output/scan-029-ai
```

需要已登录且可用的 Codex CLI；默认使用本机模型配置，可通过 `--model` 指定。`--config` 默认读取 kernel-tools.json 的 ai.codex；配置文件不存在时使用 PATH 中的 codex。scan 不需要 NPU 配置，不连接服务器，也不生成 case。

## 内部流程

1. 固定代码读取两个明确的 Git tag，提取只读源码快照，不切换用户分支。
2. 固定 AST 程序提供定义、启动点、调用关系和变化的候选线索；这些线索可能遗漏或误报。
3. 通过 codex exec 调用 AI，传入 release-scan skill 和源码快照。AI 独立复核新增、改名、helper，以及被 GPU 目录模块显式 import 并直接 launch 的外部 Triton kernel。
4. 固定代码检查返回结构及新增定义是否存在，生成 review.md 和 review.json。内部快照和成功调用日志随临时目录清理；失败保留完整日志。

## skill 的固定规则

- 包含定义在 `vllm/v1/worker/gpu` 下且有直接 launch 的 kernel。
- 目录外算子只有在 GPU 目录模块显式 import 该 Triton kernel，并在同一模块直接 `kernel[grid](...)` launch 时才纳入。
- 不沿实例方法、类型信息、接口实现、继承、context/metadata 对象、registry、backend dispatch 或模型专属状态链递归搜索。导入一个类再调用其方法，不算直接 import Triton 算子。
- 必须找到 kernel[grid](...) 或等价直接启动证据；仅有名字、装饰器、导入或测试不够。
- 排除仅被另一个 JIT 调用的 helper。
- 每项记录定义、直接 launcher、wrapper、worker 调用路径和触发条件。
- 同一 kernel 多条调用路径去重计数；改名/移动、已有算子修改和真正新增分开。
- 无法确认的项列为待核实，不静默忽略，不强行确认为算子。

## 不能保证什么

skill 是 AI 的分析规范，不是完整性证明。固定 AST 和 AI 都可能漏掉动态别名、装饰器包装或运行时 patch。这里的完整性只针对上述严格边界，不覆盖通过对象方法或动态分发可达的 Triton kernel。

报告中的“确认新增”表示 AI 根据源码证据作出了判断，仍可被复核纠正。没有待核实项也不证明绝无遗漏。静态可达不等于整网实际命中；数值正确性、设备兼容性和性能必须由后续运行验证。

优先阅读 review.md：确认新增、待核实、已有/非新增分开呈现。review.json 保留结构化证据和内部静态差分，后者的 changed 数量不能解释成新增数量。AI 调用失败会明确输出失败报告，不会把静态候选当最终答案。

不设置 `--output` 时使用系统用户数据目录的 `scans/`，不会写入工具仓库。显式输出目录必须为新目录或空目录，避免覆盖已有结果。原有纯静态版本的 scan-029/review.md 不会自动变成 AI 结果，需重新执行到新目录。
