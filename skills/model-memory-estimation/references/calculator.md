# 计算工具输入与运行

工具仅依赖 Python 3 标准库，不加载权重或访问网络。输入是一个 JSON 对象，或多个独立对象组成的列表；输出 JSON。列表项独立计算，不自动引用前项输出。数值计算使用 Decimal，最终展示为浮点数。

```bash
python3 scripts/estimate_memory.py references/kimi-k3-calculations.json
python3 scripts/test_estimate_memory.py
```

从 skill 目录运行；也可使用绝对路径。自定义输入文件作为第一个参数，或用 `-` 从标准输入读取。错误返回退出码 2。

所有显存输入都写成 `{"value": 64, "unit": "GiB"}`。支持 B、KiB、MiB、GiB、KB、MB、GB，严格按 SI/IEC 定义转换；不接受 `G` 或 `gb`。先确认源日志含义再选择单位。参数量与 token 数用完整数值，不能写 `2.8T`。

## 操作

每个对象用 `operation` 指定操作。下面列出的字段均必填，除非明确说明可选。未知字段不会参与计算，检查输入拼写。

### state

- `public_parameters`、`expert_parameters`：本次账本的参数量，非负整数。
- `public_bytes_per_param`、`expert_bytes_per_param`：权重、梯度和优化器等状态合计字节数，正数。
- `public_shards`、`ep`、`efsdp`：正整数。

计算公共项 /public_shards 与专家项 /(ep×efsdp)，同时返回未分片状态总量。TP/PP 时传入其实际本地参数量，不要先除 FSDP 后再次分片。

### checkpoint

`boundaries` 是同时保存、互不重复的边界数量，可为零；`batch`、`sequence`、`hidden` 是正整数本地张量维度，`bytes_per_element` 是正数。计算同形边界总量，异形边界分别算后加总。

### extrapolate

- `measured_small`、`known_small`：显存对象。
- `measurement_metric`、`known_metric`：必须一致，取 `allocated` 或 `device`。reserved 不能作为可加总张量账本的替代值。
- `expert_tokens_small`、`expert_tokens_large`：每卡专家任务数，正整数。

工具只算残差和比例，不判断残差是否以专家激活为主。负残差直接报错；分开计算固定项再传入 known。此操作也不会自动补回大模型已知项。

### training_grid

仅用于 TP1/CP1/PP1、公共与专家状态统一分片的网格。

- `state_total`、`checkpoint`、`capacity`：显存对象。
- `budget_fraction`：大于 0、不大于 1。
- `experts`、`devices_per_node`：正整数。
- `tp`、`cp`、`pp`：均显式填 1。
- `available_devices`：可选，正整数或 null。
- `candidates`：非空列表，每项包含正整数 `ep`、`efsdp` 和显存对象 `remaining`。

remaining 是该候选除状态、checkpoint 以外的全部额外峰值预算。理论路径传入临时参数、活动激活及其他开销之和；实测路径传入外推残差及已单独计算的其他项，同一项不能出现两次。逐候选给值，不能默认不同 EP 的额外开销相同。

返回 F=EP×eFSDP、设备/节点数、每卡专家数、该额外预算下的分片下界和峰值、余量及筛选原因。`passes_supplied_checks=true` 仅表示所提供的数值和整除约束满足，算子与通信兼容性另核验。

### inference_grid

仅用于无 PP、EP=DP×TP、专家按 EP 分配、公共参数按 TP 或复制的布局。

- `experts`、`tp`、`devices_per_node`：正整数。
- `expert_parameters`、`public_tp_parameters`、`replicated_parameters`：非负整数。
- `expert_weight_bytes`、`public_weight_bytes`：正数字节数；量化元数据另计。
- `capacity`、`budget_fraction`：同上。
- `max_instance_devices`：实际部署允许的单实例设备上限，未知填 null。
- `candidates`：每项含正整数 `experts_per_device`，显存对象 `resident`、`cache`，以及 `temporary`。

temporary 已知时为显存对象，未知时必须为 null，不能填零冒充已测。零仅表示该项确认为零。resident、cache、temporary 都是每卡量，按候选实际配置给出；cache 选择需求量或实际分配池之一。量化额外存储可计入 resident 并在报告说明。

返回权重、静态量、动态量已知时的预计峰值、物理与预算余量。拓扑上限或动态峰值未知且无明确失败时，`passes_supplied_checks=null`；明确违反任一条件则为 false。prefill 和 decode 需要分别核算，不能从静态余量自动推导“能跑通”。

## 范围

工具支持手工构造的候选，不自动搜索最优吞吐；不验证 head 整除、网络性能、路由不均、后端支持和未提供的拓扑规则。这些由 skill 按实际代码和环境分析。参见 [Kimi-K3 输入](kimi-k3-calculations.json)，其中相同开销跨候选复用是案例的简化假设。
