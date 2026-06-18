# TransferQueueBridge 方法概览

本文简要说明 `vime/utils/transfer_queue.py` 中 `TransferQueueBridge` 的核心方法职责。该类是 VIME 接入外部 TransferQueue 的适配层，统一封装 TQ 的初始化、连接、rollout 写入、训练侧读取和数据格式转换。

## 1. 总体职责

`TransferQueueBridge` 主要承担三类工作：

1. 管理 TQ 生命周期：在 driver 侧初始化 TQ，在 rollout manager 和 train actor 中连接同一个 TQ 实例。
2. 管理数据流：rollout 侧写入 `train_{rollout_id}` partition，训练侧按 task name 和 DP rank 读取。
3. 管理数据格式：在 VIME 的 list-based rollout batch 与 TQ 使用的 `TensorDict` 之间转换。

## 2. 初始化与连接

### `enabled(args)`

判断是否启用 VIME 的 TransferQueue 路径。所有 TQ 分支都应通过这个方法判断开关。

### `initialize(args)`

driver 侧调用，用于创建 TQ controller/storage 配置，并将配置写回 `args.tq_config`。该方法必须在 Ray actors 创建前执行。

### `connect(args)`

Ray actor 进程侧调用，用于基于 `args.tq_config` 连接已初始化的 TQ，并返回带 client 的 `TransferQueueBridge` 实例。

### `close()`

关闭当前进程中的 TQ runtime，通常在 rollout manager 或 train actor 退出时调用。

## 3. Rollout 写入

### `transfer_rollout_data(rollout_id, train_data)`

rollout 侧主写入入口。它将一个 rollout step 的训练数据写入 `train_{rollout_id}` partition。

该方法负责写入前的数据规范化、格式转换、partition 写入和 metadata 更新。

### `wait_for_staleness()`

写入前的简单回压控制。它根据当前未消费的 `train_` partition 数量和 `max_staleness` 决定是否等待。

### `clear_partition(rollout_id)`

清理某个 rollout 对应的 TQ partition，用于释放已经消费完成的数据。

### `partition_id(rollout_id)`

统一生成 partition 名称，当前格式为 `train_{rollout_id}`。

## 4. 训练侧读取

### `get_data(rollout_id, task_name, data_fields=None)`

actor/critic 训练侧主读取入口。它根据 rollout id、task name、当前 DP rank 和请求字段，从 TQ 读取本地 batch。

读取后会将数据广播到同一个模型并行组内的其他 rank，并补齐训练侧需要的本地调度字段。

### `_broadcast_payload(payload, device)`

将代表 rank 从 TQ 读取到的数据广播到 TP/PP/CP 并行组内其他 rank，避免所有模型并行 rank 重复访问 TQ。

### `_ensure_schedule_fields(rollout_data)`

读取 TQ 数据后补齐训练侧需要的调度字段，例如 `global_batch_sizes`、`num_microbatches` 和 `micro_batch_indices`。

## 5. 数据字段与格式

### `default_train_data_fields(args)`

返回训练侧默认从 TQ 读取的字段集合。该方法定义了 rollout 写入侧和训练读取侧之间的基础数据契约。

### `actor_train_data_fields(args)`

返回 actor 训练需要读取的字段。在启用 critic values TQ 回写时，会在默认字段基础上追加 `values`。

### `critic_values_via_transfer_queue(args)`

判断 critic values 是否通过 TQ 回写。当前主要受 TQ 开关、`use_critic` 和 `context_parallel_size` 约束。

### `normalize_train_data(train_data)`

写入前规范化 rollout 训练数据，确保必需字段存在，并补齐部分默认字段。

### `dict_to_tensordict(data, batch_size=None, device=None)`

将 VIME 的 list-based rollout batch 转换为 TQ 需要的 `TensorDict`。

### `tensordict_to_rollout_data(data)`

将从 TQ 读取到的 `TensorDict` 转回 VIME 训练侧使用的 list-based rollout batch。

## 6. 内部辅助方法

### `env_vars(args)`

返回启用 TQ 时需要设置的环境变量。

### `_build_sampler(args, tq)`

根据参数选择 TQ sampler，例如普通 GRPO group sampler 或按长度均衡的 sampler。

### `data_parallel_size(args)`

在 Megatron 初始化前，根据 actor world size 和模型并行配置推导 DP size。

### `add_total_lengths(train_data)`

根据 `tokens` 字段补齐 `total_lengths`。

### `_set_total_length_custom_meta(metadata, total_lengths)`

将 `total_lengths` 写入 TQ batch metadata。

### `put_data(rollout_id, data, data_fields, batch_meta)`

向已有 TQ batch 写回派生字段。它依赖读取阶段返回的 `batch_meta`，不作为普通 rollout 数据写入入口。

### `_unwrap_non_tensor_stack(value, non_tensor_data_cls)`

解包 TQ 中的非 tensor 字段，例如 metadata、多模态输入和 prompt。

### `_require_client()`

确保当前 bridge 已连接 TQ client。需要访问 TQ 的方法会依赖这个检查。

## 7. 典型调用链

### Driver 初始化

```text
TransferQueueBridge.initialize(args)
```

### Rollout 写入

```text
TransferQueueBridge.connect(args)
transfer_rollout_data(rollout_id, train_data)
```

### Actor/Critic 读取

```text
TransferQueueBridge.connect(args)
get_data(rollout_id, task_name, data_fields)
```

### Partition 清理

```text
clear_partition(rollout_id)
```

