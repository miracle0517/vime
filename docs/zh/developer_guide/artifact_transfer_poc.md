# vLLM + VIME + TransferQueue Artifact Transfer POC 设计与实现说明

本文档基于以下两个实现快照整理：

- vLLM: `huangyibo/vllm@2e32778fed1cc0ada15153d74e71787f66a12434`
- VIME + TransferQueue consumer: `huangyibo/vime@65e8af26e6e2c7a5781db2ce9b4cfc5ae7c27148`

## 1. 背景与目标

现有 post-training pipeline 中，rollout 侧会产生完整 trajectory，包括 prompt token、response token、sampled-token logprob、reward、mask、routing metadata 等训练所需数据。如果这些大对象直接跟随主控制路径返回，会放大 OpenAI API 响应、Ray object、HTTP router 和调度器的负载。

本 POC 引入 artifact handoff 机制：

1. vLLM 在 rollout request 完成后，将 trajectory artifact 发布到 TransferQueue。
2. vLLM 主响应只返回轻量级 `ArtifactHandle`，用于定位外部 artifact。
3. VIME 训练侧新增 TransferQueue trajectory source，从 TransferQueue 消费 `vllm.trajectory/v1alpha1` artifact。
4. VIME 将 artifact 转换为原生 `Sample`，继续复用现有训练、loss、权重更新闭环。

核心设计目标是把“大对象数据面”和“主控制面”解耦，同时让 artifact backend 可替换、可扩展。

## 2. 总体架构

```text
Prompt dispatcher / VIME rollout driver
        |
        | OpenAI request
        | artifact_transfer_params={run_id, policy_version, group_id, sample_index, ...}
        v
vLLM OpenAI server
        |
        | SamplingParams.extra_args["artifact_transfer_params"]
        v
vLLM v1 Scheduler
        |
        | accumulate accepted token ids + sampled-token logprobs
        v
TransferQueueArtifactConnector
        |
        | kv_put / async_kv_batch_put
        v
TransferQueue partition: rollout-{run_id}-{policy_version}
        |
        | StreamingDataset / point lookup by ArtifactHandle
        v
VIME TransferQueueTrajectoryConsumer
        |
        | TrajectoryArtifactV1Alpha1.from_transfer_queue()
        v
trajectory_to_sample()
        |
        v
VIME native Sample -> existing training and update_weights path
```

TransferQueue 只负责 rollout artifact handoff。闭环训练里的 policy weight update 仍走 VIME 已有的 rollout-engine 管理和 `actor_model.update_weights()` 路径；`--tq-manage-rollout-servers` 用来在消费 TransferQueue 样本时仍由 VIME 管理 vLLM rollout engines。

## 3. 关键数据契约

### 3.1 Artifact schema

artifact schema 名称为 `vllm.trajectory/v1alpha1`。

TransferQueue record 由四部分组成：

- `partition_id`: `rollout-{urlquote(run_id)}-{urlquote(policy_version)}`
- `key`: `{urlquote(run_id)}:{urlquote(policy_version)}:{urlquote(request_id)}:{sample_index}`
- `fields`: Tensor payload
- `tag`: schema、run、policy、request、engine、model 等 metadata

必需 fields：

- `prompt_token_ids`: 一维整数 tensor，不能为空
- `response_token_ids`: 一维整数 tensor
- `response_logprobs`: 一维浮点 tensor，长度必须等于 `response_token_ids`

可选 fields：

- `prompt_logprobs`
- `rewards`
- `values`
- `loss_mask`
- `routed_experts`

必需 tags：

- `schema_name = "vllm.trajectory"`
- `schema_version = "v1alpha1"`
- `status = "complete"`
- `run_id`
- `request_id`
- `engine_id`
- `model_id`
- `policy_version`
- `created_at_ns`

可选 tags：

- `group_id`
- `sample_index`
- `finish_reason`

### 3.2 ArtifactHandle

vLLM 返回给调用方的是轻量 handle：

```json
{
  "artifact_handle": {
    "backend": "transfer_queue",
    "artifact_id": "run-a:1:req-a:0",
    "location": {
      "partition_id": "rollout-run-a-1",
      "key": "run-a:1:req-a:0"
    },
    "fields": ["prompt_token_ids", "response_token_ids", "response_logprobs"],
    "metadata": {
      "schema_name": "vllm.trajectory",
      "schema_version": "v1alpha1",
      "run_id": "run-a",
      "policy_version": 1,
      "status": "complete"
    }
  }
}
```

VIME consumer 支持两种读取方式：

- `get_from_handle(handle)`: 使用 handle 中的 `partition_id` 和 `key` 点查 artifact。
- `iter_artifacts()`: 使用 `StreamingDataset` 按 partition 持续消费训练样本。

## 4. vLLM 侧架构设计

### 4.1 配置入口

新增 `ArtifactTransferConfig`，挂到 `VllmConfig` 和 `EngineArgs`：

- `artifact_connector`: connector 名称，例如 `TransferQueueArtifactConnector`
- `engine_id`: artifact producer engine id，未设置时自动生成 UUID
- `artifact_role`: `artifact_producer`、`artifact_consumer` 或 `artifact_both`
- `transfer_mode`: `final`、`streaming`、`chunked`
- `export_fields`: 预留给 worker-side artifact export 的字段选择
- `failure_policy`: `fail_request`、`fallback_to_request_output`、`ignore`
- `artifact_connector_extra_config`: connector 私有配置
- `artifact_connector_module_path`: 外部 connector 动态加载路径

CLI 新增 `--artifact-transfer-config`，OpenAI completion/chat/responses 协议新增 `artifact_transfer_params` 字段。请求侧该字段被放入 `SamplingParams.extra_args`，最终进入 v1 `Request.artifact_transfer_params`；响应侧同名字段用于回传 `ArtifactHandle`。

### 4.2 Connector 抽象

vLLM 新增 artifact connector 抽象，形态类似 v1 KV connector，但对象是 rollout artifact，不是 KV cache block。

核心类型：

- `ArtifactHandle`: 可序列化外部 artifact 引用。
- `ArtifactConnectorMetadata`: scheduler 发给 worker 的 step metadata。
- `ArtifactConnectorWorkerMetadata`: worker 回传 scheduler 的 metadata。
- `ArtifactConnectorOutput`: worker 每步输出，包括 `finished_sending`、`handles`、`worker_meta`、`errors`。
- `ArtifactConnectorBase_V1`: scheduler/worker 双侧接口。
- `ArtifactConnectorFactory`: 注册并创建 `DummyArtifactConnector` 与 `TransferQueueArtifactConnector`。

### 4.3 vLLM 调度器集成

vLLM v1 scheduler 在初始化时根据 `artifact_transfer_config` 创建 scheduler-side connector。

请求进入时：

1. `Scheduler.add_request()` 调用 `artifact_connector.on_new_request(request)`。
2. `TransferQueueArtifactConnector` 从 `request.artifact_transfer_params` 读取 `run_id`、`policy_version`、`model_id`、`group_id`、`sample_index`。
3. connector 保存 prompt token，并确保 sampled-token logprob 可用。

生成过程中：

1. `Scheduler.update_from_output()` 在 stop-token trimming 后提取 accepted `new_token_ids` 和对应 sampled-token logprobs。
2. `record_request_output()` 校验 logprob token id 与 accepted output token id 对齐。
3. 对齐后把 logprob 追加到 request accumulator。

请求结束时：

1. `Scheduler._artifact_connector_finished()` 调用 `request_finished(request)`。
2. connector 构造 `TrajectoryArtifactV1Alpha1`。
3. 调用 `to_transfer_queue_record()` 得到 `key`、`partition_id`、`fields`、`tag`。
4. 发布到 TransferQueue。
5. 返回 `record.to_handle()`，scheduler 序列化为 `{"artifact_handle": handle.to_dict()}`。

### 4.4 TransferQueue connector

`TransferQueueArtifactConnector` 是当前 POC 的 producer 实现。

TransferQueue 初始化支持两种模式：

- 通过 `transfer_queue_init_config` 或 `transfer_queue_config_path` 使用已导出的 client config 初始化本地 TQ client，不主动连接 service Ray。
- 未提供 client config 时，connector 按 `ray_address` 初始化 Ray，并调用 `transfer_queue.init()` 连接共享服务。

发布模式：

- `sync`: request finish 时同步调用 `kv_put`。
- `async`: request finish 时把 record 放入后台队列，由 daemon thread 调用 `kv_put`。
- `async_batch`: 后台线程按 `partition_id` 聚合 batch，调用 `async_kv_batch_put`。

异步 batch 处理 ragged tensor 时会把同形状字段 `torch.stack`，不同形状字段转换为 nested tensor，再封装成 `TensorDict(batch_size=[N])`。

失败策略：

- `fail_request`: 发布失败时抛出 `RuntimeError`，请求失败。
- `fallback_to_request_output`: 发布失败时记录异常并返回普通输出，不返回 handle。
- `ignore`: 发布失败只告警。

可选 metrics：

- `artifact_metrics_path` 指向 JSONL 文件。
- 同步模式记录 `schema_build_ms`、`tq_init_ms`、`kv_put_ms`、`total_ms`。
- 异步模式额外记录 `queued`、`queue_depth`、`queue_wait_ms`、batch size 等信息。

### 4.5 当前 vLLM 侧边界

当前 TQ connector 的主要可用路径是 `transfer_mode="final"` 的 scheduler-side 发布。虽然框架已经有 worker-side hook 和 `streaming/chunked` 配置枚举，但 `TransferQueueArtifactConnector` 没有实现 `record_step_artifacts()`、`get_finished()` 等 worker-side staged publish 逻辑。

当前 `request_finished()` 构造的 artifact 包含 prompt token、response token、response logprob、finish reason 等基础 trajectory 数据；`rewards`、`loss_mask`、`routed_experts` 等字段在 schema 中支持，但当前 connector 没有从 vLLM 路径填充。

## 5. VIME + TransferQueue 侧架构设计

### 5.1 新增 trajectory source

VIME 新增 `--trajectory-source`：

- `generated`: 原有路径，调用 rollout function 生成样本。
- `transfer_queue`: 从 TransferQueue 消费外部发布的 trajectory artifact。

当选择 `transfer_queue`：

1. `RolloutManager` 不初始化普通 `data_source` 和 rollout function。
2. 初始化 `TransferQueueSampleSource.from_args(args)`。
3. `_get_rollout_data()` 从 sample source 拉取 `rollout_batch_size * n_samples_per_prompt` 条样本。
4. 返回 metrics: `{"transfer_queue/samples": len(data)}`。
5. `eval()`、`save()`、`load()` 对 transfer_queue source 直接跳过。

如果传入 `--tq-manage-rollout-servers`，VIME 仍会启动和管理 rollout servers，用于保持原有 closed-loop weight update 路径。

### 5.2 TransferQueueTrajectoryConsumer

`TransferQueueTrajectoryConsumer` 封装 TQ 读取逻辑。

初始化方式：

- 如果提供 `service_config_path`，直接读取 pickle client config，适合不希望当前训练进程连接 service Ray 的场景。
- 否则初始化 Ray，调用 `transfer_queue.init()`，通过 Ray actor `TransferQueueController` 获取 TQ config。

读取方式：

- `get_artifact(key, partition_id, timeout_s)`: 点查，用 `kv_batch_get()` 获取 fields，用 `kv_list()` 获取 tag。
- `get_from_handle(handle)`: 校验 backend 为 `transfer_queue`，再调用点查。
- `iter_artifacts()`: 构造 `transfer_queue.StreamingDataset`，迭代 `(fields, batch_meta)`；每条样本从 `fields` 里切片，从 `batch_meta.custom_meta[index]` 取 tag。

`_select_sample()` 对普通 batched tensor 使用 `value[index:index+1]`，对 nested tensor 使用 `unbind()[index].unsqueeze(0)`，保持与 schema decoder 的单样本 unwrap 逻辑兼容。

### 5.3 artifact 到 VIME Sample 的转换

`TrajectoryArtifactV1Alpha1.from_transfer_queue()` 负责 consumer-side schema 校验：

- schema name/version/status 必须匹配。
- required tags 和 required fields 必须存在。
- token/logprob 字段必须为一维 tensor。
- response token 与 response logprob 长度必须一致。
- `loss_mask` 如存在，长度必须等于 response token。

`trajectory_to_sample()` 转换为 VIME 原生 `Sample`：

- `tokens = prompt_token_ids + response_token_ids`
- `response_length = len(response_token_ids)`
- `reward`: 默认要求单标量 reward；缺失时如果 `require_reward=True` 会报错。
- `loss_mask`: 缺省为全 1；如存在必须是二值。
- `group_index`: 基于 `run_id`、`policy_version`、`group_id/request_id` 的稳定 hash。
- `rollout_id` 和 `index`: 基于 `run_id`、`request_id`、`sample_index` 的稳定 hash。
- `weight_versions = [str(policy_version)]`
- `rollout_log_probs = response_logprobs.tolist()`
- `status`: `stop/eos/None -> COMPLETED`，`length/1 -> TRUNCATED`，`abort/aborted -> ABORTED`，其他为 `FAILED`。
- metadata 保留 artifact schema、run、request、policy、engine、model、finish reason 等。

### 5.4 辅助脚本

`scripts/transfer_queue/` 提供了 POC 运行辅助：

- `tq_service.py`: 在共享 Ray 上启动长驻 TransferQueue controller。
- `start_services.sh`: 启动 Ray head、TQ service，并导出 client config。
- `export_config.py`: 从 Ray actor 获取 TQ config 并 pickle 到文件。
- `health_check.py`: 检查 TQ service 和 Ray actor。
- `dispatch_rollouts.py`: 读取 JSONL prompts，向 OpenAI-compatible vLLM completion endpoint 发送请求，并写入 `artifact_transfer_params`。
- `phase6_prompts.jsonl`: smoke test prompts。

`ExternalVllmRolloutDispatcher` 会为每个 prompt fan-out `n_samples_per_prompt` 个请求，并设置：

- `X-Request-Id = {run_id}-{policy_version}-{group_id}-{sample_index}`
- `logprobs = 0`
- `artifact_transfer_params = {run_id, policy_version, model_id, group_id, sample_index}`

## 6. 运行配置示例

### 6.1 启动 TransferQueue service

```bash
scripts/transfer_queue/start_services.sh
```

该脚本会输出：

- Ray GCS 地址
- Ray Client 地址
- TransferQueue PID
- 导出的 `client_config.pkl`

### 6.2 启动 artifact producer vLLM

示例 `--artifact-transfer-config`：

```json
{
  "artifact_connector": "TransferQueueArtifactConnector",
  "artifact_connector_module_path": "vllm.distributed.artifact_transfer.artifact_connector.v1.transfer_queue_connector",
  "artifact_role": "artifact_producer",
  "transfer_mode": "final",
  "failure_policy": "fail_request",
  "artifact_connector_extra_config": {
    "transfer_queue_config_path": "/path/to/client_config.pkl",
    "run_id": "run-a",
    "policy_version": "1",
    "model_id": "qwen3-4b",
    "publish_mode": "sync",
    "artifact_metrics_path": "/tmp/artifact_metrics.jsonl"
  }
}
```

如需降低 request finish 同步开销，可改用：

```json
{
  "publish_mode": "async_batch",
  "publish_batch_size": 8,
  "publish_flush_interval_ms": 2.0,
  "publish_queue_maxsize": 4096,
  "publish_drain_on_shutdown": true
}
```

### 6.3 分发 rollout 请求

```bash
python scripts/transfer_queue/dispatch_rollouts.py \
  --endpoint http://<vime-managed-router>/v1/completions \
  --model <served-model-name> \
  --run-id run-a \
  --policy-version 1 \
  --prompts scripts/transfer_queue/phase6_prompts.jsonl \
  --n-samples-per-prompt 2 \
  --max-tokens 32 \
  --output /tmp/dispatched_rollouts.jsonl
```

### 6.4 VIME 训练侧消费

```text
--trajectory-source transfer_queue
--tq-manage-rollout-servers
--tq-ray-address ray://<transfer-queue-host>:20001
--tq-service-config-path /path/to/client_config.pkl
--tq-run-id run-a
--tq-policy-version 1
--tq-partition-prefix rollout
--num-rollout <explicit-count>
```

注意：vLLM producer 当前默认写入 `rollout-{run_id}-{policy_version}` partition；VIME CLI 当前默认 `--tq-partition-prefix train`。端到端运行时需要显式设置 `--tq-partition-prefix rollout`，或者统一修改双方 partition 规则。

## 7. 实现文件索引

### 7.1 vLLM

| 文件 | 作用 |
| --- | --- |
| `vllm/config/artifact_transfer.py` | Artifact transfer 配置对象与合法性校验 |
| `vllm/distributed/artifact_transfer/schema.py` | `vllm.trajectory/v1alpha1` wire schema、TQ record、handle 构造 |
| `vllm/distributed/artifact_transfer/artifact_connector/v1/base.py` | Artifact connector v1 抽象接口 |
| `vllm/distributed/artifact_transfer/artifact_connector/factory.py` | Connector 注册、动态加载、实例化 |
| `vllm/distributed/artifact_transfer/artifact_transfer_state.py` | worker-side connector 全局状态与 TP engine_id 同步 |
| `vllm/distributed/artifact_transfer/artifact_connector/v1/transfer_queue_connector.py` | TransferQueue producer connector |
| `vllm/v1/core/sched/scheduler.py` | scheduler-side accumulator、artifact publish、handle 回传 |
| `vllm/v1/worker/gpu/artifact_connector.py` | worker-side hook 框架 |
| `vllm/entrypoints/openai/*/protocol.py` | OpenAI API 请求/响应透传 `artifact_transfer_params` |
| `vllm/outputs.py`, `vllm/v1/outputs.py` | RequestOutput / EngineCoreOutput 携带 artifact transfer params |

### 7.2 VIME

| 文件 | 作用 |
| --- | --- |
| `vime/rollout/trajectory_artifact.py` | VIME consumer-side schema decoder |
| `vime/rollout/trajectory_sample.py` | artifact 转 VIME `Sample` |
| `vime/rollout/transfer_queue_consumer.py` | TQ point lookup、handle lookup、StreamingDataset 消费 |
| `vime/rollout/trajectory_source.py` | RolloutManager 可用的 TQ sample source |
| `vime/ray/rollout.py` | `--trajectory-source transfer_queue` 接入训练主流程 |
| `vime/utils/arguments.py` | TQ 相关 CLI 参数与校验 |
| `vime/rollout/rollout_dispatcher.py` | 外部 vLLM rollout dispatch helper |
| `scripts/transfer_queue/*` | TQ service、config export、health check、dispatch 脚本 |

## 8. 测试覆盖

vLLM 侧新增测试覆盖：

- schema key/partition 构造、字段规范化、handle 构造、TQ record round-trip。
- scheduler producer 发布完整 `vllm.trajectory/v1alpha1` artifact。
- exported TQ config 初始化路径避免启动 Ray。
- debug metrics JSONL。
- `sync`、`async`、`async_batch` 发布路径。
- unknown publish mode 校验。
- sampled-token logprob token id mismatch 触发 `fail_request`。
- `fallback_to_request_output` 失败策略。
- scheduler handle 只消费一次。

VIME 侧新增测试覆盖：

- consumer-side schema decode 与错误路径。
- artifact 到 `Sample` 的转换、reward 缺失、loss mask、finish reason 状态映射。
- `TransferQueueTrajectoryConsumer` 点查、handle lookup、StreamingDataset 迭代、partition step。
- `TransferQueueSampleSource` batch 拉取与字段选择。
- `RolloutManager` 从 TQ source 读取 batch。
- `--tq-manage-rollout-servers` 下仍启动 VIME-managed rollout servers。
- CLI 参数默认值与必填项校验。
- dispatcher 生成稳定 request id 和 artifact metadata。

## 9. 已知约束与后续工作

1. 当前 TQ connector 实际使用的是 final publish，streaming/chunked 还只是接口预留。
2. vLLM 当前发布的基础 artifact 不包含 reward。训练 ready 路径需要在消费前补齐 `rewards`，或者在调试场景使用 `--tq-allow-missing-rewards`。
3. VIME 默认 `--tq-partition-prefix train` 与 vLLM 默认 `rollout` 不一致，端到端运行需要显式对齐。
4. `export_fields` 目前主要服务 worker hook 框架；TQ final publish 路径固定写 schema 必需字段。
5. `async` / `async_batch` 模式在 request 返回 handle 时，后台写入可能仍在队列中。需要依赖 queue drain、metrics 和 consumer retry/polling 处理最终可见性。
6. TransferQueue backend 当前由部署配置决定，后续可替换为更高性能 backend，例如 RDMA/Mooncake/Yuanrong 等。
7. 点查 `get_artifact()` 当前通过 `kv_batch_get()` 取 fields、再通过 `kv_list()` 取 tag；大 partition 下 `kv_list()` 的成本需要关注。
8. 生产化需要补齐跨进程/跨节点 observability，包括 artifact backlog、publish failure、consumer lag、partition retention 和清理策略。
