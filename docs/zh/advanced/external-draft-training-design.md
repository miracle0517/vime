# VIME 外部 Draft 模型在线训练设计

> 状态：已实现并完成 NPU smoke 验证
> 目标版本：当前补丁版本
> 首期算法：EAGLE3
> 首期运行模式：同步在线训练、Draft 与 Actor rank 0 共卡、`use_logits=false`

## 实现状态

当前代码已完成阶段 1 MVP，并提前合入了部分阶段 2 能力：

- Megatron Actor 在参数更新前采集 aux hidden、final-norm hidden、token、mask 和 position；
- Draft Trainer 嵌入 Megatron Actor rank 0，Actor 阶段结束后在同一 NPU 上串行训练 EAGLE3；
- Feature、Target LM Head 和 Draft checkpoint 均携带 Target/Draft 版本信息；
- Target 与外部 Draft 在同一次 generation pause 中依次完成 NCCL 热更新；
- Draft checkpoint 保存 model、optimizer、scheduler、版本和 architecture fingerprint；
- 普通与 streaming vLLM rollout 都会透传 speculative counters 和 weight version；
- 当前强制 `Megatron + full/NCCL + PP=1 + VPP=1 + CP=1 + rollout 非共卡`，并拒绝 MTP、routing replay、`keep-old-actor` 和有损 acceptance。

Draft checkpoint 若自带 Transformers `auto_map` 训练实现可直接加载；否则必须通过
`--draft-model-factory-path package.module.factory` 提供与 vLLM Draft 权重命名一致的
EAGLE3 `torch.nn.Module`。factory 接收 `(args, device)`；模型也可以实现
`export_for_vllm(dtype, device)`，用于显式转换发布参数名。

## 1. 背景

VIME 已经可以通过 vLLM 的 `SpeculativeConfig` 加载外部 EAGLE Draft 模型进行投机解码，但当前在线训练只覆盖 Target 模型内部的 MTP 层，外部 Draft 模型会随着 RL 过程中 Target 参数持续变化而逐渐失配。失配会降低 Draft token 接受率和平均接受长度，最终可能使投机解码的额外 Draft 与验证开销超过其收益。

本设计参考 verl-SpeCo 的实现，将目标模型在 RL 数据上的隐藏状态转化为 Draft 训练样本，在每轮或固定间隔的 Actor 更新期间训练独立 Draft 模型，并将新权重热更新到 vLLM 中的外部 Draft Model。

VIME 当前已经具备以下基础能力：

- vLLM 外部 EAGLE Draft 推理配置；
- Megatron Actor 的旧策略/current policy log-prob 前向；
- Ray 训练 Actor 与 placement group；
- Target 权重向 vLLM 的在线同步；
- `start_draft_weight_update` Draft 权重更新入口；
- rollout 暂停、cache flush、权重更新和恢复生成流程；
- Sample 中的投机接受率和接受长度数据结构。

需要新增的核心能力是：

1. 从 Megatron Target 前向中采集 Draft 训练特征；
2. 独立的外部 Draft 模型训练进程、优化器和 checkpoint；
3. Target 与 Draft 权重发布协调；
4. Target/Draft/Feature 的版本一致性管理；
5. 完整的 Draft 训练和投机解码观测指标。

## 2. 设计目标

### 2.1 功能目标

- 在 VIME RL 训练过程中在线训练外部 EAGLE3 Draft 模型；
- 从 Actor 的旧策略前向采集辅助层 hidden state 和 final hidden state；
- 保证 hidden state、冻结 Target LM Head 与监督分布来自同一 Target 版本；
- 周期性将 Draft 新权重热更新到所有 vLLM rollout engine；
- 支持 Draft 模型、优化器、scheduler 和版本状态保存与恢复；
- 提供 collect-only 特征落盘能力，为离线 Draft 训练和问题诊断预留接口；
- 投机解码继续保持 Target 分布精确，不改变 PPO 样本的目标分布。

### 2.2 性能目标

- 特征采集只选择部分样本和 token 窗口，不保存整批全序列 hidden state；
- 大特征张量不经过 Python driver，以 Ray ObjectRef 或等价的点对点通道传输；
- Draft 训练不阻塞 rollout 的时间应可配置和可观测；
- 生产版本在一次 rollout 暂停窗口内完成 Target 和 Draft 两类权重更新；
- 在线训练后的端到端 rollout tokens/s 应优于关闭投机解码的基线。

### 2.3 非目标

首期不包含以下内容：

- 不从头设计新的投机解码算法；
- 不将外部 Draft 模型改写成 Megatron 模型；
- 不支持有损 speculative acceptance；
- 不在首期支持 PP/CP 下的跨 stage 特征汇聚；
- 不在首期支持 Actor、rollout 与 Draft 三者复杂共卡；
- 不在首期实现 DFlash、DSpark、Domino 和 P-EAGLE；
- 不在首期实现完全异步的多版本 Draft 训练。

## 3. 参考实现分析

verl-SpeCo 通过三层扩展构成闭环：

- `SpecoTaskRunner` 替换上游 PPO Trainer，并为 rollout worker 注入 Draft 权重发布能力；
- `SpecoRayPPOTrainer` hook rollout、old-logprob、Actor 更新和 rollout 权重发布；
- 独立 `SpecoWorker` 接收 CPU/Ray ObjectRef 特征，周期性训练和发布 Draft。

参考文件：

- [TaskRunner 集成](../../../../verl-SpeCo/verl_speco/integration/task_runner.py)
- [PPO Trainer 集成](../../../../verl-SpeCo/verl_speco/trainer/speco_ray_trainer.py)
- [Draft Worker](../../../../verl-SpeCo/verl_speco/workers/speco_worker.py)
- [Draft Trainer](../../../../verl-SpeCo/verl_speco/trainer/base_trainer.py)
- [特征数据格式](../../../../verl-SpeCo/verl_speco/trainer/feature_store.py)

### 3.1 参考时序

```mermaid
flowchart LR
    A["Target Tn 与 Draft Dn 生成 rollout"] --> B["旧 Actor Tn 前向"]
    B --> C["采集 aux hidden、final hidden、token 和 position"]
    C --> D["同步 Tn LM Head 到 Draft Worker"]
    D --> E["Actor PPO 更新：Tn 到 Tn+1"]
    E --> F["Draft 使用 Tn 监督训练：Dn 到 Dn+1"]
    F --> G["发布 Target Tn+1"]
    G --> H["发布 Draft Dn+1"]
    H --> A
```

该时序具有两个性质：

1. 一条 Draft 训练样本内部的 aux hidden、final hidden 和 LM Head 严格属于同一版本 `Tn`；
2. 下一轮服务组合是 `Target Tn+1 + Draft Dn+1`，Draft 相对新 Target 存在一轮有界滞后。

本设计首期保持这一行为。相比在 Actor 更新后额外执行一次 `Tn+1` 特征前向，它显著减少训练成本；同时通过版本字段和接受率指标显式监控滞后影响。

### 3.2 参考算法能力

| 算法 | 监督方式 | 本设计处理 |
|---|---|---|
| EAGLE3 | 多层 aux hidden；final hidden 经冻结 LM Head 生成软标签 | 首期实现 |
| EAGLE1/2 | hidden SmoothL1 回归与 token soft-CE | 后续 |
| DFlash | 多 anchor 并行 block 训练 | 第二阶段候选 |
| DSpark | DFlash、Markov bias 和 L1 分布匹配 | 后续 |
| Domino | DFlash 与 GRU 因果修正头 | 后续 |
| P-EAGLE | COD 下采样并行预测与 KL 训练 | 后续 |

### 3.3 不直接复制参考编排的原因

verl-SpeCo 的 Target/rollout 基于 verl worker 体系，Draft Trainer 使用 HF Transformers 与 FSDP。VIME 的 Actor 是 Megatron，数据采用 packed sequence，并存在 TP、PP、CP、VPP 等并行布局。因此：

- 可以移植 Draft model、loss、数据对齐规则和特征 schema；
- 不应直接移植 verl worker、dispatch、配置和进程组代码；
- Target hidden 采集必须针对 Megatron pipeline 和 packed sequence 单独实现；
- Target 到 vLLM 的权重转换继续使用 VIME 现有 Megatron-to-HF 路径；
- Draft 到 vLLM 的权重发布使用独立 HF Draft 参数迭代器。

## 4. VIME 现状与扩展点

### 4.1 训练主循环

当前 [train.py](../../../train.py) 的主时序为：

```text
rollout_manager.generate
    -> actor_model.async_train
    -> actor_model.save_model
    -> actor_model.update_weights
```

适合增加以下扩展点：

```text
rollout_manager.generate
    -> actor_model.async_train，并采集 Draft 特征
    -> draft_model.collect/train
    -> save Actor/Draft
    -> Target/Draft 联合发布
```

### 4.2 Actor 前向

[MegatronTrainRayActor](../../../vime/backends/megatron_utils/actor.py) 已通过 `compute_log_prob` 调用 `forward_only`。这是首期特征采集入口，因为：

- 该前向发生在 Actor 参数更新之前；
- 当前 Actor 权重通常与本轮 rollout Target 权重一致；
- 输入已经包含 PPO 训练所需完整 prompt/response token；
- 无需修改 vLLM 的生成响应协议以传回大规模 hidden state。

现有 `custom_megatron_before_log_prob_hook` 只能在前向前调用，无法取得每层输出、batch token 位置和 final hidden，因此需要在 `forward_only` 内增加结构化 collector，而不是仅依赖现有 hook。

### 4.3 vLLM Draft 权重更新

[VLLMEngine](../../../vime/backends/vllm_utils/vllm_engine.py) 已暴露：

- `start_weight_update`；
- `start_draft_weight_update`；
- `finish_weight_update`。

VIME 的 vLLM patch 会在 Draft 更新会话中将 weight transfer target 切换为外部 `DraftModelSpeculator` 中的 Draft 模型。当前 MTP 更新已经具备以下时序：

```text
pause generation
flush cache
start target update
send target weights
finish target update
start draft update
send draft weights
finish draft update
continue generation
```

外部 Draft 训练复用该生命周期，但 Draft 权重由 Actor rank 0 内嵌的独立 Draft 模型提供，不能像 MTP 路径一样再次发送 Actor 权重。

## 5. 总体架构

```mermaid
flowchart TB
    subgraph Rollout["Rollout 资源组"]
        VLLM["vLLM Target + External Draft"]
    end

    subgraph Actor["Actor Megatron 资源组（TP）"]
        Train["PPO 训练"]
        Collect["DraftFeatureCollector"]
        Head["Target LM Head 导出"]
        subgraph Rank0["Actor rank 0 共卡"]
            Queue["Versioned Feature Queue"]
            DTrain["单卡 HF Draft Trainer"]
            DCkpt["Draft Checkpoint"]
            DPublish["Draft Weight Exporter"]
        end
    end

    Driver["ExternalDraftCoordinator"]

    VLLM -->|"rollout tokens"| Train
    Train --> Collect
    Collect -->|"Ray ObjectRef"| Queue
    Head -->|"同版本 LM Head"| Queue
    Queue --> DTrain
    DTrain --> DCkpt
    DTrain --> DPublish
    Driver --> Train
    Driver --> DTrain
    Driver --> VLLM
    DPublish -->|"draft update session"| VLLM
    Train -->|"target update session"| VLLM
```

### 5.1 资源拓扑

不新增 `draft` placement group。资源拓扑为：

```python
pgs = {
    "actor": ...,
    "critic": ...,
    "rollout": ...,
    "draft": None,
}
```

Draft Trainer 的资源规则：

- 模型只在 Actor 全局 rank 0 初始化，并与该 rank 使用同一张训练卡；
- Actor TP ranks 完成当前训练阶段后，driver 再触发 Draft，两个反向不会并发；
- Draft Trainer 以 `distributed=False` 运行，不加入 Actor TP 进程组，也不触发 DDP collective；
- 所有 Actor rank 产生的 CPU feature ObjectRef 汇聚到 rank 0 的有界队列；
- 只有 Actor rank 0 保存 checkpoint 和生成发布快照；
- rollout 仍使用独立卡，Draft 新权重通过在线更新接口发布给全部 vLLM workers。

该方案省去一张专用 Draft 卡，同时引入以下约束：

- Actor rank 0 需要同时容纳 Target TP shard、Draft 参数、梯度和优化器状态；
- Draft 训练期间其余 Actor ranks 空闲等待，不适合较大规模的 Draft 训练；
- Draft 本地反向必须与 Actor collective 严格串行；
- 后续若启用训练 offload，需要单独验证 Draft 分配是否被 memory-saver 生命周期影响。

## 6. 模块设计

### 6.1 `ExternalDraftCoordinator`

职责：

- 根据 rollout id 判断本轮是否采集、训练、保存和发布；
- 将 Actor 返回的特征 manifest 转交给 Draft 资源组；
- 在 Actor 更新前同步对应 Target LM Head；
- 驱动 Draft 训练并收集 metrics；
- 协调 Target/Draft 的 rollout 热更新；
- 维护 rollout id、Target version、Draft version 和已发布版本；
- 在异常情况下选择跳过 Draft 更新，而不是破坏 PPO 主流程。

建议接口：

```python
class ExternalDraftCoordinator:
    def should_collect(self, rollout_id: int) -> bool: ...
    def should_train(self, rollout_id: int) -> bool: ...
    def collect(self, rollout_id: int, manifests: list[FeatureManifest]) -> dict: ...
    def sync_target_head(self, rollout_id: int, head_ref: ObjectRef) -> dict: ...
    def train(self, rollout_id: int) -> DraftTrainResult: ...
    def save(self, rollout_id: int, force_sync: bool = False) -> None: ...
    def update_rollout_weights(self, actor_model, rollout_manager) -> dict: ...
```

### 6.2 `DraftFeatureCollector`

Collector 在 Actor `forward_only` 生命周期内工作：

1. 根据 rollout id、sample rate 和 token budget 生成采集计划；
2. 在配置的 Transformer 层安装 forward hook；
3. 在 LM Head/output layer 上安装 forward-pre-hook，取得 final norm 后的 hidden；
4. 运行现有 log-prob 前向；
5. 恢复 packed microbatch 到原始 sample 顺序；
6. 按采集窗口截取 token、hidden 和 position；
7. 异步复制到 CPU BF16 pinned memory；
8. 创建 Ray ObjectRef，只向上层返回 manifest；
9. 在 `finally` 中移除所有 hook，防止重复注册和引用泄漏。

采集计划必须是确定性的。建议使用以下字段计算 hash：

```text
seed = hash(global_seed, rollout_id, original_sample_id)
```

这样各 rank 可以独立得出相同窗口，不依赖 Python 进程随机状态。

### 6.3 final hidden 采集点

EAGLE3 `use_logits=false` 要求 final hidden 与 Target LM Head 输入严格一致。应捕获：

```text
Transformer layers
    -> final norm
    -> [capture here]
    -> LM Head
```

不能直接使用最后一个 Transformer layer 的原始输出代替，尤其对于 pre-norm 模型，否则冻结 LM Head 重建出的分布会与 Actor logits 不一致。

初始化时和定期运行时执行一致性检查：

```text
argmax(TargetLMHead(captured_final_hidden))
    == argmax(Actor logits)
```

同时记录最大 logit 误差或 top-k 一致率，超出阈值时拒绝该批特征。

### 6.4 Actor 共卡 Draft Trainer

`ExternalDraftTrainGroup` 只保留 driver 侧协调接口，实际计算委托给 Actor rank 0：

```python
class MegatronTrainRayActor:
    def collect_external_draft_features(self, feature_refs, target_head, target_version): ...
    def train_external_draft(self, rollout_id): ...
    def save_external_draft(self, rollout_id): ...
    def prepare_external_draft_publish_snapshot(self): ...
```

Draft 内部可以复用/移植 verl-SpeCo 的：

- EAGLE3 模型结构；
- vocab mapping；
- target embedding 加载；
- future-step 概率蒸馏 loss；
- token-mean 梯度归一化；
- Draft checkpoint 元数据。

需要替换的部分包括：

- OmegaConf/verl config 访问；
- verl dispatch 和 worker group；
- verl rollout TP/SP mesh；
- Actor LM Head 导出接口；
- SGLang hidden-state 布局；
- verl checkpoint manager hook。

### 6.5 Draft 权重发布器

Draft 发布器必须输出与 vLLM Draft checkpoint 相同的 HF 参数名和张量布局。

职责包括：

- 从 Actor rank 0 的本地 Draft state 中导出 publish state；
- 过滤 optimizer、冻结 Target LM Head 和非服务参数；
- 根据配置转换 BF16/FP16 发布 dtype；
- 按 bucket 发送，限制单次 Ray/NCCL payload；
- 计算参数名、shape、dtype 和 checksum；
- 调用 `start_draft_weight_update` 后发送 Draft 权重；
- 所有 engine 成功后才提交新的 Draft version。

如果任一 rollout engine 更新失败：

- 本轮 Draft version 不得标记为已发布；
- generation 保持暂停，直到 Target/Draft 状态恢复一致或明确回滚；
- 首期直接 fail-fast，不尝试静默使用部分更新的 engine 集群。

## 7. 数据契约

### 7.1 `DraftFeatureSample`

建议使用显式 schema version：

```python
@dataclass
class DraftFeatureSample:
    schema_version: int
    algorithm: str

    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    hidden_positions: torch.Tensor

    aux_hidden_states: torch.Tensor
    final_hidden_states: torch.Tensor | None
    target_topk_logprobs: torch.Tensor | None

    rollout_id: int
    target_weight_version: str
    original_sample_id: str
    prompt_length: int
    response_length: int
    window_start: int
    window_end: int
    aux_layer_ids: list[int]
    hidden_layout: str
```

首期约束：

- `algorithm == "EAGLE3"`；
- `final_hidden_states` 必须存在；
- `target_topk_logprobs` 必须为空；
- `aux_hidden_states` 最后一维布局为按层拼接的 `[L * H]`；
- tensor 转 CPU 后必须 contiguous；
- hidden 使用 BF16，token/position 使用 int64，loss mask 使用 float32 或 bool；
- `window_end - window_start` 与 hidden row 数必须严格一致。

### 7.2 EAGLE3 对齐

假设 `h_aux[p]` 是位置 `p` 的辅助层 hidden，`h_final[p]` 是位置 `p` 输入 LM Head 的 final hidden，则首期训练样本对齐为：

```text
Draft feature:      h_aux[p]
Draft token input:  x[p+1]
Target distribution: LMHead_Tn(h_final[p+1])
Target token mask:  loss_mask[p+2]
```

因此一个有效训练窗口至少需要比 Draft feature row 多保留后续 token/final-hidden 行。窗口构造和裁剪不能在不理解 shift 的情况下分别处理各 tensor。

### 7.3 loss mask

- prompt token 不参与 Draft token loss；
- response 中的 padding、无效或截断占位 token 不参与 loss；
- 只有能够形成完整 future target 的位置参与 loss；
- 不允许因为窗口截断将 prompt 最后一个 token 错误计入 response loss；
- 多轮 agent 数据必须沿用 VIME 已有 `loss_masks`，不能仅按 prompt length 重建。

### 7.4 版本契约

每条特征必须携带：

```text
target_weight_version
rollout_id
lm_head_version
feature_schema_version
draft_architecture_fingerprint
```

首期训练规则：

```text
sample.target_weight_version == current_target_head_version
```

不满足时直接拒绝样本并记录 `draft/feature_rejected_version_mismatch`。

如果未来启用跨 step buffer，必须为每个 Target version 保存对应 LM Head snapshot，或切换到保存 target top-k logits 的监督模式。不能用当前 LM Head 处理旧版本 final hidden。

## 8. Megatron 并行处理

### 8.1 DP

- 每个 Actor DP rank 只采集本地样本；
- manifest 中保留原始全局 sample id；
- Draft DP rank 通过一致性 hash 或 round-robin 接收样本；
- 同一样本只归属一个 Draft DP owner，避免重复训练；
- loss 和 token count 在 Draft DP 组内做全局归约。

### 8.2 TP

Collector 必须识别 `sequence_parallel`：

- hidden 在 TP ranks 上复制时，仅 TP rank 0 导出；
- hidden 按 sequence 切分时，先在 TP 组 all-gather，再按 packed `cu_seqlens` 重建；
- hidden 按 hidden dimension 切分的模型，需要按正确维度 gather；
- gather 完成前不能创建 ObjectRef；
- 增加 shape 和 position 连续性断言，避免静默拼错。

### 8.3 PP/VPP

首期限定 `PP=1`、`VPP=1`。

后续支持 PP 时：

- 根据全局 layer id 找到所在 PP/VPP stage；
- 各 stage 只捕获本地目标层；
- 使用专用通信组或 Ray refs 汇聚同一样本的多个层；
- final hidden 由 pipeline last stage 导出；
- 只有全部目标层齐全的样本才能进入 Draft queue。

### 8.4 CP

首期限定 `CP=1`。

后续支持 CP 时，需要还原 CP 对序列首尾或分块切分后的原始 token 顺序，并同时恢复 position、loss mask、aux hidden 和 final hidden。仅 gather hidden 而不重建 mask/position 会造成难以发现的监督错位。

## 9. 在线训练时序

### 9.1 初始化

1. 解析并校验 Draft 配置；
2. 创建 rollout manager；
3. 创建 Actor/Critic；
4. 创建 Draft 资源组；
5. 加载 Draft model、optimizer、scheduler 和 checkpoint；
6. 校验 Draft checkpoint 与 vLLM speculative config；
7. 发布初始 Target 权重；
8. 必要时发布或校验初始 Draft 权重；
9. 比较 rollout engine 中的 Target/Draft checksum；
10. 开始 rollout。

### 9.2 每轮训练

| 阶段 | Target 版本 | Draft 版本 | 行为 |
|---|---:|---:|---|
| rollout | `Tn` | `Dn` | vLLM 生成样本 |
| feature capture | `Tn` | `Dn` | 旧 Actor 前向采集 `Tn` 特征 |
| head sync | `Tn` | `Dn` | 同步 `Tn` LM Head |
| Actor train | `Tn -> Tn+1` | `Dn` | PPO 更新 |
| Draft train | `Tn+1` | `Dn -> Dn+1` | 使用 `Tn` 监督训练 |
| publish | `Tn+1` | `Dn+1` | 依次更新 vLLM Target/Draft |

### 9.3 采集和训练周期

采集与训练分别控制：

```text
collect_interval_rollouts
train_interval_rollouts
publish_interval_rollouts
```

约束：

- 未采集到有效样本时不得触发 Draft optimizer step；
- Draft 未成功训练时不得发布空快照；
- Draft 训练成功但未到发布周期时可以继续保存在训练端；
- 下一次发布应发布最新完整 Draft 状态，而不是参数增量链；
- checkpoint 周期和发布周期可以不同。

## 10. 权重更新设计

### 10.1 首期实现

为了降低改造范围，首期允许：

```text
actor_model.update_weights()
draft_model.update_weights()
```

这可能造成两次 generation pause，仅用于验证正确性和端到端收益。

### 10.2 生产实现

将现有 Target updater 的生命周期拆成：

```python
session = rollout_manager.begin_weight_update(
    pause_generation=True,
    flush_cache=True,
)

actor_model.send_target_weights(session)
draft_model.send_draft_weights(session)

rollout_manager.commit_weight_update(session)
```

内部仍然使用两个 vLLM update session：

1. `start_weight_update` / `finish_weight_update`；
2. `start_draft_weight_update` / `finish_weight_update`。

但 generation 只暂停和恢复一次。

### 10.3 原子性

一次联合发布包含：

```text
target_version = N+1
draft_version = M+1
pair_id = hash(target_version, draft_version)
```

所有 engine 都成功后才提交 pair。任一 engine 失败时不得让路由层继续向部分更新的 engine 分发新请求。

首期采用 fail-fast；后续可增加：

- 从 CPU publish snapshot 重试；
- 将失败 engine 从 router 暂时摘除；
- 恢复上一组 Target/Draft pair；
- engine 重启后按版本重新同步。

## 11. 配置设计

建议新增参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--enable-external-draft-training` | false | 启用外部 Draft 在线训练 |
| `--draft-algorithm` | eagle3 | Draft 算法 |
| `--draft-model-path` | null | Draft HF checkpoint |
| `--draft-model-factory-path` | null | 无 Transformers `auto_map` 时使用的 EAGLE3 模型 factory |
| `--draft-target-embedding-path` | `--hf-checkpoint` | 初始化 Draft embedding 的 Target checkpoint |
| `--draft-target-embedding-key` | `model.embed_tokens.weight` | Target embedding 参数名 |
| `--draft-feature-layer-ids` | null | Target 辅助层 id |
| `--draft-collect-interval` | 1 | 特征采集间隔 |
| `--draft-collection-sample-rate` | 1.0 | 样本采集率 |
| `--draft-max-samples-per-rollout-per-dp` | 16 | 每个 Actor DP 的样本预算 |
| `--draft-max-tokens-per-rollout-per-dp` | 16384 | 每个 Actor DP 的 token 预算 |
| `--draft-hidden-window-mode` | front | `front` 或 `random` |
| `--draft-hidden-window-tokens` | 512 | 单样本窗口长度 |
| `--draft-train-interval` | 1 | Draft 训练间隔 |
| `--draft-train-steps-per-trigger` | 10 | 单次触发 optimizer steps |
| `--draft-batch-size-per-gpu` | 4 | Draft batch size |
| `--draft-learning-rate` | 1e-5 | Draft learning rate |
| `--draft-lr-scheduler-type` | constant | `constant` 或 `cosine` |
| `--draft-lr-warmup-steps` | 0 | Draft scheduler warmup steps |
| `--draft-lr-total-steps` | 0 | cosine 总步数；0 表示从 rollout 计划推导 |
| `--draft-publish-interval` | 1 | Draft 发布间隔 |
| `--draft-publish-dtype` | bf16 | 发布 dtype |
| `--draft-checkpoint-path` | null | Draft checkpoint 目录 |
| `--draft-save-interval` | null | Draft 保存间隔 |
| `--draft-collect-only` | false | 只采集特征，不在线训练 |
| `--draft-feature-store-path` | null | 特征存储路径 |
| `--draft-offload-model` | false | 空闲时 offload Draft model |
| `--draft-offload-optimizer` | false | 空闲时 offload optimizer |

阶段 1 的最小启用示例（其余 Actor/rollout 参数沿用现有训练脚本）：

```bash
--enable-external-draft-training \
--draft-model-path /models/eagle3 \
--draft-model-factory-path your_package.eagle3.build_model \
--draft-feature-layer-ids 2,16,29 \
--draft-checkpoint-path /checkpoints/draft \
--draft-train-steps-per-trigger 10 \
--draft-batch-size-per-gpu 4 \
--update-weight-mode full \
--update-weight-transport nccl \
--vllm-speculative-config '{"method":"eagle3","model":"/models/eagle3","num_speculative_tokens":3}'
```

如果 Draft checkpoint 自带可训练的 Transformers `auto_map`，应省略
`--draft-model-factory-path`。Target embedding 默认从 `--hf-checkpoint` 的
`model.embed_tokens.weight` 加载；不同模型命名需要覆盖
`--draft-target-embedding-key`。

启动校验：

- `vllm_speculative_config.method == "eagle"`；
- speculative config 的 `model` 与 `draft-model-path` 架构一致；
- Draft `target_hidden_size` 与 Actor hidden size 一致；
- Draft 需要的 aux layer 数与 `draft-feature-layer-ids` 一致；
- Draft vocab mapping 覆盖所有 Draft vocab 行；
- `PP=1`、`CP=1` 首期约束成立；
- 未启用有损 acceptance method；
- checkpoint 中的 Draft architecture fingerprint 与当前配置一致。

## 12. 代码改动清单

### 12.1 修改现有文件

| 文件 | 改动 |
|---|---|
| `train.py` | 创建 Actor 共卡 Draft 协调包装；插入 collect/train/save/publish 时序 |
| `vime/utils/arguments.py` | 新增 Draft 参数与启动校验 |
| `vime/ray/placement_group.py` | 创建 Actor 共卡 Draft 包装，不分配额外 accelerator bundle |
| `vime/ray/actor_group.py` | 增加 Draft 发布快照向 Actor rank 0 的委托接口 |
| `vime/backends/megatron_utils/actor.py` | 启用特征采集；在 rank 0 持有并驱动本地 Draft Trainer |
| `vime/backends/megatron_utils/model.py` | collector context、layer hook、final hidden hook、packed 样本还原 |
| `vime/backends/megatron_utils/update_weight/*` | 抽取可复用的权重更新 session 生命周期 |
| `vime/backends/vllm_utils/vllm_engine.py` | 增加 Draft 更新入口与权重版本处理 |
| `vime/rollout/vllm_rollout.py`、`vllm_streaming_rollout.py` | 将 vLLM speculative metrics 写入 `meta_info` |

### 12.2 新增文件

```text
vime/backends/speculative_training/
    __init__.py
    config.py
    feature_schema.py
    feature_collector.py
    draft_group.py
    draft_trainer.py
    backends/
        __init__.py
        eagle3.py
    factories/
        __init__.py
        verl_speco_eagle3.py
```

### 12.3 测试文件

```text
tests/speculative_training/
    test_config.py
    test_draft_group.py
    test_draft_trainer_helpers.py
    test_eagle3_backend.py
    test_feature_collector.py
    test_feature_schema.py
```

## 13. 可观测性

### 13.1 特征采集

```text
draft/feature_candidate_samples
draft/feature_collected_samples
draft/feature_collected_rows
draft/feature_payload_mib
draft/feature_rejected_alignment
draft/feature_rejected_version_mismatch
draft/feature_rejected_missing_layer
timing/draft_feature_plan
timing/draft_feature_gpu_to_cpu
timing/draft_feature_ray_put
```

### 13.2 Draft 训练

```text
draft/loss
draft/top1_accuracy
draft/top5_accuracy
draft/valid_tokens
draft/optimizer_steps
draft/learning_rate
draft/grad_norm
draft/target_version
draft/trained_version
timing/draft_activate
timing/draft_train
timing/draft_cleanup
```

### 13.3 权重发布

```text
draft/publish_attempted
draft/published
draft/publish_version
draft/publish_payload_mib
draft/publish_checksum_mismatch
timing/target_weight_update
timing/draft_weight_update
timing/combined_weight_update_pause
```

### 13.4 投机解码

```text
spec/accept_rate
spec/accept_length
spec/draft_tokens
spec/accepted_tokens
spec/verify_count
spec/rollout_tokens_per_second
spec/speedup_vs_no_spec
```

当前 `Sample.SpecInfo` 已有接受 token、Draft token 和 verify count 字段，但 rollout adapter 需要把 vLLM 响应中的对应字段写入 `meta_info`。

## 14. 异常处理

### 14.1 可降级异常

以下异常默认跳过本轮 Draft 训练，但 PPO 继续：

- 没有采集到有效特征；
- 特征窗口过短；
- 未到训练或发布周期；
- 单个样本 position 不连续；
- 单个样本缺少目标层；
- Draft batch 有效 token 数为零。

### 14.2 必须失败的异常

以下异常必须 fail-fast：

- Target LM Head 与 final hidden 版本不一致；
- Draft checkpoint 架构与 vLLM Draft 模型不一致；
- 参数名或 shape 无法映射到 vLLM Draft；
- 集群中只有部分 engine 完成 Draft 权重更新；
- TP/PP/CP 重建后 token position 与 hidden row 不一致；
- Target/Draft update session 状态机非法；
- checkpoint 恢复后 Draft version 倒退。

## 15. 测试与验收

### 15.1 单元测试

- schema 序列化、反序列化和版本校验；
- EAGLE3 `p/p+1/p+2` 对齐；
- prompt/response 边界和 loss mask；
- front/random window 的确定性；
- padding、截断和最短序列；
- final hidden 与 LM Head logits 一致性；
- 旧版本 feature 被当前 LM Head 拒绝；
- HF Draft 参数到 vLLM 参数的 name/shape mapping；
- checkpoint 保存和恢复 optimizer/scheduler/version。

### 15.2 分布式测试

- DP 样本无重复、无遗漏；
- TP replicated hidden 只导出一次；
- TP sequence-parallel hidden gather 顺序正确；
- Actor 共卡 Draft 的 loss/token mean 与独立单卡参考结果一致；
- 多 rollout engine 更新无死锁；
- 一个 engine 更新失败时不会恢复 generation。

### 15.3 端到端测试

建立三组基线：

1. 不启用投机解码；
2. 使用固定外部 Draft；
3. 使用在线训练外部 Draft。

验收指标：

- 在线训练 Draft loss 呈下降趋势；
- Draft top-1/top-5 accuracy 相比固定 Draft 改善；
- accept rate 和 accept length 相比固定 Draft 改善；
- Target 最终输出分布与关闭 speculative decoding 的基线一致；
- PPO reward、KL、entropy 不出现系统性回退；
- 在线 Draft 训练后的端到端 rollout tokens/s 高于不投机基线；
- 保存恢复后训练和发布版本连续；
- 连续运行多个发布周期无显存持续增长或 Ray ObjectRef 泄漏。

## 16. 分阶段实施

### 阶段 0：验证 vLLM 外部 Draft 热更新

- 加载外部 EAGLE checkpoint；
- 手工修改一个 Draft 参数并发布；
- 校验 vLLM Draft checksum 发生变化；
- 校验 Target 参数没有被误写；
- 校验更新后的生成、CUDA graph 和 cache 行为。

### 阶段 1：正确性 MVP

- EAGLE3、`use_logits=false`；
- Draft 与 Actor rank 0 共卡并串行训练；
- `PP=1`、`CP=1`；
- 同步训练；
- 当前 rollout 特征，不跨版本缓存；
- Actor old-logprob 特征采集；
- Draft checkpoint；
- 首期允许 Target/Draft 分别暂停更新。

### 阶段 2：生产级同步闭环

- TP sequence-parallel hidden 重建；
- Target/Draft 一次暂停联合发布；
- 异步 CPU copy、Ray ObjectRef 分块；
- 参数 bucket 与发布 checksum；
- 完整 acceptance/throughput metrics；
- collect-only 和离线 feature store。

### 阶段 3：拓扑和性能扩展

- PP/VPP 多 stage 特征汇聚；
- CP 序列重建；
- Draft model/optimizer offload；
- Draft 多卡训练或与 rollout 共卡；
- 多版本异步队列和最大 stale version；
- DFlash 与其他算法后端。

## 17. 主要风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| hidden 与 token 错位 | loss 可下降但接受率不提升 | 强制 position schema、对齐单测、运行时 top-k 检查 |
| final hidden 捕获在 final norm 之前 | 重建监督分布错误 | 在 LM Head 输入处捕获并与 Actor logits 比较 |
| LM Head 与 hidden 版本不一致 | 软标签错误 | 样本和 head 都携带版本；不匹配直接拒绝 |
| old-logprob 被复用优化跳过 | 某些配置无特征 | 采集周期强制专用 forward 或关闭该次复用 |
| TP/PP/CP 重建错误 | 分布式场景静默训练坏数据 | 首期限制拓扑；逐项加入 shape/position 断言 |
| 两次权重更新暂停过长 | 降低 rollout 吞吐 | MVP 后合并为一次 pause、两个 update session |
| Draft 发布参数名不匹配 | vLLM 更新失败或部分更新 | 启动时 dry-run mapping；发布 checksum；fail-fast |
| Actor rank 0 显存不足 | Draft 初始化或 optimizer step OOM | 采样预算、窗口、训练间隔、Draft optimizer offload |
| Draft 相对 Target 一轮滞后 | 接受率改善受限 | 记录版本差；必要时改为 Actor 更新后额外特征前向 |
| Ray ObjectRef 未释放 | CPU/object store 内存增长 | 有界队列、消费确认、超时清理、泄漏压力测试 |

## 18. 设计决策总结

1. **首期选择 EAGLE3 `use_logits=false`。** VIME rollout 不返回 hidden/top-k logits，而 Actor old-logprob 前向天然可提供 hidden；使用同版本 LM Head 可以恢复完整软标签。
2. **Target 保持 Megatron，Draft 使用单卡 HF Trainer。** 两者通过显式 feature schema 和权重发布协议解耦。
3. **从 Actor old-logprob 前向采集。** 不扩展 vLLM HTTP 协议传输大 hidden state。
4. **Draft 与 Actor rank 0 共卡。** 两个训练阶段严格串行，Draft 不加入 Actor TP/DDP collective，不额外占用训练卡。
5. **首期不跨 Target 版本混合数据。** 避免 final hidden 与 LM Head 版本不一致。
6. **复用 vLLM Draft update target。** 不重复实现 rollout engine 内部权重替换。
7. **生产版本采用一次 pause、两次 update session。** 保持 Target/Draft 集群发布的一致性并减少生成停顿。
8. **用接受率和端到端吞吐验收。** Draft loss 下降只是中间指标，不能替代真实 speculative decoding 收益。
