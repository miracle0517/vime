# VIME DSpark 投机模型增量训练设计

## 实现状态（Qwen 首期）

本文档的 Qwen/Speculators 路径已在 VIME 中按最小改动实现：

- 复用现有 External EAGLE3 的 Actor old-policy 特征采集、版本队列、检查点和发布编排；
- 新增 `draft_algorithm=dspark`，首期只接受 Qwen3 系列的 Speculators DSpark checkpoint；
- 训练模型由 `speculators.models.dspark.DSparkDraftModel` 加载，loss、Markov head 和 confidence head
  均使用 Speculators 原生实现；
- DSpark batch 使用单序列多 document 拼接，每个 document 的尾部 block 被屏蔽，禁止 anchor 跨样本；
- Target LM Head 按 Target 版本同步到 DSpark 的 `lm_head` 和 `verifier_lm_head`；
- Draft 发布复用 `start_draft_weight_update`，并把一个 DSpark snapshot 保持为单次
  `model.load_weights`，满足 Ascend confidence head 的原子加载要求；
- 未修改 vLLM 和 vLLM Ascend。运行环境必须已经提供 DSpark 推理加载和 Draft 热更新入口。

首期不支持 Gemma、DeepSeek-V4 内嵌 `mtp.*` DSpark、FSDP Draft worker 或跨 Target 版本异步训练。
Qwen serving 路径限定 `markov_head_type=vanilla`；启用 confidence head 时要求
`confidence_head_with_markov=true`，与当前 vLLM/vLLM Ascend Qwen DSpark 参数结构一致。

### 启动参数

训练环境需要单独安装与 checkpoint 和 vLLM 版本匹配的 `speculators`。VIME 不把它加入基础依赖，
避免 Speculators 的 PyTorch 版本约束覆盖 Ascend 环境中的定制 PyTorch/torch_npu。

```bash
--enable-external-draft-training \
--draft-algorithm dspark \
--draft-model-path /models/qwen3-dspark-speculators \
--draft-target-embedding-path /models/qwen3-target \
--draft-train-interval 1 \
--draft-train-steps-per-trigger 10 \
--draft-batch-size-per-gpu 4 \
--draft-dspark-max-anchors 64 \
--draft-dspark-loss-fn '{"ce":0.1,"tv":0.9}' \
--save /outputs/actor-checkpoints \
--save-interval 10 \
--draft-save-hf /outputs/dspark-{rollout_id} \
--vllm-speculative-config '{"method":"dspark","model":"/models/qwen3-dspark-speculators"}'
```

`--draft-save-hf` 使用与主模型 `--save-hf` 相同的 `{rollout_id}` 路径模板习惯。Draft 达到
`--draft-save-interval` 或最后一个 rollout 时，将完整参数转为 CPU 连续张量后调用 `save_pretrained`，
原子导出 `config.json`、PyTorch HuggingFace 权重和完成标记；它不依赖 Actor 的 `--save-interval`。
失败的导出不会覆盖上一个有效目录。导出目录不得与原始 `--draft-model-path` 或 Actor 的
`--save-hf` 目录相同。目前此选项仅支持 DSpark。

仅设置 `--vllm-speculative-config method=dspark` 只代表 rollout engine 使用 DSpark 推理，本身没有可保存的
训练副本。设置 `--draft-save-hf` 后，VIME 会自动启用 External Draft 训练副本、将算法归一化为 DSpark，
并在未显式设置 `--draft-model-path` 时使用 speculative config 中的 `model`。如果 Actor rank 0 未返回经过
完整性校验的导出结果，训练会直接报错，不再静默跳过保存。

导出实际发生在 Actor rank 0 所在主机，成功日志会包含该主机名、绝对路径、权重文件列表和总字节数。
Ray 多机运行时应将 `--draft-save-hf` 指向所有节点可见的共享文件系统；否则模型已经写入 worker 本地磁盘，
但在提交任务的 head 节点上看不到。

本地 checkpoint 的 `block_size` 和 `aux_hidden_state_layer_ids` 会从 `config.json` 读取；远程
checkpoint 必须显式提供 `--draft-dspark-block-size` 和 `--draft-feature-layer-ids`。`max_anchors`
控制每步实际构造的并行 Draft block 数，在线增量训练建议从 64 开始，根据显存和有效 token 数调整。

## 1. 背景

本文设计 VIME 在强化学习训练过程中对 DSpark Draft 模型进行持续增量训练，
并将新 Draft 权重热更新到 vLLM/vLLM Ascend rollout engine。

本文基于以下本地代码版本分析：

- VIME：`c346561b`，已包含 External EAGLE3 在线训练框架；
- vLLM：`568afb3a13`，v0.26.0；
- vLLM Ascend：`30b44103d`。

当前前提如下：

1. DSpark 基础 checkpoint 已存在；
2. vLLM/vLLM Ascend 已能加载该 checkpoint 并完成投机推理；
3. 本设计不负责重新实现 DSpark 推理；
4. 本设计解决训练副本初始化、在线特征生成、增量优化、checkpoint、版本一致性和 Draft 热更新。

DSpark 由并行 DFlash-style backbone、顺序 Markov head 和可选 confidence
head 组成。训练时应继续使用有监督蒸馏目标，不把 PPO policy loss 或 reward
直接施加到 Draft 模型上。Draft 的职责是逼近当前 Target 分布，提高接受长度，
而不是改变最终生成分布。

## 2. 目标与非目标

### 2.1 目标

- 从 vLLM 可加载的 DSpark 基础 checkpoint 创建可训练副本；
- 复用 Actor 前向产生的 token、辅助层 hidden state 和 final hidden state；
- 使用与 DSpark checkpoint 布局一致的 block alignment、Markov teacher forcing、
  distribution matching 和 confidence supervision；
- 支持只训练 Markov/confidence head，以及解冻 Draft backbone 的两阶段训练；
- 严格绑定 Target/Draft/feature/checkpoint 版本；
- 将训练状态导出为 vLLM Draft loader 可接受的 serving state；
- 在一次 generation pause 内更新 Target 和 Draft；
- 提供发布前离线门控、发布后在线指标和失败回滚能力；
- 保持现有 EAGLE3 路径行为不变。

### 2.2 非目标

- 不在 vLLM inference model 上直接执行 backward；
- 不依赖 rollout engine 返回完整 Target logits；
- 第一阶段不做异步跨多个 Target 版本的 stale feature 训练；
- 第一阶段不做 Draft 参数 delta 编码；每次发布完整的 serving-owned state；
- 不改变 speculative verification 的无损语义；
- 不把 confidence score 当作跳过 Target verification 的依据。

## 3. 当前代码结论

### 3.1 VIME 已有能力

现有 External EAGLE3 框架已经具备以下可复用组件：

- `train.py` 中的 collect/train/save/publish 调度；
- `DraftFeatureCollector` 的 Megatron layer hook；
- `DraftFeatureSample` 和按 Target 版本隔离的 feature queue；
- Actor rank 0 上的 `ExternalDraftTrainer`；
- Draft checkpoint、optimizer、scheduler 恢复；
- `start_draft_weight_update` 入口；
- generation pause 下的 Target/Draft 连续更新。

但当前算法边界仍写死 EAGLE3：

- `speculative_training/config.py` 只允许 `draft_algorithm=eagle3`；
- `feature_schema.py` 固定 `eagle3_aux_plus_last`；
- `draft_trainer.py` 直接调用 `collate_eagle3_samples` 和
  `compute_eagle3_loss`；
- 默认 exporter 只发送 `requires_grad=True` 的参数；
- feature 的最短窗口和 shift 是 EAGLE3 的 `p/p+1/p+2` 规则。

这些逻辑需要抽象，但现有调度和传输主流程不需要推倒重写。

### 3.2 vLLM DSpark checkpoint 类型

当前代码至少存在两类 serving layout。

#### 独立 Draft checkpoint

Qwen/Gemma/Speculators-style checkpoint 作为独立 Draft model 加载。主要权重包括：

- `embed_tokens`，可能与 Target 共享，也可能独立；
- `fc`、`hidden_norm`；
- 多层 Draft decoder；
- `norm`、`lm_head`；
- `markov_head.markov_w1`；
- `markov_head.markov_w2`；
- `confidence_head.proj`；
- 可选 `t2d`/`d2t` vocabulary mapping。

vLLM 的 `Qwen3DSparkForCausalLM.load_weights()` 接受接近 HF/Speculators
checkpoint 的参数名，并在内部完成 QKV、gate/up 等 stacked parameter mapping。

#### Target 内嵌 Draft checkpoint

DeepSeek-V4 DSpark 权重位于 Target checkpoint 的 `mtp.{stage}.*` namespace。
vLLM Ascend 的 `DSparkDeepseekV4ForCausalLM.load_weights()` 会把该 namespace
映射到运行时 Draft model 的 `model.layers.*`、Markov head、embedding 和 LM head。

因此训练后端不能把 Qwen serving name 硬编码给所有模型，必须使用 checkpoint
profile 和显式 exporter。

#### 首期支持矩阵

| Checkpoint/模型类型 | 增量训练优先级 | 首期范围 |
| --- | --- | --- |
| Qwen/Speculators-style 独立 Draft | P0/P1 | 完整支持 backbone、Markov、confidence 训练和热更新 |
| 使用 Qwen-style Draft architecture 的其他 verifier | P1 | checkpoint profile 和 loader parity 通过后启用 |
| Gemma 独立 Draft | P2 | 复用协议，单独验证 Gemma parameter mapping |
| DeepSeek-V4 `mtp.*` 内嵌 Draft | P2/P3 | 先完成 exporter/load parity，再启用 MoE backbone 训练 |

首个可交付版本应只承诺 Qwen/Speculators-style 路径。当前 Ascend confidence
head 和 dynamic verification 已明确覆盖 Qwen DSpark，而 DeepSeek loader 会忽略
`confidence_head.*`，其 MoE/MHC 参数映射和训练资源需求也明显不同。将两类模型
同时放入首期会使 feature/loss 问题和 serving mapper 问题难以独立定位。

### 3.3 两种 block alignment

当前 vLLM 同时处理两种 DSpark query layout：

- `sample_from_anchor`：anchor query 自身预测第一个 Draft token；
- `bonus_anchor`：anchor 只是 bonus/context token，后续 mask query 才产生 Draft token。

Speculators-format checkpoint 会由 vLLM config 转换逻辑设置
`dspark_bonus_anchor=True`。其他 DSpark checkpoint 可能采用
`sample_from_anchor=True`。训练必须读取 checkpoint config，禁止通过
`draft_algorithm=dspark` 推断 alignment。

### 3.4 Draft 热更新前提和发布约束

vLLM upstream 已提供：

- `GPUWorker.get_draft_model()`；
- `start_draft_weight_update()`；
- `WeightTransferEngine.set_weight_update_target()`；
- session 结束后的 target reset。

本实现按用户环境的现状，把 vLLM/vLLM Ascend 已支持 `start_draft_weight_update`
以及 DSpark Draft target 切换作为运行前提。VIME 只调用现有入口，不修改或 patch
vLLM/vLLM Ascend。缺少该入口的旧运行时会在启动 Draft update session 时明确失败，
不属于 VIME 训练侧的兼容范围。

另一个阻塞项是 Draft loader 的分桶语义：

- HCCL non-packed 路径可能对每个 tensor 调用一次 `model.load_weights()`；
- Qwen Ascend confidence loader 在一次调用没有收到 confidence 权重时会关闭
  confidence head；
- Qwen DSpark loader 会在 `load_weights()` 末尾重建 fused KV buffer；
- 分桶 session 中间执行上述动作可能观察到不完整权重集合。

VIME 侧将一个 DSpark serving snapshot 放入单个 packed buffer，使其只触发一次
`model.load_weights()`。这样 confidence weight/bias 在同一次 loader 调用中可见，后续
也不会出现“不含 confidence 的第二个桶”。代价是发布时 trainer 和 rollout worker
需要容纳完整 Draft snapshot 的临时 packed buffer；这是“不修改 vLLM Ascend”约束下的
显式内存取舍，必须纳入部署显存预算。

## 4. 总体决策

采用“训练副本、服务副本、发布协议”三层解耦架构。

```text
Megatron Actor T(n)
    |
    | old-policy pre-update forward
    v
Versioned DSpark Feature Store
    |
    v
Trainable DSpark Replica D-train(n)
    |
    | evaluate + export
    v
Serving Snapshot D-serve(n)
    |
    | HCCL/IPC draft update session
    v
vLLM DSpark Runtime D-runtime(n)
```

三个状态不得混用：

- `D-train` 包含 optimizer 需要的标准训练参数；
- `D-serve` 使用 checkpoint loader 接受的名字和布局；
- `D-runtime` 可能包含 TP shard、stacked QKV、量化或旋转后的 kernel tensor。

VIME 只生成 `D-serve`，由 vLLM/vLLM Ascend loader 负责转换到
`D-runtime`。VIME 不直接生成 NPU kernel layout。

## 5. 组件设计

### 5.1 DraftAlgorithmBackend

将现有 `ExternalDraftTrainer` 中的 EAGLE3 逻辑移入算法后端。

建议接口如下：

| 接口 | 作用 |
| --- | --- |
| `load_training_model()` | 从基础 checkpoint 创建训练模型 |
| `feature_spec()` | 声明层号、hidden 语义、lookahead 和 document 要求 |
| `validate_feature()` | 校验 schema、版本、shape、alignment |
| `collate()` | 组 batch、生成 anchor 和 attention metadata |
| `compute_loss()` | 返回 graph-connected loss 和 count-normalized metrics |
| `sync_teacher_state()` | 同步当前 Target LM head/embedding/norm |
| `trainable_parameter_groups()` | 返回 staged freeze 和差分 LR 参数组 |
| `export_serving_state()` | 生成 loader-facing state |
| `architecture_manifest()` | 生成不可变兼容指纹 |

后端注册表至少包含：

- `eagle3 -> Eagle3Backend`；
- `dspark -> DSparkBackend`。

Trainer 只负责 optimizer、scheduler、DDP、queue、checkpoint 和版本状态，
不再导入任何 `compute_eagle3_*` 函数。

### 5.2 DSparkCheckpointProfile

启动时从基础 checkpoint 和 vLLM speculative config 构造 profile：

```text
profile_id
checkpoint_format
architecture
block_size
alignment_mode
aux_hidden_state_layer_ids
target_hidden_size
draft_hidden_size
draft_num_layers
draft_vocab_size
mask_token_id
markov_rank
markov_head_type
enable_confidence_head
confidence_head_with_markov
has_own_embed_tokens
has_own_lm_head
vocab_mapping_fingerprint
serving_name_schema
```

`checkpoint_format` 初期支持：

- `speculators_external`；
- `dense_external`；
- `deepseek_mtp_embedded`。

Profile 必须参与 feature、trainer checkpoint 和 serving snapshot 的 fingerprint。
任意影响 query layout 或参数 shape 的字段变化都要求重启，不能在线切换。

### 5.3 DraftFeatureSpec

算法后端向 collector 返回显式采集契约：

```text
algorithm = dspark
schema_version = 2
aux_layer_ids = [...]
aux_layer_id_semantics = checkpoint_original
final_hidden_semantics = post_final_norm
minimum_future_tokens = K
requires_document_ids = true
requires_target_lm_head = true
```

必须使用原始 checkpoint 的 `aux_hidden_state_layer_ids`。vLLM 为运行时 hidden
capture 可能对 layer id 做 `-1/+1` 转换，该转换不应泄漏到 VIME collector。

### 5.4 DraftFeatureSample V2

建议 schema：

```text
input_ids:                 [T]
loss_mask:                [T]
position_ids:             [T]
document_ids:             [T]
aux_hidden_states:        [T, N_aux * H_target]
final_hidden_states:      [T, H_target]
target_weight_version:    str
rollout_id:               int
sample_id:                str
window_start/window_end:  int
prompt_length:            int
response_length:          int
algorithm:                dspark
architecture_fingerprint: str
alignment_mode:           str
```

张量使用 CPU contiguous BF16 保存，token/mask/position/document 使用明确 dtype。

`final_hidden_semantics` 非常重要。当前 collector 在 output layer pre-hook 处获取
的通常是 final norm 后、LM head 前的 hidden。如果训练后端再次执行 verifier
norm，会导致 double normalization。训练后端应根据 feature spec 决定是否应用
Target norm。

## 6. 特征采集时序

### 6.1 推荐：Actor 更新后采集

当前 External EAGLE3 在 Actor optimizer step 前采集特征，然后把由旧 Target
监督的 Draft 和新 Target 一起发布。对 DSpark 尤其是 confidence calibration，
这种一轮滞后不理想。

推荐时序：

```text
rollout 使用 (Tn, Dn)
Actor 使用 rollout 数据完成 optimizer step: Tn -> Tn+1
在抽样数据上执行 post-update forward
采集 feature(Tn+1) 并导出 teacher state(Tn+1)
训练 Draft: Dn -> Dn+1, supervised_by=Tn+1
发布 (Tn+1, Dn+1)
```

post-update forward 只处理 Draft sample budget 覆盖的样本，不必重新前向整个
rollout batch。采样计划应在 forward 前确定，避免先计算全部样本再丢弃。

### 6.2 兼容模式：更新前采集

保留 `pre_update` 模式作为低开销选项，但必须记录：

```text
feature_target_version = Tn
published_target_version = Tn+1
teacher_lag = 1
```

当 `teacher_lag` 超过配置阈值时不得继续发布 Draft。confidence head 默认不使用
该模式，因为 confidence 的绝对概率校准比普通 top-1 accuracy 更容易受 Target
漂移影响。

### 6.3 样本和窗口选择

只对 response 区域创建 anchor，同时保留足够的左侧 context。推荐：

- 每个 Actor DP 每 rollout 采集 2 到 8 个窗口；
- 每个窗口 512 到 2048 token；
- 每窗口随机选择 32 到 128 个 anchor；
- 一个 batch 内可以 pack 多个 document，但必须使用不同 `document_ids`；
- anchor 不得跨 document、padding 或 window 尾部；
- 短于 query lookahead 的样本直接跳过。

固定只取 response 前部会使 DSpark 偏向回答开头，应默认采用确定性随机窗口，
随机种子由 `rollout_id + sample_id + target_version` 派生。

## 7. DSpark alignment

### 7.1 Sample-from-anchor

设 speculative block 长度为 `K`，anchor 为 token 位置 `p`：

```text
query input:      x[p], MASK, ..., MASK       # K 个 query
prediction:       x[p+1], ..., x[p+K]
Markov previous:  x[p], ..., x[p+K-1]
```

训练时 Markov previous token 使用 ground truth；推理时使用前一个已采样 Draft
token。每个 anchor 必须保留 `K` 个 future token。

### 7.2 Bonus-anchor

设 speculative token 数为 `K`：

```text
query input:      x[p], MASK, ..., MASK       # 1 + K 个 query
prediction:             x[p+1], ..., x[p+K]
Markov previous:        x[p], ..., x[p+K-1]
```

anchor query 只提供 context，不计入 token loss。Speculators-format checkpoint
应走该路径，与 vLLM 的 `dspark_bonus_anchor=True` 保持一致。

### 7.3 Loss mask

VIME 需要把现有“token 是否属于 response”的 mask 转换为“该 prediction 是否
参与 Draft loss”的 mask。对于预测 `x[q]` 的 slot，应该读取 `token_loss_mask[q]`，
不能直接复用 anchor 位置的 mask。

需要用手工小序列测试以下边界：

- prompt 最后一个 token 作为 anchor，首个 response token 作为 target；
- response 中间 anchor；
- response 末尾不足一个 block；
- truncation 后 loss mask 中间出现零；
- packed document 边界。

## 8. Teacher state 与词表

### 8.1 Target LM head

不传输完整 Target logits。Actor 每个 Target 版本只导出一次 LM head，Draft
trainer 使用 `final_hidden_states @ lm_head.T` 重建 Target logits。

Target head 必须：

- 完成 Megatron TP all-gather；
- 去掉 `padded_vocab_size` 的 padding row；
- 保持与 feature 相同的 `target_weight_version`；
- 在 Draft trainer 上以 BF16 保存，softmax/loss 使用 FP32；
- tied embedding 时使用真实 shared embedding/output tensor。

### 8.2 Embedding、LM head 和 norm 的共享策略

Checkpoint profile 决定运行时权重所有权：

| 策略 | 训练行为 | 发布行为 |
| --- | --- | --- |
| Draft own embedding/head | 从基础 Draft checkpoint 加载并冻结 | serving snapshot 保留；未改变可不重复发送 |
| Draft shares Target | 训练前同步当前 Target 权重，`requires_grad=False` | 不以 Draft name 发布，由 runtime alias Target |
| DeepSeek embedded | 按 checkpoint `mtp.*` 规则加载 | exporter 输出 `mtp.*` loader name |

不能根据数值是否相等动态决定所有权。所有权应由基础 checkpoint 的实际参数集合
和 runtime manifest 固定，否则重启后可能得到不同 sharing 行为。

### 8.3 Reduced vocabulary

如 `draft_vocab_size < target_vocab_size`，必须使用 checkpoint 自带的 `t2d/d2t`
mapping：

- hard CE label 先完成 Target-to-Draft 映射；
- Target probability 使用映射指定的 row；
- 不允许简单截取 Target vocab 前 `draft_vocab_size` 行；
- mapping tensor 和 fingerprint 属于 serving snapshot；
- 发布前验证所有 Draft id 都有合法 Target id。

## 9. Loss 设计

### 9.1 基础目标

默认与 DSpark/Speculators 训练配方对齐：

```text
L = alpha_ce * L_ce
  + alpha_tv * L_tv
  + alpha_conf * L_conf
```

推荐初始值：

```text
alpha_ce   = 0.1
alpha_tv   = 0.9
alpha_conf = 1.0
```

其中：

- `L_ce`：真实 next token 的交叉熵；
- `L_tv`：Draft 与 Target probability 的 total variation distance；
- `L_conf`：confidence logit 与 analytical acceptance label 的 BCE。

confidence label 为：

```text
c_star = sum_v min(P_draft(v), P_target(v))
       = 1 - TV(P_draft, P_target)
```

`c_star` 必须 detach，confidence loss 不应通过标签反向影响 Draft logits。

### 9.2 Position weighting

block 后部 token 对 prefix accepted length 的贡献较低，使用 position decay。
第一阶段直接复用基础 checkpoint 对应 Speculators 版本的 weighting 定义，不在
VIME 中创建新公式。

必须记录每个位置的：

- loss；
- greedy accuracy；
- analytical acceptance rate；
- conditional prefix survival；
- confidence MAE；
- confidence cumulative-product bias。

### 9.3 大词表内存控制

禁止一次性长期保留 `[batch, anchors, K, vocab]` 的 Draft 和 Target FP32 tensor。
建议：

- anchor microbatch；
- vocab logits BF16，softmax/TV reduction FP32；
- 每个 microbatch 完成 loss accumulation 后立即释放 logits；
- 以全局 valid token count 归一化；
- DDP 下保持 graph-connected zero，确保空 token rank 仍参与 collective。

## 10. 增量优化策略

### 10.1 初始化

训练副本必须从 rollout engine 使用的同一个基础 Draft checkpoint 初始化，禁止
随机初始化 DSpark backbone。推荐顺序：

1. 加载 checkpoint config、模型和 vocab mapping；
2. 计算 architecture fingerprint；
3. 校验 training model state 与 serving manifest；
4. 同步当前 Target teacher state；
5. 加载 VIME Draft resume checkpoint；
6. 校验 resume checkpoint 的 base checkpoint hash 和 optimizer version。

### 10.2 两阶段训练

#### 阶段 A：Head adaptation

训练：

- Markov head；
- confidence head。

冻结：

- Draft backbone；
- `fc`、norm；
- embedding、LM head。

该阶段开销小，适合验证 feature alignment、loss、checkpoint 和发布链路。建议
先运行 20 到 100 个 optimizer step，再根据 validation gate 决定是否进入阶段 B。

#### 阶段 B：Backbone adaptation

解冻：

- `fc`；
- Draft decoder layers；
- Draft norm；
- Markov/confidence head。

继续冻结 embedding 和 LM head。使用差分学习率，例如：

| 参数组 | 建议 LR |
| --- | --- |
| Draft backbone | `1e-5` 到 `3e-5` |
| Markov head | `5e-5` 到 `3e-4` |
| Confidence head | `5e-5` 到 `3e-4` |

实际默认值应通过目标模型和 RL domain 的短实验确定，不写死在 backend 中。

### 10.3 Feature queue

默认只使用当前 Target 版本的 feature：

- 收到 `Tn+1` teacher state 后清理其他版本；
- batch 中禁止混合 Target 版本；
- 未收集到足够 anchor 时跳过 optimizer step；
- feature 可以 repeat，但每次重新采样 anchor；
- validation split 通过 sample id 的稳定 hash 产生，不能从 train batch 临时抽取。

第一阶段不建议跨 Target 版本 replay。若未来需要 replay，必须同时保存对应版本的
LM head、norm 和 architecture fingerprint，显存和存储代价较高。

### 10.4 资源放置

DSpark 通常比单层 EAGLE3 更大，推荐恢复真正的 Draft placement 配置：

- `dedicated`：一到多张独立 NPU，生产默认；
- `actor_rank0`：与 Actor rank 0 串行共卡，仅用于小模型或 smoke test。

`dedicated` 模式下 Actor 只产生 CPU BF16 feature refs，Draft worker 自己维护
model/optimizer。多卡 Draft 采用独立 DDP group，不能加入 Actor TP/DP collective。

## 11. 版本状态机

### 11.1 版本定义

```text
target_version       Target 完整权重版本
feature_version      feature 对应的 Target 版本
teacher_version      LM head/norm 对应的 Target 版本
draft_train_version  每次成功训练后递增
draft_serve_version  最近成功发布的 Draft 版本
pair_version         hash(target_version, draft_serve_version)
```

`feature_version == teacher_version` 是执行 optimizer step 的硬条件。

### 11.2 正常状态转换

```text
SERVING(Tn, Dn)
  -> ACTOR_TRAINING
  -> FEATURE_READY(Tn+1)
  -> DRAFT_TRAINING(Tn+1, Dn+1)
  -> VALIDATING
  -> SNAPSHOT_READY(Tn+1, Dn+1)
  -> PUBLISHING
  -> SERVING(Tn+1, Dn+1)
```

### 11.3 Draft 训练失败

Draft 训练失败不应回滚已经完成的 PPO optimizer step。允许发布：

```text
(Tn+1, Dn)
```

但必须：

- 记录 `draft_teacher_lag`；
- confidence dynamic verification 超过 lag 阈值时关闭；
- stale Draft 仍由 Target 完整 verification，不能改变最终分布；
- 下一轮优先重新训练 Draft；
- lag 超过硬阈值后禁用 speculative decoding，而不是继续使用失配 confidence。

### 11.4 部分 engine 发布失败

这是 fail-closed 情况：

- generation 保持 pause；
- 未提交新的 pair version；
- 对全部 engine 重试同一完整 snapshot；
- 无法恢复时重启失败 engine 并加载最近 committed pair；
- 不允许部分 engine 使用新 Draft、部分使用旧 Draft 后继续接收请求。

## 12. Checkpoint 设计

### 12.1 Trainer checkpoint

`draft_latest.pt` 保存：

```text
training_model_state
optimizer_state
scheduler_state
grad_scaler_state
rng_state
optimizer_steps
draft_train_version
draft_serve_version
target_version
last_rollout_id
base_checkpoint_id
architecture_fingerprint
feature_schema_version
training_stage
```

该 checkpoint 用于 VIME 恢复，不直接交给 vLLM。

### 12.2 Serving snapshot

Serving snapshot 保存：

```text
named_tensors
source_name_schema
target_loader_schema
architecture_fingerprint
base_checkpoint_id
trained_against_target_version
draft_version
pair_id
tensor_manifest
snapshot_checksum
```

`tensor_manifest` 对每个 tensor 记录 name、shape、dtype、numel、checksum 和
ownership。发布器在发送前校验 manifest，engine 在 load 后返回 checksum 或至少
loaded-name/shape summary。

### 12.3 发布完整状态而非 optimizer delta

第一阶段每次发布完整的 serving-owned trainable state，原因是：

- 大部分 backbone 参数经过 optimizer 后都发生变化；
- vLLM loader 需要完成 stacked/TP/quant processing；
- delta patch 会绕开标准 `load_weights()` 语义；
- 完整 snapshot 更容易重试和回滚。

共享的 Target embedding/LM head、明确未改变且 runtime 独立持有的冻结权重可以
不发送，但该集合必须来自 ownership manifest，而不是 `requires_grad`。

## 13. Serving exporter

### 13.1 Qwen/Speculators exporter

优先输出基础 checkpoint 的 HF/Speculators names：

- `layers.*`；
- `fc.*`；
- `hidden_norm.*`；
- `norm.*`；
- `markov_head.*`；
- `confidence_head.*`；
- 必要时 `embed_tokens.*`、`lm_head.*`、`d2t`。

由 `Qwen3DSparkForCausalLM.load_weights()` 和 Ascend subclass 完成 runtime
mapping。若 Ascend 启用了 QuaRot，`fc` 应发送训练空间的未旋转权重，由 loader
执行 `process_weight()`，VIME 不应提前旋转。

confidence head 当前在 Ascend 中使用 FP32 `ReplicatedLinear`。Exporter 应保留
confidence 参数的 FP32 serving dtype，其他模型参数根据 checkpoint/runtime 要求
使用 BF16 或配置 dtype。

### 13.2 DeepSeek embedded exporter

DeepSeek exporter 输出 loader 接受的 `mtp.{stage}.*` names，而不是 runtime
`model.layers.*` names。需要显式维护：

- Draft stage 到 runtime layer index 的映射；
- `main_proj/main_norm` 位于首 stage；
- `norm/markov_head` 位于末 stage；
- embedding/head 的特殊名字；
- expert、gate/up/down 和 scale 的映射；
- TP/EP 前保持 checkpoint full tensor layout。

该 mapper 必须以 vLLM/vLLM Ascend loader parity test 约束。

## 14. vLLM Ascend P0 改动要求

虽然本文主要设计 VIME，以下推理侧能力是在线 Draft 发布的必要条件。

### 14.1 Draft update target

在 `NPUWorker` 增加与 upstream `GPUWorker` 等价的能力：

```text
get_draft_model()
_set_draft_weight_update_target()
start_draft_weight_update()
```

Draft model 获取规则：

- model runner v1：`model_runner.drafter.model`；
- model runner v2：`model_runner.speculator.model`。

调用 `WeightTransferEngine.set_weight_update_target(draft_model,
draft_model_config)`，并在成功、异常、finish 三条路径都执行 reset。

### 14.2 Loader session 化

将“收到一个 bucket”和“完整 Draft reload session 结束”区分开：

- bucket 中没有 confidence tensor 时不得关闭 confidence head；
- 只有初次 checkpoint load 才能根据完整 manifest 判断 confidence 是否缺失；
- fused KV buffer 只在 session finish 后重建；
- session finish 校验所有 manifest 中声明的参数均已加载；
- 未更新的层恢复旧 kernel tensor；
- post-load hook 在 layerwise finalize 后执行。

建议为 Draft model 增加统一 hook：

```text
begin_weight_update(manifest)
load_weights(bucket)
finish_weight_update() -> loaded_manifest/checksum
abort_weight_update()
```

WeightTransferEngine 仍负责 transport，model hook 负责算法相关的完整性和 derived
buffer 重建。

### 14.3 HCCL/IPC 返回值

当前 update API 主要是命令式调用。为了可靠提交 pair，建议 finish 返回：

```text
target = draft
architecture_fingerprint
loaded_tensor_count
loaded_numel
missing_names
unexpected_names
snapshot_checksum
```

VIME 只有在所有 engine 返回一致结果后才提交 `draft_serve_version`。

## 15. 发布协议

初期沿用一次 pause、两个 update session：

```text
pause_generation(all engines)
flush_cache(all engines)

start_weight_update(target)
send Target Tn+1
finish_weight_update(target)

start_draft_weight_update()
send Draft Dn+1 serving snapshot
finish_weight_update(draft)

validate pair/checksum(all engines)
continue_generation(all engines)
```

Target 和 Draft session 不能并行，因为 weight transfer engine 只有一个 active
target。两次 session 共享同一次 pause。

中期建议 upstream 增加 pair transaction：

```text
begin_model_pair_update(pair_id)
update_target(...)
update_draft(...)
commit_model_pair_update(pair_id)
```

transaction 不要求复制两套完整权重，但必须保证 generation 不会观察到中间状态。

## 16. 发布门控

不能以“训练 loss 有限”作为唯一发布条件。每次训练触发保留 current-version
validation features，计算：

- total validation loss；
- CE、TV、confidence loss；
- 每位置 greedy accuracy；
- analytical mean acceptance length；
- confidence MAE/ECE；
- cumulative confidence bias；
- 与当前 serving Draft 的 paired comparison。

默认发布条件建议为：

```text
optimizer_steps > 0
all parameters finite
validation TV 不显著回退
analytical accept length 不低于当前 serving Draft - tolerance
confidence ECE < threshold
architecture/teacher/feature version 完全一致
serving export dry-run 通过
```

head-only 阶段还应检查后部位置 acceptance 是否改善；只看 position 0 会掩盖
Markov head 没有学习到 intra-block dependency 的问题。

## 17. 在线观测

### 17.1 Feature

```text
draft/feature_candidate_samples
draft/feature_collected_samples
draft/feature_collected_tokens
draft/valid_anchors
draft/feature_payload_mib
draft/feature_rejected_short_block
draft/feature_rejected_version
draft/feature_rejected_fingerprint
```

### 17.2 Training

```text
draft/loss
draft/ce_loss
draft/tv_loss
draft/confidence_loss
draft/accept_rate
draft/analytical_accept_length
draft/confidence_ece
draft/confidence_cumprod_bias
draft/grad_norm
draft/optimizer_steps
draft/teacher_lag
```

### 17.3 Serving

```text
spec/draft_tokens
spec/accepted_tokens
spec/acceptance_rate_per_position
spec/mean_accepted_length
spec/verify_length
spec/draft_latency_ms
spec/verify_latency_ms
spec/end_to_end_tokens_per_second
spec/draft_version
spec/target_draft_pair_id
```

最终验收以真实 mean accepted length 和端到端 throughput 为准，训练 loss 只是中间
指标。

## 18. 参数设计

保留通用调度参数，将算法参数放入结构化配置，避免继续扩张平铺 CLI。

建议通用参数：

```text
--enable-external-draft-training
--draft-algorithm dspark
--draft-model-path
--draft-checkpoint-path
--draft-placement dedicated|actor_rank0
--draft-feature-timing post_update|pre_update
--draft-collect-interval
--draft-train-interval
--draft-publish-interval
--draft-save-interval
--draft-algorithm-config <json/yaml>
```

DSpark algorithm config 示例字段：

```text
block_size
alignment_mode                  # 默认从 checkpoint 读取，只允许校验
max_anchors_per_window
feature_window_tokens
loss_fn                         # {ce: 0.1, tv: 0.9}
confidence_head_alpha
position_loss_weight
head_only_steps
backbone_learning_rate
markov_learning_rate
confidence_learning_rate
max_grad_norm
publish_validation_tolerance
confidence_ece_threshold
```

结构字段如 block size、alignment、layer ids、markov rank 必须以 checkpoint 为
source of truth。配置只能断言相等，不能覆盖后继续运行。

## 19. 代码改动边界

### 19.1 VIME

| 文件/模块 | 改动 |
| --- | --- |
| `train.py` | 支持 post-update feature/train/publish 状态机 |
| `speculative_training/config.py` | 注册 DSpark，解析结构化算法配置 |
| `feature_schema.py` | 增加 schema V2、document id、fingerprint |
| `feature_collector.py` | 按 backend feature spec 采集，预先做样本预算 |
| `draft_trainer.py` | 算法无关化、staged optimizer、validation gate |
| `backends/eagle3.py` | 迁移现有逻辑，不改变行为 |
| `backends/dspark.py` | alignment、collate、loss、metrics |
| `factories/speculators_dspark.py` | 构造可训练 DSpark 模型 |
| `exporters/qwen_dspark.py` | Speculators/HF serving export |
| `exporters/deepseek_dspark.py` | `mtp.*` serving export |
| `update_weight_from_distributed.py` | manifest、checksum、pair publish |

### 19.2 vLLM Ascend

| 文件/模块 | 改动 |
| --- | --- |
| `worker/worker.py` | Draft weight update target 和异常 reset |
| `distributed/weight_transfer/*` | Draft session manifest/finish result |
| `models/qwen3_dspark.py` | confidence loader 支持分桶 reload |
| Qwen DFlash/DSpark model | fused buffer 改为 session finish 后重建 |
| DeepSeek DSpark model | 增量 reload manifest 和 mapper parity |

## 20. 测试方案

### 20.1 纯算法单测

- sample-from-anchor token alignment；
- bonus-anchor token alignment；
- Markov previous-token teacher forcing；
- prompt/response loss mask shift；
- document boundary；
- reduced vocab mapping；
- analytical confidence label；
- position decay；
- 零有效 token 的 graph-connected loss。

### 20.2 Speculators parity

固定随机 seed、同一 checkpoint 和同一输入，比较：

- backbone hidden；
- Markov-corrected logits；
- CE/TV/confidence loss；
- 每位置 metrics；
- 一步 backward gradient；
- 一步 optimizer 后的参数。

### 20.3 Export/load parity

- training state -> Qwen serving names；
- training state -> DeepSeek `mtp.*` names；
- vLLM loader loaded-name 集合完整；
- TP=1/2/4 shard shape；
- confidence FP32 dtype；
- QuaRot fc 只旋转一次；
- 分多个 bucket 与单 bucket 的最终参数一致；
- reload 后 fused KV buffer 引用新权重。

### 20.4 NPU 数值测试

- CPU/CUDA reference 与 NPU eager forward；
- BF16 loss tolerance；
- one-step optimizer loss 下降；
- head-only 和 backbone stage 的 grad allowlist；
- Draft DDP token-count normalization。

### 20.5 热更新测试

1. 启动固定 DSpark rollout engine；
2. 记录 Draft 参数 checksum 和固定 prompt proposal；
3. 修改 Markov/confidence 参数；
4. 通过 `start_draft_weight_update` 分桶发布；
5. 验证 Target checksum 不变；
6. 验证 Draft checksum 改变；
7. 验证所有 engine pair id 一致；
8. 验证 generation 恢复且无 session 残留。

### 20.6 端到端对比

同一 Target checkpoint 和请求集比较：

1. 关闭 speculative decoding；
2. 固定基础 DSpark；
3. 只增量训练 Markov/confidence head；
4. 增量训练完整 DSpark backbone。

必须验证：

- 最终输出与无投机 baseline 保持 lossless；
- mean accepted length 改善；
- confidence calibration 没有退化；
- 端到端 tokens/s 改善；
- 训练和发布开销没有抵消 rollout 收益。

## 21. 分阶段落地

### P0：打通可靠 Draft reload

- vLLM Ascend `start_draft_weight_update`；
- Draft target reset；
- Qwen confidence 分桶 reload；
- fused KV buffer post-update hook；
- loaded manifest/checksum；
- 手工参数变更热更新 smoke test。

P0 未通过前，不开始在线 Draft optimizer 集成。

### P1：Collect-only

- schema V2；
- 从 checkpoint 解析 alignment 和 layer ids；
- post-update feature capture；
- feature 落盘；
- 使用离线 reference trainer 验证 loss 和 alignment。

### P2：Head-only 增量训练

- 训练 Markov/confidence head；
- current-version queue；
- trainer checkpoint；
- serving exporter；
- 发布前 validation gate；
- 单 NPU E2E。

### P3：Backbone 增量训练

- 解冻 fc/layers/norm；
- anchor/vocab microbatch；
- 独立 Draft worker/DDP；
- 多 NPU 数值和性能验证。

### P4：生产强化

- pair transaction；
- canary/rollback；
- stale Draft policy；
- publish threshold 自适应；
- feature 和 optimizer offload；
- 长期 checkpoint 清理。

## 22. 推荐初始配置

第一条可验证链路建议采用：

```text
checkpoint_format       = speculators_external
feature_timing          = post_update
placement               = dedicated, 1 NPU
training_stage          = head_only
block/alignment         = checkpoint value
window_tokens           = 512
anchors_per_window      = 32
samples_per_actor_dp    = 2
train_steps_per_trigger = 10
publish_interval        = 1
loss                    = CE 0.1 + TV 0.9 + confidence 1.0
publish                 = full serving-owned state
```

完成 P0-P2、确认在线 accepted length 改善后，再把窗口、anchor、训练步数和可训练
backbone 范围扩大。不要一开始就在 Actor rank 0 上训练完整 3 到 5 层 DSpark，
否则显存和串行训练时间会同时干扰 PPO 基线定位。

## 23. 验收标准

功能验收：

- checkpoint/profile/feature/teacher 版本严格一致；
- DSpark optimizer 可以连续保存和恢复；
- serving exporter 与当前 vLLM loader 完全匹配；
- Draft 热更新不修改 Target 参数；
- 所有 rollout engine 提交相同 pair id；
- 任意失败不会让 generation 运行在部分更新状态。

质量验收：

- validation TV loss 相比基础 Draft 下降；
- 后部位置 conditional acceptance 改善；
- confidence ECE 在门限内；
- 真实 mean accepted length 相比固定 Draft 改善；
- speculative on/off 输出保持无损一致。

性能验收：

- 在线训练与发布摊销后，rollout 端到端吞吐高于固定 Draft 或无投机基线；
- generation pause、feature transfer 和 Draft train 时间均可单独观测；
- 在目标并发度下，dynamic confidence verification 不降低总体吞吐。

## 24. 参考资料

- [DSpark 论文](https://arxiv.org/abs/2607.05147)
- [vLLM Speculators](https://github.com/vllm-project/speculators)
- [Speculators DSpark 文档](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/)
- VIME `docs/zh/advanced/external-draft-training-design.md`
- vLLM `vllm/model_executor/models/qwen3_dspark.py`
- vLLM `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`
- vLLM `vllm/v1/worker/gpu_worker.py`
- vLLM Ascend `vllm_ascend/models/qwen3_dspark.py`
- vLLM Ascend `vllm_ascend/models/deepseek_v4_dspark.py`
- vLLM Ascend `vllm_ascend/worker/worker.py`
