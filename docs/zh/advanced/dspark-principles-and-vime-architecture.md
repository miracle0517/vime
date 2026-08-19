# DSpark 原理、训练与 VIME 实现架构

> 文档状态：实现说明与架构基线<br>
> 最后核对：2026-08-19<br>
> 论文：[DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)<br>
> VIME 基线：<code>9bc3a9e0</code>、<code>1cb41579</code> 及其后的 DSpark-only 精简补丁<br>
> Speculators：<code>af3f1795495b5393c7c9aed3f1d55a42d47878ae</code><br>
> 本地推理源码：<code>vllm@568afb3a1380</code>、<code>vllm-ascend@30b44103d376</code>

本文分成两个层次：

1. 解释 DSpark 为什么能在一次并行 Draft backbone 后恢复块内 token 依赖，以及 confidence scheduling 为什么必须同时考虑接受概率和硬件负载。
2. 说明 VIME 如何在 RL 生命周期内采集当前 Actor 特征、单独训练 DSpark、冻结可发布 candidate，并在同一次 generation pause 中热更新 Target 与 serving Draft。

更偏工程契约、参数校验和故障边界的逐项说明见
[VIME DSpark 在线增量训练设计](dspark-incremental-training-design.md)。本文重点回答“算法为什么成立”和“各模块如何连起来”。

本文及对应精简补丁的功能范围只包含 DSpark。既有 EAGLE3 model、loss、feature schema、保存和发布语义保持不变；
共享代码中的新增严格校验、immutable candidate 与 legacy NPUWorker compatibility hook 都按
<code>algorithm=dspark</code> 收口。

## 1. 先给结论

- DSpark 不是另一种 RL 算法。Target/Actor 继续使用 PPO、GRPO 等 RL loss；Draft 使用冻结 Target 提供的 token、hidden state 和分布做监督蒸馏。
- DSpark 由三部分组成：DFlash 风格并行 backbone、低秩 Markov 顺序头、confidence head。前者用一次重计算出整块基础 logits，Markov 头用很轻的串行计算恢复块内依赖，confidence head估计每个位置被 Target 接受的条件概率。
- 标准 rejection sampling 才负责“输出分布等价于 Target”。Draft 越差通常只会降低接受率和吞吐；synthetic、typical 或近似接受路径不具有相同保证。
- VIME 的独立 Draft 训练唯一开关是 <code>--enable-external-draft-training</code>。<code>--draft-save-hf</code> 只控制 HF/Speculators 目录导出；不设置它时 Draft 仍可训练和热发布。
- 当前 Draft trainer 与 Actor global rank 0 共置，具有独立模型和 AdamW optimizer，但在 Actor phase 后串行执行；没有独立 Ray worker、独立 placement bundle 或 Actor/Draft 并行 backward。
- 论文中的完整 hardware-aware scheduler 与当前运行时能力不能画等号。当前 Ascend MRV1 有 confidence 驱动的预算分配实现，但它是阈值、周期预算和全局 top-K 近似；本地 vLLM GPU 与 Ascend MRV2 路径没有等价的完整论文调度闭环。
- “接受率 60%”本身不能保证加速。投机 1 个 token 时，每轮期望产出是 <code>1 + 0.6 = 1.6</code> 个 token；只要 Draft、2-token Target verify、采样和调度总耗时达到普通 decode step 的 1.6 倍，就已经不再有收益。

## 2. 名词与边界

| 名词 | 本文含义 |
| --- | --- |
| Target / verifier | 定义最终采样分布的大模型；VIME 中是持续接受 RL 更新的 Actor |
| Draft / speculator | 提议候选 token 的小模型；候选必须由 Target 验证 |
| anchor / bonus token | 上一验证轮由 Target 确定的最后一个 token，是下一 Draft block 的起点 |
| block size $\gamma$ | DSpark 一次最多建模的 Draft block 长度 |
| proposal length $K$ | 运行时本轮实际提出并准备验证的 Draft token 数，$K\leq\gamma$ |
| acceptance rate | 被接受 Draft token 数除以提出 Draft token 数 |
| accepted length | 每轮最终前进的 token 数；论文指标默认包含 Target 产生的 bonus/correction token |
| candidate | 最近一次至少完成一个有效 optimizer step 后冻结的、可保存或发布的完整 Draft 版本 |
| live Draft | Actor rank 0 中继续接收 Target LM Head 同步、用于下一轮训练的模型实例 |
| serving Draft | vLLM/vLLM-Ascend rollout engine 内只做推理的 Draft 副本 |

本文的“独立训练”是模型与 optimizer 意义上的独立，并不等于资源或时间上的并行。

## 3. 从标准投机解码开始

### 3.1 一轮生成

设 Draft 在第 $k$ 个位置的分布为 $p_k^d$，Target 分布为 $p_k^t$，Draft 采样出 $x_k$。标准 speculative rejection sampling 以

$$
\alpha_k(x_k)=\min\left(1,\frac{p_k^t(x_k)}{p_k^d(x_k)}\right)
$$

接受该 token。验证从左到右进行；第一个被拒绝的位置之后的 Draft suffix 全部作废。发生拒绝时，Target 从归一化残差分布

$$
p_k^{\mathrm{res}}(v)
=
\frac{[p_k^t(v)-p_k^d(v)]_+}
{\sum_u[p_k^t(u)-p_k^d(u)]_+}
$$

采样 correction token。若整块都通过，Target 还能给出一个 bonus token。正确实现时，这一接受/修正过程恢复 Target 的精确采样分布；Draft 不是 PPO 的行为策略，也不需要单独计算 RL importance ratio。

~~~mermaid
flowchart LR
    C["已确认上下文"] --> D["Draft 提议 x1...xK"]
    D --> V["Target 一次并行计算 K+1 个位置"]
    V --> A{"从左到右接受？"}
    A -->|"连续接受"| P["输出 accepted prefix"]
    A -->|"第一个拒绝"| R["Target residual correction"]
    P --> B["Target bonus / correction token"]
    R --> B
    B --> C
~~~

### 3.2 吞吐的真正目标

论文把单 token 平均时延写成

$$
L=\frac{T_{\mathrm{draft}}+T_{\mathrm{verify}}}{\tau},
$$

其中 $\tau$ 是一轮前进的期望 token 数。因此加速只有三个方向：

- 降低 Draft 时间 $T_{\mathrm{draft}}$；
- 提高被 Target 接受的前缀长度 $\tau$；
- 减少无价值候选占用的 Target verify 计算。

自回归 Draft 通常提高第二项，却让 Draft 时间近似随 $K$ 线性增加。纯并行 Draft 让 Draft 时间接近常数，却因为块内位置互相独立而出现 suffix acceptance decay。DSpark 正是为这个矛盾设计。

## 4. DSpark 的核心原理

### 4.1 论文原始总览图

[打开论文 Figure 1 原图](https://arxiv.org/html/2607.05147v1/model_arch.svg)

![DSpark 论文 Figure 1：架构与解码周期](https://arxiv.org/html/2607.05147v1/model_arch.svg)

图中重计算分为两段：

1. 重型并行 backbone 一次产生整块的 hidden states 和基础 logits。
2. 轻型顺序头根据已经采样出的前一个 token，逐位置修正基础 logits。

confidence head 同时给出每个位置的条件接受概率；调度器只把有正收益的前缀交给 Target 验证。

> 图片来源：Cheng et al., DSpark Figure 1，arXiv:2607.05147v1，论文以
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 发布。本文直接引用 arXiv 官方 SVG，
> 不修改原图；若文档渲染器拦截远程 SVG，可点击图上方链接。

### 4.2 DFlash 风格的并行 backbone

Target 从多个中间层抽取上下文特征：

$$
H_{\mathrm{ctx}}
=
\operatorname{RMSNorm}\left(
W_c[H^{(l_1)};\ldots;H^{(l_m)}]
\right).
$$

Draft 每层把这些上下文特征注入 KV：

$$
K_i=[W_i^K H_{\mathrm{ctx}};W_i^K H_d],
\qquad
V_i=[W_i^V H_{\mathrm{ctx}};W_i^V H_d].
$$

输入由 anchor token 与 mask token 组成。论文 DSpark 的默认
<code>sample_from_anchor=true</code> 使用 <code>anchor + (γ-1) masks</code> 共 $\gamma$ 个 query，
并让 anchor slot 本身预测第一个 Draft token；原始 DFlash 则可理解为
<code>anchor + γ masks</code> 后只从 mask positions 预测。块内位置使用非因果或按层配置的 attention，一次
backbone forward 就得到 $h_1,\ldots,h_\gamma$ 和基础 logits
$U_1,\ldots,U_\gamma$。重型计算不再随每个 Draft token 重跑，因此可以使用比浅层自回归 Draft 更深的 backbone。

纯并行预测的问题是：每个位置只对“可能的前缀”做边缘化，不能看到本轮真实采样出的前一个 token。例如两个合理模式分别是 “of course” 和 “no problem”，独立位置可能拼出 “of problem”。越靠后的 token 受这种多模态碰撞影响越明显。

### 4.3 低秩 Markov head：少量自回归恢复块内依赖

DSpark 在基础 logits 上叠加前缀相关偏置：

$$
p_k(v\mid x_0,x_{<k})
=
\operatorname{softmax}\left(U_k(v)+B_k(x_0,x_{<k},v)\right).
$$

默认 vanilla Markov head 只依赖紧邻的前一个 token：

$$
B(x_{k-1},\cdot)
=
W_1[x_{k-1}]W_2,
\qquad
W_1\in\mathbb{R}^{V\times r},
\quad
W_2\in\mathbb{R}^{r\times V}.
$$

完整 $V\times V$ 转移矩阵成本不可接受，所以用秩 $r$ 分解；论文和 Speculators 常用 $r=256$。每个位置只需要一次 embedding lookup、低秩投影和采样，远小于重新运行 Transformer backbone。

~~~mermaid
flowchart TB
    H["一次并行 backbone<br/>h1...hK, U1...UK"] --> S1
    X0["anchor x0"] --> M1["W1[x0] · W2"]
    M1 --> S1["softmax(U1 + bias)"]
    S1 --> X1["sample x1"]
    X1 --> M2["W1[x1] · W2"]
    H --> S2["softmax(U2 + bias)"]
    M2 --> S2
    S2 --> X2["sample x2"]
    X2 --> DOT["... 直到 xK"]
~~~

论文还讨论 gated 和 RNN 变体。当前 VIME 的可部署 Qwen3 契约只允许
<code>markov_head_type=vanilla</code>；这既缩小兼容面，也与论文默认选择一致。

### 4.4 Confidence head：预测“此前都通过时，本位置也通过”的概率

confidence head 输出

$$
c_k
=
\sigma\left(w^\top[h_k;W_1[x_{k-1}]]\right),
$$

其监督目标不是 token 是否恰好 argmax 相同，而是标准 rejection sampling 的解析接受概率：

$$
c_k^*
=
1-\frac{1}{2}\lVert p_k^d-p_k^t\rVert_1
=
\sum_v\min(p_k^d(v),p_k^t(v)).
$$

这里 $c_k$ 是条件概率。第 $j$ 个位置所在 Draft prefix 全部存活的概率为

$$
a_{r,j}=\prod_{i\leq j}c_{r,i}.
$$

因此 confidence 的绝对数值是否校准很重要，不能只看排序。论文使用 Sequential Temperature Scaling
对累积生存概率逐位置校准。当前 VIME 训练 confidence head，但没有实现论文的 STS 校准阶段。

### 4.5 Hardware-aware prefix scheduling

一批有 $R$ 个请求，对请求 $r$ 选择验证长度 $\ell_r$。Target 需要处理的 token batch 为

$$
B=\sum_{r=1}^{R}(1+\ell_r),
$$

期望前进 token 数为

$$
\tau
=
\sum_{r=1}^{R}\left(1+\sum_{j=1}^{\ell_r}a_{r,j}\right).
$$

如果预先测得硬件在 batch token 数 $B$ 下的 step/s 曲线 $\operatorname{SPS}(B)$，目标就是

$$
\Theta(\ell_1,\ldots,\ell_R)=\tau\cdot\operatorname{SPS}(B).
$$

论文算法把所有可扩展 prefix 按 $a_{r,j}$ 排序，逐个加入，并在吞吐第一次不再提升时停止。这个 early-stop
不仅是性能启发式，也用于满足 non-anticipating 条件：当前 token 的 admission 决策不能依赖尚未合法生成的未来 token，否则会引入 selection bias。

当前本地实现需要明确区分：

| 路径 | Confidence / 动态长度现状 |
| --- | --- |
| 本地 vLLM GPU | Qwen3 DSpark loader 明确跳过 confidence head；固定长度 proposal/verify |
| vLLM-Ascend MRV1 | 显式启用 <code>additional_config.dynamic_spec_config.method=dspark</code> 后，加载 FP32 confidence head；周期更新共享预算，再按累积生存概率做 top-K 分配 |
| vLLM-Ascend MRV2 | 继承上游 DSpark speculator；当前源码未看到 MRV1 对等的 confidence-budget 调用 |
| DSpark 论文 | STS 校准 + profiled SPS 曲线 + hardware-aware prefix scheduler |

因此 MRV1 当前实现可称为“confidence-driven 动态 verify-length”，但不能称为论文调度器的逐项复现。

### 4.6 训练目标

Target、共享 embedding 与两颗 LM Head 冻结，更新 Draft backbone、Markov head 和 confidence head。每个位置使用

$$
w_k=\exp(-(k-1)/\gamma)
$$

衰减权重，优先优化更可能进入最终接受前缀的前部位置。三项 loss 为

$$
\mathcal{L}_{\mathrm{ce}}
=-\sum_{k=1}^{\gamma}w_k\log p_k^d(x_k^*),
$$

$$
\mathcal{L}_{\mathrm{tv}}
=\sum_{k=1}^{\gamma}w_k\lVert p_k^d-p_k^t\rVert_1,
$$

$$
\mathcal{L}_{\mathrm{conf}}
=-\sum_{k=1}^{\gamma}w_k
\left[c_k^*\log c_k+(1-c_k^*)\log(1-c_k)\right].
$$

总目标为

$$
\mathcal{L}
=
\alpha_{\mathrm{ce}}\mathcal{L}_{\mathrm{ce}}
+\alpha_{\mathrm{tv}}\mathcal{L}_{\mathrm{tv}}
+\alpha_{\mathrm{conf}}\mathcal{L}_{\mathrm{conf}}.
$$

论文默认 $\alpha_{\mathrm{ce}}=0.1$、$\alpha_{\mathrm{tv}}=0.9$、
$\alpha_{\mathrm{conf}}=1.0$。VIME 默认 DSpark loss 配置同样使用
<code>{"ce":0.1,"tv":0.9}</code>，confidence 权重由
<code>--draft-dspark-confidence-head-alpha</code> 控制。

论文公式与固定 <code>speculators@af3f179...</code> 的数值实现还有四个必须记录的差异：

- Speculators 的 TV loss 是标准 total variation
  $1-\sum_v\min(p^d(v),p^t(v))=\frac{1}{2}\lVert p^d-p^t\rVert_1$；
  论文 Equation 10 直接写 L1。因此相同的数值系数不代表逐项 loss scale 完全相同。
- Speculators 的 CE hard label 来自 <code>argmax(target_logits)</code>，而论文公式把 $x_k^*$ 描述为 ground-truth token。
- VIME 默认 <code>--draft-dspark-decay-gamma=4.0</code>，位置权重实现为
  $\exp(-\text{zero-based-position}/4)$；这个 decay 超参不自动等于 checkpoint block size $\gamma$。
- 固定 revision 在调用方不传 loss config 时会 fallback 到 KL；VIME 总是显式传 CE+TV，所以在线路径不使用该 fallback。

### 4.7 论文实验图如何解读

[打开 Figure 2 原图](https://arxiv.org/html/2607.05147v1/position_cond_accept.svg)

![论文 Figure 2：各位置条件接受率](https://arxiv.org/html/2607.05147v1/position_cond_accept.svg)

Figure 2 展示：DFlash 的后部位置条件接受率明显衰减；DSpark 通过很轻的块内顺序依赖减缓这个趋势。

[打开 Figure 3 原图](https://arxiv.org/html/2607.05147v1/layer_comparison.svg)

![论文 Figure 3：Draft depth 消融](https://arxiv.org/html/2607.05147v1/layer_comparison.svg)

Figure 3 展示论文条件下增加 DSpark backbone depth 的收益，并报告 2-layer DSpark 已超过 5-layer DFlash。
这说明轻量顺序依赖的参数效率，但不是“层数越多端到端一定越快”的结论。

[打开 Figure 4 原图](https://arxiv.org/html/2607.05147v1/block_size_comparison.svg)

![论文 Figure 4：proposal length 与延迟](https://arxiv.org/html/2607.05147v1/block_size_comparison.svg)

Figure 4 的结论有严格实验条件：论文使用特定模型、block、draft depth、batch 和硬件。它证明 Markov 顺序头在该条件下相对 DFlash
开销很小，不证明任意系统中 DSpark 相对纯 Target 都会加速，尤其不能外推到 <code>K=1</code> 的高并发场景。

[打开 Figure 5 原图](https://arxiv.org/html/2607.05147v1/confidence_threshold_sweep.svg)

![论文 Figure 5：confidence threshold sweep](https://arxiv.org/html/2607.05147v1/confidence_threshold_sweep.svg)

[打开 Figure 6 原图](https://arxiv.org/html/2607.05147v1/calibration_alpaca.svg)

![论文 Figure 6：confidence reliability diagram](https://arxiv.org/html/2607.05147v1/calibration_alpaca.svg)

Figure 5 说明提高阈值会丢弃更多低价值 suffix，并提高“已提交验证 token”的接受率；Figure 6 说明原始 confidence
仍可能过度自信，必须同时看 ECE/Brier 等校准指标。Figure 2–6 与 Figure 1 相同，均直接引用论文
CC BY 4.0 官方 SVG。

## 5. VIME 的总体实现架构

### 5.1 组件图

~~~mermaid
flowchart LR
    subgraph Rollout["rollout 服务（vLLM / vLLM-Ascend）"]
        ST["serving Target T"]
        SD["serving DSpark Draft D"]
        RS["standard rejection sampler"]
        ST --> RS
        SD --> RS
    end

    subgraph Actor["Megatron Actor ranks"]
        AF["Actor pre-update forward"]
        RL["RL loss + Actor optimizer"]
        HOOK["aux/final hidden hooks"]
        HEAD["Target LM Head snapshot"]
        AF --> HOOK
        AF --> HEAD
        AF --> RL
    end

    subgraph Rank0["Actor global rank 0（同卡、串行）"]
        Q["按 Target version 隔离的 feature queue"]
        DT["独立 DSpark model + AdamW"]
        C["immutable candidate metadata<br/>+ frozen CPU LM Head"]
        Q --> DT --> C
    end

    subgraph Publish["保存与发布控制面"]
        CKPT["draft_latest.pt"]
        HF["config.json + model.safetensors"]
        SNAP["完整 CPU serving snapshot"]
        XFER["NCCL/HCCL packed transfer"]
    end

    Rollout -->|"responses"| AF
    HOOK --> Q
    HEAD --> Q
    C --> CKPT
    C --> HF
    C --> SNAP --> XFER
    RL -->|"新 Target weights"| XFER
    XFER -->|"pause 内顺序更新"| ST
    XFER -->|"切换 transfer target"| SD
~~~

### 5.2 所有权与资源

| 对象 | 所有者 | 更新方式 | 是否参与 RL gradient |
| --- | --- | --- | --- |
| Megatron Actor/Target | 所有 Actor ranks | RL optimizer | 是 |
| feature hooks | 每个 Actor rank | 只读捕获 | 否 |
| DSpark 训练副本 | Actor global rank 0 | 独立 AdamW | 否 |
| Draft queue/scheduler | Actor global rank 0 | Python 控制状态 | 否 |
| candidate LM Head | Actor rank 0 CPU | 成功 Draft step 后复制冻结 | 否 |
| serving Draft | 每个 rollout engine | 热加载 candidate snapshot | 否 |
| serving Target | 每个 rollout engine | Actor 权重发布 | 否 |

当前没有 dedicated Draft Ray actor。<code>--draft-num-nodes</code> 与
<code>--draft-num-gpus-per-node</code> 是废弃兼容参数，不会新增资源。Actor rank 0 在完成本轮 Actor phase 后执行 Draft
backward，所以端到端 iteration time 会包含两段串行训练。

## 6. 一轮 RL + DSpark 的精确时序

~~~mermaid
sequenceDiagram
    participant R as rollout: Target Tn / Draft Dp
    participant A as Megatron Actor
    participant F as Feature Collector
    participant D as DSpark Trainer @ Actor rank0
    participant S as Save / Snapshot
    participant U as Weight Updater

    R->>A: rollout samples + speculative counters
    A->>A: pre-update forward
    A->>F: capture aux hidden, final hidden, token/mask
    A->>F: export LM Head(Tn)
    A->>A: RL optimizer: Tn -> Tn+1
    F->>D: payloads tagged target_version=Tn
    D->>D: sync frozen training heads to LM Head(Tn)
    D->>D: version-matched queue + native DSpark loss
    D->>D: successful step -> candidate Dp+1, teacher=Tn
    opt publish interval命中
        D->>S: materialize full CPU serving snapshot
    end
    opt save interval命中
        D->>S: draft_latest.pt
        D->>S: optional HF two-file export
    end
    A->>U: Target Tn+1 weights
    S->>U: staged Draft candidate Dp+1
    U->>R: pause + flush
    U->>R: publish Target Tn+1
    U->>R: retarget transfer engine and publish Draft Dp+1
    U->>R: reset target + resume
~~~

这里有三个容易混淆的版本：

- <code>target_weight_version</code>：采集特征和同步 teacher LM Head 时的 Target 版本。
- <code>draft_version</code>：至少完成一个有效 Draft optimizer step 才递增。
- <code>last_published_draft_version</code>：driver 已确认 rollout engines 更新完成的版本。

同一 rollout 中训练并发布时，serving Target 通常是 $T_{n+1}$，Draft 的 teacher 是 $T_n$，形成最小一版本 lag。
如果 collect/train/publish interval 错开，pending candidate 可能更旧，gap 可以大于 1；metadata 保留真实
<code>trained_against_target_version</code>，但当前没有 gap 自动门控。

### 6.1 为什么需要 immutable candidate

每次 collect 会把 live Draft 的 <code>lm_head</code> 与 <code>verifier_lm_head</code> 同步到最新 Target。
若把 live state 直接留作延迟发布：

1. backbone 可能是上一次训练成功后的版本；
2. LM Head 却可能已经来自下一 Target；
3. metadata 还可能错误声称两者是同一个 teacher。

当前 DSpark 路径在成功 optimizer step 后冻结：

- <code>candidate_draft_version</code>；
- <code>candidate_target_weight_version</code>；
- 一份 CPU <code>lm_head.weight</code>。

发布、native checkpoint 和 HF export 在物化完整 state 时用这份 frozen head 同时覆盖 Draft 的两颗训练头。
driver 再把 snapshot envelope 的 Draft/Target 版本与最后一次成功训练结果做双重校验。这样后续 collect、零步训练或 publish
cadence 错开不会静默污染 candidate。

## 7. 特征采集与训练数据契约

### 7.1 从 Actor forward 捕获什么

<code>DraftFeatureCollector</code> 在 Actor 的 pre-update forward 注册：

- 指定 Target 层的 forward hooks；
- output layer 的 pre-hook，用于捕获进入 Target LM Head 前的 final hidden；
- token、loss mask、position、prompt/response 长度；
- 当前 <code>target_weight_version</code>。

多个 auxiliary hidden 按最后一维拼接，final hidden 单独保存。使用 sequence parallel 时，先在 TP group
内 all-gather；每个 DP replica 的 TP rank 0 才导出 CPU feature payload，减少重复。Target LM Head
只由 Actor global rank 0 导出一次；group 要求所有 feature manifest 的 Target version 一致，再把唯一 head
交给同一个 rank 0 trainer。

~~~mermaid
flowchart LR
    TOK["tokens + response mask"] --> WIN["anchor-bounded window"]
    L1["Target layer l1"] --> CAT["concat aux hidden"]
    L2["Target layer l2"] --> CAT
    LM["pre LM-head final hidden"] --> PAY["DraftFeatureSample v1"]
    CAT --> PAY
    WIN --> PAY
    VER["target version"] --> PAY
    PAY --> Q["VersionedFeatureQueue"]
~~~

DSpark 的窗口至少需要 <code>block_size + 2</code> 行，response 至少需要
<code>block_size + 1</code> 个 token。窗口从 prompt 末尾附近开始，从而保留第一个 response token 所需的 anchor
上下文；可按 front 或确定性 random offset 截取。

### 7.2 Feature schema v1

| 字段 | 形状/类型 | 含义 |
| --- | --- | --- |
| <code>input_ids</code> | <code>[rows]</code> | 窗口 token |
| <code>loss_mask</code> | <code>[rows]</code> | 有效监督位置 |
| <code>position_ids</code> | <code>[rows]</code> | 原序列位置 |
| <code>hidden_positions</code> | <code>[rows]</code> | hidden 与 token 对齐证明 |
| <code>aux_hidden_states</code> | <code>[rows, m*d]</code> | 多层 Target hidden 拼接 |
| <code>final_hidden_states</code> | <code>[rows, d]</code> | Target LM Head 前 hidden |
| <code>aux_layer_ids</code> | tuple | 特征层身份与顺序 |
| <code>target_weight_version</code> | string | teacher 版本 |
| <code>algorithm</code> | <code>dspark</code> | 算法 discriminator |
| <code>hidden_layout</code> | <code>qwen_dspark_aux_plus_last</code> | DSpark 严格布局 |

queue 从不隐式混合不同 Target 版本。同步新 LM Head 后调用 <code>clear_except(new_version)</code>；
旧 feature 被丢弃，训练只从与当前 teacher head 完全一致的 version bucket 取样。

### 7.3 Pack 多个独立窗口

Speculators 接受单个 packed sequence。VIME 为每个样本构造独立 <code>document_id</code>，拼接 token、aux hidden、
final hidden、position 和 mask。每个 document 尾部最后 <code>block_size</code> 个 loss positions 强制置零，防止
Speculators 的 anchor selector 跨越文档边界。

~~~mermaid
flowchart TB
    A["doc A: ... anchor + future"] --> PA["tail K positions mask=0"]
    B["doc B: ... anchor + future"] --> PB["tail K positions mask=0"]
    PA --> PACK["cat -> [1, total_rows, ...]"]
    PB --> PACK
    PACK --> NATIVE["Speculators DSpark forward"]
~~~

## 8. DSpark 模型构建与配置投影

### 8.1 训练侧与 serving 侧需要不同 schema

Speculators 使用嵌套 <code>transformer_layer_config</code>；本地 vLLM 的
<code>Qwen3DSparkModel</code> 还需要顶层 Qwen 字段。VIME 保留训练 schema，同时生成 direct Qwen3 hybrid：

~~~json
{
  "model_type": "qwen3",
  "architectures": ["Qwen3DSparkModel"],
  "speculators_model_type": "dspark",
  "sample_from_anchor": true,
  "dspark_bonus_anchor": false,
  "target_layer_ids": [1, 9, 17, 25, 33],
  "eagle_aux_hidden_state_layer_ids": [1, 9, 17, 25, 33],
  "dflash_config": {
    "mask_token_id": 151669,
    "target_layer_ids": [1, 9, 17, 25, 33]
  }
}
~~~

关键映射是：

- <code>dspark_bonus_anchor = not sample_from_anchor</code>；
- vLLM 的 <code>target_layer_ids = aux_hidden_state_layer_ids - 1</code>；
- <code>dflash_config.target_layer_ids</code> 决定 context projection 输入宽度；
- <code>sliding_window_non_causal=true</code> 时才写全局 <code>causal=false</code> override。

这不是“从权重猜配置”。VIME 只投影 raw config 已明确给出的字段；缺少结构、aux layer 身份、proposal alignment
或完整权重时 fail fast。

通用 <code>model_type=speculators</code> adapter 在 GPU/MRV2 与 Ascend MRV1 对 anchor 字段的读取规则不同：
一侧使用 <code>dspark_bonus_anchor</code>，另一侧直接使用
<code>sample_from_anchor</code>。VIME 因而拒绝用远程 canonical adapter checkpoint 直接做 online training，
要求先下载到可写本地目录，生成同时包含两组一致字段的 direct Qwen3 hybrid。proposal capacity 也由 VIME
根据 checkpoint <code>block_size</code> 与 <code>sample_from_anchor</code> 自行 fail-fast 校验，而不依赖
不同 runtime revision 的内部默认推导。

### 8.2 当前支持矩阵

| 项目 | 当前范围 |
| --- | --- |
| backbone | dense Qwen3 |
| Target/Draft hidden width | 相同 |
| Markov | vanilla |
| confidence | 可训练；serving 能力依 runner 而异 |
| checkpoint | 独立 Speculators DSpark，必须有明确 config 与权重 |
| reduced vocabulary | checkpoint 内必须有严格成对的 <code>t2d</code>/<code>d2t</code> |
| acceptance | <code>rejection_sample_method=standard</code> |
| PP/CP/VPP | 均为 1 |
| weight update | full + NCCL；Ascend 对应使用 HCCL engine |
| remote checkpoint | 必须已经是完整 direct Qwen3 hybrid |

reduced-vocab 的 <code>d2t</code> 保存的是 offset，不是绝对 Target token id。VIME 验证

$$
\operatorname{arange}(V_d)+d2t
=
\operatorname{nonzero}(t2d),
$$

从而保证训练选择的 Target LM Head rows 与 serving token 映射完全一致。DSpark 禁止仅在 trainer 中用外部 mapping
覆盖 checkpoint，因为 rollout engine 会在第一次热更新前从原 serving checkpoint 加载 mapping。

## 9. Draft 训练器内部

### 9.1 初始化

Actor rank 0 构造 <code>ExternalDraftTrainer(distributed=False)</code>：

1. 用严格 config 建立 Speculators DSpark；
2. 加载 checkpoint 并检查 missing、unexpected、mismatched keys；
3. 从 Target checkpoint 加载 embedding，默认冻结；
4. 验证或解析 draft vocab mapping；
5. 获取 Speculators 原生训练参数；
6. 只把 <code>requires_grad=True</code> 参数交给独立 AdamW 与 scheduler；
7. 创建按 Target version 分桶的有限 FIFO queue；
8. 可选恢复 <code>draft_latest.pt</code>。

<code>lm_head</code> 和 <code>verifier_lm_head</code> 每次从同一 Target LM Head 同步并冻结：

- verifier head把 captured final hidden 投影成 teacher distribution；
- Draft LM Head把 Draft hidden 投影成 draft distribution；
- 两者必须使用相同 vocabulary rows。

VIME 的 strict checkpoint 契约要求独立 DSpark 自带完整 Draft LM Head。虽然本地 vLLM 在某些缺权重场景下可以把
Draft embedding/LM Head alias 到 Target，但该共享所有权不适用于 Draft-only 热更新：更新 Draft 时可能触碰 Target
共享 tensor。因此 VIME 不把这种 fallback 当作可发布 external DSpark 的合法所有权模型。

### 9.2 一次训练 trigger

~~~mermaid
flowchart TD
    G{"有 teacher head<br/>和同版本 features?"}
    G -->|"否"| Z["trained=0，不创建新 candidate"]
    G -->|"是"| B["按 batch_size repeat 取样"]
    B --> C["collate documents + tail masking"]
    C --> F["Speculators native forward"]
    F --> L["CE + TV + confidence loss"]
    L --> N{"loss 与 grad finite?"}
    N -->|"否"| B
    N -->|"是"| O["clip grad -> AdamW -> scheduler"]
    O --> M{"达到 trigger steps<br/>或无更多有效样本"}
    M -->|"继续"| B
    M -->|"至少 1 个成功 step"| V["draft_version += 1<br/>freeze candidate"]
~~~

NPU 上由 VIME 请求 eager loss 实现，避免依赖 CUDA Triton fused kernel。固定的
<code>speculators@af3f179...</code> 会通过兼容参数接收该 hint，并在非 CUDA 环境选择其可用实现；升级
Speculators 后必须重新跑 config、one-step backward 与指标契约测试。

### 9.3 指标

训练结果包含：

- loss、top-1 accuracy、valid tokens；
- grad norm、successful steps、optimizer steps、learning rate；
- queue samples；
- 可用时的 acceptance rate、expected accepted length、confidence loss。

这些是训练侧估计，不应替代 rollout engine 的真实 accepted token counters 和端到端 tokens/s。

## 10. Candidate 保存与导出

| 产物 | 开关 | 内容 | 用途 |
| --- | --- | --- | --- |
| <code>draft_latest.pt</code> | <code>--draft-checkpoint-path</code> | model、optimizer、scheduler、steps、Draft/Target version、rollout、fingerprint | 恢复在线训练 |
| HF/Speculators 目录 | <code>--draft-save-hf</code> | <code>config.json</code>、<code>model.safetensors</code> | serving 或后续离线/在线训练 |
| publish snapshot | publish interval | serving tensors、Draft/teacher version、fingerprint、algorithm | 热更新 rollout engines |

HF 目录使用 staging + 可回滚目录交换；它能处理捕获到的写入/rename 异常，但不是进程崩溃意义上的多文件事务。
checkpoint/export 代表 trainer candidate，不代表 rollout engine 已经发布该版本。

resume 后 trainer 会恢复 candidate，但 driver-side <code>last_train_result</code> 不持久化；当前必须再完成一次成功
Draft 训练，才会把恢复后的/新 candidate staging 到 rollout engine。

## 11. Target + Draft 热更新架构

### 11.1 单次 pause 内顺序发布

~~~mermaid
sequenceDiagram
    participant Driver
    participant Actor0 as Actor PP source
    participant Engine as rollout engine
    participant Transfer as NCCL/HCCL transfer engine

    Driver->>Actor0: stage CPU Draft snapshot
    Driver->>Actor0: update_weights()
    Actor0->>Engine: pause_generation + flush_cache
    Actor0->>Engine: start_weight_update(Target)
    Actor0->>Transfer: send Target buckets
    Actor0->>Engine: finish_weight_update
    Actor0->>Engine: start_draft_weight_update
    Engine->>Transfer: set target = serving Draft
    Actor0->>Transfer: send one packed DSpark snapshot
    Actor0->>Engine: finish_weight_update
    Engine->>Transfer: reset target = serving Target
    Actor0->>Engine: continue_generation
~~~

本地 vLLM route 已包含 <code>/start_draft_weight_update</code>。本地 vLLM-Ascend 的
<code>NPUWorker</code> 尚无原生同名方法，因此 VIME worker extension 在 DSpark 模式下补齐控制面：

1. 检查 transfer engine 的 <code>supports_draft_weight_update</code> 有效值；
2. 检查 <code>set_weight_update_target</code>/<code>reset_weight_update_target</code>；
3. MRV1 优先从 <code>model_runner.drafter</code> 获取 Draft；
4. MRV2 fallback 到公开 <code>get_draft_model()</code>，再 fallback 到 <code>speculator</code>；
5. update/finish/异常路径都恢复 transfer target。

compatibility hook 明确只允许 DSpark，不改变既有 EAGLE3 runtime API。

### 11.2 为什么 DSpark 强制一个 packed snapshot

当前 Ascend confidence loader 在一次 <code>load_weights()</code> 看不到 confidence tensors 时，可能把
<code>enable_confidence_head</code> 永久置为 false。若把 DSpark 分成多个普通 bucket，前一个不含 confidence 的 bucket
就可能破坏状态。

VIME 因此让一个 candidate 的所有 serving tensor：

- 只触发一次 Draft <code>load_weights()</code>；
- 使用一个 packed transfer buffer；
- confidence tensors 保持 FP32，其余浮点权重按 publish dtype；
- 训练专用 verifier head/norm 不发到 serving Draft；
- 非浮点 mapping/buffer 保留原 dtype。

代价是峰值内存。发送端会同时持有 CPU snapshot、设备 tensor 列表和近似
<code>sum(tensor_bytes)</code> 的 packed buffer，接收端也要有对应 buffer。大 Draft 上必须把这项纳入 HBM/host RAM
容量评估；“单次 load 的一致性”不是零成本事务。

## 12. 本地 vLLM / vLLM-Ascend 推理数据流

### 12.1 固定长度路径

~~~mermaid
flowchart LR
    TH["Target aux hidden states"] --> FC["DSpark context projection"]
    AN["anchor + masks"] --> BB["Qwen3 DFlash-style backbone"]
    FC --> BB
    BB --> BL["base logits U1...UK"]
    BL --> MK["Markov sequential sampling"]
    MK --> DT["draft token ids"]
    DT --> EXP["scheduler expands Target query to 1+K"]
    EXP --> TV["Target verify forward"]
    TV --> REJ["standard rejection sampler"]
    REJ --> OUT["accepted prefix + correction/bonus"]
~~~

<code>sample_from_anchor</code> 与 <code>dspark_bonus_anchor</code> 的语义相反。VIME hybrid 同时写两者，
使 GPU/MRV2 与 Ascend MRV1 在 query 长度和第一个 proposal 位置上对齐。

上图强调算子依赖，但一个稳态 speculative round 的真实先后顺序是：Target 先验证上一轮已经生成的 Draft，
rejection sampler 提交最长连续接受前缀以及 correction/bonus；随后，同一次 Target forward 输出的 aux hidden
才被 DSpark 用来生成下一轮 Draft：

~~~mermaid
sequenceDiagram
    participant S as Scheduler
    participant T as Target model
    participant R as Rejection sampler
    participant D as DSpark Draft

    S->>T: context + 上轮 draft[1..K]
    T->>T: 一次 forward 产生 K 个 verifier logits + 1 个 bonus logit
    T->>R: Target logits、draft ids、可选 draft logits
    R->>R: 从左到右接受最长连续前缀
    alt 首个 Draft token 被拒绝
        R-->>S: 已接受前缀 + Target residual correction
    else K 个 Draft token 全部接受
        R-->>S: K 个 Draft token + Target bonus
    end
    T->>D: 本轮 Target aux hidden
    D->>D: context-KV 预填 + 一次并行 backbone
    loop i = 1..K
        D->>D: base logits[i] + Markov bias(previous token)
        D->>D: 生成 draft[i]
    end
    D-->>S: 下一轮 draft[1..K]
~~~

因此性能剖析必须把“本轮 Target verify”和“下一轮 Draft proposal”视为稳态流水线中的相邻阶段，不能把本轮
aux hidden误认为是在同一次 Target verify 之前生成当前待验证 Draft。

运行时 <code>num_speculative_tokens=K</code> 不会因为 checkpoint 的
<code>block_size=7</code> 自动计算完整 7-token Draft；Qwen3 DSpark speculator 按运行时 $K$ 分配和循环。
checkpoint block size 只定义训练/容量上限。

### 12.2 Confidence 路径的实现差异

显式设置 <code>additional_config.dynamic_spec_config.method=dspark</code> 后，Ascend MRV1 会从 Draft backbone
hidden 与 Markov embedding 计算 confidence logits，并：

1. 仍先完整生成配置的 $K$ 个 Draft token；
2. 每隔 <code>budget_update_interval</code> 根据阈值估计共享 budget；
3. 至少为每个请求保留一个 Draft token；
4. 对其余位置的累积 survival probability 做全局 top-K；
5. 产生每请求不同的 verify length。

它没有读取论文的 profiled <code>SPS(B)</code> 表，也没有 STS calibration，因此应单独评估吞吐和分布正确性。
动态长度只减少下一轮提交给 Target verifier 的 token 数，不减少本轮 DSpark backbone/Markov 完整生成 $K$ 个
proposal 的成本。
MRV1 当前还强制 DSpark Draft eager；MRV2 走另一套 graph/speculator 路径，不能把 MRV1 profile 结果直接外推。
若没有显式启用上述 dynamic config，confidence 权重即使已经训练和发布，也不会改变固定
<code>num_speculative_tokens</code> 的验证长度。

采样能力也不同：当前 Ascend MRV1 DSpark proposer 明确拒绝
<code>draft_sample_method=probabilistic</code>，只支持 greedy Draft proposal；本地 upstream GPU speculator
存在 probabilistic/Gumbel 路径。Ascend MRV2 虽继承该 Draft 路径，但当前 NPU 非 greedy rejection sampler
把接受判定使用的随机量固定为 0，因此该组合尚不能证明恢复精确 Target 分布；正确性敏感的 RL rollout
应暂时视为未支持，直到修复并完成 spec on/off 分布一致性测试。这里说的是 Draft proposal 后端能力，
不应与用户对 Target 设置 temperature/top-p 的采样语义混为一谈。

## 13. 训练开关与运行模式

### 13.1 真值表

| enable external training | draft_save_hf | 结果 |
| --- | --- | --- |
| false | 未设置 | 只做 vLLM DSpark 推理；不创建 trainer |
| true | 未设置 | 创建独立 Draft + optimizer；collect/train/publish 正常；不导出 HF |
| false | 已设置 | 参数校验失败，绝不隐式开启训练 |
| true | 已设置 | 在线训练和热发布，并在保存周期导出 HF |

若同时不设置 <code>--draft-checkpoint-path</code> 与 <code>--draft-save-hf</code>，任务期间仍然训练并热更新，
但退出后不留下可恢复的 native checkpoint 或 HF 目录，也不会覆盖原
<code>--draft-model-path</code>。

### 13.2 最小 DSpark 在线训练配置

~~~bash
--enable-external-draft-training \
--draft-algorithm dspark \
--draft-model-path /models/qwen3-dspark \
--draft-target-embedding-path /models/qwen3-target \
--draft-feature-layer-ids 2,10,18,26,34 \
--draft-collect-interval 1 \
--draft-train-interval 1 \
--draft-publish-interval 1 \
--draft-train-steps-per-trigger 10 \
--draft-batch-size-per-gpu 4 \
--draft-dspark-loss-fn '{"ce":0.1,"tv":0.9}' \
--vllm-speculative-config \
'{"method":"dspark","model":"/models/qwen3-dspark","num_speculative_tokens":4,"rejection_sample_method":"standard"}'
~~~

需要恢复训练时加 <code>--draft-checkpoint-path</code>。需要 serving 目录时再加
<code>--draft-save-hf</code>。为保持 Actor/Draft resume rollout 对齐，推荐不显式设置
<code>--draft-save-interval</code>，让它跟随 Actor <code>--save-interval</code>。

## 14. 为什么 K=1、接受率 60% 仍可能慢

### 14.1 严格盈亏公式

设同一活跃请求数 $R$ 下：

- 普通 decode 一轮耗时 $T_0(R)$，产出 $R$ 个 token；
- DSpark $K=1$ 一轮总耗时 $T_s(R)$；
- Draft token 接受概率为 $a$。

拒绝时 Target correction 仍让请求前进 1 token；接受时 Draft token 加 Target bonus，共前进 2 token。因此

$$
\mathbb{E}[\text{tokens/request/round}]=1+a.
$$

吞吐加速比为

$$
S
=
\frac{(1+a)T_0(R)}{T_s(R)}.
$$

令 $C=T_s/T_0$，则 break-even 条件是

$$
C<1+a.
$$

当 $a=0.6$ 时，必须满足 $C<1.6$。若要至少 10% 加速，则

$$
C\leq\frac{1.6}{1.1}\approx1.455.
$$

### 14.2 $T_s$ 包含哪些成本

$$
T_s
=
T_{\mathrm{verify}}(2R)
+T_{\mathrm{draft}}(R,1)
+T_{\mathrm{sampler}}
+T_{\mathrm{hidden}}
+T_{\mathrm{TP/HCCL}}
+T_{\mathrm{scheduler}}
-T_{\mathrm{overlap}}.
$$

- Target verify 的 token batch 从 $R$ 扩成 $2R$；
- 即使只投机一个 token，DSpark 仍需读取 Draft 权重、运行 context projection/backbone/LM Head；
- Target 必须额外输出多层 aux hidden；
- standard rejection sampler 要处理两个分布和 residual/Gumbel 等 vocab 操作；
- TP=4 时 Target 与 Draft 都可能增加 collective；
- Ascend MRV1 DSpark 当前走 eager，若纯 Target 命中 graph，差距会进一步放大；
- 高并发下 Target 已 compute/KV/通信饱和，$T_{\mathrm{verify}}(2R)/T_0(R)$ 可接近 2。

若 verify 单项已经达到 $1.6T_0$，即使 Draft 和 sampler 免费，60% 接受率也不可能加速。高并发极限下若
$T_{\mathrm{verify}}(2R)\approx2T_0$，理论上限只有 $1.6/2=0.8$ 倍，尚未计算其他开销。

### 14.3 正确的 A/B 与指标

1. 禁止 online Draft training 和 weight sync，仅比较纯 rollout。
2. 同一 checkpoint、prompt、seed、temperature/top-p/top-k、输出长度和 TP。
3. sweep 实际活跃请求数 $R\in\{1,2,4,8,16,32,64,128,256\}$。
4. 单测 Target $R$ 与 $2R$ token shape 的 forward，得到
   $r_v=T_{\mathrm{verify}}(2R)/T_0(R)$。
5. 用设备 event 分别测 Draft、Target verify、sampler、hidden/TP communication，不能只看异步 CPU launch 时间。
6. 汇总原始计数，而不是逐样本比率的简单平均：

$$
\text{global accept rate}
=
\frac{\sum \text{spec\_accept\_token\_num}}
{\sum \text{spec\_draft\_token\_num}},
$$

$$
\text{global accepted length}
=
\frac{\sum \text{completion\_token\_num}}
{\sum \text{spec\_verify\_ct}}.
$$

VIME 当前 <code>rollout/spec_accept_rate</code> 和 <code>rollout/spec_accept_length</code>
是逐样本比率平均；严谨诊断应从 debug rollout data 或 engine metrics 汇总 raw counters。

## 15. 正确性、故障和版本边界

### 15.1 必须 fail closed 的条件

- speculative method 不是 <code>dspark</code> 或 model path 不一致；
- 未显式提供正整数 <code>num_speculative_tokens</code>；
- proposal 超过 checkpoint capacity；
- acceptance 不是 standard rejection sampling；
- Ascend MRV2 使用尚未验收的 probabilistic Draft/rejection 组合；
- aux layer IDs、anchor alignment、Qwen layout 或 vocab mapping 不完整；
- checkpoint missing/unexpected/mismatched weights；
- Target/Draft hidden width不一致；
- publish snapshot 为空、重复 tensor 名、缺 candidate LM Head；
- snapshot 的 Draft/teacher version 与成功训练结果不一致；
- rollout engine 不支持 Draft transfer target 切换。

### 15.2 当前不是事务的部分

- Target 与 Draft 在一次 pause 内顺序更新，但没有 pair checksum、两阶段 commit 或自动 rollback；
- Target 更新成功、Draft 更新失败时 job 会 fail fast，不能声称已原子提交一对模型；
- pause 之后的更新路径若抛错，当前不保证执行 <code>continue_generation</code>；engine 可能保持 pause，需要显式恢复或重启；
- HF 两文件目录交换不是 crash-atomic；
- resume 不自动把已恢复 candidate 发布到 serving engine；
- candidate/Target version gap 只有 metadata，没有自动阈值与回退策略。

生产化建议增加 pair ID、完整 manifest/checksum、prepare/commit/abort、启动恢复和 target-version-gap gate。

### 15.3 分布正确性的适用边界

标准 speculative rejection sampling 的“输出分布等于 Target”是算法结论，不是对任意 runner、sampler kernel
和参数组合的 blanket 保证。VIME 当前参数校验会要求
<code>rejection_sample_method=standard</code>，但这不足以替代后端实现验收：

- Ascend MRV1 的 greedy Draft proposal 可继续配合随机 Target sampling；“Draft 只支持 greedy”不等于请求必须
  <code>temperature=0</code>；
- 当前 Ascend MRV2 probabilistic 路径存在上述固定接受随机量问题，不能声称无损；
- confidence 动态截断、batch 调度或未来近似 acceptance 变更，都必须重新做相同 prompt/seed 的 spec on/off
  分布一致性测试，并检查 rollout logprob 语义；
- 在完成端到端验证前，生产 RL 应优先使用已验收的 MRV1 greedy Draft 路径，或关闭 speculation。

## 16. 可观测性与验收

### 16.1 训练侧

- collect received/accepted/queued/version mismatch；
- loss、top-1、valid tokens、grad norm；
- acceptance surrogate、expected accepted length、confidence loss；
- optimizer steps、Draft version、teacher Target version；
- published Draft version。

### 16.2 推理侧

- raw proposed/accepted/verify-round/completion counters；
- position-wise conditional acceptance；
- Target calls/token；
- output tokens/s、per-user tokens/s、TTFT、ITL、p50/p99；
- running/waiting requests、scheduled tokens、KV cache、preemption；
- Draft/Target/accept sampler 分段设备时间；
- graph hit/miss、HCCL/NCCL 时间、NPU/GPU utilization；
- confidence ECE、Brier score、实际/预测 prefix survival。

### 16.3 最低验收矩阵

| 层级 | 必测项 |
| --- | --- |
| config | canonical + hybrid parse；非法 anchor/layer/vocab/capacity fail-fast |
| checkpoint | strict load；HF export 后 Speculators 与 vLLM real reload |
| training | NPU one-step backward；CE/TV/confidence finite；LM heads 冻结 |
| candidate | collect B 不污染 candidate A；零步训练保留 A；metadata drift fail |
| transport | DSpark 一次 packed send/load；任何异常 reset Target transfer target |
| inference correctness | 相同 prompts 的 spec on/off 分布一致性测试 |
| performance | concurrency sweep；真实 raw counters；端到端 confidence interval |
| restart | Actor/Draft rollout 对齐；导出目录覆盖/回滚；恢复后重新训练再发布 |

## 17. 当前限制与下一步

### 已实现

- dense Qwen3 DSpark strict checkpoint；
- pre-update aux/final hidden + Target LM Head 采集；
- version-isolated queue 与 document-safe packing；
- Speculators 原生 DSpark loss；
- Actor rank 0 独立 optimizer；
- immutable candidate head/version；
- native checkpoint 与 HF 两文件导出；
- Target 后顺序 Draft 热更新；
- Ascend MRV1/MRV2 Draft model lookup compatibility；
- DSpark-only single packed snapshot。

### 尚未实现或未完成端到端验收

- Actor optimizer 后重新 forward 的 strict aligned teacher mode；
- dedicated Draft workers、DDP 与 Actor/Draft 并行训练；
- 论文 STS calibration 与完整 profiled-SPS scheduler；
- MRV2 confidence-driven verify length；
- validation gate、自动禁用负收益 batch；
- Target/Draft pair transaction、checksum、rollback；
- post-resume 自动 stage/publish；
- Gemma、DeepSeek/MoE、跨 hidden-width；
- 真实 NPU 长时间训练、热更新和故障注入验收；
- 以 target version gap、真实吞吐和 confidence calibration 为发布门控。

## 18. 代码导航

### VIME

| 模块 | 职责 |
| --- | --- |
| [config.py](../../../vime/backends/speculative_training/config.py) | 开关、硬约束、DSpark hybrid config、preflight |
| [feature_collector.py](../../../vime/backends/speculative_training/feature_collector.py) | Actor hidden/LM-head 前特征捕获 |
| [feature_schema.py](../../../vime/backends/speculative_training/feature_schema.py) | schema v1 与 versioned queue |
| [backends/dspark.py](../../../vime/backends/speculative_training/backends/dspark.py) | pack、native loss、metrics、双 LM Head 同步 |
| [factories/speculators_dspark.py](../../../vime/backends/speculative_training/factories/speculators_dspark.py) | strict config/model load |
| [draft_trainer.py](../../../vime/backends/speculative_training/draft_trainer.py) | optimizer、candidate、checkpoint、HF export |
| [draft_group.py](../../../vime/backends/speculative_training/draft_group.py) | Actor-rank0 RPC、candidate publish gate |
| [actor.py](../../../vime/backends/megatron_utils/actor.py) | pre-update capture 与 Draft RPC |
| [train.py](../../../train.py) | collect/train/save/publish 调度 |
| [update_weight_from_distributed.py](../../../vime/backends/megatron_utils/update_weight/update_weight_from_distributed.py) | Target/Draft 顺序发布与 packed transfer |
| [update_weight_from_tensor.py](../../../vime/backends/megatron_utils/update_weight/update_weight_from_tensor.py) | Ascend worker Draft target compatibility |

### 当前本地推理源码

| 模块 | 职责 |
| --- | --- |
| [vLLM Qwen3 DSpark](../../../../vllm/vllm/model_executor/models/qwen3_dspark.py) | Draft model、Markov head、weight loader |
| [vLLM DSpark speculator](../../../../vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py) | GPU/MRV2 proposal loop |
| [Ascend Qwen3 DSpark](../../../../vllm-ascend/vllm_ascend/models/qwen3_dspark.py) | confidence head 与 NPU weight load |
| [Ascend MRV1 proposer](../../../../vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py) | eager proposal 与动态 verify budget |
| [Ascend MRV1 runner](../../../../vllm-ascend/vllm_ascend/worker/model_runner_v1.py) | aux hidden、proposal、per-request verify length |
| [Ascend MRV2 speculator](../../../../vllm-ascend/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py) | MRV2 DSpark 接入 |

## 19. 一手资料

- [DSpark 论文 HTML](https://arxiv.org/html/2607.05147)
- [DSpark 论文 PDF](https://arxiv.org/pdf/2607.05147)
- [Speculators DSpark 文档](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/)
- [Speculators 固定 revision](https://github.com/vllm-project/speculators/tree/af3f1795495b5393c7c9aed3f1d55a42d47878ae)
- [固定 revision 的 DSpark core](https://github.com/vllm-project/speculators/blob/af3f1795495b5393c7c9aed3f1d55a42d47878ae/src/speculators/models/dspark/core.py)
- [固定 revision 的 Markov/Confidence heads](https://github.com/vllm-project/speculators/blob/af3f1795495b5393c7c9aed3f1d55a42d47878ae/src/speculators/models/dspark/model_definitions.py)
- [固定 revision 的 DSpark metrics](https://github.com/vllm-project/speculators/blob/af3f1795495b5393c7c9aed3f1d55a42d47878ae/src/speculators/models/dspark/metrics.py)
- [固定 revision 的 DFlash backbone](https://github.com/vllm-project/speculators/blob/af3f1795495b5393c7c9aed3f1d55a42d47878ae/src/speculators/models/dflash/core.py)
- [vLLM speculative decoding](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md)
