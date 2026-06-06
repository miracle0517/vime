# Delta 权重同步(Delta Weight Sync)

> 移植自 THUDM/slime #1806/#1946/#1991,翻译到 vime 的 vLLM rollout。

## 它解决什么

RL 训练每步更新 policy 权重后,需要把新权重同步到 rollout 推理引擎(vLLM)。默认的**全量同步**每次把整套参数广播过去 —— 大模型(尤其大 MoE)或跨数据中心场景下,每步全量传输的带宽开销很大。

**Delta 权重同步只传"自上次同步以来变化的字节"**:训练侧对当前权重和上次广播的 pinned-CPU 快照做**逐字节 diff**,只打包变化的位置 + 新值发出去;接收侧把这些字节**原样覆盖**进 live 模型(无损、无算术、不累积漂移)。

> 注意:这是**互补的带宽优化,不是必需**。**colocate(训练/推理同机)用 CUDA-IPC 直传更快,无需 delta**。delta 面向 **non-colocate**(尤其大模型 / 跨 DC / 带宽受限)。

## 开启

```bash
--update-weight-mode delta            # 默认 full
--update-weight-transport nccl        # 或 disk
--update-weight-encoding deltas_zstd  # indices | deltas | deltas_zstd
```

### 参数

| 参数 | 说明 |
|---|---|
| `--update-weight-mode {full,delta}` | `full`=每次广播整参数(默认);`delta`=只发逐字节变化 |
| `--update-weight-transport {nccl,disk}` | delta 的每桶载体。`nccl`=NCCL 广播(低延迟、同 DC);`disk`=写 safetensors 落共享盘 + 每轮一次 HTTP 唤醒引擎读取(跨 DC、带宽受限) |
| `--update-weight-encoding {indices,deltas,deltas_zstd}` | 位置编码。`indices`=int32 绝对下标(最大、计算最省);`deltas`=uint16 间隔差(更小);`deltas_zstd`=`deltas` + zstd 压缩(最小) |
| `--update-weight-delta-dir <path>` | `disk` 传输时,每轮 delta safetensors 的目录(训练侧与引擎侧共享文件系统) |
| `--update-weight-delta-keep-files` | 保留各轮 delta 文件(默认清理) |
| `--update-weight-delta-chunk-bytes <N>` | 接收侧每次 `load_weights` 的字节预算(解码后分块 apply) |

## 工作原理(vime 实现)

1. **训练侧**(`UpdateWeightFromDistributedDelta`,继承全量的 `UpdateWeightFromDistributed`):留一份上次广播权重的 pinned-CPU 快照;每次同步做逐字节 diff → `(positions, values)` + 每参数解码清单 `DeltaSpec`。第一次同步只 seed 快照、不联系引擎(引擎 init 时已加载同一 HF checkpoint)。
2. **wire 格式**:`__positions__`(uint8 字节偏移 blob)+ `__values__`(参数 dtype 的值)+ JSON `DeltaSpec`(encoding + 每参数切片 + checksum)。`nccl`/`disk` 两种传输共享。
3. **接收侧**(vime 的 `vLLMColocateWorkerExtension` 上经 `/collective_rpc` 调用的方法 + `delta_receiver.py`):
   - `nccl`:按训练侧广播顺序在 `model_update_group` 上收 `(positions, values)`;
   - `disk`:读 + 解压 safetensors;
   - 解码:positions 解包成下标(`indices` 绝对 / `deltas` 间隔差 cumsum 还原),`index_copy_` 到一个填 NaN 的全形 tensor(NaN=未变);
   - apply:`with delta_apply_context(model): model.load_weights(chunk)` —— vLLM 的 `load_weights` 照常分片,期间 `torch.Tensor.copy_/fill_` 被临时拦截,**只把非-NaN 位写进模型参数存储**;`post_load_weights`(fp8 scale 等派生量)用原始 copy_ 照常重算。
4. **无需重建镜像**:接收侧是 vime 运行时注入的 worker-extension 方法 / monkey-patch(经容器内 `pip install -e .` 加载),不改 vLLM 引擎源、不改 Dockerfile。

## 已知限制(重要)

- **投机解码(EAGLE/MTP)的 draft 模型不会被同步刷新**。vLLM 在任何权重同步(全量 / delta)下都**只更新主模型**(`gpu_model_runner.reload_weights` 只 load 主模型的 `named_parameters`;drafter 仅在 init 时加载一次)。因此 RL + spec-decode 下,draft 专属层(EAGLE transformer / MTP head)在同步后会 stale —— 这是 **vLLM 层面的全局限制,与 delta 无关**,全量同步也一样。slime 曾用 #1993 在 sglang 侧给 EAGLE draft-worker 转发 delta 文件来修,但 vLLM 没有等价机制;**此 PR 不实现 draft 覆盖**,待上游(vLLM)补 drafter-on-update 后再跟进。
- `disk` 传输要求训练侧与引擎侧**共享文件系统**;`nccl` 传输用于同机/同 DC。
- delta diff/编码有 CPU 开销;收益在"变化字节量 ≪ 整模型"时(大模型 / 跨 DC)才显著。

## 验证

`tests/test_delta_weight_update.py`(e2e):delta 同步后,rollout/train logprob 与全量同步一致;zero-delta(无变化)轮次的 weight_version 仍正确递增(不触发 actor 的 version 一致性断言)。
