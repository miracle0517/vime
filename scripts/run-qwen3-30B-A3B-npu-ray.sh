#!/bin/bash
#set -ex

# Usage:
#   # worker node
#   VIME_RAY_ROLE=worker HEAD_NODE_IP=<head-ip> bash /root/vime/scripts/run-qwen3-30B-A3B-npu-ray.sh
#
#   # head node
#   VIME_RAY_ROLE=head HEAD_NODE_IP=<head-ip> bash /root/vime/scripts/run-qwen3-30B-A3B-npu-ray.sh

VIME_RAY_ROLE="${VIME_RAY_ROLE:-head}"
NNODES="${NNODES:-2}"
NPUS_PER_NODE="${NPUS_PER_NODE:-8}"
TOTAL_NPUS=$((NNODES * NPUS_PER_NODE))
NODE_IP="${NODE_IP:-$(hostname -I | awk '{print $1}')}"
if [[ "${VIME_RAY_ROLE}" == "head" ]]; then
   HEAD_NODE_IP="${HEAD_NODE_IP:-${NODE_IP}}"
else
   : "${HEAD_NODE_IP:?Set HEAD_NODE_IP on worker node}"
fi

# cleanup
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 2
npu-smi info 2>/dev/null | grep rayWorker | awk '{print $4}' | xargs -r kill -9 2>/dev/null || true
sleep 3

# Ray isolation: independent temp-dir, ports, and cleanup
export RAY_TMPDIR=/tmp/ray_vime_npu_30b_a3b
export RAY_PORT=6388
export RAY_DASHBOARD_PORT=8274
export RAY_AGENT_PORT=52378
unset RAY_ADDRESS RAY_REDIS_ADDRESS

ray stop --force 2>/dev/null || true
rm -rf "${RAY_TMPDIR}"
sleep 2

# NPU environment
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH="/root/Megatron-LM:/root/vllm_src:/root/vllm-ascend:/root/vime:/root/Megatron-Bridge:/root/mbridge:/root/MindSpeed:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_DETERMINISTIC=true
export HYDRA_FULL_ERROR=1
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export ASCEND_COREDUMP_SIGNAL=None
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export ATB_LLM_LCOC_ENABLE=0
export TASK_QUEUE_ENABLE=1
#export VLLM_VERSION=0.17.0
export RAY_DISABLE_SIGINT_OVERRIDE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/cann/lib64:${LD_LIBRARY_PATH}

# Start Ray with isolated temp-dir and ports
unset ASCEND_RT_VISIBLE_DEVICES https_proxy http_proxy proxy
if [[ "${VIME_RAY_ROLE}" == "worker" ]]; then
   ray start \
      --address="${HEAD_NODE_IP}:${RAY_PORT}" \
      --node-ip-address "${NODE_IP}" \
      --num-gpus 0 \
      --resources "{\"NPU\": ${NPUS_PER_NODE}}" \
      --disable-usage-stats \
      --block
   exit 0
fi

ray start --head \
   --temp-dir="${RAY_TMPDIR}" \
   --port="${RAY_PORT}" \
   --dashboard-port="${RAY_DASHBOARD_PORT}" \
   --dashboard-agent-listen-port="${RAY_AGENT_PORT}" \
   --node-ip-address "${HEAD_NODE_IP}" \
   --num-gpus 0 \
   --resources "{\"NPU\": ${NPUS_PER_NODE}}" \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0

python3 - << EOF
import time

import ray

ray.init(address="${HEAD_NODE_IP}:${RAY_PORT}", ignore_reinit_error=True)
deadline = time.time() + 600
while time.time() < deadline:
    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    npus = int(ray.cluster_resources().get("NPU", 0))
    if len(alive_nodes) >= ${NNODES} and npus >= ${TOTAL_NPUS}:
        print(f"Ray cluster ready: nodes={len(alive_nodes)}, NPU={npus}")
        break
    print(f"Waiting for Ray cluster: nodes={len(alive_nodes)}/${NNODES}, NPU={npus}/${TOTAL_NPUS}")
    time.sleep(5)
else:
    raise SystemExit("Timed out waiting for Ray cluster resources")
ray.shutdown()
EOF

SCRIPT_DIR="/root/vime/scripts/"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B-npu.sh"
MODEL_ROOT="${MODEL_ROOT:-/root}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/models/Qwen3-30B-A3B/"
   --load "${MODEL_ROOT}/models/Qwen3-30B-A3B/"
   --no-load-optim
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data "${MODEL_ROOT}/datasets/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 300
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --global-batch-size 256
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
   --megatron-to-hf-mode bridge
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

VLLM_ARGS=(
   --rollout-backend vllm
   --rollout-num-gpus-per-engine 8
   --vllm-weight-sync-mode native
   --vllm-enable-sleep-mode
   --vllm-enable-expert-parallel
   --vllm-gpu-memory-utilization 0.6
   --vllm-max-model-len 20480
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --micro-batch-size 1
   --use-flash-attn
   --no-gradient-accumulation-fusion
   --train-memory-margin-bytes 2147483648
)

RUNTIME_ENV_JSON=$(cat << EOF
{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM:/root/vllm_src:/root/vllm-ascend:/root/vime:/root/Megatron-Bridge:/root/mbridge:/root/MindSpeed:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
    "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
    "HCCL_CONNECT_TIMEOUT": "7200",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
    "MASTER_ADDR": "${HEAD_NODE_IP}",
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${HEAD_NODE_IP}",
    "LD_LIBRARY_PATH": "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/cann/lib64:/usr/local/Ascend/cann/aarch64-linux/devlib"
  }
}
EOF
)

ray job submit --address="http://${HEAD_NODE_IP}:${RAY_DASHBOARD_PORT}" \
--runtime-env-json="${RUNTIME_ENV_JSON}" \
--working-dir="/root/vime" \
-- python3 train.py \
--train-backend megatron \
--actor-num-nodes "${NNODES}" \
--actor-num-gpus-per-node "${NPUS_PER_NODE}" \
--num-gpus-per-node "${NPUS_PER_NODE}" \
--rollout-num-gpus "${TOTAL_NPUS}" \
--colocate \
${MODEL_ARGS[@]} \
${CKPT_ARGS[@]} \
${ROLLOUT_ARGS[@]} \
${OPTIMIZER_ARGS[@]} \
${GRPO_ARGS[@]} \
${PERF_ARGS[@]} \
${VLLM_ARGS[@]} \
${MISC_ARGS[@]}
