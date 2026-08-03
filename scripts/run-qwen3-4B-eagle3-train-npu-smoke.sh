#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
export HYDRA_FULL_ERROR=1
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export VIME_EXTERNAL_DRAFT_SMOKE_SKIP_ACTOR_UPDATE="${VIME_EXTERNAL_DRAFT_SMOKE_SKIP_ACTOR_UPDATE:-1}"
export VLLM_USE_AOT_COMPILE=0

VIME_ROOT="${VIME_ROOT:-/home/w00664509/vime-speculative-final}"
TARGET_MODEL="${TARGET_MODEL:-/home/data/weights/Qwen3-4B-Instruct-2507}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/data/weights/Qwen3-4B-Instruct-2507-eagle3}"
PROMPT_DATA="${PROMPT_DATA:-${VIME_ROOT}/scripts/data/qwen3_eagle3_smoke_math.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/w00664509/vime-speculative-output}"
DRAFT_CHECKPOINT_PATH="${DRAFT_CHECKPOINT_PATH:-${OUTPUT_ROOT}/draft-smoke}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/home/c00944022/vime-proj}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6385}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
RAY_DASHBOARD_AGENT_PORT="${RAY_DASHBOARD_AGENT_PORT:-52375}"
RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-46540}"
RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-52376}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_vime_external_draft_smoke_${RAY_GCS_PORT}_${RAY_DASHBOARD_PORT}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
DRAFT_TRAIN_STEPS_PER_TRIGGER="${DRAFT_TRAIN_STEPS_PER_TRIGGER:-1}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-128}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"

REWARD_POST_PROCESS_ARGS=()
if [[ "${VIME_EXTERNAL_DRAFT_SMOKE_REWARD_FALLBACK:-1}" == "1" ]]; then
  REWARD_POST_PROCESS_ARGS=(
    --custom-reward-post-process-path vime.backends.speculative_training.smoke_rewards.ensure_nonzero_grpo_signal
  )
fi

export PYTHONPATH="${VIME_ROOT}:${MEGATRON_ROOT}/Megatron-Bridge/src:${MEGATRON_ROOT}/Megatron-LM:${MEGATRON_ROOT}/MindSpeed:${PYTHONPATH:-}"

cd "${VIME_ROOT}"
source "${VIME_ROOT}/scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
  --hf-checkpoint "${TARGET_MODEL}"
  --load "${TARGET_MODEL}"
  --ref-load "${TARGET_MODEL}"
  --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type math
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size 2
  --n-samples-per-prompt 2
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --global-batch-size 4
  --balance-data
  "${REWARD_POST_PROCESS_ARGS[@]}"
)

PERF_ARGS=(
  --tensor-model-parallel-size 4
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --use-dynamic-batch-size
  # Keep the tiny smoke batch in one dynamic microbatch. The current Ascend
  # baseline can stall when this case is split between repeated NPU forwards.
  --max-tokens-per-gpu 1024
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --kl-coef 0.0
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
  --rollout-num-gpus-per-engine 4
  --vllm-gpu-memory-utilization 0.55
  --vllm-max-model-len 512
  --vllm-enforce-eager
  --vllm-enable-sleep-mode
  --vllm-weight-sync-mode native
  --vllm-speculative-config "{\"method\":\"eagle3\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":3}"
)

DRAFT_ARGS=(
  --enable-external-draft-training
  --draft-algorithm eagle3
  --draft-model-path "${DRAFT_MODEL}"
  --draft-model-factory-path vime.backends.speculative_training.factories.verl_speco_eagle3.build_model
  --draft-feature-layer-ids 2,18,33
  --draft-hidden-window-tokens 64
  --draft-max-samples-per-rollout-per-dp 2
  --draft-max-tokens-per-rollout-per-dp 256
  --draft-collect-interval 1
  --draft-train-interval 1
  --draft-publish-interval 1
  --draft-train-steps-per-trigger "${DRAFT_TRAIN_STEPS_PER_TRIGGER}"
  --draft-batch-size-per-gpu 1
  --draft-checkpoint-path "${DRAFT_CHECKPOINT_PATH}"
  --draft-save-interval 1
  --update-weight-mode full
  --update-weight-transport nccl
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --micro-batch-size 1
  --use-flash-attn
)

TRAIN_ARGS=(
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${VLLM_ARGS[@]}" \
  "${DRAFT_ARGS[@]}" \
  "${MISC_ARGS[@]}"
)

if [[ "${VIME_PARSE_ONLY:-0}" == "1" ]]; then
  /usr/local/python3.12.13/bin/python3.12 -c \
    'from vime.utils.arguments import parse_args; a = parse_args(); print("PARSE_OK", a.draft_feature_layer_ids, a.vllm_speculative_config)' \
    "${TRAIN_ARGS[@]}"
  exit 0
fi

if [[ "${VIME_CLEANUP_FIRST_8_NPUS:-1}" == "1" && "${VIME_HOST_NPU_CLEANUP_DONE:-0}" != "1" ]]; then
  echo "ERROR: real smoke runs must be started with scripts/run-qwen3-4B-eagle3-train-npu-smoke-host.sh" >&2
  echo "The host wrapper releases physical NPU devices 0-7 before entering the container." >&2
  exit 1
fi

ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address 127.0.0.1 \
  --port="${RAY_GCS_PORT}" \
  --temp-dir="${RAY_TEMP_DIR}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_PORT}" \
  --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
  --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}" \
  --disable-usage-stats

if [[ "${VIME_RAY_JOB_SUBMIT:-1}" == "1" ]]; then
  ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" -- \
    /usr/local/python3.12.13/bin/python3.12 train.py "${TRAIN_ARGS[@]}"
else
  /usr/local/python3.12.13/bin/python3.12 train.py "${TRAIN_ARGS[@]}"
fi
