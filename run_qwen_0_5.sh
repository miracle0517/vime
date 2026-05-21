#!/bin/bash

export PYTHONPATH=/root/Megatron-LM

export CUDA_DEVICE_MAX_CONNECTIONS=1

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
VIME_ROOT="${VIME_ROOT:-${REPO_ROOT}/vime}"
SCRIPT_DIR="${SCRIPT_DIR:-${VIME_ROOT}/scripts}"
source "${SCRIPT_DIR}/models/qwen2.5-0.5B.sh"

export PYTHONUNBUFFERED=1
LOG_ROOT="${LOG_ROOT:-/mnt/nvme3n1/n0090/SLIME_PJ/vime_pj/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${LOG_ROOT}/tb_qwen2.5-0.5B}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/train_qwen2.5-0.5B_vllm.log}"
mkdir -p "${TENSORBOARD_DIR}" "${LOG_ROOT}"

TRAIN_ENV_VARS_JSON="{\"TENSORBOARD_DIR\":\"${TENSORBOARD_DIR}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}"

cd "${VIME_ROOT}"

HF_CKPT="${HF_CKPT:-/root/Qwen2.5-0.5B-Instruct}"
REF_LOAD="${REF_LOAD:-/root/Qwen2.5-0.5B-Instruct_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/root/gsm8k/train.parquet}"

python train.py \
  --train-backend megatron \
  --train-env-vars "${TRAIN_ENV_VARS_JSON}" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 2 \
  --rollout-num-gpus 2 \
  --rollout-num-gpus-per-engine 1 \
  ${MODEL_ARGS[@]} \
  \
  --hf-checkpoint "${HF_CKPT}" \
  --ref-load "${REF_LOAD}" \
  \
  --prompt-data "${PROMPT_DATA}" \
  --input-key messages \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type deepscaler \
  --vllm-gpu-memory-utilization 0.2 \
  \
  --num-rollout 200 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len 8192 \
  --rollout-temperature 1.0 \
  --global-batch-size 256 \
  --balance-data \
  \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  \
  --tensor-model-parallel-size 1 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 8192 \
  --colocate \
  --no-offload-train \
  \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  \
  --train-memory-margin-bytes 2147483648 \
  --use-tensorboard \
  2>&1 | tee -a "${LOG_FILE}"