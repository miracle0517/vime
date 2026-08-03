#!/bin/bash

set -euo pipefail

# Run this wrapper on the host. Ascend 910 devices 0-7 are shown by npu-smi as
# four dual-chip NPU cards (card ids 0-3). A container cannot signal processes
# in another container's PID namespace, so cleanup has to happen here.
CONTAINER_NAME="${CONTAINER_NAME:-wxx-vllm-022}"
VIME_ROOT="${VIME_ROOT:-/home/w00664509/vime-speculative-final}"
CLEANUP_NPU_CARD_IDS="${CLEANUP_NPU_CARD_IDS:-0,1,2,3}"
CLEANUP_TERM_TIMEOUT="${CLEANUP_TERM_TIMEOUT:-10}"
VIME_CLEANUP_FIRST_8_NPUS="${VIME_CLEANUP_FIRST_8_NPUS:-1}"

collect_npu_pids() {
  npu-smi info | awk -F '|' -v wanted="${CLEANUP_NPU_CARD_IDS}" '
    BEGIN {
      count = split(wanted, ids, ",")
      for (i = 1; i <= count; ++i) selected[ids[i] + 0] = 1
    }
    {
      card = $2
      pid = $4
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", card)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", pid)
      if (card ~ /^[0-9]+$/ && pid ~ /^[0-9]+$/ && selected[card + 0]) {
        print pid
      }
    }
  ' | sort -nu
}

cleanup_container_runtime() {
  echo "[NPU cleanup] Stopping stale Ray and vLLM processes in ${CONTAINER_NAME}."
  docker exec "${CONTAINER_NAME}" ray stop --force >/dev/null 2>&1 || true
  # A failed/finished Ray run can leave orphaned EngineCore and Worker processes
  # under the container init process. Kill those before checking npu-smi so they
  # cannot recreate device workers after the host-side cleanup.
  docker exec "${CONTAINER_NAME}" bash -lc \
    "pkill -TERM -f '[V]LLM::' 2>/dev/null || true" || true
  sleep 2
  docker exec "${CONTAINER_NAME}" bash -lc \
    "pkill -KILL -f '[V]LLM::' 2>/dev/null || true" || true
}

cleanup_first_8_npus() {
  cleanup_container_runtime

  mapfile -t pids < <(collect_npu_pids)
  if ((${#pids[@]} == 0)); then
    echo "[NPU cleanup] Physical devices 0-7 are already free."
  else
    echo "[NPU cleanup] Sending TERM to processes on physical devices 0-7: ${pids[*]}"
    kill -TERM "${pids[@]}" 2>/dev/null || true

    local deadline=$((SECONDS + CLEANUP_TERM_TIMEOUT))
    local -a remaining=()
    while ((SECONDS < deadline)); do
      remaining=()
      mapfile -t remaining < <(collect_npu_pids)
      ((${#remaining[@]} == 0)) && break
      sleep 1
    done

    mapfile -t remaining < <(collect_npu_pids)
    if ((${#remaining[@]} > 0)); then
      echo "[NPU cleanup] TERM timeout; sending KILL to: ${remaining[*]}"
      kill -KILL "${remaining[@]}" 2>/dev/null || true
      sleep 2
    fi
  fi

  local -a remaining=()
  mapfile -t remaining < <(collect_npu_pids)
  if ((${#remaining[@]} > 0)); then
    echo "[NPU cleanup] Failed to release physical devices 0-7: ${remaining[*]}" >&2
    exit 1
  fi

  # Require the devices to stay free briefly; this catches a surviving parent
  # process that respawns workers immediately after they are killed.
  sleep 3
  mapfile -t remaining < <(collect_npu_pids)
  if ((${#remaining[@]} > 0)); then
    echo "[NPU cleanup] Processes respawned on physical devices 0-7: ${remaining[*]}" >&2
    exit 1
  fi
  echo "[NPU cleanup] Physical devices 0-7 are free."
}

if [[ "${VIME_CLEANUP_FIRST_8_NPUS}" == "1" ]]; then
  cleanup_first_8_npus
fi

PASS_ENV_NAMES=(
  ASCEND_RT_VISIBLE_DEVICES
  DRAFT_CHECKPOINT_PATH
  DRAFT_TRAIN_STEPS_PER_TRIGGER
  NUM_ROLLOUT
  OUTPUT_ROOT
  PROMPT_DATA
  RAY_DASHBOARD_AGENT_GRPC_PORT
  RAY_DASHBOARD_AGENT_PORT
  RAY_DASHBOARD_PORT
  RAY_GCS_PORT
  RAY_RUNTIME_ENV_AGENT_PORT
  RAY_TEMP_DIR
  ROLLOUT_MAX_RESPONSE_LEN
  ROLLOUT_TEMPERATURE
  VIME_EXTERNAL_DRAFT_SMOKE_REWARD_FALLBACK
  VIME_EXTERNAL_DRAFT_SMOKE_SKIP_ACTOR_UPDATE
  VIME_PARSE_ONLY
  VIME_RAY_JOB_SUBMIT
)

DOCKER_ENV_ARGS=()
for name in "${PASS_ENV_NAMES[@]}"; do
  if [[ -v "${name}" ]]; then
    DOCKER_ENV_ARGS+=(--env "${name}=${!name}")
  fi
done

# The container entrypoint rejects a real launch without this marker, because
# only the host can clean processes owned by other PID namespaces.
DOCKER_ENV_ARGS+=(--env VIME_HOST_NPU_CLEANUP_DONE=1)

exec docker exec -i "${DOCKER_ENV_ARGS[@]}" "${CONTAINER_NAME}" \
  bash -lc "cd '${VIME_ROOT}' && exec bash scripts/run-qwen3-4B-eagle3-train-npu-smoke.sh"
