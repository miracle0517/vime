from __future__ import annotations

import ast
import json
import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


NUM_GPUS = 16

_RANK_LOSS_PREFIX = "VIME_RANK_LOSS "
_TRAIN_STEP_RE = re.compile(r"actorstep\s+(\d+):\s+(\{.*\})")
_RANK_LOSS_RE = re.compile(re.escape(_RANK_LOSS_PREFIX) + r"(\{.*\})")


def _mpu_int(name: str, default: int = -1, **kwargs) -> int:
    from megatron.core import mpu

    func = getattr(mpu, name, None)
    if func is None:
        return default
    try:
        return int(func(**kwargs))
    except TypeError:
        return int(func())


def rank_logging_policy_loss(args, batch, logits, sum_of_sample_mean):
    """Policy loss wrapper used by the e2e test.

    It returns the normal production policy loss unchanged, but emits one JSON
    log line per loss-reporting rank. The logged loss is reduced with the same
    DP-with-CP group formula as the single-step train logger, so the test can
    assert rank consistency from the captured training log without modifying
    training code.
    """
    import torch
    import torch.distributed as dist
    from megatron.core import mpu

    from vime.backends.megatron_utils.loss import policy_loss_function

    loss, log = policy_loss_function(args, batch, logits, sum_of_sample_mean)

    if dist.is_available() and dist.is_initialized() and "loss" in log:
        reduced_loss = log["loss"].detach().float().mean().clone()
        dist.all_reduce(reduced_loss, group=mpu.get_data_parallel_group(with_context_parallel=True))
        reduced_loss = reduced_loss / float(args.global_batch_size)

        payload = {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "tp_rank": _mpu_int("get_tensor_model_parallel_rank"),
            "pp_rank": _mpu_int("get_pipeline_model_parallel_rank"),
            "dp_rank": _mpu_int("get_data_parallel_rank", with_context_parallel=True),
            "cp_rank": _mpu_int("get_context_parallel_rank"),
            "ep_rank": _mpu_int("get_expert_model_parallel_rank"),
            "loss": float(reduced_loss.item()),
        }
        print(_RANK_LOSS_PREFIX + json.dumps(payload, sort_keys=True), flush=True)

    return loss, log


def _run_bash(command: str, *, timeout: int = 1800) -> str:
    proc = subprocess.run(
        ["bash", "-lc", command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AssertionError(f"Command failed with exit code {proc.returncode}:\n{command}\n\n{proc.stdout}")
    return proc.stdout


def _setup_npu_env_bash() -> str:
    return r"""
source_if_exists() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    source "${path}"
  fi
}

source_if_exists /usr/local/Ascend/driver/bin/setenv.bash
source_if_exists /usr/local/Ascend/ascend-toolkit/set_env.sh
source_if_exists /usr/local/Ascend/nnal/atb/set_env.sh

export PYTHONPATH="/root/Megatron-LM:/root/vllm_src:/root/vllm-ascend:/root/vime:/root/Megatron-Bridge:/root/mbridge:/root/MindSpeed:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_DETERMINISTIC=true
export VLLM_ASCEND_ENABLE_NZ=0
export ASCEND_COREDUMP_SIGNAL=None
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export ATB_LLM_LCOC_ENABLE=0
export TASK_QUEUE_ENABLE=1
export RAY_DISABLE_SIGINT_OVERRIDE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/cann/lib64:/usr/local/Ascend/cann/aarch64-linux/devlib:${LD_LIBRARY_PATH:-}"
"""


def _cleanup_bash(ray_tmpdir: str) -> str:
    return f"""
pkill -9 -f "vllm serve|VLLM::" 2>/dev/null || true
npu-smi info 2>/dev/null | grep rayWorker | awk '{{print $4}}' | xargs -r kill -9 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 redis 2>/dev/null || true
rm -rf {shlex.quote(ray_tmpdir)}
"""


def _local_ip() -> str:
    return _run_bash("hostname -I | awk '{print $1}'").strip()


def _worker_ips(head_ip: str) -> list[str]:
    if worker_ips := os.environ.get("WORKER_IPS"):
        return [ip.strip() for ip in worker_ips.split(",") if ip.strip()]

    hostfile = os.environ.get("HOSTFILE")
    if not hostfile:
        raise AssertionError("Set WORKER_IPS or HOSTFILE so the test can start the second Ray node.")

    ips = []
    for line in Path(hostfile).read_text().splitlines():
        fields = line.split()
        if fields:
            ips.append(fields[0])
    return [ip for ip in ips if ip != head_ip]


def _start_ray_cluster(*, head_ip: str, worker_ips: list[str], ray_tmpdir: str, ray_port: int, dashboard_port: int):
    npu_per_node = 8
    _run_bash(_setup_npu_env_bash() + _cleanup_bash(ray_tmpdir), timeout=120)

    _run_bash(
        _setup_npu_env_bash()
        + f"""
unset ASCEND_RT_VISIBLE_DEVICES https_proxy http_proxy proxy
ray start --head \
  --temp-dir={shlex.quote(ray_tmpdir)} \
  --port={ray_port} \
  --dashboard-port={dashboard_port} \
  --node-ip-address={shlex.quote(head_ip)} \
  --num-gpus 0 \
  --resources '{{"NPU": {npu_per_node}}}' \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0
""",
        timeout=120,
    )

    for worker_ip in worker_ips:
        remote_command = (
            _setup_npu_env_bash()
            + _cleanup_bash(ray_tmpdir)
            + f"""
unset ASCEND_RT_VISIBLE_DEVICES https_proxy http_proxy proxy
ray start \
  --address={shlex.quote(f"{head_ip}:{ray_port}")} \
  --node-ip-address={shlex.quote(worker_ip)} \
  --num-gpus 0 \
  --resources '{{"NPU": {npu_per_node}}}' \
  --disable-usage-stats
"""
        )
        _run_bash(f"ssh root@{shlex.quote(worker_ip)} {shlex.quote(remote_command)}", timeout=180)

    _wait_for_ray_cluster(head_ip=head_ip, ray_port=ray_port, expected_nodes=1 + len(worker_ips), expected_npus=16)


def _wait_for_ray_cluster(*, head_ip: str, ray_port: int, expected_nodes: int, expected_npus: int):
    code = f"""
import ray
import time

ray.init(address={f"{head_ip}:{ray_port}"!r}, ignore_reinit_error=True)
deadline = time.time() + 600
while time.time() < deadline:
    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    npus = int(ray.cluster_resources().get("NPU", 0))
    if len(alive_nodes) >= {expected_nodes} and npus >= {expected_npus}:
        print(f"Ray cluster ready: nodes={{len(alive_nodes)}}, NPU={{npus}}")
        break
    print(f"Waiting for Ray cluster: nodes={{len(alive_nodes)}}/{expected_nodes}, NPU={{npus}}/{expected_npus}")
    time.sleep(5)
else:
    raise SystemExit("Timed out waiting for Ray cluster resources")
ray.shutdown()
"""
    _run_bash("python3 - <<'PY'\n" + code + "PY\n", timeout=660)


def _runtime_env_json(head_ip: str) -> str:
    return json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": (
                    "/root/Megatron-LM:/root/vllm_src:/root/vllm-ascend:/root/vime:"
                    "/root/Megatron-Bridge:/root/mbridge:/root/MindSpeed:"
                    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:"
                    "/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge"
                ),
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
                "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
                "HCCL_CONNECT_TIMEOUT": "7200",
                "HCCL_DETERMINISTIC": "true",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "MASTER_ADDR": head_ip,
                "no_proxy": f"localhost,127.0.0.1,0.0.0.0,{head_ip}",
                "LD_LIBRARY_PATH": (
                    "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:"
                    "/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/ascend-toolkit/latest/lib64:"
                    "/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:"
                    "/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:"
                    "/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/:"
                    "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/cann/lib64:"
                    "/usr/local/Ascend/cann/aarch64-linux/devlib"
                ),
            }
        }
    )


def _submit_one_step_train_job(*, head_ip: str, dashboard_port: int) -> str:
    model_root = os.environ.get("MODEL_ROOT", "/root")
    script_dir = os.environ.get("SCRIPT_DIR", "/root/vime/scripts")
    runtime_env = shlex.quote(_runtime_env_json(head_ip))

    return _run_bash(
        _setup_npu_env_bash()
        + f"""
source {shlex.quote(script_dir)}/models/qwen3-30B-A3B-npu.sh
RUNTIME_ENV_JSON={runtime_env}

ray job submit --address="http://{head_ip}:{dashboard_port}" \
  --runtime-env-json="${{RUNTIME_ENV_JSON}}" \
  --working-dir="/root/vime" \
  -- python3 train.py \
  --train-backend megatron \
  --actor-num-nodes 2 \
  --actor-num-gpus-per-node 8 \
  --num-gpus-per-node 8 \
  --rollout-num-gpus 16 \
  --colocate \
  "${{MODEL_ARGS[@]}}" \
  --hf-checkpoint {shlex.quote(model_root)}/models/Qwen3-30B-A3B/ \
  --load {shlex.quote(model_root)}/models/Qwen3-30B-A3B/ \
  --no-load-optim \
  --megatron-to-hf-mode bridge \
  --prompt-data {shlex.quote(model_root)}/datasets/dapo-math-17k/dapo-math-17k.jsonl \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type math \
  --num-rollout 1 \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len 128 \
  --rollout-temperature 0.0 \
  --global-batch-size 4 \
  --advantage-estimator grpo \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --kl-coef 0.0 \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --loss-type custom_loss \
  --custom-loss-function-path tests.test_ci_rank_loss_consistency.rank_logging_policy_loss \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --tensor-model-parallel-size 2 \
  --sequence-parallel \
  --pipeline-model-parallel-size 2 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 4 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --max-tokens-per-gpu 4096 \
  --rollout-backend vllm \
  --rollout-num-gpus-per-engine 8 \
  --vllm-weight-sync-mode native \
  --vllm-enable-sleep-mode \
  --vllm-enable-expert-parallel \
  --vllm-gpu-memory-utilization 0.45 \
  --vllm-max-model-len 1024 \
  --vllm-max-num-seqs 64 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --use-flash-attn \
  --micro-batch-size 1 \
  --moe-token-dispatcher-type alltoall \
  --no-gradient-accumulation-fusion \
  --train-memory-margin-bytes 2147483648
""",
        timeout=int(os.environ.get("VIME_RANK_LOSS_JOB_TIMEOUT", "7200")),
    )


def _parse_train_loss(log_text: str) -> float:
    matches = _TRAIN_STEP_RE.findall(log_text)
    step_logs = []
    for step, raw_dict in matches:
        data = ast.literal_eval(raw_dict)
        if int(step) == 0 and "train/loss" in data:
            step_logs.append(data)

    assert len(step_logs) == 1, f"Expected exactly one train step log with train/loss, got {len(step_logs)}"
    loss = float(step_logs[0]["train/loss"])
    assert math.isfinite(loss), f"train/loss is not finite: {loss}"
    assert int(step_logs[0]["train/step"]) == 0
    return loss


def _parse_rank_losses(log_text: str) -> list[dict]:
    losses = [json.loads(match) for match in _RANK_LOSS_RE.findall(log_text)]
    unique = {}
    for item in losses:
        unique[item["rank"]] = item
    return [unique[rank] for rank in sorted(unique)]


def test_qwen3_30b_a3b_two_node_one_step_rank_loss_from_train_log():
    head_ip = os.environ.get("HEAD_NODE_IP") or os.environ.get("MASTER_ADDR") or _local_ip()
    worker_ips = _worker_ips(head_ip)
    assert len(worker_ips) == 1, f"Expected exactly one worker node for 2*8 test, got {worker_ips}"

    ray_tmpdir = os.environ.get("RAY_TMPDIR", f"/tmp/ray_vime_rank_loss_{int(time.time())}")
    ray_port = int(os.environ.get("RAY_PORT", "6388"))
    dashboard_port = int(os.environ.get("RAY_DASHBOARD_PORT", "8274"))
    log_path = Path(os.environ.get("VIME_RANK_LOSS_LOG_PATH", "/tmp/vime_rank_loss_consistency_train.log"))

    _start_ray_cluster(
        head_ip=head_ip,
        worker_ips=worker_ips,
        ray_tmpdir=ray_tmpdir,
        ray_port=ray_port,
        dashboard_port=dashboard_port,
    )

    log_text = _submit_one_step_train_job(head_ip=head_ip, dashboard_port=dashboard_port)
    log_path.write_text(log_text)

    train_loss = _parse_train_loss(log_text)
    rank_losses = _parse_rank_losses(log_text)
    expected_loss_reporting_ranks = NUM_GPUS // 2
    assert len(rank_losses) == expected_loss_reporting_ranks, (
        f"Expected {expected_loss_reporting_ranks} rank loss log lines from PP last stage, got {len(rank_losses)}. "
        f"Training log saved to {log_path}"
    )

    values = [float(item["loss"]) for item in rank_losses]
    assert all(math.isfinite(value) for value in values), f"Non-finite rank loss values: {rank_losses}"
    max_diff = max(abs(value - train_loss) for value in values)
    assert max_diff <= 1e-5, (
        f"Rank loss mismatch against train/loss={train_loss}, max_diff={max_diff}. "
        f"rank_losses={rank_losses}. Training log saved to {log_path}"
    )
