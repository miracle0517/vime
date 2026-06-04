#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


IMAGE = "inferactinc/public:vime-vllm-cu129-latest"
# Lightweight image for the always-on CPU checks (pre-commit, plugin contracts):
# the Buildkite equivalent of a GitHub-hosted ubuntu-latest runner, so these
# don't pull the heavy CUDA training image.
CPU_IMAGE = "python:3.10"
CPU_QUEUE = "small_cpu_queue_premerge"
# All GPU jobs (4- and 8-GPU) run on the shared H100 Kubernetes pool; each job
# gets its own pod sized to its GPU count (the pool supports up to 8 GPUs/pod).
H100_QUEUE = "mithril-h100-pool"

HF_CACHE = "/fsx/hf_cache"   # HF cache on the CPU (docker-plugin) agents
K8S_CACHE = "/mnt/hf-cache"  # hostPath cache root on the H100 nodes


@dataclass(frozen=True)
class TestJob:
    test_file: str
    num_gpus: int
    test_args: str = ""
    use_deepep: str = "0"
    use_fp8_rollout: str = "0"
    enable_eval: str = "1"
    timeout_in_minutes: int | None = None
    image: str = IMAGE


SUITES: dict[str, list[TestJob]] = {
    "short": [
        TestJob("test_qwen3.5_0.8B_gsm8k_async_short.py", 4, timeout_in_minutes=120),
        TestJob("test_qwen3.5_0.8B_gsm8k_short.py", 4, timeout_in_minutes=120),
        TestJob("test_qwen2.5_0.5B_ppo_critic_only_short.py", 4, timeout_in_minutes=120),
        TestJob("test_qwen2.5_0.5B_fully_async_short.py", 4, timeout_in_minutes=120),
    ],
    "vllm-config": [
        TestJob("test_qwen2.5_0.5B_vllm_config.py", 8, timeout_in_minutes=180),
        TestJob("test_qwen2.5_0.5B_vllm_config_distributed.py", 8, timeout_in_minutes=180),
        TestJob("test_vllm_config_mixed_offload.py", 8, timeout_in_minutes=180),
        TestJob("test_vllm_config_mixed_offload_ft.py", 8, timeout_in_minutes=180),
    ],
    "megatron": [
        TestJob("test_quick_start_glm4_9B.py", 8, timeout_in_minutes=240),
        TestJob("test_glm4.7_30B_A3B_pd_mooncake.py", 8, timeout_in_minutes=240),
        TestJob("test_qwen3_30B_A3B.py", 8, use_deepep="1", use_fp8_rollout="1", timeout_in_minutes=240),
        TestJob("test_qwen3.6_35B_A3B_pd_mooncake.py", 8, use_deepep="1", timeout_in_minutes=240),
        TestJob(
            "test_qwen3_30B_A3B_r3.py",
            8,
            use_deepep="1",
            use_fp8_rollout="1",
            enable_eval="0",
            timeout_in_minutes=240,
        ),
        TestJob("test_qwen3_30B_A3B_r3.py", 8, enable_eval="0", timeout_in_minutes=240),
        TestJob("test_qwen3_4B_ppo.py", 8, timeout_in_minutes=240),
        TestJob("test_qwen3_4B_ppo_disaggregate.py", 8, timeout_in_minutes=240),
        TestJob("test_qwen3_4B_ppo_train_critic_only.py", 8, timeout_in_minutes=240),
        TestJob("test_moonlight_16B_A3B.py", 8, timeout_in_minutes=240),
        TestJob("test_moonlight_16B_A3B_r3.py", 8, enable_eval="0", timeout_in_minutes=240),
        TestJob("test_qwen2.5_0.5B_debug_rollout_then_train.py", 8, timeout_in_minutes=240),
        TestJob("test_qwen2.5_0.5B_opd_vllm.py", 8, timeout_in_minutes=240),
    ],
    "precision": [
        TestJob("test_qwen3_0.6B_parallel_check.py", 8, timeout_in_minutes=180),
    ],
    "ckpt": [
        TestJob("test_qwen3_4B_ckpt.py", 8, timeout_in_minutes=180),
        TestJob("test_qwen3_4B_ckpt.py", 8, test_args="--async-save", timeout_in_minutes=180),
    ],
    "image": [
        TestJob("test_qwen3.5_0.8B_gsm8k_async_short.py", 4, timeout_in_minutes=120),
        TestJob("test_qwen3.5_0.8B_gsm8k_short.py", 4, timeout_in_minutes=120),
        TestJob("test_quick_start_glm4_9B.py", 8, timeout_in_minutes=240),
        TestJob("test_glm4.7_30B_A3B_pd_mooncake.py", 8, timeout_in_minutes=240),
        TestJob("test_qwen3_30B_A3B.py", 8, timeout_in_minutes=240),
        TestJob("test_qwen3.6_35B_A3B_pd_mooncake.py", 8, use_deepep="1", timeout_in_minutes=240),
        TestJob("test_qwen3_4B_ppo.py", 8, timeout_in_minutes=240),
        TestJob("test_moonlight_16B_A3B.py", 8, timeout_in_minutes=240),
        TestJob("test_qwen3_0.6B_parallel_check.py", 8, timeout_in_minutes=180),
        TestJob("test_qwen3_4B_ckpt.py", 8, timeout_in_minutes=180),
        TestJob("test_qwen3_4B_ckpt.py", 8, test_args="--async-save", timeout_in_minutes=180),
        TestJob("test_qwen2.5_0.5B_debug_rollout_then_train.py", 8, timeout_in_minutes=240),
        TestJob("test_qwen2.5_0.5B_opd_vllm.py", 8, timeout_in_minutes=240),
    ],
}

PLUGIN_CONTRACTS = [
    "test_megatron_argument_validation.py",
    "plugin_contracts/test_plugin_rollout_contracts.py",
    "plugin_contracts/test_plugin_runtime_hook_contracts.py",
    "plugin_contracts/test_plugin_path_loading_contracts.py",
    "plugin_contracts/test_plugin_generate_contracts.py",
]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: upload_suite.py <core|short|vllm-config|megatron|precision|ckpt|image|changed>")

    suite = sys.argv[1]
    if suite == "core":
        pipeline = {"steps": core_steps()}
    elif suite == "changed":
        pipeline = {"steps": changed_steps()}
    elif suite in SUITES:
        pipeline = {"steps": suite_group(suite, SUITES[suite])}
    else:
        raise SystemExit(f"unknown suite: {suite}")

    json.dump(pipeline, sys.stdout, indent=2)
    sys.stdout.write("\n")


def core_steps() -> list[dict[str, Any]]:
    return [
        {
            "label": "Pre-commit",
            "key": "pre-commit",
            "timeout_in_minutes": 30,
            "commands": [
                "python -m pip install -q pre-commit --break-system-packages || python -m pip install -q pre-commit",
                "python -m pre_commit run --all-files --show-diff-on-failure --color=always",
            ],
            "agents": {"queue": CPU_QUEUE},
            "plugins": [lite_docker_plugin()],
            "retry": retry_agent_lost(),
        },
        {
            "group": "Plugin Contracts",
            "depends_on": "pre-commit",
            "steps": [
                test_step(
                    "plugin-contracts",
                    TestJob(test_file, 0, timeout_in_minutes=30),
                    group_label="Plugin Contracts",
                    depends_on=["pre-commit"],
                    cpu_lite=True,
                )
                for test_file in PLUGIN_CONTRACTS
            ],
        },
        {
            "label": "Unit & utils tests",
            "key": "unit-utils-tests",
            "depends_on": ["pre-commit"],
            "timeout_in_minutes": 45,
            "commands": [
                "python -m pip install -e . --no-deps --break-system-packages || python -m pip install -e . --no-deps",
                "python -m pip install -q pytest --break-system-packages || python -m pip install -q pytest",
                "python -m pytest tests/unit tests/utils",
            ],
            "agents": {"queue": CPU_QUEUE},
            "plugins": [docker_plugin()],
            "retry": retry_agent_lost(),
        },
    ]


def suite_group(suite: str, jobs: list[TestJob]) -> list[dict[str, Any]]:
    title = {
        "short": "Short Tests",
        "vllm-config": "vLLM Config Tests",
        "megatron": "Megatron Tests",
        "precision": "Precision Tests",
        "ckpt": "Checkpoint Tests",
        "image": "Image Release Tests",
    }[suite]

    return [
        {
            "group": title,
            "steps": [
                test_step(suite, job, group_label=title, depends_on=["pre-commit"])
                for job in jobs
            ],
        }
    ]


def changed_steps() -> list[dict[str, Any]]:
    files = changed_test_files()
    if not files:
        return [
            {
                "label": "Changed tests - none",
                "commands": ["echo 'No changed tests found.'"],
                "agents": {"queue": CPU_QUEUE},
            }
        ]

    jobs = []
    for file_path in files:
        num_gpus = parse_num_gpus(Path(file_path))
        jobs.append(TestJob(file_path, num_gpus, timeout_in_minutes=240 if num_gpus >= 8 else 120))

    return [
        {
            "group": "Changed Tests",
            "steps": [
                test_step("changed", job, group_label="Changed Tests", depends_on=["pre-commit"])
                for job in jobs
            ],
        }
    ]


def test_step(
    suite: str,
    job: TestJob,
    *,
    group_label: str,
    depends_on: list[str],
    cpu_lite: bool = False,
) -> dict[str, Any]:
    # cpu_lite -> always-on lightweight CPU runner (plugin contracts): a plain
    # python image on the CPU queue that pip-installs CPU deps, instead of the
    # heavy CUDA image / H100 pod used by GPU and in-image jobs.
    key = step_key(suite, job)
    step = {
        "label": f"{group_label} - {display_name(job)}",
        "key": key,
        "depends_on": depends_on,
        "commands": [".buildkite/scripts/run_test.sh"],
        "agents": {"queue": CPU_QUEUE if cpu_lite else agent_queue(job.num_gpus)},
        "env": test_env(job, cpu_lite=cpu_lite),
        "plugins": [lite_docker_plugin() if cpu_lite else plugin_for(job)],
        "retry": retry_agent_lost(),
    }
    if job.timeout_in_minutes:
        step["timeout_in_minutes"] = job.timeout_in_minutes
    return step


def test_env(job: TestJob, *, cpu_lite: bool = False) -> dict[str, str]:
    env = {
        "TEST_FILE": job.test_file,
        "TEST_ARGS": job.test_args,
        "NUM_GPUS": str(job.num_gpus),
        # Each k8s pod is sized to exactly NUM_GPUS, so the GPU lock should see
        # that many devices.
        "TOTAL_GPUS": str(job.num_gpus),
        "VIME_TEST_USE_DEEPEP": job.use_deepep,
        "VIME_TEST_USE_FP8_ROLLOUT": job.use_fp8_rollout,
        "VIME_TEST_ENABLE_EVAL": job.enable_eval,
    }
    if cpu_lite:
        # Lightweight image has no torch/deps baked in; run_test.sh installs the
        # CPU wheel set before running the contract file.
        env["VIME_INSTALL_CPU_DEPS"] = "1"
    return env


def plugin_for(job: TestJob) -> dict[str, Any]:
    # Every GPU job runs as its own H100 k8s pod; only CPU jobs use the
    # in-image docker plugin.
    if job.num_gpus >= 1:
        return h100_k8s_plugin(job)
    return docker_plugin(image=job.image)


def lite_docker_plugin(image: str = CPU_IMAGE) -> dict[str, Any]:
    # Lightweight CPU runner (no CUDA image) for pre-commit and plugin
    # contracts -- the Buildkite equivalent of GitHub-hosted ubuntu-latest.
    return {
        "docker#v5.2.0": {
            "image": image,
            "always-pull": True,
            "propagate-environment": True,
            "environment": [
                "http_proxy",
                "https_proxy",
                "HTTP_PROXY",
                "HTTPS_PROXY",
            ],
        }
    }


def docker_plugin(*, image: str = IMAGE) -> dict[str, Any]:
    # In-image CPU runner for the unit/utils suite (needs megatron + torch from
    # the CUDA image). GPU work goes through h100_k8s_plugin instead.
    config: dict[str, Any] = {
        "image": image,
        "always-pull": True,
        "propagate-environment": True,
        "network": "host",
        "ipc": "host",
        "shm-size": "16g",
        "environment": [
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "WANDB_API_KEY",
            "HF_TOKEN",
            f"HF_HOME={HF_CACHE}",
        ],
        "volumes": [
            f"{HF_CACHE}:{HF_CACHE}",
        ],
    }
    return {"docker#v5.2.0": config}


def h100_k8s_plugin(job: TestJob) -> dict[str, Any]:
    env = [
        {"name": "HF_HOME", "value": "/root/.cache/huggingface"},
        {"name": "VIME_TEST_USE_DEEPEP", "value": job.use_deepep},
        {"name": "VIME_TEST_USE_FP8_ROLLOUT", "value": job.use_fp8_rollout},
        {"name": "VIME_TEST_ENABLE_EVAL", "value": job.enable_eval},
        {"name": "TEST_FILE", "value": job.test_file},
        {"name": "TEST_ARGS", "value": job.test_args},
        {"name": "NUM_GPUS", "value": str(job.num_gpus)},
        {"name": "TOTAL_GPUS", "value": "8"},
        {
            "name": "HF_TOKEN",
            "valueFrom": {
                "secretKeyRef": {
                    "name": "hf-token-secret",
                    "key": "token",
                }
            },
        },
        {
            # Optional: if `wandb-api-key-secret` is absent the var is simply
            # left unset (wandb runs unauthenticated) instead of failing the pod.
            "name": "WANDB_API_KEY",
            "valueFrom": {
                "secretKeyRef": {
                    "name": "wandb-api-key-secret",
                    "key": "api-key",
                    "optional": True,
                }
            },
        },
    ]

    return {
        "kubernetes": {
            "podSpec": {
                "containers": [
                    {
                        "image": job.image,
                        "imagePullPolicy": "Always",
                        "resources": {"limits": {"nvidia.com/gpu": job.num_gpus}},
                        "volumeMounts": [
                            {"name": "devshm", "mountPath": "/dev/shm"},
                            {"name": "hf-cache", "mountPath": "/root/.cache/huggingface"},
                            {"name": "vime-cache", "mountPath": "/data/vime_ci"},
                            {"name": "vime-models", "mountPath": "/root/models"},
                            {"name": "vime-datasets", "mountPath": "/root/datasets"},
                        ],
                        "env": env,
                    }
                ],
                "nodeSelector": {"node.kubernetes.io/instance-type": "gpu-h100-sxm"},
                "volumes": [
                    {"name": "devshm", "emptyDir": {"medium": "Memory"}},
                    {
                        "name": "hf-cache",
                        "hostPath": {"path": K8S_CACHE, "type": "DirectoryOrCreate"},
                    },
                    {
                        "name": "vime-cache",
                        "hostPath": {"path": f"{K8S_CACHE}/vime", "type": "DirectoryOrCreate"},
                    },
                    {
                        "name": "vime-models",
                        "hostPath": {"path": f"{K8S_CACHE}/vime/models", "type": "DirectoryOrCreate"},
                    },
                    {
                        "name": "vime-datasets",
                        "hostPath": {"path": f"{K8S_CACHE}/vime/datasets", "type": "DirectoryOrCreate"},
                    },
                ],
            }
        }
    }


def agent_queue(num_gpus: int) -> str:
    if num_gpus == 0:
        return CPU_QUEUE
    # 4- and 8-GPU jobs both run on the H100 pool, sized per-pod.
    return H100_QUEUE


def retry_agent_lost() -> dict[str, Any]:
    return {
        "automatic": [
            {"exit_status": -1, "limit": 1},
            {"exit_status": -10, "limit": 1},
        ]
    }


def display_name(job: TestJob) -> str:
    name = job.test_file.removeprefix("tests/")
    flags = []
    if job.test_args:
        flags.append(job.test_args)
    if job.use_deepep == "1":
        flags.append("deepep")
    if job.use_fp8_rollout == "1":
        flags.append("fp8-rollout")
    if job.enable_eval == "0":
        flags.append("no-eval")
    if flags:
        return f"{name} ({', '.join(flags)})"
    return name


def step_key(suite: str, job: TestJob) -> str:
    raw = "-".join(
        [
            suite,
            job.test_file,
            job.test_args,
            f"deepep-{job.use_deepep}",
            f"fp8-{job.use_fp8_rollout}",
            f"eval-{job.enable_eval}",
        ]
    )
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-").lower()[:180]


def changed_test_files() -> list[str]:
    base_branch = os.environ.get("BUILDKITE_PULL_REQUEST_BASE_BRANCH", "main")
    commit = os.environ.get("BUILDKITE_COMMIT", "HEAD")
    base_ref = resolve_base_ref(base_branch)
    if not base_ref:
        return []

    proc = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=AM",
            f"{base_ref}...{commit}",
            "--",
            "tests/test_*.py",
            "tests/plugin_contracts/test_*.py",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line]


def resolve_base_ref(base_branch: str) -> str | None:
    candidates = [f"origin/{base_branch}", base_branch]
    for candidate in candidates:
        if rev_parse(candidate):
            return candidate

    subprocess.run(["git", "fetch", "--depth=200", "origin", base_branch], check=False)
    for candidate in candidates:
        if rev_parse(candidate):
            return candidate
    return None


def rev_parse(ref: str) -> bool:
    return subprocess.run(
        ["git", "rev-parse", "--verify", ref],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def parse_num_gpus(path: Path) -> int:
    if "plugin_contracts" in path.parts:
        default = 0
    else:
        default = 8

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return default

    match = re.search(r"^NUM_GPUS\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    if not match:
        return default
    return int(match.group(1))


if __name__ == "__main__":
    main()
