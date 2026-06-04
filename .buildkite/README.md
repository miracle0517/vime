# vime Buildkite CI

This pipeline follows the lightweight `vllm-omni` pattern: Buildkite loads
`.buildkite/pipeline.yml`, a bootstrap step resolves docs-only skip-ci, then
label-gated upload steps add the actual test suites.

Always-on gates:

- `Pre-commit`
- `Plugin Contracts`
- `Unit & utils tests`

PR labels and matching environment overrides:

- `run-ci-short` / `RUN_CI_SHORT=1`
- `run-ci-vllm-config` / `RUN_CI_VLLM_CONFIG=1`
- `run-ci-megatron` / `RUN_CI_MEGATRON=1`
- `run-ci-precision` / `RUN_CI_PRECISION=1`
- `run-ci-ckpt` / `RUN_CI_CKPT=1`
- `run-ci-image` / `RUN_CI_IMAGE=1`
- `run-ci-changed` / `RUN_CI_CHANGED=1`
- `run-ci-all` / `RUN_ALL=1` or `VIME_RUN_ALL=1`

The static test matrix is defined in `.buildkite/scripts/upload_suite.py`.

All GPU jobs (4- and 8-GPU) run on the shared H100 Kubernetes pool
(`mithril-h100-pool`); each job gets its own pod sized to its GPU count
(`nvidia.com/gpu: <num_gpus>`, up to 8), and `tests/ci/gpu_lock_exec.py` locks
the pod's visible GPUs. CPU jobs (pre-commit / plugin-contracts / unit) run the
CI image on `small_cpu_queue_premerge` via the docker plugin.

Cluster prerequisites:

- Nodes must be able to pull the public image
  `inferactinc/public:vime-vllm-cu129-latest`.
- Secret `hf-token-secret` (key `token`) for gated HF model downloads.
- Optional secret `wandb-api-key-secret` (key `api-key`) for wandb logging; if
  absent, wandb runs unauthenticated.
- vime models/datasets must be reachable inside the pod at `/root/models` and
  `/root/datasets` (hostPath under `/mnt/hf-cache/vime` on the H100 nodes), or
  resolvable via HF using `HF_HOME=/root/.cache/huggingface`.
