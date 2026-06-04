# vime Buildkite CI

vime PR CI is split into two layers:

- **Always-on lightweight checks** run automatically on every PR.
- **GPU end-to-end suites** are label-gated and run only when the matching
  `run-ci-*` label is added.

The default PR path is intentionally cheap; GPU validation is explicit and
targeted. (This is the Buildkite port of `.github/workflows/pr-test.yml`.)

## Trigger model

| GitHub Actions concept | Buildkite equivalent |
|---|---|
| `pull_request` (`synchronize`, `labeled`) on `main` | Pipeline GitHub settings: build PRs, and rebuild on label change |
| Add a `run-ci-*` label | Normal PR path — gates the matching suite via `build.pull_request.labels includes "..."` |
| `workflow_dispatch` (manual "Run workflow", debug/override) | Manual **New Build** with `RUN_CI_*` / `RUN_ALL` env vars (broader than one label) |

Adding a label such as `run-ci-short` is the normal way to request a GPU suite
on a PR. A manual build with env overrides is the debug/override path.

## Always-on jobs (every PR, no label)

| Job | Runner | Image | Runs |
|---|---|---|---|
| `Pre-commit` | CPU (`small_cpu_queue_premerge`) | `python:3.10` (lightweight) | `pre-commit run --all-files` |
| `Plugin Contracts` | CPU (`small_cpu_queue_premerge`) | `python:3.10` (lightweight, **0 GPU**) | each contract file as `python tests/<file>.py` (installs CPU deps first) |
| `Unit & utils tests` | CPU (`small_cpu_queue_premerge`) | `inferactinc/public:vime-vllm-cu129-latest` (**in-image**) | `python -m pytest tests/unit tests/utils` |

Plugin contracts are the *lightweight* CPU path (no CUDA image), matching
GitHub-hosted `ubuntu-latest`. Unit/utils run **in the CI image** because the
`tests/unit/backends/megatron_utils` tests need megatron + torch at import time.

## Label-gated GPU suites

Run only when the PR carries the label (or via a manual build with the env
override). The static matrix lives in `.buildkite/scripts/upload_suite.py`.

| Label / env override | Suite | GPU | Matrix |
|---|---|---:|---:|
| `run-ci-short` / `RUN_CI_SHORT=1` | Short | 4 | 4 |
| `run-ci-vllm-config` / `RUN_CI_VLLM_CONFIG=1` | vLLM config | 8 | 4 |
| `run-ci-megatron` / `RUN_CI_MEGATRON=1` | Megatron e2e | 8 | 13 |
| `run-ci-precision` / `RUN_CI_PRECISION=1` | Precision/parallel check | 8 | 1 |
| `run-ci-ckpt` / `RUN_CI_CKPT=1` | Checkpoint (incl. `--async-save`) | 8 | 2 |
| `run-ci-image` / `RUN_CI_IMAGE=1` | Image-validation subset | 4 or 8 | 13 |
| `run-ci-changed` / `RUN_CI_CHANGED=1` | Changed test files | dynamic | dynamic |
| `run-ci-all` / `RUN_ALL=1` (or `VIME_RUN_ALL=1`) | All GPU suites | — | — |

All GPU jobs run on the shared H100 Kubernetes pool (`mithril-h100-pool`); each
job gets its own pod sized to its GPU count (`nvidia.com/gpu: <num_gpus>`, up to
8), and `tests/ci/gpu_lock_exec.py` locks the pod's visible GPUs.

### `run-ci-changed` behavior

File-based (not dependency-based). It diffs the PR branch against `origin/main`
with `git diff --diff-filter=AM` and matches only added/modified
`tests/test_*.py` and `tests/plugin_contracts/test_*.py`. For each file it reads
the top-level `NUM_GPUS = <N>` (default `8`); `0` runs CPU-only with no GPU lock,
otherwise it runs via `gpu_lock_exec.py --count <N>`. Changing *application* code
does not infer which suites to run — add the relevant fixed `run-ci-*` label for
that.

## Pipeline mechanics

Buildkite loads `.buildkite/pipeline.yml` (two-document layout). Document 1 runs
`scripts/upload_pipeline_with_skip_ci.sh`, which resolves docs-only skip-CI and
substitutes the `__UPLOAD_*_IF__` placeholders in document 2 with the trigger
conditions above, then uploads them. Each upload step calls
`scripts/upload_suite.py <suite>` to emit that suite's steps;
`scripts/run_test.sh` installs vime editable and runs each test.

## Cluster prerequisites

- Nodes able to pull `inferactinc/public:vime-vllm-cu129-latest` (GPU/unit jobs)
  and `python:3.10` (lightweight CPU jobs).
- Secret `hf-token-secret` (key `token`) for gated HF model downloads.
- Optional secret `wandb-api-key-secret` (key `api-key`) for wandb logging; if
  absent, wandb runs unauthenticated.
- vime models/datasets reachable inside the pod at `/root/models` and
  `/root/datasets` (hostPath under `/mnt/hf-cache/vime` on the H100 nodes), or
  resolvable via HF using `HF_HOME=/root/.cache/huggingface`.
