#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE_YML="${ROOT}/${1:-.buildkite/pipeline.yml}"

is_docs_only_change() {
  local file_path
  local has_any=0

  while IFS= read -r file_path; do
    [[ -z "${file_path}" ]] && continue
    has_any=1

    if [[ "${file_path}" == docs/* ]]; then
      continue
    fi
    if [[ "${file_path}" == *.md ]]; then
      continue
    fi
    if [[ "${file_path}" == "README_zh.md" ]]; then
      continue
    fi
    return 1
  done

  [[ "${has_any}" -eq 1 ]]
}

changed_files() {
  local base_branch base_ref

  if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]]; then
    base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"
    if ! git rev-parse --verify "origin/${base_branch}" >/dev/null 2>&1; then
      git fetch --depth=200 origin "${base_branch}" >/dev/null 2>&1 || true
    fi

    if git rev-parse --verify "origin/${base_branch}" >/dev/null 2>&1; then
      base_ref="origin/${base_branch}"
    elif git rev-parse --verify "${base_branch}" >/dev/null 2>&1; then
      base_ref="${base_branch}"
    else
      return 1
    fi

    git diff --name-only "${base_ref}...${BUILDKITE_COMMIT}"
    return
  fi

  if [[ "${BUILDKITE_BRANCH:-}" == "main" ]] && git rev-parse --verify "${BUILDKITE_COMMIT}^" >/dev/null 2>&1; then
    git diff --name-only "${BUILDKITE_COMMIT}^..${BUILDKITE_COMMIT}"
    return
  fi

  return 1
}

skip_ci=0
if files="$(changed_files 2>/dev/null)" && is_docs_only_change <<< "${files}"; then
  skip_ci=1
  buildkite-agent annotate ":memo: vime CI skipped - docs-only change" --style info || true
fi

if [[ ! -f "${PIPELINE_YML}" ]]; then
  echo "Missing ${PIPELINE_YML}" >&2
  exit 1
fi

export PIPELINE_YML skip_ci
python3 <<'PY' | buildkite-agent pipeline upload
import os
from pathlib import Path

text = Path(os.environ["PIPELINE_YML"]).read_text(encoding="utf-8")
sep = "\n---\n"
_, continuation = text.split(sep, 1) if sep in text else ("", text)

run_all = (
    'build.env("RUN_ALL") == "1" || '
    'build.env("VIME_RUN_ALL") == "1" || '
    '(build.branch != "main" && build.pull_request.labels includes "run-ci-all")'
)

conditions = {
    "__UPLOAD_CORE_IF__": "'true'",
    "__UPLOAD_SHORT_IF__": f"'({run_all}) || build.env(\"RUN_CI_SHORT\") == \"1\" || (build.branch != \"main\" && build.pull_request.labels includes \"run-ci-short\")'",
    "__UPLOAD_VLLM_CONFIG_IF__": f"'({run_all}) || build.env(\"RUN_CI_VLLM_CONFIG\") == \"1\" || (build.branch != \"main\" && build.pull_request.labels includes \"run-ci-vllm-config\")'",
    "__UPLOAD_MEGATRON_IF__": f"'({run_all}) || build.env(\"RUN_CI_MEGATRON\") == \"1\" || (build.branch != \"main\" && build.pull_request.labels includes \"run-ci-megatron\")'",
    "__UPLOAD_PRECISION_IF__": f"'({run_all}) || build.env(\"RUN_CI_PRECISION\") == \"1\" || (build.branch != \"main\" && build.pull_request.labels includes \"run-ci-precision\")'",
    "__UPLOAD_CKPT_IF__": f"'({run_all}) || build.env(\"RUN_CI_CKPT\") == \"1\" || (build.branch != \"main\" && build.pull_request.labels includes \"run-ci-ckpt\")'",
    "__UPLOAD_IMAGE_IF__": f"'({run_all}) || build.env(\"RUN_CI_IMAGE\") == \"1\" || (build.branch != \"main\" && build.pull_request.labels includes \"run-ci-image\")'",
    "__UPLOAD_CHANGED_IF__": f"'build.env(\"RUN_CI_CHANGED\") == \"1\" || (build.branch != \"main\" && build.pull_request.labels includes \"run-ci-changed\")'",
}

if os.environ.get("skip_ci") == "1":
    conditions = {key: "'false'" for key in conditions}

for key, value in conditions.items():
    continuation = continuation.replace(key, value)

print(continuation, end="")
PY
