#!/usr/bin/env bash
set -euo pipefail

: "${TEST_FILE:?TEST_FILE is required}"

NUM_GPUS="${NUM_GPUS:-0}"
TEST_ARGS="${TEST_ARGS:-}"
TOTAL_GPUS="${TOTAL_GPUS:-${NUM_GPUS}}"

if [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" || -z "${BUILDKITE_PULL_REQUEST:-}" ]]; then
  pr_suffix="non-pr"
else
  pr_suffix="${BUILDKITE_PULL_REQUEST}"
fi

export GITHUB_COMMIT_NAME="${GITHUB_COMMIT_NAME:-${BUILDKITE_COMMIT:-local}_${pr_suffix}}"
export VIME_TEST_ENABLE_INFINITE_RUN="${VIME_TEST_ENABLE_INFINITE_RUN:-false}"
export VIME_TEST_USE_DEEPEP="${VIME_TEST_USE_DEEPEP:-0}"
export VIME_TEST_USE_FP8_ROLLOUT="${VIME_TEST_USE_FP8_ROLLOUT:-0}"
export VIME_TEST_ENABLE_EVAL="${VIME_TEST_ENABLE_EVAL:-1}"

# Lightweight CPU runner (plugin contracts) starts from a bare python image, so
# install the CPU wheel set first. Mirrors the cpu path in pr-test.yml.
if [[ "${VIME_INSTALL_CPU_DEPS:-0}" == "1" ]]; then
  echo "--- :python: Install CPU test deps"
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  python -m pip install pytest numpy packaging pyyaml omegaconf tqdm httpx pybase64 pylatexenc sympy aiohttp pillow
fi

echo "--- :python: Install vime editable"
python -m pip install -e . --no-deps --break-system-packages || python -m pip install -e . --no-deps

test_path="${TEST_FILE}"
if [[ "${test_path}" != tests/* ]]; then
  test_path="tests/${test_path}"
fi

if [[ -n "${TEST_ARGS}" ]]; then
  read -r -a test_args_array < <(printf "%s\n" "${TEST_ARGS}")
else
  test_args_array=()
fi

echo "--- :test_tube: Run ${test_path}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "VIME_TEST_USE_DEEPEP=${VIME_TEST_USE_DEEPEP}"
echo "VIME_TEST_USE_FP8_ROLLOUT=${VIME_TEST_USE_FP8_ROLLOUT}"
echo "VIME_TEST_ENABLE_EVAL=${VIME_TEST_ENABLE_EVAL}"

if [[ "${NUM_GPUS}" == "0" ]]; then
  if [[ "${test_path}" == *.sh ]]; then
    bash "${test_path}" "${test_args_array[@]}"
  else
    python "${test_path}" "${test_args_array[@]}"
  fi
else
  if [[ "${test_path}" == *.sh ]]; then
    python tests/ci/gpu_lock_exec.py --count "${NUM_GPUS}" --total-gpus "${TOTAL_GPUS}" -- bash "${test_path}" "${test_args_array[@]}"
  else
    python tests/ci/gpu_lock_exec.py --count "${NUM_GPUS}" --total-gpus "${TOTAL_GPUS}" -- python "${test_path}" "${test_args_array[@]}"
  fi
fi
