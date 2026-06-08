# Tau bench
This example shows vime training in an agentic multi-turn tool use environment.


## Environment Setup
Install vime and tau-bench dependencies:

```bash
cd /root/
git clone https://github.com/vllm-project/vime.git
cd vime
pip install -e . --no-deps --no-build-isolation
# for tau bench
cd /root/
git clone https://github.com/JD-ETH/tau-bench.git tau-bench-src
cd tau-bench-src
git checkout feature/litellm-retry
pip install -e . --no-deps
pip install litellm
```

Use the following script to generate mock data for vime training.

```bash
cd /root/vime/examples/tau-bench
python tau1_mock.py --local_dir /root/tau-bench/
```

Initialize the Qwen3-4B-Instruct-2507 model needed for tool use:

```bash
# hf checkpoint
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507

# mcore checkpoint
cd /root/vime
source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B-Instruct-2507 \
    --save /root/Qwen3-4B-Instruct-2507_torch_dist
```

## Running the Script

Configure user simulation for tau-bench (defaults are applied in `generate_with_tau._ensure_tau_args`):

```python
# Defaults (override via train.py CLI flags):
#   tau_env=retail
#   tau_user_model=openai/local-qwen3-4b
#   tau_user_model_provider=openai
#   tau_task_split=train
#   tau_user_strategy=llm
#   max_turns=10
#
# Gemini user simulator example:
#   export GEMINI_API_KEY="YOUR KEY"
#   --tau-user-model gemini-2.0-flash-lite --tau-user-model-provider gemini
#
# Local mock user (no external API), set in run_qwen3_4B.sh:
#   export TAU_BENCH_MOCK=1
```

Rollout uses vLLM (`/v1/chat/completions/render` + `/inference/v1/generate`).
Tool calls are parsed locally via `vllm_tool_parser.py` and `openai_tool_adapter.py`.

And run:


```bash
cd /root/vime
bash examples/tau-bench/run_qwen3_4B.sh
```
