# Profiling

In vime, you can profile the **rollout (vLLM inference)** path in detail using vLLM's profiling HTTP API. Profiling targets the vLLM engine side, not the Megatron training side.

Typical flow:

- Start train (`sleep_rollout` + `vllm-profiler-config`)
- Wait until vLLM engines and the router are ready
- Read router/worker addresses from logs
- `start_profile`
- Send a few inference requests
- (Optional) `stop_profile`; or traces flush automatically when `max_iterations` is reached
- Inspect trace files under `torch_profiler_dir`



## 1. Put Rollout into a Wait State (`sleep_rollout`)

For flexible stress testing and profiling, rollout usually waits after initialization instead of generating immediately.

Replace `rollout_function_path` in `train.py` startup args—no code changes required:

```bash
python train.py \
    --rollout-function-path vime.rollout.sleep_rollout.sleep \
    ... (other arguments)
```

This puts the rollout process in an infinite wait loop so you can send HTTP requests or run stress tools manually.

## 2. Enable the vLLM Profiler (at train startup)

vLLM registers `/start_profile` and `/stop_profile` only when started with `--profiler-config`. In vime, pass **`--vllm-profiler-config`** through to the `vllm serve` subprocess.

### 2.1 Pass the full config as JSON

```bash
--vllm-profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/logs/vllm_profile","max_iterations":3,"ignore_frontend":true}'
```

Common JSON fields:

| Field | Description |
|------|------|
| `profiler` | `"torch"` or `"cuda"` |
| `torch_profiler_dir` | Trace output directory (absolute path) |
| `max_iterations` | Worker auto-stops and flushes after more than N steps (condition is `> N`) |
| `ignore_frontend` | Recommended `true`: profile workers only, lower frontend overhead |

**Avoid RPC timeout on `stop_profile`:** vLLM APIServer talks to EngineCore/workers over internal RPC. Manually calling `stop_profile` to flush traces can take minutes, while the default `VLLM_RPC_TIMEOUT` is only **10 seconds** (10000 ms), which can interrupt flush or leave traces incomplete. For profiling, set **30 minutes** (1800000 ms).

Set this variable **before starting train and launching vLLM**, in the Ray worker environment (a local shell `export` may not reach the Ray job). Pass it via `runtime-env-json` on `ray job submit`, for example:

```bash
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"VLLM_RPC_TIMEOUT\": \"${VLLM_RPC_TIMEOUT}\"
  }
}"

ray job submit --address=\"http://127.0.0.1:8265\" \
  --runtime-env-json=\"${RUNTIME_ENV_JSON}\" \
  -- python3 train.py \
  ... \
  --vllm-profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/root/logs/vllm_profile\",...}'
```


### 2.2 Verify it took effect

After train starts, confirm all three in logs (missing any means the profiler is not enabled correctly):

1. **Args parsed**: `vllm_profiler_config ... profiler='torch'` (and `torch_profiler_dir` path).
2. **Forwarded to vLLM subprocess**: `Launching vLLM server: ... --profiler-config {"profiler":"torch",...}`.
3. **HTTP routes registered**: vLLM startup route list includes `/start_profile` and `/stop_profile` (otherwise `POST /start_profile` returns 404).

## 3. Get Router and Worker Addresses

vLLM engines (workers) register on the vllm-router. Example startup log:

```text
Router launched at 127.0.0.1:3521, Prometheus port: 4153
Ports for engine 0: {'host': '127.0.0.1', 'port': 15000, ...}
Starting vLLM server on http://127.0.0.1:15000
```

**Note: the router port may change on every job** (random in 3000–4000 by default). Do not reuse the previous port. Verify with curl:

```bash
curl http://127.0.0.1:3521/workers
```

Returns each worker's `url` and `is_healthy`.

## 4. Use `tools/profile_rollout.py`

The script reads the router's `/workers` list and calls `/start_profile` or `/stop_profile` on every worker.

### Start Profiling

```bash
cd /root/vime
python tools/profile_rollout.py \
    --router-url http://127.0.0.1:3521 \
    --action start
```

### Stop Profiling (optional)

If `--vllm-profiler-config` sets `max_iterations`, the worker **auto-stops and flushes** after enough steps. In practice, traces often appear under `torch_profiler_dir` right after inference—you **do not** need to call `stop_profile` manually. Use this only to end collection early:

```bash
python tools/profile_rollout.py \
    --router-url http://127.0.0.1:3521 \
    --action stop
```

## 5. Send Inference Requests

While `sleep_rollout` is waiting:

1. `profile_rollout.py --action start`
2. Send a few completion requests to the router or **directly to a worker** (2–4 is enough; traces get large)
3. (Optional) `profile_rollout.py --action stop`; or wait for `max_iterations` to auto-flush
4. Inspect traces under `torch_profiler_dir`

Example request (`model` is the HF checkpoint path):

```bash
curl -X POST http://127.0.0.1:15000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/models/Qwen3-4B","prompt":"Hello","max_tokens":32}'
```


## 6. View Traces

### Perfetto

1. Open [https://ui.perfetto.dev/](https://ui.perfetto.dev/)
2. **Open trace file**, pick `*.trace.json.gz`
3. Inspect GPU kernels, CPU ops, and the timeline

### Chrome Tracing

Open `chrome://tracing` in the browser and **Load** a trace file.

### Analysis Tool

```bash
cd /root/vime
python tools/analyze_profile.py --profile-dir /root/logs/vllm_profile --all-ranks
```


## 7. Troubleshooting

| Symptom | Fix |
|------|------|
| `POST /start_profile` 404 | Pass `--vllm-profiler-config` as JSON; restart the job |
| Start OK but empty output dir | Confirm curl hits a worker and returns 200; increase `max_iterations` or send more requests |
| Router 503 | Confirm the current job's router port; connect directly to a worker |
| Slow or timed-out stop | Increase `VLLM_RPC_TIMEOUT`; reduce request count |
