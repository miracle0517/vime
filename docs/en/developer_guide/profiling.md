# Profiling

In vime, we can perform detailed performance analysis of the rollout process using the profiling interface provided by vLLM.

## 1. Sleeping the Rollout Process

For more flexible stress testing and profiling, it is often useful to make the vime rollout process enter a waiting state after initialization, instead of starting generation immediately.

You can achieve this by replacing the `rollout_function_path` in your startup arguments without modifying the source code:

```bash
python train.py \
    --rollout-function-path vime.rollout.sleep_rollout.sleep \
    ... (other arguments)
```

This function will make the rollout process enter an infinite wait loop, allowing you to manually send requests or run stress testing tools.

## 2. Enabling the vLLM Profiler

vLLM only registers the `/start_profile` and `/stop_profile` endpoints when started with a profiler config. In vime, pass it through to the `vllm serve` subprocess with `--vllm-profiler-config`:

```bash
--vllm-profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/logs/vllm_profile","max_iterations":3,"ignore_frontend":true}'
```

**Key Fields:**
* `profiler`: `"torch"` or `"cuda"`.
* `torch_profiler_dir`: Trace output directory (absolute path).
* `max_iterations`: Worker auto-stops and flushes after this many steps.
* `ignore_frontend`: Recommended `true`; profile workers only.

## 3. Obtaining vLLM Engine List

vLLM engines (workers) are registered with the router. You can retrieve the list of all active engines by accessing the `/workers` endpoint of the router.

The router address is typically printed in the startup logs:
```
Router launched at 127.0.0.1:3000
```

You can use `curl` to view the workers:
```bash
curl http://127.0.0.1:3000/workers
```

## 4. Using Automated Profiling Tool

To simplify profiling across multiple engines simultaneously, we provide an automated script: `tools/profile_rollout.py`.

### Starting Profiling

By default, this tool starts profiling on all workers:

```bash
python tools/profile_rollout.py --router-url http://127.0.0.1:3000 --action start
```

### Stopping Profiling Manually

If you set `max_iterations`, the worker auto-stops and flushes. To stop early:

```bash
python tools/profile_rollout.py --router-url http://127.0.0.1:3000 --action stop
```

## 5. Running Stress Tests

While the Rollout process is in a waiting state via `sleep_rollout`, you can:
1. Start profiling using `tools/profile_rollout.py`.
2. Use stress testing tools to send requests to the router or directly to the engines.
3. Wait for profiling to complete (if `max_iterations` was set) or stop it manually.
4. Collect the `.json` trace files from the `torch_profiler_dir` and view them using `chrome://tracing` in Chrome or [Perfetto](https://ui.perfetto.dev/).
