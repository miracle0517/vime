# Docker release rule

vime ships one image based on the official vllm image, published as
`aosheninferact/vime-vllm:cu129` (also `latest`).

Base image: `vllm/vllm-openai:v0.21.0-cu129-ubuntu2404`.

Build locally:

```bash
docker build -f docker/Dockerfile -t vime/pr-9-vllm:cu129 .
```

## vLLM 0.21.0 patch: `/inference/v1/generate` + `routed_experts`

Upstream v0.21.0 already records MoE routing when `--enable-return-routed-experts`
is set, and exposes it on `/v1/completions` as:

- top-level `prompt_routed_experts` — prompt token routing
- `choices[].routed_experts` — generated token routing

The tokens API `/inference/v1/generate` did not wire these fields into the HTTP
response. `docker/patch/${PATCH_VERSION}/vllm.patch` backports that wiring (see
`vllm/entrypoints/serve/disagg/protocol.py` and `serving.py`).

### Verify inside the image

Start vLLM with routing enabled, then:

```bash
curl -s http://127.0.0.1:8000/inference/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL",
    "token_ids": [1, 2, 3],
    "sampling_params": {"max_tokens": 8, "temperature": 0.0},
    "stream": false
  }' | python3 -c "
import json, sys
d = json.load(sys.stdin)
c = d['choices'][0]
pre = d.get('prompt_routed_experts')
gen = c.get('routed_experts')
print('prompt_routed_experts rows:', len(pre) if pre else None)
print('choices[0].routed_experts rows:', len(gen) if gen else None)
assert pre is not None and gen is not None, 'missing routed experts fields'
"
```

Expected: both `prompt_routed_experts` and `choices[0].routed_experts` are
non-null nested lists. For SGLang-style `(len(tokens)-1, L, K)` merge in slime,
concatenate prompt + gen rows in rollout (step 2).

Container内临时打补丁与验证见：仓库根目录 `run_script/README.md`（A100 `vime2`，工作区 `/data/nfs_87/xky/new_rl`）。

## Release matrix

Before tagging a new stable image, the following matrix must pass. All four
are currently TODO — none has been wired into CI yet:

- [ ] Qwen3-4B sync
- [ ] Qwen3-4B async
- [ ] Qwen3-30B-A3B sync
- [ ] Qwen3-30B-A3B fp8 sync
