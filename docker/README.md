# Docker release rule

We will publish 2 kinds of docker images:
1. stable version, which is based on an official vLLM release.
2. latest version, which aligns to the current vLLM-based Dockerfile.

current stable version is:
- vLLM v0.21.0, megatron dev 3714d81d418c9f1bca4594fc35f9e8289f652862

The command to build:

```bash
just release
```

Before each update, we will test the following models with 64xH100:

- Qwen3-4B sync
- Qwen3-4B async
- Qwen3-30B-A3B sync
- Qwen3-30B-A3B fp8 sync
- GLM-4.5-355B-A32B sync
