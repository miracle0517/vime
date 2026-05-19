ARG BASE_IMAGE=vllm/vllm-openai:v0.20.2
FROM ${BASE_IMAGE}

# ======================================== Arguments =============================================

ARG VLLM_VERSION=0.20.2
ARG MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
ARG PATCH_VERSION=v0.5.9
ARG INSTALL_VLLM=0
ARG VLLM_PIP_INSTALL_ARGS=
ARG ENABLE_CUDA_13=1
ARG SGLANG_VERSION=0.5.9

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

WORKDIR /root

RUN apt-get update && \
    apt-get install -y --no-install-recommends git rsync dnsutils nvtop \
      cmake ninja-build build-essential \
      cuda-nvrtc-dev-13-0 cuda-nvml-dev-13-0 cuda-profiler-api-13-0 cuda-nvtx-13-0 \
      libcusparse-dev-13-0 libcusolver-dev-13-0 libcufft-dev-13-0 libcurand-dev-13-0 \
      libcudnn9-dev-cuda-13 && \
    rm -rf /var/lib/apt/lists/*

# If the base image is not already vllm/vllm-openai:<version>, set
# INSTALL_VLLM=1 to install the requested wheel explicitly.
RUN if [ "${INSTALL_VLLM}" = "1" ]; then \
      python3 -m pip install --force-reinstall "vllm==${VLLM_VERSION}" ${VLLM_PIP_INSTALL_ARGS}; \
    fi

# TE does not have wheel on cuda 13 yet, thus need to install from source.
# nvidia-mathdx: latest is 25.6.0 on PyPI (26.6.0 doesn't exist).
# pybind11 + setuptools_scm: required by TE's pyproject.toml build under --no-build-isolation.
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      pip install --no-deps nvidia-mathdx pybind11 setuptools_scm && \
      pip -v install --no-build-isolation --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.10; \
    else \
      pip -v install --no-build-isolation --no-deps "transformer_engine[pytorch]==2.10.0"; \
    fi && \
    pip install --no-deps onnxscript onnx_ir onnx ml_dtypes

# Keep sglang installed (with --no-deps) so training-side imports keep working.
# IPython: sglang/utils.py unconditionally `from IPython.display import HTML, display`
#   on module load — without it, `import slime` fails.
# numpy<2: Megatron-LM rejects numpy 2.x (vllm/vllm-openai base ships 2.x).
RUN pip install --no-deps "sglang==${SGLANG_VERSION}" sglang-router==0.3.2 torch_memory_saver==0.0.9 && \
    pip install IPython && \
    pip install "numpy<2"

# mbridge + Megatron-Bridge + modelopt: required by slime's --megatron-to-hf-mode bridge
# path which loads HF checkpoints into Megatron actors. Without these, slime's
# checkpoint.py:_load_checkpoint_hf asserts `--megatron-to-hf-mode == "bridge"`
# but the bridge code fails to import.
RUN pip install --no-deps git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c && \
    pip install --no-deps --no-build-isolation git+https://github.com/radixark/Megatron-Bridge.git@bridge && \
    pip install --no-build-isolation "nvidia-modelopt[torch]>=0.37.0"

# flash-linear-attention (fla) provides ShortConvolution / FusedRMSNormGated /
# chunk_gated_delta_rule used by slime_plugins/models/qwen3_5.py and qwen3_next.py
# (linear-attention model variants). Some fla fused kernels dispatch through
# tilelang at runtime — install both. Tilelang has no cu13 wheel yet; cu128
# wheel is what upstream slime uses on CUDA 13 (runtime-compatible).
RUN pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/
RUN pip install flash-linear-attention==0.4.1

# Install slime runtime requirements while excluding SGLang packages. Future
# code paths should provide vLLM equivalents instead of importing SGLang.
COPY requirements.txt /tmp/slime-requirements.txt
RUN python3 - <<'PY' && \
    python3 -m pip install -r /tmp/slime-requirements-vllm.txt
from pathlib import Path

source = Path("/tmp/slime-requirements.txt")
target = Path("/tmp/slime-requirements-vllm.txt")
skip_prefixes = ("sglang", "sglang-router")

lines = []
for raw_line in source.read_text().splitlines():
    normalized = raw_line.split("#", 1)[0].strip().lower()
    if normalized.startswith(skip_prefixes):
        continue
    lines.append(raw_line)

target.write_text("\n".join(lines) + "\n")
PY

# slime's Megatron integration needs the local Megatron patch. This build does
# not apply docker/patch/*/sglang.patch.
RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive /root/Megatron-LM && \
    cd /root/Megatron-LM && \
    git checkout "${MEGATRON_COMMIT}"

COPY docker/patch/${PATCH_VERSION}/megatron.patch /root/Megatron-LM/megatron.patch
RUN cd /root/Megatron-LM && \
    git update-index --refresh && \
    git apply megatron.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Megatron patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
    rm megatron.patch && \
    python3 -m pip install -e . --no-deps

# megatron-bridge HEAD (qwen_provider.py) imports
# `get_transformer_block_with_experimental_attention_variant_spec` from
# Megatron-LM's experimental_attention_variant_module_specs, but the pinned
# MEGATRON_COMMIT doesn't define that name. We're not using experimental
# attention variants for the validated models (Qwen3-0.6B), so append a
# NotImplementedError stub so the import succeeds.
RUN python3 - <<'PY'
import importlib.util
spec = importlib.util.find_spec("megatron.core.models.gpt.experimental_attention_variant_module_specs")
path = spec.origin
with open(path, "r") as f: src = f.read()
if "get_transformer_block_with_experimental_attention_variant_spec" not in src:
    with open(path, "a") as f:
        f.write("""

# Stub: megatron-bridge expects this name. The pinned Megatron-LM commit doesn't
# provide it. Code paths that actually use experimental attention variants will
# fail at call time with a clear error.
def get_transformer_block_with_experimental_attention_variant_spec(*args, **kwargs):
    raise NotImplementedError(
        "get_transformer_block_with_experimental_attention_variant_spec is a no-op stub; "
        "the pinned Megatron-LM commit does not provide it. Bump MEGATRON_COMMIT to a "
        "release that defines this function if you need experimental attention variants."
    )
""")
    print("appended stub to", path)
else:
    print("stub already present at", path)
PY

ENV PYTHONPATH="/root/Megatron-LM:${PYTHONPATH}"

# Install slime itself without dependency resolution so the vLLM base image
# keeps control of CUDA/PyTorch/Triton/transformers compatibility.
COPY . /root/slime
RUN cd /root/slime && \
    python3 -m pip install -e . --no-deps

WORKDIR /root/slime

RUN python3 - <<'PY'
import importlib.metadata as metadata
import vllm

print("vllm import ok", getattr(vllm, "__version__", metadata.version("vllm")))
print("sglang version", metadata.version("sglang"))
print("transformer_engine version", metadata.version("transformer_engine"))
import sglang  # noqa: F401  (verify deep import works, not just dist version)
print("sglang import ok", sglang.__version__)
import sglang_router  # noqa: F401
print("sglang_router import ok", sglang_router.__version__)
from megatron.bridge import AutoBridge  # noqa: F401  (verify slime's bridge mode dep)
print("megatron.bridge import ok")
PY

CMD ["/bin/bash"]
