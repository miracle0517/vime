ARG BASE_IMAGE=vllm/vllm-openai:v0.21.0
FROM ${BASE_IMAGE}

# ======================================== Arguments =============================================

ARG VLLM_VERSION=0.21.0
ARG MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
ARG PATCH_VERSION=v0.5.9
ARG INSTALL_VLLM=0
ARG VLLM_PIP_INSTALL_ARGS=

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

WORKDIR /root

RUN apt-get update && \
    apt-get install -y --no-install-recommends git rsync dnsutils nvtop && \
    rm -rf /var/lib/apt/lists/*

# If the base image is not already vllm/vllm-openai:<version>, set
# INSTALL_VLLM=1 to install the requested wheel explicitly.
RUN if [ "${INSTALL_VLLM}" = "1" ]; then \
      python3 -m pip install --force-reinstall "vllm==${VLLM_VERSION}" ${VLLM_PIP_INSTALL_ARGS}; \
    fi

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
try:
    metadata.version("sglang")
except metadata.PackageNotFoundError:
    print("sglang not installed")
else:
    raise SystemExit("sglang should not be installed in the vLLM image")
PY

CMD ["/bin/bash"]
