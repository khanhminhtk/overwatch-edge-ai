# syntax=docker/dockerfile:1.7

ARG TRITON_IMAGE=nvcr.io/nvidia/tritonserver:24.07-py3-igpu
FROM ${TRITON_IMAGE}

WORKDIR /workspace

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_CACHE_DIR=/root/.cache/uv

COPY pyproject.toml uv.lock /workspace/
COPY deploy/docker/triton-detection-builder/build_detector_plan.sh /usr/local/bin/build_detector_plan.sh

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    apt-get update \
    && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libxcb1 \
      libsm6 \
      libxext6 \
      libxrender1 \
    && pip3 install uv \
    && uv sync --frozen --no-dev --no-install-project \
    && chmod +x /usr/local/bin/build_detector_plan.sh \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/local/bin/build_detector_plan.sh"]
