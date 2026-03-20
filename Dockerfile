# =============================================================================
# 🧠 LLM Fine-Tuner v3.2 — GPU Dockerfile (CUDA 12.6 / cuDNN 9)
# =============================================================================
#
# Multi-stage build:
#   Stage 1 (builder) — compile llama.cpp with CUDA support + install all deps
#   Stage 2 (runtime) — lean final image, copies from builder
#
# Requirements:
#   • Docker Engine ≥ 24
#   • NVIDIA Container Toolkit  (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
#   • NVIDIA driver ≥ 525 (CUDA 12.x capable)
#
# Build:
#   docker build -t llm-fine-tuner:gpu .
#
# Run:
#   docker run --gpus all -p 7860:7860 llm-fine-tuner:gpu
#
# =============================================================================

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04 AS builder

LABEL maintainer="Yog-Sotho"
LABEL description="LLM Fine-Tuner v3.2 build stage"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        git \
        wget \
        curl \
        build-essential \
        cmake \
        ninja-build \
        libopenblas-dev \
        # CUDA build tools needed for flash-attn & llama.cpp
        cuda-nvcc-12-6 \
        libcublas-dev-12-6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

# ── pip & build tools ─────────────────────────────────────────────────────────
RUN python3 -m pip install --upgrade pip setuptools wheel

# ── PyTorch (CUDA 12.6 wheel) ─────────────────────────────────────────────────
RUN pip install \
        torch==2.5.1+cu126 \
        torchvision==0.20.1+cu126 \
        torchaudio==2.5.1+cu126 \
        --index-url https://download.pytorch.org/whl/cu126

# ── Core dependencies ─────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
# Strip optional heavy packages that require special build steps
# (flash-attn, vllm, auto-gptq, exllamav2 are installed separately below)
RUN grep -vE "^(flash-attn|vllm|auto-gptq|exllamav2|#)" /tmp/requirements.txt \
    | pip install -r /dev/stdin

# ── Flash Attention 2 (requires CUDA headers, built here in builder) ──────────
# Non-fatal: some older driver versions can't build it
RUN pip install flash-attn --no-build-isolation || \
    echo "⚠️  Flash Attention 2 build failed — continuing without it"

# ── Unsloth (must come AFTER torch is present) ────────────────────────────────
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
        --no-deps || \
    echo "⚠️  Unsloth install failed — continuing without it"

# ── llama.cpp (GGUF export fallback) ─────────────────────────────────────────
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp && \
    cmake /opt/llama.cpp -B /opt/llama.cpp/build \
        -DLLAMA_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja && \
    cmake --build /opt/llama.cpp/build --config Release -j$(nproc) && \
    echo "✅ llama.cpp built with CUDA support"

# ── NLTK data (downloaded once at build time) ─────────────────────────────────
RUN python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04 AS runtime

LABEL maintainer="Yog-Sotho"
LABEL description="LLM Fine-Tuner v3.2 — GPU runtime"
LABEL version="3.2"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # HuggingFace
    HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # llama.cpp on PATH
    PATH="/opt/llama.cpp/build/bin:${PATH}" \
    # Gradio
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# ── Minimal runtime system packages ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip \
        libopenblas0 \
        git \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

# ── Copy Python packages from builder ─────────────────────────────────────────
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/nltk_data /root/nltk_data

# ── Copy llama.cpp binaries ───────────────────────────────────────────────────
COPY --from=builder /opt/llama.cpp/build/bin /opt/llama.cpp/build/bin

# ── Application code ──────────────────────────────────────────────────────────
WORKDIR /app
COPY . /app/

# ── Directory structure for persistent volumes ────────────────────────────────
RUN mkdir -p \
        /app/cache/huggingface \
        /app/data \
        /app/models \
        /app/outputs

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 7860

# Volumes that users should mount for persistence
VOLUME ["/app/cache/huggingface", "/app/data", "/app/models", "/app/outputs"]

ENTRYPOINT ["/docker-entrypoint.sh"]
# Default: launch Gradio UI. Pass CLI args to override:
#   docker run ... train --model gpt2 --data /app/data/train.csv
CMD []
