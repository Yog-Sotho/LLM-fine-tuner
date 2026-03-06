#!/usr/bin/env bash
set -euo pipefail


# =============================================================================
# 🧠 LLM Fine-Tuner Installation Script (2026 Edition)
# =============================================================================
# Fully addressed every bug: no bc, proper CUDA parsing, trap cleanup,
# script directory anchoring, dependency arrays, robust launcher, etc.
# =============================================================================


print_step()    { echo -e "\n\033[1;34m==>\033[0m \033[1m$1\033[0m"; }
print_success() { echo -e "\033[1;32m✔\033[0m $1"; }
print_warning() { echo -e "\033[1;33m⚠\033[0m $1"; }
print_error()   { echo -e "\033[1;31m✘\033[0m $1"; exit 1; }


# ----------------------------------------------------------------------------
# Anchor to script directory (fixes relative path issues)
# ----------------------------------------------------------------------------
cd "\( (dirname " \){BASH_SOURCE[0]}")" || print_error "Failed to cd to script directory"
PROJECT_ROOT="$(pwd)"


# ----------------------------------------------------------------------------
# Trap for clean interrupt
# ----------------------------------------------------------------------------
cleanup() {
    print_warning "Installation interrupted. Cleaning up..."
    [[ -d "$VENV_DIR" ]] && rm -rf "$VENV_DIR"
    exit 1
}
trap cleanup INT TERM


# ----------------------------------------------------------------------------
# Non-interactive mode
# ----------------------------------------------------------------------------
NON_INTERACTIVE=false
[[ "${AUTO_INSTALL:-false}" == "true" || "$1" == "--yes" || "$1" == "-y" ]] && NON_INTERACTIVE=true


ask() {
    if [[ "$NON_INTERACTIVE" == true ]]; then
        echo -e "\033[1;33m→\033[0m Auto-accepting: $1"
        return 0
    fi
    read -p "$1 (y/N) " -n 1 -r
    echo
    [[ \( REPLY =\~ ^[Yy] \) ]]
}


# ----------------------------------------------------------------------------
# Prerequisites (pure Bash/Python – no bc)
# ----------------------------------------------------------------------------
print_step "Checking prerequisites"


command -v python3 >/dev/null || print_error "Python 3 is not installed."
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=${PY_VERSION%%.*}
PY_MINOR=${PY_VERSION#*.}
if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 10 ]]; }; then
    print_error "Python 3.10+ required (found $PY_VERSION)"
fi
print_success "Python $PY_VERSION found"


command -v pip3 >/dev/null || print_error "pip3 is not installed."


# Optional build tools warning
for tool in git cmake make; do
    command -v "$tool" >/dev/null || print_warning "Build tool '$tool' missing (sudo apt install build-essential cmake git)"
done


# ----------------------------------------------------------------------------
# CUDA Detection (clean & reliable)
# ----------------------------------------------------------------------------
print_step "Detecting hardware"


CUDA_AVAILABLE=0
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
CUDA_VERSION_FULL="none"


if command -v nvidia-smi >/dev/null; then
    CUDA_VERSION_FULL=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' || echo "0")
    CUDA_MAJOR=${CUDA_VERSION_FULL%%.*}
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)


    print_success "GPU detected: $GPU_NAME (CUDA $CUDA_VERSION_FULL)"


    if [[ "$CUDA_MAJOR" -ge 12 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu126"
    elif [[ "$CUDA_MAJOR" -eq 11 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu117"
    fi
    CUDA_AVAILABLE=1
else
    print_warning "No NVIDIA GPU → CPU-only mode"
fi


# ----------------------------------------------------------------------------
# Virtual Environment
# ----------------------------------------------------------------------------
print_step "Creating virtual environment"
VENV_DIR="$PROJECT_ROOT/llm_finetuner_env"


if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists"
    if ask "Delete and recreate?"; then
        rm -rf "$VENV_DIR"
    else
        print_error "Aborted."
    fi
fi


python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
print_success "Virtual environment created & activated"


# ----------------------------------------------------------------------------
# Base tools
# ----------------------------------------------------------------------------
print_step "Upgrading pip & build tools"
pip install --upgrade pip setuptools wheel


# ----------------------------------------------------------------------------
# PyTorch
# ----------------------------------------------------------------------------
print_step "Installing PyTorch"
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
print_success "PyTorch installed (${TORCH_INDEX##*/})"


# ----------------------------------------------------------------------------
# Dependency arrays (clean & maintainable)
# ----------------------------------------------------------------------------
CORE_DEPS=(
    transformers datasets accelerate peft bitsandbytes trl
    gradio typer pandas numpy matplotlib tqdm huggingface-hub
    safetensors einops hf_transfer
)


print_step "Installing core dependencies"
pip install "${CORE_DEPS[@]}"


# Flash Attention 2
print_step "Installing Flash Attention 2 (huge speed win)"
pip install flash-attn --no-build-isolation || print_warning "Flash Attention 2 optional – continuing"


# ----------------------------------------------------------------------------
# Optional packages (grouped)
# ----------------------------------------------------------------------------
print_step "Optional high-value dependencies"


ask "Install Unsloth (4–5× faster training)?" && {
    pip install unsloth
    print_success "Unsloth installed"
}


ask "Install vLLM (high-throughput inference)?" && {
    pip install vllm
    print_success "vLLM installed"
}


ask "Install quantization tools (AutoGPTQ + exllamav2)?" && {
    pip install auto-gptq exllamav2
    print_success "Quantization tools installed"
}


ask "Install evaluation & data tools (nltk, rouge, bert-score, nlpaug, PDF/Excel)?" && {
    pip install nltk rouge-score bert-score evaluate nlpaug PyPDF2 openpyxl
    python -c "import nltk; nltk.download('punkt', quiet=True)"
    print_success "Evaluation & data tools installed"
}


pip install psutil wandb   # always useful


# ----------------------------------------------------------------------------
# llama.cpp (GGUF export)
# ----------------------------------------------------------------------------
print_step "llama.cpp (GGUF export)"
if ask "Clone & build llama.cpp with CUDA support?"; then
    if [ ! -d "llama.cpp" ]; then
        git clone https://github.com/ggerganov/llama.cpp.git
        cd llama.cpp
        if [ "$CUDA_AVAILABLE" -eq 1 ]; then
            make LLAMA_CUDA=1 -j"$(nproc || echo 4)"
        else
            make -j"$(nproc || echo 4)"
        fi
        cd ..
        print_success "llama.cpp built"
    else
        print_warning "llama.cpp already exists – skipping"
    fi
fi


# ----------------------------------------------------------------------------
# Launcher (absolute paths + HF transfer)
# ----------------------------------------------------------------------------
print_step "Creating launcher"
SCRIPT_PATH="$PROJECT_ROOT/LLM_fine_tuner_v3.2.py"
LAUNCHER="$VENV_DIR/bin/llm-finetune"


if [ -f "$SCRIPT_PATH" ]; then
    chmod +x "$SCRIPT_PATH"
    cat > "$LAUNCHER" <<EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="\$PATH:$PROJECT_ROOT/llama.cpp"
python "$SCRIPT_PATH" "\$@"
EOF
    chmod +x "$LAUNCHER"
    print_success "Launcher created → $LAUNCHER"
else
    print_warning "Main script not found at $SCRIPT_PATH"
fi


# ----------------------------------------------------------------------------
# Final message
# ----------------------------------------------------------------------------
print_step "Installation complete – LLM Fine-Tuner v5.0"
echo ""
echo -e "\033[1;32m🎉 Ready to fine-tune!\033[0m"
echo ""
echo "   Activate:   source $VENV_DIR/bin/activate"
echo "   Run UI:     llm-finetune"
echo "   CLI help:   llm-finetune --help"
echo ""
echo "Next time use: ./install.sh --yes   or   AUTO_INSTALL=true ./install.sh"
echo "Happy training – may your loss curves always go down! 🧠"