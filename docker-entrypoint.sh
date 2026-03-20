#!/usr/bin/env bash
# =============================================================================
# 🧠 LLM Fine-Tuner v3.2 — Docker Entrypoint
# =============================================================================
#
# Behaviour:
#   • No arguments  → Launch Gradio UI on port 7860
#   • Any arguments → Pass directly to the CLI (main.py)
#
# Environment variables (all optional, can be set via docker run -e or compose):
#   HF_TOKEN          — HuggingFace API token (for Hub push / gated models)
#   SHARE             — Set to "true" to get a public Gradio share URL
#   EXTRA_ARGS        — Additional args appended to UI launch (e.g. --auth user:pass)
#
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

log()  { echo -e "${CYAN}[entrypoint]${RESET} $*"; }
ok()   { echo -e "${GREEN}[entrypoint] ✅${RESET} $*"; }
warn() { echo -e "${YELLOW}[entrypoint] ⚠️ ${RESET} $*"; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║        🧠  LLM Fine-Tuner v3.2 — Docker Container        ║${RESET}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── HuggingFace token ─────────────────────────────────────────────────────────
if [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HuggingFace token detected — logging in"
    python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>/dev/null || \
        warn "HuggingFace login failed — token may be invalid"
else
    warn "HF_TOKEN not set. Hub push and gated models (Llama, Mistral) will require a token."
    warn "Set it with:  docker run -e HF_TOKEN=hf_xxx ..."
fi

# ── CUDA / hardware check ─────────────────────────────────────────────────────
log "Checking hardware..."
python3 - <<'PYCHECK'
import torch, psutil, os

gpu = "No GPU detected"
vram = ""
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"

ram = psutil.virtual_memory().total / 1e9
print(f"  GPU:  {gpu}{vram}")
print(f"  RAM:  {ram:.1f} GB")
print(f"  PyTorch: {torch.__version__}")

# Warn if no GPU
if not torch.cuda.is_available():
    print("\n  ⚠️  No NVIDIA GPU found — running in CPU mode.")
    print("  ⚠️  Training large models will be very slow.")
    print("  ⚠️  Recommended: stick to gpt2 or distilgpt2 in CPU mode.")
PYCHECK

echo ""

# ── Volume check ──────────────────────────────────────────────────────────────
for dir in /app/cache/huggingface /app/data /app/models /app/outputs; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        warn "Created missing directory: $dir"
    fi
done

# ── Dispatch ──────────────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    # ── UI mode ───────────────────────────────────────────────────────────────
    log "No arguments — launching Gradio UI on port 7860"

    SHARE_FLAG=""
    if [[ "${SHARE:-false}" == "true" ]]; then
        SHARE_FLAG="--share"
        ok "Share mode enabled — a public URL will be printed below"
    fi

    ok "UI will be available at:  http://localhost:7860"
    echo ""

    exec python3 main.py ${SHARE_FLAG} ${EXTRA_ARGS:-}
else
    # ── CLI mode ──────────────────────────────────────────────────────────────
    log "Arguments detected — running CLI: $*"
    echo ""
    exec python3 main.py "$@"
fi
