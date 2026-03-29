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
# C-3 FIX: Token is no longer passed as a -c argument (which exposed it in `ps aux`
# and /proc/<pid>/cmdline). It is now read from the environment inside the heredoc,
# keeping it out of the process command line entirely.
if [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HuggingFace token detected — logging in"
    python3 - <<'PYEOF' 2>/dev/null || warn "HuggingFace login failed — token may be invalid"
import os
from huggingface_hub import login
token = os.environ.get("HF_TOKEN", "")
if token:
    login(token=token)
PYEOF
else
    warn "HF_TOKEN not set. Hub push and gated models (Llama, Mistral) will require a token."
    warn "Set it with:  docker run -e HF_TOKEN=hf_xxx ..."
fi

# ── CUDA / hardware check ─────────────────────────────────────────────────────
log "Checking hardware..."
python3 - <<'PYCHECK'
# M-17 FIX: Wrap hardware check in try/except so a bad install doesn't crash the container.
try:
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
except ImportError as _e:
    print(f"  ⚠️  Hardware check skipped: {_e}")
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

    # L-8 FIX: EXTRA_ARGS is now split into a proper array to handle spaces in
    # argument values correctly (e.g. EXTRA_ARGS="--auth user:pass").
    # The previous unquoted ${EXTRA_ARGS:-} relied on word-splitting which is
    # fragile and can cause unexpected argument splitting.
    EXTRA_ARGS_ARRAY=()
    if [[ -n "${EXTRA_ARGS:-}" ]]; then
        # M-18 FIX: Use read -ra for proper shell word splitting instead of
        # unquoted variable expansion which breaks on arguments with spaces.
        read -ra EXTRA_ARGS_ARRAY <<<"${EXTRA_ARGS}"
    fi

    ok "UI will be available at:  http://localhost:7860"
    echo ""

    exec python3 main.py ${SHARE_FLAG} "${EXTRA_ARGS_ARRAY[@]+"${EXTRA_ARGS_ARRAY[@]}"}"
else
    # ── CLI mode ──────────────────────────────────────────────────────────────
    log "Arguments detected — running CLI: $*"
    echo ""
    exec python3 main.py "$@"
fi
