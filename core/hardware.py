"""
core/hardware.py
=================
Layer 1 — hardware introspection and model-selection helpers.
Imports: config.constants only (+ stdlib + torch).

Fix log
-------
  M5 (Medium): auto_recommend_model steered users with 8–15 GB VRAM to
     TinyLlama-1.1B. In 2026 hardware terms, RTX 3060 (12 GB), RTX 4070
     (12 GB), and RTX 3080 (10 GB) all fell into that bucket. With QLoRA,
     Mistral-7B trains comfortably at 10–12 GB VRAM. Updated thresholds
     to reflect current hardware reality and added intermediate tiers.
  L6 (Low): VRAM and RAM were reported in decimal GB (`/ 1e9`) but OS,
     GPU drivers, and storage all report in binary GiB (`/ 1024**3`).
     A 12 GB card showed as "12.9 GB" with decimal division. Changed
     to `/ (1024 ** 3)` and updated display labels to "GiB".
"""

import torch

from config.constants import (
    HAS_PSUTIL,
    HAS_OPENPYXL,
    HAS_PDF,
    HAS_HUB,
    HAS_UNSLOTH,
    HAS_TRL,
    HAS_REWARD_TRAINER,
    HAS_PPO,
    HAS_ORPO,
    HAS_EVALUATE,
    HAS_BERTSCORE,
    HAS_NLTK,
    HAS_NLPAUG,
    HAS_VLLM,
    LORA_TARGET_MAP,
)


def get_hardware_summary() -> str:
    """Return a multi-line string describing GPU, RAM and optional-dep status."""
    lines: list = []

    # GPU
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        # L6 FIX: binary GiB (1024**3), not decimal GB (1e9).
        vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        lines.append(f"🟢  GPU:  {name}  |  VRAM: {vram_gib:.1f} GiB ")
    else:
        lines.append("🟡  GPU:  Not available — training will use CPU (slow) ")

    # System RAM
    if HAS_PSUTIL:
        try:
            import psutil
            # L6 FIX: binary GiB.
            ram_gib = psutil.virtual_memory().total / (1024 ** 3)
            lines.append(f"💾  System RAM:  {ram_gib:.1f} GiB ")
        except Exception:
            lines.append("💾  System RAM:  unavailable ")
    else:
        lines.append("💾  System RAM:  install `psutil` to see this ")

    # PyTorch version
    lines.append(f"🐍  PyTorch:  {torch.__version__} ")

    # Core optional deps
    deps = []
    deps.append("openpyxl ✓ "              if HAS_OPENPYXL else "openpyxl ✗ (no Excel) ")
    deps.append("PyPDF2 ✓ "               if HAS_PDF      else "PyPDF2 ✗ (no PDF) ")
    deps.append("huggingface_hub ✓ "      if HAS_HUB      else "huggingface_hub ✗ (no Hub push) ")
    deps.append("psutil ✓ "               if HAS_PSUTIL   else "psutil ✗ ")
    deps.append("unsloth ✓ "              if HAS_UNSLOTH  else "unsloth ✗ (install for 2-5× speed) ")
    deps.append("trl ✓ (DPO + SFT ready)" if HAS_TRL      else "trl ✗ (pip install trl for DPO) ")
    lines.append("📦  Optional deps: " + " | ".join(deps))

    # v2.7 RLHF / eval deps
    v27 = []
    v27.append("RewardTrainer ✓" if HAS_REWARD_TRAINER else "RewardTrainer ✗")
    v27.append("PPO ✓"           if HAS_PPO            else "PPO ✗")
    v27.append("ORPO ✓"          if HAS_ORPO           else "ORPO ✗")
    v27.append("evaluate ✓"      if HAS_EVALUATE       else "evaluate ✗")
    v27.append("bert_score ✓"    if HAS_BERTSCORE      else "bert_score ✗")
    v27.append("nltk ✓"          if HAS_NLTK           else "nltk ✗")
    v27.append("nlpaug ✓"        if HAS_NLPAUG         else "nlpaug ✗")
    v27.append("vLLM ✓ (cached)" if HAS_VLLM           else "vLLM ✗")
    lines.append("🆕  v2.7 deps: " + " | ".join(v27))

    return "\n".join(lines)


def auto_recommend_model() -> str:
    """Return the largest model ID that comfortably fits in available VRAM.

    M5 FIX: Thresholds updated for 2026 GPU landscape. A 10–12 GiB card
    (RTX 3060, RTX 4070, RTX 3080) can run Mistral-7B under QLoRA comfortably.
    The previous single threshold of 16 GiB for 7B models left these common
    GPUs using a 1.1B model unnecessarily.

    Tiers
    -----
      < 4 GiB   → gpt2 (124M)                  — CPU/integrated GPU
      4–7 GiB   → facebook/opt-350m             — older 6 GB cards
      8–11 GiB  → TinyLlama-1.1B                — 8 GB cards with headroom
      12–15 GiB → mistralai/Mistral-7B-v0.1     — QLoRA fits easily
      ≥ 16 GiB  → mistralai/Mistral-7B-v0.1     — full precision option
    """
    if not torch.cuda.is_available():
        return "gpt2"
    # L6 FIX: binary GiB for consistent comparison with driver-reported values.
    vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gib < 4:
        return "gpt2"
    elif vram_gib < 8:
        return "facebook/opt-350m"
    elif vram_gib < 12:
        return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    else:
        # M5 FIX: 12 GiB+ → Mistral-7B via QLoRA (was previously 16 GiB threshold).
        return "mistralai/Mistral-7B-v0.1"


def get_model_info(model_id: str) -> str:
    """Return a short parameter-count / VRAM-estimate string for known models."""
    m = model_id.lower()
    table = {
        "gpt2-xl":     ("1.5B",  "6 GiB"),
        "gpt2-large":  ("774M",  "3 GiB"),
        "gpt2-medium": ("355M",  "1.5 GiB"),
        "gpt2":        ("124M",  "0.5 GiB"),
        "distilgpt2":  ("82M",   "0.3 GiB"),
        "opt-125m":    ("125M",  "0.5 GiB"),
        "opt-350m":    ("350M",  "1.4 GiB"),
        "opt-1.3b":    ("1.3B",  "2.7 GiB"),
        "pythia-70m":  ("70M",   "0.3 GiB"),
        "pythia-160m": ("160M",  "0.6 GiB"),
        "tinyllama":   ("1.1B",  "2.2 GiB"),
        "llama-2-7b":  ("7B",    "14 GiB"),
        "mistral-7b":  ("7B",    "14 GiB"),
        "llama-2-13b": ("13B",   "26 GiB"),
    }
    for key, (params, mem) in table.items():
        if key in m:
            return f" Parameters:  {params}  |   Estimated RAM/VRAM:  {mem} "
    return " Parameters:  unknown  |   Estimated RAM/VRAM:  unknown "


def get_lora_targets(model_name: str) -> list:
    """Return the correct LoRA target module names for the given model family."""
    m = model_name.lower()
    for key, targets in LORA_TARGET_MAP.items():
        if key in m:
            return targets
    return LORA_TARGET_MAP["default"]


def is_unsloth_supported(model_name: str) -> bool:
    """Return True if Unsloth natively supports this model family."""
    supported = ["llama", "mistral", "gemma", "qwen", "phi", "tinyllama", "opt"]
    return any(s in model_name.lower() for s in supported)
