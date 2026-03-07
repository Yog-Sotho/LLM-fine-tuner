"""
core/hardware.py
=================
Layer 1 — hardware introspection and model-selection helpers.
Imports: config.constants only (+ stdlib + torch).
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
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        lines.append(f"🟢  GPU:  {name}  |  VRAM: {vram:.1f} GB ")
    else:
        lines.append("🟡  GPU:  Not available — training will use CPU (slow) ")

    # System RAM
    if HAS_PSUTIL:
        try:
            import psutil
            ram = psutil.virtual_memory().total / 1e9
            lines.append(f"💾  System RAM:  {ram:.1f} GB ")
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
    """Return the largest model ID that comfortably fits in available VRAM."""
    if not torch.cuda.is_available():
        return "gpt2"
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram < 4:
        return "gpt2"
    elif vram < 8:
        return "facebook/opt-350m"
    elif vram < 16:
        return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    else:
        return "mistralai/Mistral-7B-v0.1"


def get_model_info(model_id: str) -> str:
    """Return a short parameter-count / VRAM-estimate string for known models."""
    m = model_id.lower()
    table = {
        "gpt2-xl":     ("1.5B",  "6 GB"),
        "gpt2-large":  ("774M",  "3 GB"),
        "gpt2-medium": ("355M",  "1.5 GB"),
        "gpt2":        ("124M",  "0.5 GB"),
        "distilgpt2":  ("82M",   "0.3 GB"),
        "opt-125m":    ("125M",  "0.5 GB"),
        "opt-350m":    ("350M",  "1.4 GB"),
        "opt-1.3b":    ("1.3B",  "2.7 GB"),
        "pythia-70m":  ("70M",   "0.3 GB"),
        "pythia-160m": ("160M",  "0.6 GB"),
        "tinyllama":   ("1.1B",  "2.2 GB"),
        "llama-2-7b":  ("7B",    "14 GB"),
        "mistral-7b":  ("7B",    "14 GB"),
        "llama-2-13b": ("13B",   "26 GB"),
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
