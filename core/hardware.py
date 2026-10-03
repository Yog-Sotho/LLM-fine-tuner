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

import os

import torch

from config.constants import (
    HAS_BERTSCORE,
    HAS_EVALUATE,
    HAS_GRPO,
    HAS_HUB,
    HAS_KTO,
    HAS_LIGER,
    HAS_NLPAUG,
    HAS_NLTK,
    HAS_OPENPYXL,
    HAS_ORPO,
    HAS_PDF,
    HAS_PSUTIL,
    HAS_REWARD_TRAINER,
    HAS_TRL,
    HAS_UNSLOTH,
    HAS_VLLM,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    LORA_VARIANTS,
    MOE_EXPERT_FIELDS,
    MOE_ROUTER_MODULES,
    QLORA_ENHANCED_BNB_KWARGS,
)


def get_hardware_summary() -> str:
    """Return a multi-line string describing GPU, RAM and optional-dep status."""
    lines: list = []

    # GPU
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        # L6 FIX: binary GiB (1024**3), not decimal GB (1e9).
        vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        lines.append(f"- 🟢 **GPU:** {name} &nbsp;|&nbsp; **VRAM:** {vram_gib:.1f} GiB")
    else:
        lines.append("- 🟡 **GPU:** Not available &mdash; training will use CPU (slow)")

    # System RAM
    if HAS_PSUTIL:
        try:
            import psutil

            # L6 FIX: binary GiB.
            ram_gib = psutil.virtual_memory().total / (1024**3)
            lines.append(f"- 💾 **System RAM:** {ram_gib:.1f} GiB")
        except Exception:
            lines.append("- 💾 **System RAM:** Unavailable")
    else:
        lines.append("- 💾 **System RAM:** Install `psutil` to see this")

    # PyTorch version
    lines.append(f"- 🐍 **PyTorch:** {torch.__version__}")

    # Core optional deps
    deps = []
    deps.append("openpyxl ✓" if HAS_OPENPYXL else "openpyxl ✗ (no Excel)")
    deps.append("pypdf ✓" if HAS_PDF else "pypdf ✗ (no PDF)")
    deps.append("huggingface_hub ✓" if HAS_HUB else "huggingface_hub ✗ (no Hub push)")
    deps.append("psutil ✓" if HAS_PSUTIL else "psutil ✗")
    deps.append("unsloth ✓" if HAS_UNSLOTH else "unsloth ✗ (install for 2-5× speed)")
    deps.append("trl ✓ (DPO + SFT ready)" if HAS_TRL else "trl ✗ (pip install trl for DPO)")
    lines.append("- 📦 **Optional deps:** " + " &nbsp;|&nbsp; ".join(deps))

    # RLHF / eval deps
    v27 = []
    v27.append("Reward model ✓" if HAS_REWARD_TRAINER else "Reward model ✗")
    v27.append("GRPO ✓" if HAS_GRPO else "GRPO ✗")
    v27.append("KTO ✓" if HAS_KTO else "KTO ✗")
    v27.append("Liger ✓" if HAS_LIGER else "Liger ✗ (optional, CUDA)")
    v27.append("ORPO ✓" if HAS_ORPO else "ORPO ✗")
    v27.append("evaluate ✓" if HAS_EVALUATE else "evaluate ✗")
    v27.append("bert_score ✓" if HAS_BERTSCORE else "bert_score ✗")
    v27.append("nltk ✓" if HAS_NLTK else "nltk ✗")
    v27.append("nlpaug ✓" if HAS_NLPAUG else "nlpaug ✗")
    v27.append("vLLM ✓ (cached)" if HAS_VLLM else "vLLM ✗")
    lines.append("- 🧩 **Training & eval:** " + " &nbsp;|&nbsp; ".join(v27))

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
    vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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
        "gpt2-xl": ("1.5B", "6 GiB"),
        "gpt2-large": ("774M", "3 GiB"),
        "gpt2-medium": ("355M", "1.5 GiB"),
        "gpt2": ("124M", "0.5 GiB"),
        "distilgpt2": ("82M", "0.3 GiB"),
        "opt-125m": ("125M", "0.5 GiB"),
        "opt-350m": ("350M", "1.4 GiB"),
        "opt-1.3b": ("1.3B", "2.7 GiB"),
        "pythia-70m": ("70M", "0.3 GiB"),
        "pythia-160m": ("160M", "0.6 GiB"),
        "tinyllama": ("1.1B", "2.2 GiB"),
        "llama-2-7b": ("7B", "14 GiB"),
        "mistral-7b": ("7B", "14 GiB"),
        "llama-2-13b": ("13B", "26 GiB"),
    }
    for key, (params, mem) in table.items():
        if key in m:
            return f" Parameters:  {params}  |   Estimated RAM/VRAM:  {mem} "
    return " Parameters:  unknown  |   Estimated RAM/VRAM:  unknown "


def get_lora_targets() -> str:
    """LoRA target modules: every linear layer of the transformer blocks ("all-linear")."""
    return LORA_TARGET_MODULES


def lora_variant_kwargs(variant: str) -> dict[str, bool]:
    """LoraConfig options for a LoRA variant (LoRA, rsLoRA or DoRA)."""
    if variant not in LORA_VARIANTS:
        raise ValueError(f"Unknown LoRA variant '{variant}'. Choose from: {list(LORA_VARIANTS)}")
    return dict(LORA_VARIANTS[variant])


def is_unsloth_supported(model_name: str) -> bool:
    """Return True if Unsloth natively supports this model family."""
    supported = ["llama", "mistral", "gemma", "qwen", "phi", "tinyllama", "opt"]
    return any(s in model_name.lower() for s in supported)


def select_precision(device: str) -> dict[str, bool]:
    """Mixed-precision flags for TrainingArguments: bf16 where the GPU supports it, else fp16.

    CPU training runs in full precision (both False).
    """
    if device != "cuda" or not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    if torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": True}


def compute_dtype(device: str) -> torch.dtype:
    """Weight/compute dtype matching ``select_precision`` (so AMP and model dtype agree)."""
    precision = select_precision(device)
    if precision["bf16"]:
        return torch.bfloat16
    if precision["fp16"]:
        return torch.float16
    return torch.float32


def training_device_args(device: str) -> dict[str, bool]:
    """TrainingArguments for the device: precision, plus ``use_cpu`` on CPU.

    ``use_cpu`` also lets a multi-process CPU launch (torchrun) train data-parallel
    (gloo); on GPUs accelerate detects multi-GPU launches by itself.
    """
    return {**select_precision(device), "use_cpu": device != "cuda"}


def world_size() -> int:
    """Number of training processes (``torchrun`` / ``accelerate launch`` set WORLD_SIZE)."""
    return int(os.environ.get("WORLD_SIZE") or 1)


def is_main_process() -> bool:
    """True in single-process runs and on rank 0 — the process that saves outputs."""
    return int(os.environ.get("RANK") or 0) == 0


def quantized_device_map():
    """``device_map`` for 4-bit (QLoRA) loads.

    One process: ``"auto"``. Multi-process (data-parallel) training: this process's
    own GPU — ``"auto"`` would spread every copy of the model over all GPUs.
    """
    return {"": int(os.environ.get("LOCAL_RANK") or 0)} if world_size() > 1 else "auto"


def nf4_quantization_config(device: str, quant_storage: torch.dtype | None = None):
    """QLoRA's 4-bit config: NF4 + double quantisation, computing in the training precision.

    ``quant_storage``: dtype the packed 4-bit weights are stored in; sharded QLoRA
    (FSDP / ZeRO-3) needs it to equal the model dtype (``sharded_quant_storage``).
    """
    from transformers import BitsAndBytesConfig  # lazy: keeps transformers out of startup

    kwargs = {**QLORA_ENHANCED_BNB_KWARGS, "bnb_4bit_compute_dtype": compute_dtype(device)}
    if quant_storage is not None:
        kwargs["bnb_4bit_quant_storage"] = quant_storage
    return BitsAndBytesConfig(**kwargs)


def sharding_backend() -> str | None:
    """``"fsdp"`` / ``"deepspeed"`` when started by ``accelerate launch`` with such a config.

    The launcher sets the variables on CPU too, but accelerate only shards on GPUs (on
    CPU it runs plain data-parallel), so no CUDA means no sharding.
    """
    if not torch.cuda.is_available():
        return None
    if os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true":
        return "fsdp"
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true":
        return "deepspeed"
    return None


def sharding_unsupported(trainer: str) -> str | None:
    """Error for trainers that only support single-GPU / data-parallel runs."""
    if backend := sharding_backend():
        return (
            f"❌ {trainer} runs on one GPU or data-parallel (DDP) only, not with "
            f"{backend.upper()} sharding. Sharded training is available for `train` (SFT/DPO); "
            "launch this with configs/accelerate/multi_gpu.yaml instead."
        )
    return None


def sharded_quant_storage(device: str) -> torch.dtype:
    """4-bit storage dtype for sharded QLoRA (FSDP / ZeRO-3): the model's dtype.

    bf16 when training in bf16; float32 for fp16 mixed precision (PEFT's guidance).
    """
    return torch.bfloat16 if select_precision(device)["bf16"] else torch.float32


def moe_expert_count(model) -> int:
    """Experts per MoE layer (0 for dense models)."""
    config = getattr(model, "config", None)
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    for field in MOE_EXPERT_FIELDS:
        value = getattr(text_config, field, None)
        if isinstance(value, int) and value > 1:
            return value
    return 0


def lora_dropout(model) -> float:
    """LoRA dropout for ``model``: 0 for MoE models, else LORA_DROPOUT.

    PEFT (0.18+, Transformers 5) adapts fused MoE expert weights through a parameter
    wrapper that refuses any dropout ("ParamWrapper does not work with lora_dropout != 0").
    """
    return 0.0 if moe_expert_count(model) else LORA_DROPOUT


def setup_moe(model, router_aux_loss: bool = True, freeze_router: bool = True) -> dict | None:
    """Prepare a (PEFT-wrapped or full) MoE model for fine-tuning; None for dense models.

    • ``router_aux_loss``: switches on the router load-balancing loss — TRL's SFT, ORPO,
      reward and GKD trainers add it only when ``output_router_logits`` is set, and most
      MoE configs ship with it off. Pass False for DPO, KTO and GRPO: with the config
      flag, their training fails (checkpoint recompute / generation shape errors, TRL
      0.29–1.14); current TRL's DPO adds the loss itself.
    • ``freeze_router``: freezes adapter weights on the routers (LoRA's B starts at zero,
      so routing stays as pretrained); training the router risks expert collapse. PEFT's
      Transformers-5 MoE conversion ignores ``exclude_modules``, hence freezing. Pass
      False for DPO and KTO: a frozen router with gradient checkpointing fails there
      ("Recomputed values … have different metadata").
    Each trainer passes the combination verified on TRL 0.29 and 1.14 (tests/test_smoke_training).
    Full fine-tuning still trains the router, balanced by the auxiliary loss.
    """
    experts = moe_expert_count(model)
    if not experts:
        return None
    if router_aux_loss:
        for config in {id(c): c for c in (model.config, model.config.get_text_config())}.values():
            config.output_router_logits = True
    router_trained = False
    for name, param in model.named_parameters():
        if not any(part in MOE_ROUTER_MODULES for part in name.split(".")):
            continue
        if freeze_router and any(key in name for key in ("lora_", "ia3_")):
            param.requires_grad = False
        router_trained = router_trained or param.requires_grad
    return {
        "experts": experts,
        "router_aux_loss": router_aux_loss,
        "router_trained": router_trained,
    }


def full_finetune_dtype(device: str) -> torch.dtype:
    """Weight dtype for full fine-tuning (every weight trained, no quantisation).

    bf16 where supported; otherwise fp32 — fp16 mixed precision needs fp32 master
    weights (the Trainer refuses to unscale fp16 gradients).
    """
    return torch.bfloat16 if select_precision(device)["bf16"] else torch.float32
