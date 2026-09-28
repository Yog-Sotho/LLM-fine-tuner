"""
llm_fine_tuner/config/constants.py
===================================
Layer 0 — no internal project imports.

Contains:
  • Column-name constants
  • File-extension constants
  • Model / training configuration presets
  • ALL optional-dependency try/except guards (HAS_* flags)
  • Lazy-imported objects exposed only when their package is present

Rule: nothing in this file may import from any other llm_fine_tuner module.
"""

import importlib.util
import os
import shutil
import warnings

import torch

# H-2 FIX: Removed the previous global `warnings.filterwarnings("ignore")` call.
# That line silenced ALL Python warnings for the entire process, hiding critical
# deprecation notices from transformers/peft/trl that help users diagnose failures.
# We now apply only narrowly-scoped suppressions for known-noisy but harmless warnings.
warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
    module="huggingface_hub",
)
warnings.filterwarnings(
    "ignore",
    message=".*use_reentrant.*",
    category=UserWarning,
)

# ── Remote code execution opt-in ───────────────────────────────────────────
# Hub repos with custom modeling code run that Python on this machine when loaded
# with trust_remote_code enabled. Off unless the operator explicitly enables it.
ALLOW_REMOTE_CODE: bool = os.environ.get("ALLOW_REMOTE_CODE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}

# ── Reproducibility ────────────────────────────────────────────────────────
# UI training runs are saved as <RUNS_DIR>/<run name>/ so they persist and can be resumed.
RUNS_DIR: str = os.environ.get("LFT_RUNS_DIR", "runs")
RUN_CONFIG_FILENAME = "run_config.yaml"
RUN_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
DEFAULT_SEED = 42

# Experiment trackers usable as TrainingArguments.report_to — only those installed.
# (Trackio needs huggingface_hub>=1.10, i.e. Transformers 5; it is simply absent otherwise.)
TRACKING_BACKENDS: list[str] = ["none"] + [
    name
    for name, module in (
        ("trackio", "trackio"),
        ("wandb", "wandb"),
        ("mlflow", "mlflow"),
        ("tensorboard", "tensorboard"),
    )
    if importlib.util.find_spec(module) is not None
]
_env_report_to = os.environ.get("LFT_REPORT_TO", "none").strip().lower()
DEFAULT_REPORT_TO: str = _env_report_to if _env_report_to in TRACKING_BACKENDS else "none"

# ── Column name constants ──────────────────────────────────────────────────
COL_INSTRUCTION = "instruction"
COL_OUTPUT = "output"
COL_TEXT = "text"
COL_PROMPT = "prompt"
COL_CHOSEN = "chosen"
COL_REJECTED = "rejected"
COL_COMPLETION = "completion"  # TRL prompt-completion / KTO format
COL_LABEL = "label"  # KTO: True = desirable completion, False = undesirable
COL_REFERENCE = "reference"  # GRPO: expected answer used by the reference-match reward

# Prompt layout for instruction data when no chat template is used.
SFT_PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# ── File extension constants ───────────────────────────────────────────────
FILE_EXT_CSV = ".csv"
FILE_EXT_JSONL = ".jsonl"
FILE_EXT_JSON = ".json"
FILE_EXT_TXT = ".txt"
FILE_EXT_XLSX = ".xlsx"
FILE_EXT_PDF = ".pdf"

# ── GGUF quantisation presets ──────────────────────────────────────────────
GGUF_QUANT_PRESETS: dict[str, dict[str, str]] = {
    "q8_0": {"desc": "Near-lossless (99% quality)", "size": "~7 GB (7B)"},
    "q6_k": {"desc": "Best balance — recommended default", "size": "~5.5 GB (7B)"},
    "q5_k_m": {"desc": "Good quality, smaller", "size": "~4.7 GB (7B)"},
    "q4_k_m": {"desc": "Max compression", "size": "~4 GB (7B)"},
}

# ── QLoRA Enhanced configuration ──────────────────────────────────────────
# Used by train_model() (QLoRA Enhanced branch).
QLORA_ENHANCED_LORA_CONFIG: dict = {
    "r": 64,
    "lora_alpha": 128,
    "target_modules": [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "lora_dropout": 0.05,
    "bias": "none",
}

# NOTE: bnb_4bit_compute_dtype uses torch.bfloat16 as default; callers must
# override to torch.float16 when torch.cuda.is_bf16_supported() returns False.
QLORA_ENHANCED_BNB_KWARGS: dict = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_use_double_quant": True,
}

# ── vLLM / evaluation constants ───────────────────────────────────────────
VLLM_QUANT_OPTIONS: list[str] = ["none", "awq", "gptq", "bnb"]
LLM_JUDGE_CRITERIA: list[str] = [
    "helpfulness",
    "accuracy",
    "coherence",
    "safety",
    "relevance",
]

# ── HuggingFace Hub constants ─────────────────────────────────────────────
# HuggingFace write tokens always start with this prefix and are >= 36 chars.
HF_TOKEN_PREFIX: str = "hf_"
HF_TOKEN_MIN_LEN: int = 36

# ── LoRA target module map ─────────────────────────────────────────────────
# Used by get_lora_targets() in core/hardware.py
LORA_TARGET_MAP: dict[str, list[str]] = {
    "gpt2": ["c_attn"],
    "gpt_neo": ["q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "llama": ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "pythia": ["query_key_value"],
    "falcon": ["query_key_value"],
    "tinyllama": ["q_proj", "v_proj"],
    "default": ["q_proj", "v_proj"],
}

# ══════════════════════════════════════════════════════════════════════════════
# Optional dependency guards — ALL HAS_* flags defined here.
# Every other module imports only these flags; they never re-run try/except.
# ══════════════════════════════════════════════════════════════════════════════

# ── openpyxl (Excel support) ──────────────────────────────────────────────
try:
    import openpyxl  # noqa: F401

    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

# ── pypdf (PDF ingestion) ──────────────────────────────────────────────────
try:
    import pypdf  # noqa: F401

    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# ── psutil (system RAM reporting) ─────────────────────────────────────────
try:
    import psutil  # noqa: F401

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── huggingface_hub (Hub push / registry) ─────────────────────────────────
try:
    from huggingface_hub import HfApi, create_repo  # noqa: F401

    HAS_HUB = True
except ImportError:
    HAS_HUB = False

# ── peft AdapterConfig (optional fork) ────────────────────────────────────
try:
    from peft import AdapterConfig  # noqa: F401

    HAS_ADAPTER_CONFIG = True
except ImportError:
    HAS_ADAPTER_CONFIG = False

# ── Unsloth (2-5× faster training + GGUF export) ──────────────────────────
try:
    from unsloth import (
        FastLanguageModel,  # noqa: F401
        is_bfloat16_supported,  # noqa: F401
    )

    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

# ── TRL core (DPO / SFT) ──────────────────────────────────────────────────
try:
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer  # noqa: F401

    HAS_TRL = True
except ImportError:
    HAS_TRL = False

# ── TRL RewardTrainer ─────────────────────────────────────────────────────
try:
    from trl import RewardConfig, RewardTrainer  # noqa: F401

    HAS_REWARD_TRAINER = True
except ImportError:
    HAS_REWARD_TRAINER = False

# ── TRL GRPO (online RL with reward functions; replaces PPO) ───────────────
try:
    from trl import GRPOConfig, GRPOTrainer  # noqa: F401

    HAS_GRPO = True
except ImportError:
    HAS_GRPO = False

# ── TRL KTO (top-level in TRL 0.x/1.x, trl.experimental.kto also in 1.x) ──
try:
    from trl import KTOConfig, KTOTrainer  # noqa: F401

    HAS_KTO = True
except ImportError:
    try:
        from trl.experimental.kto import KTOConfig, KTOTrainer  # noqa: F401

        HAS_KTO = True
    except ImportError:
        HAS_KTO = False

# ── Liger kernels (fused Triton kernels; CUDA only) ───────────────────────
# find_spec, not import: importing liger_kernel initialises Triton at startup.
HAS_LIGER: bool = importlib.util.find_spec("liger_kernel") is not None

# ── TRL ORPO (top-level in TRL 0.x, trl.experimental.orpo in TRL 1.x) ────
warnings.filterwarnings("ignore", message=".*importing from 'trl.experimental'.*")
try:
    from trl.experimental.orpo import ORPOConfig, ORPOTrainer  # noqa: F401

    HAS_ORPO = True
except ImportError:
    try:
        from trl import ORPOConfig, ORPOTrainer  # noqa: F401

        HAS_ORPO = True
    except ImportError:
        HAS_ORPO = False

# ── AutoGPTQ (GPTQ quantised inference) ───────────────────────────────────
try:
    from auto_gptq import AutoGPTQForCausalLM  # noqa: F401

    HAS_GPTQ = True
except ImportError:
    HAS_GPTQ = False

# ── ExLlamaV2 (EXL2 inference backend) ────────────────────────────────────
try:
    from exllamav2 import ExLlamaV2, ExLlamaV2Config  # noqa: F401

    HAS_EXLLAMA = True
except ImportError:
    HAS_EXLLAMA = False

# ── HuggingFace evaluate hub ──────────────────────────────────────────────
try:
    import evaluate as hf_evaluate  # noqa: F401

    HAS_EVALUATE = True
except ImportError:
    HAS_EVALUATE = False

# ── rouge-score ───────────────────────────────────────────────────────────
try:
    from rouge_score import rouge_scorer as rouge_scorer_lib  # noqa: F401

    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

# ── bert-score ────────────────────────────────────────────────────────────
try:
    from bert_score import score as bert_score_fn  # noqa: F401

    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

# ── NLTK + BLEU ───────────────────────────────────────────────────────────
try:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    from nltk.translate.bleu_score import (  # noqa: F401
        SmoothingFunction,
        corpus_bleu,
        sentence_bleu,
    )

    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# ── nlpaug (data augmentation) ────────────────────────────────────────────
try:
    import nlpaug.augmenter.word as naw  # noqa: F401

    HAS_NLPAUG = True
except ImportError:
    HAS_NLPAUG = False

# ── vLLM (high-throughput inference) ──────────────────────────────────────
try:
    from vllm import LLM, SamplingParams  # noqa: F401

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# ── heretic-llm (post-training restriction removal) ───────────────────────
# N-5 FIX: The previous implementation spawned a live subprocess
# (`_sp.run(["heretic", "--version"], ...)`) at every application startup
# to set this flag.  This had three problems:
#   1. Added ~5s latency on every cold start when heretic is missing.
#   2. If `_sp.run` itself raised (e.g. FileNotFoundError), the bare
#      `except Exception` swallowed it but `_sp` was already bound in the
#      module namespace — `del _sp` in the success branch never ran, leaving
#      a dangling `subprocess` reference exported as `_sp`.
#   3. The flag was never actually used anywhere in the codebase.
#
# Fix: use `shutil.which` — a pure-Python PATH scan, zero subprocesses,
# zero latency, zero leak risk.  HAS_HERETIC is now used in training/sft.py
# to guard the subprocess call and emit a clear message when the binary is
# absent rather than silently skipping Heretic Mode.
HAS_HERETIC: bool = shutil.which("heretic") is not None
