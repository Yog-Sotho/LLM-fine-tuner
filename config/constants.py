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

import warnings
import torch

warnings.filterwarnings("ignore")

# ── Column name constants ──────────────────────────────────────────────────
COL_INSTRUCTION = "instruction"
COL_OUTPUT      = "output"
COL_TEXT        = "text"
COL_PROMPT      = "prompt"
COL_CHOSEN      = "chosen"
COL_REJECTED    = "rejected"

# ── File extension constants ───────────────────────────────────────────────
FILE_EXT_CSV  = ".csv"
FILE_EXT_JSONL = ".jsonl"
FILE_EXT_JSON = ".json"
FILE_EXT_TXT  = ".txt"
FILE_EXT_XLSX = ".xlsx"
FILE_EXT_PDF  = ".pdf"

# ── GGUF quantisation presets ──────────────────────────────────────────────
GGUF_QUANT_PRESETS: dict[str, dict[str, str]] = {
    "q8_0":  {"desc": "Near-lossless (99% quality)",       "size": "~7 GB (7B)"},
    "q6_k":  {"desc": "Best balance — recommended default", "size": "~5.5 GB (7B)"},
    "q5_k_m":{"desc": "Good quality, smaller",             "size": "~4.7 GB (7B)"},
    "q4_k_m":{"desc": "Max compression",                   "size": "~4 GB (7B)"},
}

# ── QLoRA Enhanced configuration ──────────────────────────────────────────
# Used by train_model() (CUDA branch), run_ppo_v27(), and load_qlora_model_v27().
QLORA_ENHANCED_LORA_CONFIG: dict = {
    "r": 64,
    "lora_alpha": 128,
    "target_modules": [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
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
    "helpfulness", "accuracy", "coherence", "safety", "relevance",
]

# ── LoRA target module map ─────────────────────────────────────────────────
# Used by get_lora_targets() in core/hardware.py
LORA_TARGET_MAP: dict[str, list[str]] = {
    "gpt2":     ["c_attn"],
    "gpt_neo":  ["q_proj", "v_proj"],
    "opt":      ["q_proj", "v_proj"],
    "llama":    ["q_proj", "v_proj"],
    "mistral":  ["q_proj", "v_proj"],
    "pythia":   ["query_key_value"],
    "falcon":   ["query_key_value"],
    "tinyllama":["q_proj", "v_proj"],
    "default":  ["q_proj", "v_proj"],
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

# ── PyPDF2 (PDF ingestion) ─────────────────────────────────────────────────
try:
    import PyPDF2  # noqa: F401
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
    from unsloth import FastLanguageModel          # noqa: F401
    from unsloth import is_bfloat16_supported      # noqa: F401
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

# ── TRL core (DPO / SFT) ──────────────────────────────────────────────────
try:
    from trl import DPOTrainer, DPOConfig, SFTTrainer, SFTConfig  # noqa: F401
    HAS_TRL = True
except ImportError:
    HAS_TRL = False

# ── TRL RewardTrainer ─────────────────────────────────────────────────────
try:
    from trl import RewardTrainer, RewardConfig  # noqa: F401
    HAS_REWARD_TRAINER = True
except ImportError:
    HAS_REWARD_TRAINER = False

# ── TRL PPO ───────────────────────────────────────────────────────────────
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead  # noqa: F401
    HAS_PPO = True
except ImportError:
    HAS_PPO = False

# ── TRL ORPO ──────────────────────────────────────────────────────────────
try:
    from trl import ORPOTrainer, ORPOConfig  # noqa: F401
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
        sentence_bleu, corpus_bleu, SmoothingFunction,
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
