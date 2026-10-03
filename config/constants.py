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

# ── Version ────────────────────────────────────────────────────────────────
# The only place the version is written: pyproject.toml reads it from here (a plain
# string literal, so setuptools parses it without importing this module).
APP_VERSION = "3.2.0"
APP_NAME = f"LLM Fine-Tuner v{APP_VERSION}"

# ── Logging ────────────────────────────────────────────────────────────────
# Level for this app's own loggers (DEBUG, INFO, WARNING, ERROR); set by main.py.
LOG_LEVEL = os.environ.get("LFT_LOG_LEVEL", "INFO").strip().upper() or "INFO"

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
# Extra folders the web UI may read and write (os.pathsep-separated). Always allowed:
# the working directory, RUNS_DIR and the temp folder (uploads, exports).
# Left out of Hub uploads and download ZIPs: resume checkpoints (optimizer state, often
# several times the adapter size) and training_args.bin (a pickle). fnmatch patterns on
# paths relative to the model folder.
OUTPUT_EXCLUDE_PATTERNS: tuple[str, ...] = ("checkpoint-*", "training_args.bin")
EXTRA_ALLOWED_PATHS: list[str] = [
    p for p in os.environ.get("LFT_ALLOWED_PATHS", "").split(os.pathsep) if p.strip()
]
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
COL_MESSAGES = "messages"  # chat format: [{"role": ..., "content": ...}, ...]
COL_TOOLS = "tools"  # tool-calling chats: JSON schemas of the available functions
COL_IMAGES = "images"  # vision chats: images for the {"type": "image"} parts, in order
COL_IMAGE = "image"  # single-image variant, normalised to COL_IMAGES
COL_CONTEXT = "context"  # synthetic data: the document passage a question was written from
COL_ANCHOR = "anchor"  # embedding pairs: the query …
COL_POSITIVE = "positive"  # … the passage that answers it …
COL_NEGATIVE = "negative"  # … and (optional) a similar passage that does not
CHAT_COLUMNS = (COL_MESSAGES, COL_TOOLS, COL_IMAGES)  # columns kept for chat data
CHAT_ROLES = ("system", "user", "assistant", "tool")
# Extra message fields kept for training (rendered by the model's chat template):
# assistant reasoning (plus ``tool_calls``), and the tool name / call id of tool results.
CHAT_REASONING_KEYS = ("reasoning_content", "thinking")
CHAT_TOOL_KEYS = ("name", "tool_call_id")

# ── Hugging Face Hub datasets ─────────────────────────────────────────────
# IDs are owner/name only: load_dataset() would also read a *local* directory.
HUB_DATASET_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*"
HUB_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}"  # config and split names
# Model ids may be bare legacy names ("gpt2") as well as owner/name.
HUB_MODEL_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?"
# GRPO/KTO checkpoints (resume after a stop or crash); older ones are deleted.
CHECKPOINT_SAVE_STEPS = 50
CHECKPOINT_TOTAL_LIMIT = 2
HUB_DEFAULT_MAX_ROWS = 20_000  # rows are streamed, so only what is used is downloaded
HUB_MAX_ROWS_LIMIT = 1_000_000

DEFAULT_EVAL_SPLIT = 0.1  # fraction of rows held out for evaluation (0 = no eval)
TOKEN_STATS_SAMPLE = 2_000  # rows tokenised to estimate token lengths

# Prompt layout for instruction data when no chat template is used.
SFT_PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# ── File extension constants ───────────────────────────────────────────────
FILE_EXT_CSV = ".csv"
FILE_EXT_JSONL = ".jsonl"
FILE_EXT_JSON = ".json"
FILE_EXT_TXT = ".txt"
FILE_EXT_XLSX = ".xlsx"
FILE_EXT_PDF = ".pdf"
FILE_EXT_DOCX = ".docx"
FILE_EXT_MD = ".md"
# Documents that "Create training data" reads as plain text.
DOCUMENT_EXTENSIONS = (FILE_EXT_PDF, FILE_EXT_DOCX, FILE_EXT_TXT, FILE_EXT_MD)

# ── Training data from documents (question / answer pairs written by an LLM) ──
# Chunking follows Meta's synthetic-data-kit defaults (4000 characters, 200 overlap):
# enough context per question, overlap so facts on a boundary aren't lost.
SYNTH_CHUNK_CHARS = 4000
SYNTH_CHUNK_OVERLAP = 200
SYNTH_PAIRS_PER_CHUNK = 5
SYNTH_MAX_CHUNKS = 50  # bounds the LLM calls per run
# LLM-judge curation: pairs rated below this (1–10) are dropped; 0 = keep all.
SYNTH_CURATE_THRESHOLD = 7
SYNTH_MAX_NEW_TOKENS = 1024
SYNTH_WRITERS = ("OpenAI-compatible server", "Local model")

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
    "target_modules": "all-linear",
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
# vLLM `quantization` values ("bitsandbytes", not "bnb": vLLM rejects the short name).
VLLM_QUANT_OPTIONS: list[str] = ["none", "awq", "gptq", "bitsandbytes"]
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

# ── LoRA targets and variants ──────────────────────────────────────────────
# LoRA on every linear layer of the transformer blocks (attention + MLP; never the
# output head) — current practice, and PEFT resolves it for any architecture.
LORA_TARGET_MODULES = "all-linear"
# Unsloth takes explicit names (its documented recommendation).
UNSLOTH_LORA_TARGETS: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]  # fmt: skip
# LoraConfig options per variant. rsLoRA scales by alpha/sqrt(r) (stable at high
# rank); DoRA learns magnitude and direction separately (better at low rank, slower).
LORA_VARIANTS: dict[str, dict[str, bool]] = {
    "LoRA": {},
    "rsLoRA": {"use_rslora": True},
    "DoRA": {"use_dora": True},
}
DEFAULT_LORA_VARIANT = "LoRA"
# Adapter types (adapter_config.json "peft_type") whose weights can be merged into the
# base model for GGUF / quantized export. Prefix/prompt tuning add tokens, not weights.
MERGEABLE_PEFT_TYPES = ("LORA", "IA3")

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

# ── python-docx (Word documents for "Create training data") ───────────────
HAS_DOCX: bool = importlib.util.find_spec("docx") is not None

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

# ── Unsloth (2-5× faster training + GGUF export) ──────────────────────────
# Imported here, at startup, on purpose: Unsloth patches transformers / TRL / PEFT and
# warns that it "should be imported before trl, transformers, peft to ensure all
# optimizations are applied" — a lazy import would lose them. Only when installed.
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

# ── TRL GKD (knowledge distillation; trl.experimental.gkd in TRL 0.29 and 1.x) ──
try:
    from trl.experimental.gkd import GKDConfig, GKDTrainer  # noqa: F401

    HAS_GKD = True
except ImportError:
    HAS_GKD = False

# ── sentence-transformers (embedding-model fine-tuning) ───────────────────
# find_spec, not import: importing it loads scikit-learn and scipy at startup.
HAS_SENTENCE_TRANSFORMERS: bool = importlib.util.find_spec("sentence_transformers") is not None

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

# ── NLTK (BLEU) ───────────────────────────────────────────────────────────
# find_spec, not import + nltk.download: BLEU splits on whitespace (no tokenizer data
# needed), and a download at every start blocks offline / proxied servers.
HAS_NLTK: bool = importlib.util.find_spec("nltk") is not None

# ── nlpaug (data augmentation) ────────────────────────────────────────────
try:
    import nlpaug.augmenter.word as naw  # noqa: F401

    HAS_NLPAUG = True
except ImportError:
    HAS_NLPAUG = False

# ── lm-evaluation-harness (standard benchmarks) ───────────────────────────
# find_spec, not import: lm_eval imports its whole task registry on import.
HAS_LM_EVAL: bool = importlib.util.find_spec("lm_eval") is not None

# Benchmarks offered in the UI/CLI — each verified to load with datasets>=4
# (no dataset loading scripts). Name → short description.
BENCHMARK_TASKS: dict[str, str] = {
    "arc_easy": "ARC-Easy — grade-school science questions",
    "arc_challenge": "ARC-Challenge — harder science questions",
    "hellaswag": "HellaSwag — commonsense sentence completion",
    "piqa": "PIQA — physical commonsense",
    "winogrande": "WinoGrande — pronoun resolution",
    "boolq": "BoolQ — yes/no reading comprehension",
    "truthfulqa_mc2": "TruthfulQA (MC2) — avoiding common falsehoods",
    "gsm8k": "GSM8K — grade-school maths (generation; slow)",
}
BENCHMARK_DEFAULT_LIMIT = 100  # examples per task; full sets take hours on CPU
BENCHMARK_MAX_LIMIT = 10_000

# ── Quantized export (llm-compressor → compressed safetensors, served by vLLM) ──
# fp8: data-free FP8 weights + dynamic per-token activations (Hopper/Ada GPUs run it
# natively). w4a16: 4-bit GPTQ weights, needs calibration text.
QUANT_EXPORT_FORMATS: dict[str, str] = {"fp8": "FP8_DYNAMIC", "w4a16": "W4A16"}
QUANT_CALIBRATION_SAMPLES = 128  # llm-compressor examples use 128–512
QUANT_CALIBRATION_MAX_LENGTH = 512
HAS_LLMCOMPRESSOR: bool = importlib.util.find_spec("llmcompressor") is not None

# ── GPU job queue (web UI) ─────────────────────────────────────────────────
# Training, evaluation, benchmarks, GGUF export, merging and vLLM runs share one queue,
# so two users (or two tabs) never load models onto the GPU at the same time.
GPU_QUEUE_ID = "gpu"
try:
    GPU_JOB_CONCURRENCY = min(max(int(os.environ.get("LFT_GPU_JOBS", "1")), 1), 8)
except ValueError:
    GPU_JOB_CONCURRENCY = 1

# ── GRPO ──────────────────────────────────────────────────────────────────
# Loss formulations supported by every TRL version in range (0.29.1 – 1.x).
# dapo (TRL's default) and dr_grpo remove the length bias of the original grpo loss.
GRPO_LOSS_TYPES: list[str] = ["dapo", "dr_grpo", "grpo", "bnpo"]
DEFAULT_GRPO_LOSS_TYPE = "dapo"
# Built-in rewards (key → UI label). "reference" and "math" need a `reference` column.
GRPO_REWARDS: dict[str, str] = {
    "reference": "Reference answer appears in the output",
    "math": "Maths answer equals the reference (math-verify)",
    "think_format": "<think>…</think> reasoning, then the answer",
    "json": "Output is valid JSON",
    "regex": "Output matches a regular expression",
}
GRPO_REWARDS_NEEDING_REFERENCE = ("reference", "math")
DEFAULT_GRPO_REWARDS: list[str] = ["reference"]
GRPO_LORA_RANK = 16
GRPO_LORA_ALPHA = 32
# vLLM generation during GRPO (colocate: shares the training GPU).
GRPO_VLLM_GPU_MEMORY = 0.3  # TRL's default share of GPU memory for vLLM
# Knowledge distillation (GKD: the student learns the teacher's next-token distribution).
# lmbda: share of batches on the student's own generations (on-policy; 0 = teacher-forced
# on the dataset answers). beta: 0 ≈ forward KL, 1 ≈ reverse KL (generalised JSD).
# TRL's defaults.
DISTILL_LMBDA = 0.5
DISTILL_BETA = 0.5
DISTILL_TEMPERATURE = 0.9
DISTILL_MAX_NEW_TOKENS = 128
DISTILL_LORA_RANK = 16
DISTILL_LORA_ALPHA = 32

# ── Embedding-model fine-tuning (sentence-transformers) ───────────────────
# Ungated, no custom code (checked on the Hub): small English → multilingual → larger.
EMBED_MODEL_SUGGESTIONS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/multilingual-e5-small",
    "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
)
EMBED_METHODS = ("Full fine-tuning", "LoRA")  # LoRA is merged into the model on save
EMBED_LORA_RANK = 16
EMBED_LORA_ALPHA = 32
EMBED_MAX_SEQ_LENGTH = 256
# Matryoshka: also train the first N dimensions, so embeddings can be truncated.
EMBED_MATRYOSHKA_DIMS = (768, 512, 256, 128, 64)
EMBED_MIN_EVAL_QUERIES = 5  # fewer held-out pairs: evaluate on the training pairs instead
# Hard negatives must score below this share of their positive's score (NV-Retriever's
# positive-aware filter), so near-duplicates of the answer are not used as negatives.
EMBED_NEGATIVE_MARGIN = 0.05
# Combining LoRA adapters trained on the same base (PEFT add_weighted_adapter).
# linear / ties / dare_* need equal ranks; cat and svd accept any. density: share kept.
ADAPTER_MERGE_METHODS: dict[str, str] = {
    "ties": "TIES — trim small changes, resolve sign conflicts (same rank)",
    "dare_ties": "DARE + TIES — random drop and rescale, then TIES (same rank)",
    "dare_linear": "DARE + weighted sum (same rank)",
    "linear": "Weighted sum of the adapters (same rank)",
    "cat": "Concatenate (exact; rank = sum of ranks)",
    "svd": "Exact sum compressed back with SVD (any ranks)",
}
ADAPTER_MERGE_DENSITY_METHODS = ("ties", "dare_ties", "dare_linear")
DEFAULT_ADAPTER_MERGE_METHOD = "ties"
DEFAULT_ADAPTER_MERGE_DENSITY = 0.5
# Mixture-of-experts models: config fields holding the expert count, and the router
# module names. The router stays frozen during fine-tuning (training it risks expert
# collapse); the router load-balancing loss is switched on (output_router_logits).
MOE_EXPERT_FIELDS = ("num_experts", "num_local_experts", "n_routed_experts")
MOE_ROUTER_MODULES = ("gate", "router")
LORA_DROPOUT = 0.05
HAS_MATH_VERIFY: bool = importlib.util.find_spec("math_verify") is not None
# Vision-language fine-tuning: VLM processors need torchvision (must match torch).
HAS_TORCHVISION: bool = importlib.util.find_spec("torchvision") is not None

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
