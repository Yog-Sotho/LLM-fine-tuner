# CLAUDE.md — AI Assistant Guide for LLM Fine-Tuner

This file provides context for AI assistants (Claude, Copilot, etc.) working in this repository. It covers the architecture, key conventions, development workflows, and common pitfalls.

---

## Repository Overview

**LLM Fine-Tuner v3.2** is a production-ready application for fine-tuning large language models. It exposes two interfaces over the same core: a Gradio web UI and a Typer CLI. The application supports supervised fine-tuning (SFT), DPO, ORPO, PPO, and reward model training, with optional acceleration via Unsloth and vLLM.

**Entry point:** `main.py` — if `sys.argv` has arguments, delegates to the Typer CLI; otherwise launches the Gradio UI on port 7860.

---

## Directory Structure

```
LLM-fine-tuner/
├── main.py                  # Entry point (UI or CLI dispatch)
├── pyproject.toml           # Package metadata, dependencies, pytest config
├── requirements.txt         # Direct pip dependencies
├── docker-compose.yml       # GPU and CPU Docker services
│
├── config/
│   └── constants.py         # ALL constants, HAS_* flags, LoRA presets — Layer 0
│
├── core/
│   ├── state.py             # AppState singleton (caches, stop_event)
│   ├── hardware.py          # VRAM/RAM detection, model recommendation
│   └── callbacks.py         # Trainer callbacks (Stop, Logging, ETA)
│
├── data/
│   ├── loader.py            # Multi-format file ingestion (CSV/JSON/PDF/Excel/ZIP)
│   ├── preprocessing.py     # Dataset validation, cleaning, deduplication
│   └── augmentation.py      # nlpaug-backed data augmentation
│
├── training/
│   ├── sft.py               # train_model() — SFT/DPO unified pipeline
│   ├── reward.py            # train_reward_model_v27()
│   ├── ppo.py               # run_ppo_v27() — PPO fine-tuning
│   └── orpo.py              # train_orpo_v27() — ORPO alignment
│
├── inference/
│   ├── generate.py          # _load_for_inference(), generate_text(), batch_generate()
│   ├── evaluation.py        # BLEU/ROUGE/BERTScore/LLM-judge evaluation
│   └── vllm_runner.py       # vLLM engine with caching
│
├── export/
│   ├── gguf.py              # on_export_gguf() — GGUF quantization via Unsloth/llama.cpp
│   ├── hub.py               # push_to_hub() — HuggingFace Hub publishing
│   ├── registry.py          # Model registry reader
│   └── utils.py             # ZIP creation, model card generation
│
├── ui/
│   ├── app.py               # build_demo() — Gradio UI builder and event wiring hub
│   ├── handlers.py          # Gradio event handler functions (thin glue layer)
│   ├── css.py               # CUSTOM_CSS styling (violet theme)
│   └── tabs/
│       ├── data_tab.py      # Data upload & preview layout
│       ├── train_tab.py     # Training configuration layout
│       ├── gguf_tab.py      # GGUF export layout
│       ├── inference_tab.py # Inference layout
│       ├── rlhf_tab.py      # Reward/PPO/ORPO layout
│       ├── evaluation_tab.py# Evaluation layout
│       └── share_tab.py     # Hub push & download layout
│
├── cli/
│   └── commands.py          # Typer CLI (train, reward, orpo, ppo, evaluate)
│
├── tests/
│   ├── conftest.py          # pytest setup (inserts repo root into sys.path)
│   ├── test_cli.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_ppo_reward_type.py
│   └── test_training_guards.py
│
├── docs/                    # User-facing documentation (01_installation.md … 13_docker.md)
└── archive/                 # Deprecated code — do not import from here
```

---

## Architecture: Unidirectional Dependency Graph

The codebase enforces a strict layered architecture. **Never introduce circular or upward imports.**

```
config/ → core/ → data/ → training/ → inference/ → export/ → ui/ / cli/
```

| Layer | Modules | May import from |
|-------|---------|-----------------|
| 0 | `config/constants.py` | stdlib only |
| 1 | `core/` | config, stdlib |
| 2 | `data/` | config, core, stdlib |
| 3 | `training/` | config, core, data, stdlib |
| 4 | `inference/` | config, core, stdlib |
| 5 | `export/` | config, core, inference, stdlib |
| Top | `ui/`, `cli/` | All layers — never imported by other modules |

---

## Key Conventions

### Constants — always use `config/constants.py`

All column names, file extensions, PEFT preset maps, and optional-dependency guards live here. **Do not hardcode these strings elsewhere.**

```python
# Column names
COL_INSTRUCTION, COL_OUTPUT, COL_TEXT, COL_PROMPT, COL_CHOSEN, COL_REJECTED

# File extension sets
FILE_EXT_CSV, FILE_EXT_JSONL, FILE_EXT_PDF, ...

# Optional dependency guards (checked once at import time)
HAS_OPENPYXL, HAS_PDF, HAS_UNSLOTH, HAS_TRL, HAS_VLLM, HAS_HERETIC, ...
```

`HAS_*` flags are defined via `try/except` at module load and should be imported where needed rather than rechecked.

### UI event wiring — only in `ui/app.py`

Tab files (`ui/tabs/*.py`) define **layout only** — no `.click()`, `.change()`, or `.submit()` calls. All event wiring happens in `build_demo()` inside `ui/app.py`. This is the single source of truth for the event graph.

### Handlers — thin glue layer

`ui/handlers.py` contains handler functions that validate inputs, call core logic, and format outputs for Gradio. They should not contain business logic — delegate to the appropriate `training/`, `inference/`, or `export/` module.

### Thread safety

- The inference model cache in `inference/generate.py` is protected by `_cache_lock`.
- Do not access shared mutable state from Gradio handlers without acquiring the lock.
- The training stop mechanism uses `app_state.stop_event` (a `threading.Event`). Do not bypass it.

### Optional dependencies

Before using an optional package, check its `HAS_*` flag from `config/constants.py` and raise a user-friendly error if unavailable. Never let a missing optional package cause an unguarded `ImportError` at call time.

### Heretic mode

`HAS_HERETIC` is checked via `shutil.which("heretic")` — **not** subprocess. Do not change this pattern; it avoids process leaks and latency.

---

## Development Workflow

### Setup

```bash
# Recommended: use the installer script
chmod +x install.sh && ./install.sh
source llm_finetuner_env/bin/activate

# Or development install
pip install -e ".[dev]"

# Unsloth (optional, install separately)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-deps
```

### Running the application

```bash
# Launch Gradio UI (port 7860)
python main.py

# CLI training
python main.py train --model gpt2 --data data.csv --output ./models/run1
python main.py --help
```

### Running tests

```bash
# From the repo root
pytest

# Verbose with short traceback (configured default via pyproject.toml)
pytest -v --tb=short

# Specific file or test
pytest tests/test_cli.py -v
pytest tests/test_cli.py::test_help_flag_exits_zero -v
```

Tests use `CliRunner` (no subprocess spawning) and patch heavy functions so they run without a GPU or downloaded models. `conftest.py` inserts the repo root into `sys.path[0]` — **do not remove this**.

### Code style

```bash
# Format
black .

# Lint
ruff check .
```

- Line length: 100 characters
- Ruff rules: E, F, W, I, UP, B (E501 and B008 ignored — see `pyproject.toml`)
- Python target: 3.10+

### Docker

```bash
# GPU
docker compose up llm-fine-tuner-gpu

# CPU-only
docker compose up llm-fine-tuner-cpu

# Pass HuggingFace token
HF_TOKEN=hf_xxx docker compose up llm-fine-tuner-gpu
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MAX_VLLM_ENGINES` | `1` | Max concurrent vLLM engines |
| `HF_TOKEN` | — | HuggingFace Hub auth (gated models, Hub push) |
| `SHARE` | `false` | Enable public Gradio link |
| `TOKENIZERS_PARALLELISM` | `false` | Suppress tokenizer parallelism warning (Docker) |
| `HF_HOME` | `/app/cache/huggingface` | HuggingFace cache path (Docker) |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Faster Hub downloads (Docker) |

---

## Key Data Flows

### Training (UI path)
```
File upload → load_dataset_from_file() → validate_and_clean_dataset()
    → [optional: augment/filter] → on_train_click() → train_model()
    → create_model_card() + create_zip_from_folder()
```

### Inference
```
Generate request → _load_for_inference() [thread-safe, cached]
    → generate_text() [single] or batch_generate() [batch]
```

### GGUF export
```
Trained model → on_export_gguf()
    → [Unsloth available] FastLanguageModel GGUF export
    → [Fallback] llama.cpp conversion
```

### Hub push
```
Trained model → push_to_hub()
    → create_repo() [if needed] → upload files + model card
```

---

## Training Modes & PEFT Methods

**Training modes:** SFT, DPO (via `training/sft.py`), ORPO (`training/orpo.py`), PPO (`training/ppo.py`), Reward modeling (`training/reward.py`)

**PEFT methods:** LoRA, QLoRA Enhanced (NF4 + double quantization), Prefix Tuning, Prompt Tuning, Adapters, Full fine-tuning

**QLoRA Enhanced** requires `torch.cuda.is_bf16_supported()` — falls back to float16 if unsupported. Config lives in `QLORA_ENHANCED_BNB_KWARGS` in `config/constants.py`.

---

## Common Pitfalls

1. **Do not import from `archive/`** — deprecated code, kept for historical reference only.
2. **Do not wire Gradio events in tab files** — only `ui/app.py:build_demo()` does this.
3. **Do not add constants outside `config/constants.py`** — column names, file extensions, and feature flags belong there.
4. **Do not re-check `HAS_*` flags with `try/except`** — import from `config/constants.py`.
5. **PPO rewards must stay float32** — mixed types cause runtime errors; see `test_ppo_reward_type.py`.
6. **Small dataset guard** — `train_model()` has a split guard for tiny datasets; do not remove it.
7. **Inference cache** — always acquire `_cache_lock` before reading/writing the cache dict; return a locally-held reference, not a re-read from the dict (prevents race conditions).
8. **sys.path in tests** — `conftest.py` inserts the repo root; do not move or remove this.

---

## Fix Log Prefix Convention

Inline patch notes in docstrings use a prefix scheme:

| Prefix | Meaning |
|--------|---------|
| C | Critical — breaking bug halting execution |
| H | Hazard — memory/threading/security issue |
| M | Medium — data/logic issue with moderate impact |
| F | Feature — enhancement or new capability |
| N | Notice/Minor — small quality or efficiency improvement |
| L | Low — documentation or display fix |

Example: `# C-1: Removed broken llm_fine_tuner.* package imports`

---

## Package & Dependency Notes

- **Core deps:** transformers, datasets, peft, trl, torch, gradio, typer, pandas, safetensors
- **Optional groups** (install via `pip install -e ".[group]"`): `eval`, `quant`, `vllm`, `heretic`, `dev`, `all`
- **Unsloth:** installed separately — not in `requirements.txt`. Provides 2–5× training speedup and native GGUF export.
- **heretic-llm:** optional dep (moved out of required in v3.2 to avoid PyPI install failures).
- **Python:** 3.10, 3.11, 3.12 supported.
