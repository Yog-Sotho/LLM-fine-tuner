# CLAUDE.md — AI Assistant Guide for LLM Fine-Tuner

This file provides context for AI assistants (Claude, Copilot, etc.) working in this repository. It covers the architecture, key conventions, development workflows, and common pitfalls.

---

## Repository Overview

**LLM Fine-Tuner** is a production-ready application for fine-tuning large language models. It exposes two interfaces over the same core: a Gradio web UI and a Typer CLI. The application supports supervised fine-tuning (SFT), DPO, ORPO, KTO, GRPO, and reward model training, with optional acceleration via Unsloth and vLLM.

**Entry point:** `main.py` — if `sys.argv` has arguments, delegates to the Typer CLI; otherwise launches the Gradio UI on port 7860.

---

## Directory Structure

```
LLM-fine-tuner/
├── main.py                  # Entry point (UI or CLI dispatch)
├── pyproject.toml           # Package metadata, dependencies, pytest config
├── requirements.txt         # Direct pip dependencies
├── docker-compose.yml       # GPU and CPU Docker services
├── configs/accelerate/      # multi_gpu, fsdp2, fsdp_qlora, deepspeed_zero2/3 launch presets
│
├── config/
│   └── constants.py         # APP_VERSION, ALL constants, HAS_* flags, LoRA presets — Layer 0
│
├── core/
│   ├── state.py             # AppState singleton (shared caches) + per-session SessionState
│   ├── hardware.py          # VRAM/RAM detection, model recommendation, precision helpers
│   ├── run_config.py        # run_config.yaml, dataset fingerprint, run folders, tracking choice
│   ├── model_card.py        # README.md model card built from the run record
│   └── callbacks.py         # Trainer callbacks (Stop, Logging, ETA)
│
├── data/
│   ├── loader.py            # Multi-format file ingestion (CSV/JSON/PDF/Excel/ZIP)
│   ├── documents.py         # read_document() (PDF/.docx/.txt/.md) + chunk_text() for synthetic data
│   ├── preprocessing.py     # Dataset validation, cleaning, deduplication
│   └── augmentation.py      # nlpaug-backed data augmentation
│
├── training/
│   ├── sft.py               # train_model() — SFT/DPO unified pipeline
│   ├── vision.py            # train_vision_sft() — image + text chats (VLMs)
│   ├── reward.py            # train_reward_model_v27() — sequence-classifier reward model
│   ├── grpo.py              # train_grpo() — GRPO (reward model and/or reference answers)
│   ├── kto.py               # train_kto() — KTO from desirable/undesirable examples
│   ├── distill.py           # train_distill() — knowledge distillation (TRL GKD, student ← teacher)
│   └── orpo.py              # train_orpo_v27() — ORPO alignment
│
├── inference/
│   ├── generate.py          # _load_for_inference(), generate_text(), batch_generate()
│   ├── evaluation.py        # BLEU/ROUGE/BERTScore/LLM-judge evaluation, base-model comparison
│   ├── benchmarks.py        # run_benchmarks() — lm-evaluation-harness (ARC, HellaSwag, GSM8K…)
│   ├── remote.py            # remote_chat() — any OpenAI-compatible /v1/chat/completions server
│   ├── synthesize.py        # synthesize_pairs() — Q/A pairs from document chunks + LLM rating
│   └── vllm_runner.py       # vLLM engine with caching
│
├── export/
│   ├── gguf.py              # on_export_gguf() — GGUF quantization via Unsloth/llama.cpp
│   ├── hub.py               # push_to_hub() — HuggingFace Hub publishing
│   ├── quantize.py          # quantize_model() — FP8 / W4A16 safetensors via llm-compressor (vLLM)
│   ├── serve.py             # build_serve_command() / serve() — llama-server (GGUF) or vllm serve
│   ├── merge.py             # merge_lora_adapters() — combine LoRA adapters (PEFT TIES/DARE/SVD/cat)
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
│       ├── rlhf_tab.py      # Reward/GRPO/ORPO/KTO layout
│       ├── evaluation_tab.py# Evaluation layout
│       └── share_tab.py     # Hub push & download layout
│
├── cli/
│   └── commands.py          # Typer CLI (train … distill, benchmark, merge, merge-adapters, export, push, serve, synthesize)
│
├── tests/
│   ├── conftest.py          # pytest setup (inserts repo root into sys.path)
│   ├── test_cli.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_training_data.py
│   ├── test_training_guards.py
│   └── …                    # one file per area (security, formats, deploy, smoke training…)
│
└── docs/                    # User-facing documentation (01_installation.md … 13_docker.md)
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
- Stop signals and temp files are **per browser session**: get them with `app_state.session_for(request)` (handlers take `request: gr.Request | None = None`; the CLI uses the default session). Pass the session's `stop_event` to `StopCallback(stop_event)`. Track temp outputs with `session.track()` / `session.release()`, never with module globals — one session must not stop or delete another's work.

### Multi-process training and the GPU queue

- Trainers must work under `accelerate launch` / `torchrun` (data-parallel): load 4-bit models with `device_map=quantized_device_map()` (never `"auto"`), pass `**training_device_args(device)` to the TRL config, and write outputs (`save_pretrained`, `save_run_config`, subprocesses) only `if is_main_process():` — all from `core/hardware.py`.
- New heavy GPU event handlers in `ui/app.py` get `**GPU_JOB` so they join the shared queue; Stop must stay outside it.
- Sharding (FSDP / DeepSpeed via `configs/accelerate/`): `sharding_backend()` (needs CUDA — the launcher sets the env on CPU too, but accelerate only shards on GPUs). Only `train_model` supports it: 4-bit loads get `bnb_4bit_quant_storage = sharded_quant_storage()` = `torch_dtype` and no `device_map`; the final save is `trainer.save_model()` on **every** process (collective; FSDP switched to `FULL_STATE_DICT` first). Other trainers return `sharding_unsupported(name)`. Presets keep `mixed_precision: 'no'` so the app's precision wins.
- MoE: call `setup_moe(model, router_aux_loss=…, freeze_router=…)` before creating the TRL trainer (and again on `trainer.model` when TRL applies LoRA). Use the combination the smoke tests verify per trainer: DPO/KTO pass `freeze_router=False` and DPO/KTO/GRPO `router_aux_loss=False` (they fail otherwise). LoRA dropout comes from `lora_dropout(model)` (0 for MoE).

### Security defaults

- Never hardcode `trust_remote_code=True`; pass `trust_remote_code=ALLOW_REMOTE_CODE` from `config/constants.py`.
- PEFT adapters must be safetensors: call `validate_adapter_dir()` before `PeftModel.from_pretrained` on a local path. Pickle weights (`.bin`/`.pt`) are rejected because loading them can execute code.

### Reproducibility

- Every trainer calls `save_run_config()` after saving the model, so `run_config.yaml` (mode, model, hyperparameters, seed, dataset SHA-256, library versions) sits next to it. New trainers must do the same. It also writes the `README.md` model card (`core/model_card.py`), replacing the one TRL/PEFT wrote — call it last. `base_model` is only set for Hub ids, never local paths (the Hub rejects those).
- Resume: `latest_checkpoint(output_dir)` from `core/run_config.py`; GRPO/KTO save every `CHECKPOINT_SAVE_STEPS`.
- Call `transformers.set_seed(seed)` **before** building the model: LoRA initialises its weights at creation, before the Trainer seeds, so seeding later does not reproduce a run (covered by `test_same_seed_gives_identical_weights…`).
- UI runs go to `run_dir_for(run_name)` (`<LFT_RUNS_DIR>/<name>/`); never build run paths from raw user input.
- Tracking backends come from `TRACKING_BACKENDS` (installed only); validate choices with `resolve_report_to()`.

### Version and logging

- The version is written once: `APP_VERSION` in `config/constants.py` (a plain literal — `pyproject.toml` reads it via `[tool.setuptools.dynamic]`, CI checks the wheel). Show it with `APP_NAME`; never write the version anywhere else.
- Library modules log with `logger = logging.getLogger(__name__)`, never `print()`. `main.py` configures logging (`LFT_LOG_LEVEL`) and is the only module that prints.

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

`tests/test_smoke_training.py` trains `hf-internal-testing/tiny-random-LlamaForCausalLM` for real (SFT, DPO, ORPO, CLI, inference). It is skipped when the model can't be downloaded, unless `REQUIRE_SMOKE_MODELS=1` (set in CI). CI lives in `.github/workflows/ci.yml` and runs ruff, mypy (every layer), pytest (3.10–3.12, coverage must stay ≥ 85 %), pip-audit, hadolint and a packaging check.

`tests/test_gpu.py` (marker `gpu`) covers what CPU CI can't: 4-bit QLoRA, half-precision full fine-tuning, Flash Attention + packing, vLLM GRPO, NCCL. It skips without CUDA; `.github/workflows/gpu.yml` runs it on a GPU runner (manual, or weekly when the `GPU_RUNNER` repository variable is set). New GPU-only code gets a test there.

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
| `MAX_VLLM_ENGINES` | `1` | Max concurrent vLLM engines (clamped to 1–8) |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Bind address (Docker images set `0.0.0.0` inside the container) |
| `GRADIO_AUTH` | — | Require login: `user:password`, comma-separated pairs |
| `ALLOW_REMOTE_CODE` | `false` | Enables `trust_remote_code` for Hub models with custom code — off by default |
| `LFT_RUNS_DIR` | `runs` (Docker: `/app/models`) | Where UI training runs are saved (`<dir>/<run name>/`) |
| `LFT_GPU_JOBS` | `1` | Heavy GPU jobs (training, eval, benchmarks, export, merge, vLLM) the web UI runs at once (1–8) |
| `LFT_SERVE_API_KEY` | — | API key required by `main.py serve` (passed to the server via its environment) |
| `LFT_SYNTH_API_KEY` | — | API key for the writer server of `main.py synthesize` |
| `LFT_LOG_LEVEL` | `INFO` | Log level for the app's own modules (other libraries log warnings only) |
| `LFT_REPORT_TO` | `none` | Default experiment tracker (`trackio`, `wandb`, `mlflow`, `tensorboard`) if installed |
| `HF_TOKEN` | — | HuggingFace Hub auth (gated models, Hub push) |
| `SHARE` | `false` | Enable public Gradio link |
| `TOKENIZERS_PARALLELISM` | `false` | Suppress tokenizer parallelism warning (Docker) |
| `HF_HOME` | `/app/cache/huggingface` | HuggingFace cache path (Docker) |
| `LFT_ALLOWED_PATHS` | — | Extra folders the web UI may read/write (os.pathsep-separated); always allowed: working directory, runs folder, temp folder |

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

### Evaluation
```
Test CSV/JSONL → on_evaluate_click() / `evaluate` → generate_predictions() [greedy]
    → [compare_base: same model, adapter_names=["__base__"] per call] → BLEU/ROUGE/BERTScore
    → [optional] llm_judge_evaluate() → parse_judge_score() ("Score: N", 1–10)
Benchmarks → run_benchmarks() → lm_eval HFLM object (never a model_args string) → simple_evaluate()
```
Never toggle adapters on the cached inference model (`disable_adapter()`, `set_adapter()`) —
it is shared between sessions; pass `adapter_names` per `generate()` call instead.

### GGUF export
```
Trained model → on_export_gguf()
    → [Unsloth available] FastLanguageModel GGUF export
    → [Fallback] LoRA adapter merged into its base (temp dir) → convert_hf_to_gguf.py
      (run with sys.executable) → llama-quantize
```

### Quantized export and serving
```
Model / LoRA adapter → quantize_model() [adapter merged to a temp dir] → llm-compressor oneshot
    (FP8_DYNAMIC data-free | W4A16 GPTQ on calibration_texts()) → save_pretrained(save_compressed)
serve --model X → build_serve_command(): .gguf → llama-server | folder/Hub id → vllm serve
    (adapter → base + --lora-modules); API key via env (LLAMA_API_KEY / VLLM_API_KEY), never argv
```

### Training data from documents
```
Documents → document_chunks() [data/documents.py: read_document + chunk_text, 4000/200 chars]
    → synthesize_pairs(chunks, ask) [inference/synthesize.py] → QA_PROMPT (JSON pairs) per chunk
    → [threshold > 0] RATING_PROMPT → parse_judge_score() → drop low/unrated + duplicate questions
ask = remote_writer() (remote_chat, any OpenAI-compatible server) | local_writer() (cached HF model)
```
Reading lives in `data/` and writing in `inference/` (inference may not import data); the UI
handler / CLI joins them. One failed chunk is counted, never fatal.

### Hub push
```
Trained model → push_to_hub()
    → card: base_model checked on the Hub (canonical id, license copied; dropped if unknown)
    → create_repo(exist_ok=True) → upload_folder()
```

---

## Training Modes & PEFT Methods

**Training modes:** SFT, DPO (via `training/sft.py`), ORPO (`training/orpo.py`), KTO (`training/kto.py`), GRPO (`training/grpo.py`), Reward modeling (`training/reward.py`), Distillation (`training/distill.py`)

**Distillation** uses `trl.experimental.gkd.GKDTrainer` (same API in TRL 0.29 and 1.x): chat data only (`to_distill_dataset()` turns instruction/output and prompt/completion into chats), the student gets LoRA, the teacher only runs forward. Check the vocab sizes match before training — TRL 0.29 doesn't.

**Adapter merging** (`export/merge.py`) uses PEFT `add_weighted_adapter`: LoRA only, one base model, no DoRA (magnitudes are dropped); linear/ties/dare_* need equal ranks. Make merged weights contiguous before saving (older PEFT leaves SVD views). mergekit is not used: 0.1.4 pins accelerate~=1.6 / pydantic~=2.10 and fails with current Transformers.

**Data formats:** `messages` (chat), `text`, `instruction`/`output`, and `prompt`/`chosen`/`rejected` (DPO). `load_hub_dataset()` streams Hub datasets (IDs must be `owner/name` and must not exist locally — `load_dataset` would read a local directory). Duplicates are detected ignoring case/whitespace.

**SFT data** goes through `to_sft_dataset()` into TRL's prompt-completion format (chat data: all turns before the last assistant reply → prompt, that reply → completion), so `SFTTrainer` trains on the response only and appends EOS. Don't pre-tokenise or build `labels` yourself.

**Precision:** use `select_precision(device)` / `compute_dtype(device)` from `core/hardware.py` (bf16 where supported, else fp16; fp32 on CPU). Always pass them explicitly — TRL configs default to bf16, which fails on CPU. Full fine-tuning is never loaded quantised (Transformers refuses to train a purely quantised model): weights use `full_finetune_dtype(device)` — bf16, else fp32, never fp16.

**Reward models** are saved merged (full sequence classifier) because `GRPOTrainer` loads a reward-model path as `AutoModelForSequenceClassification(num_labels=1)`.

**PEFT methods:** LoRA, QLoRA Enhanced (NF4 + double quantization), Prefix Tuning, Prompt Tuning, IA3 (PEFT's default layers per architecture; mergeable like LoRA, but `vllm serve` takes LoRA only — merge first), Full fine-tuning

**LoRA targets and variants:** every LoRA config uses `get_lora_targets()` (`"all-linear"`: all attention + MLP projections, never the output head) — don't hard-code module names; Unsloth alone gets `UNSLOTH_LORA_TARGETS`. Variants come from `LORA_VARIANTS` via `lora_variant_kwargs()` (LoRA, rsLoRA, DoRA). DoRA can't be switched off per request (`adapter_names`), so `is_lora_model()` excludes it from base-model comparison. PiSSA isn't offered: PEFT can't initialise it on 4-bit weights, which the GPU LoRA path uses.

**Chat data (tools, reasoning, images):** build chat datasets with `chat_dataset()` (`data/preprocessing.py`) — never `Dataset.from_list`/`from_pandas` on messages. Each conversation is one `Json` value (Arrow would merge differently shaped messages/tool-call arguments and fill nulls; a list of `Json` would also break older TRL's pyarrow truncation), `tools` is a JSON string (TRL decodes it), `images` is `List(Image())` carried undecoded (`Image(decode=False)`) so bytes are never re-encoded. Read chat columns by column access, not `to_pandas()` (it returns JSON strings on datasets 4.7). Chats with tool calls expand to one example per assistant turn. Image data routes `train_model()` to `training/vision.py` (`AutoModelForImageTextToText` + `AutoProcessor`, `max_length=None`).

**GRPO rewards:** built-ins live in `training/grpo.py` (`build_reward_funcs`). Reuse TRL's reward functions (`trl.rewards`) where they exist — they read `completion[0]["content"]`, so wrap plain-string completions with `_as_messages()`. Always set `vllm_mode` explicitly when `use_vllm` (its default differs between TRL versions).

**QLoRA Enhanced** requires `torch.cuda.is_bf16_supported()` — falls back to float16 if unsupported. Config lives in `QLORA_ENHANCED_BNB_KWARGS` in `config/constants.py`.

---

## Common Pitfalls

1. **Do not `print()` outside `main.py`** — log with `logger = logging.getLogger(__name__)` (a test enforces it); warnings a user must see during training also go to the run log via `_warn()` in `training/sft.py`.
2. **Do not wire Gradio events in tab files** — only `ui/app.py:build_demo()` does this.
3. **Do not add constants outside `config/constants.py`** — column names, file extensions, and feature flags belong there.
4. **Do not re-check `HAS_*` flags with `try/except`** — import from `config/constants.py`.
5. **GRPO mutates `reward_funcs`** — `GRPOTrainer` replaces reward-model paths in the list with loaded models; pass a copy and derive anything you need from the list beforehand.
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
- **Optional groups** (install via `pip install -e ".[group]"`): `eval`, `vllm` (TRL's vllm extra, CUDA), `heretic`, `vision`, `compress` (llm-compressor — pins recent torch/Transformers, not in `all`), `dev`, `all`
- **Unsloth:** installed separately — not in `requirements.txt`. Provides 2–5× training speedup and native GGUF export.
- **heretic-llm:** optional dep (moved out of required in v3.2 to avoid PyPI install failures).
- **Python:** 3.10, 3.11, 3.12 supported.
- **Packaging:** `pyproject.toml` is the only source of metadata (`setup.py` is an empty shim). Packages are listed in `[tool.setuptools.packages.find] include` — add any new top-level package there; `main.py` ships via `py-modules`. CI installs the built wheel and checks it (contents, entry points, version).
