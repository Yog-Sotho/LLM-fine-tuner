# Codebase Audit Report

**Date:** 2026-09-29 (supersedes the 2026-09-27 baseline, merged in below)
**Project:** llm-fine-tuner 3.2.0 (`APP_VERSION`)
**Audited revision:** branch `claude/project-audit-report-qsOS1` — `main` after PR #175 (Tier 10) + Tier 11 (PR #176) + Tier 12
**Language / Framework:** Python 3.10–3.12 · Gradio 6 · Transformers 4.56–5.x · TRL 0.29–1.x · PEFT 0.17–0.21 · Typer
**Project Type:** Web service (Gradio UI, `127.0.0.1` by default) + CLI — all 9 dimensions applicable
**Audit Mode:** Global (flat layered packages: `config → core → data → training → inference → export → ui/cli`)

---

## Executive Summary

Two days ago the application could not start (broken `main.py` imports), CI had never run,
any visitor could execute remote code, and the RLHF stack targeted removed TRL APIs (baseline
score **3.8/10**). Twelve tiers of work since then fixed every CRITICAL and HIGH finding of that
baseline: the app starts, CI runs ruff / mypy (all layers) / pytest on 3 Python versions with an
85 % coverage floor / pip-audit / hadolint / a wheel check, remote code is opt-in, adapters must be
safetensors, state is per session, and the trainers are rebuilt on current TRL (SFT, DPO, ORPO,
KTO, GRPO, reward, vision). The code-quality score is now **7.3/10**. What remains is
MEDIUM/LOW: a string-blacklist path check, a broken "Adapters" PEFT option (replaced with IA3 in Tier 13), optional
packages hard-listed in `requirements.txt`, an oversized `train_model()`, and GPU code paths that
have tests (`tests/test_gpu.py`) but have **not yet been run on a GPU**.
**Top priority:** run `.github/workflows/gpu.yml` on a GPU runner.

## Overall Score: 7.3 / 10 (baseline 3.8)

| Dimension | Score | Baseline | Priority | Status |
|---|---|---|---|---|
| Security | 8/10 | 3 | CRITICAL | PASS |
| Build & Types | 8/10 | 2 | CRITICAL | PASS |
| Concurrency | 7/10 | 4 | HIGH | PASS |
| Code Principles | 6/10 | 5 | HIGH | WARN |
| Dependencies | 7/10 | 3 | MEDIUM | WARN |
| Observability | 8/10 | 3 | MEDIUM | PASS |
| Lifecycle | 7/10 | 4 | MEDIUM | WARN |
| Code Quality | 7/10 | 5 | MEDIUM | WARN |
| Dead Code | 8/10 | 5 | LOW | PASS |

Average 7.33; no CRITICAL findings remain, so no hard caps apply → **7.3**.

### SOTA position (separate from code quality): 7 / 10 (baseline 3)

Feature parity with Unsloth / Axolotl / LlamaFactory on the 2026 core set (SFT, DPO, ORPO, KTO,
GRPO with built-in rewards and vLLM, reward models, vision SFT, tool-calling and reasoning chats,
LoRA / rsLoRA / DoRA / QLoRA / full fine-tuning, GGUF + FP8/W4A16 export, OpenAI-compatible
serving, reproducible runs). Not yet at 8: no DeepSpeed / sequence parallelism / efficient MoE
training, no distillation (TRL GKD/GOLD) or model merging, no embedding-model training, no
persistent job queue, and the GPU paths have not been exercised by an automated run.

| Area | Score | Notes |
|---|---|---|
| Training coverage | 8.5 (T13) | Core 2026 set + distillation (GKD), LoRA adapter merging (TIES/DARE/SVD), IA3; missing async/agentic RL, full-model merges |
| Export & deploy | 8 | GGUF, FP8/W4A16, Hub card, llama-server / vLLM serving, remote client |
| Reproducibility | 9 | run_config.yaml, dataset SHA-256, early seeding, resume, tracking, one version |
| Security | 8 | See Security below |
| Scale | 6.5 (T14) | DDP + FSDP2 / FSDP-QLoRA / DeepSpeed ZeRO-2/3 presets for `train`; MoE support; activation offloading, padding-free. Sharding **not yet run on GPUs**; no context/sequence parallelism |
| Code quality | 7 | mypy on all layers; `train_model()` still ~580 lines |
| Test depth | 8 (was 6) | 88 % coverage, 505 tests; GPU suite written, not yet run on a GPU |

---

## Strengths

1. **Clear layered architecture, enforced by habit and review.** Unidirectional imports; every
   Gradio event wired in `ui/app.py:build_demo`; constants and `HAS_*` flags in one module.
2. **Tests that train for real.** `tests/test_smoke_training.py` trains tiny Llama/Qwen3/Qwen2.5-VL
   models end to end (SFT, DPO, ORPO, KTO, GRPO, reward, vision, tools, DDP with 2 processes) on
   both the oldest and newest supported library versions; CI requires the models (no silent skip).
3. **Security defaults.** Remote code off unless `ALLOW_REMOTE_CODE`; login via `GRADIO_AUTH` with a
   warning when exposed; safetensors-only adapters; zip-bomb limits; token redaction; API keys via
   environment, never argv; torch ≥ 2.6 (CVE-2025-32434).
4. **Reproducibility.** Every trainer writes `run_config.yaml` and a Hub-valid model card; runs can
   be replayed (`train --config`) and resumed; seeds are set before the model is built.
5. **One source of truth** for the version (`APP_VERSION`), logging (`LFT_LOG_LEVEL`), run folders
   (`LFT_RUNS_DIR`) and the GPU queue (`LFT_GPU_JOBS`).

---

## Findings by Dimension

Status of every baseline finding: **FIXED** (with the tier that fixed it) or **OPEN**.
New findings from this audit are marked **NEW**.

### Security [8/10]

| Severity | Status | Location | Issue | Recommendation / fix |
|---|---|---|---|---|
| CRITICAL | FIXED (T1) | `main.py`, loaders | Unauthenticated UI on `0.0.0.0` + `trust_remote_code=True` everywhere → RCE | `trust_remote_code=ALLOW_REMOTE_CODE` (off); bind `127.0.0.1`; `GRADIO_AUTH`; exposure warning |
| HIGH | FIXED (T1) | Docker, CI, adapter upload | torch ≤ 2.5.1 (CVE-2025-32434) + `.bin` adapters | torch ≥ 2.6 (images: 2.14); `validate_adapter_dir()` rejects pickle weights |
| HIGH | FIXED (T1) | entrypoint / `main.py` | `SHARE` / auth passed as argv to the CLI | `GRADIO_SHARE`/`SHARE`/`GRADIO_AUTH` read from env in `main.py` |
| MEDIUM | FIXED (T1+) | dependency floors | Gradio/Transformers floors with advisories | gradio ≥ 6, transformers ≥ 4.56.2; pip-audit in CI (3 documented waivers) |
| MEDIUM | OPEN | `core/state.py:46` `validate_path_traversal` | Still a string blacklist (`..`, `\`, NUL). Absolute server paths in path fields are accepted — mitigated by auth + localhost default, not by the check | Allow-list: `realpath` must be under the temp dir, Gradio upload dir or `LFT_RUNS_DIR` (absolute paths are legitimate, so do not simply block them) |
| LOW | OPEN | `ui/handlers.py:129,424,497,561`, `inference/vllm_runner.py:95,249`, `inference/generate.py:162`, trainers' failure returns | Raw exception text shown in the UI without `redact_sensitive_info` | Route every user-facing error through `redact_sensitive_info` |

### Build & Types [8/10]

| Severity | Status | Location | Issue | Recommendation / fix |
|---|---|---|---|---|
| CRITICAL | FIXED (T1) | `main.py` | Imports of non-existent top-level modules — app could not start | `ui.app` / `cli.commands`; import + `build_demo()` smoke test |
| CRITICAL | FIXED (T1) | CI | `ci.yml` at repo root never ran | `.github/workflows/ci.yml`; replayed locally before every push |
| HIGH | FIXED (T1/T2) | trainers | `tokenizer=` instead of `processing_class=`; legacy PPO; value-head reward model | Rebuilt on current TRL: PPO → GRPO, reward model = sequence classifier |
| MEDIUM | FIXED (T6) | packaging | `main.py` not installed; generic top-level packages | `py-modules = ["main"]`, explicit package list, CI wheel check |
| MEDIUM | FIXED (T11/T12) | tests | 22 uncollected benchmark/verify scripts | Deleted; their unique checks moved into real tests |
| **NEW** MEDIUM | FIXED (T12) | CI mypy | mypy skipped `training/`, `inference/`, `ui/`, `main.py` | All layers checked |
| **NEW** MEDIUM | OPEN | GPU paths | QLoRA 4-bit, half-precision full FT, Flash Attention + packing, vLLM GRPO, NCCL have never run in automation | `tests/test_gpu.py` + `.github/workflows/gpu.yml` (manual, or weekly when `GPU_RUNNER` is set) — **run it** |

### Concurrency [7/10]

| Severity | Status | Location | Issue | Recommendation / fix |
|---|---|---|---|---|
| HIGH | FIXED (T1) | `core/state.py` | Global state shared by all sessions (Stop / cleanup crossed users) | `SessionState` per browser session (`app_state.session_for(request)`) |
| MEDIUM | FIXED (T1/T5) | Stop / evaluation | Stop event not cleared; evaluation silently empty after Stop | Cleared at the start of every job; evaluation takes the session's event |
| MEDIUM | FIXED (T1) | `inference/generate.py` | Parallel loads of the same model | Per-key load lock + `_cache_lock` |
| MEDIUM | FIXED (T5) | `inference/evaluation.py` | `fork` in a threaded server | Metrics computed in-process |
| MEDIUM | OPEN (mitigated) | `inference/vllm_runner.py:179-197` | `app_state.vllm_cache` read/evicted without a lock | UI calls are serialised by the GPU queue (`GPU_JOB`); add a lock for other callers |

### Code Principles [6/10]

| Severity | Status | Location | Issue | Recommendation |
|---|---|---|---|---|
| MEDIUM | OPEN | `training/sft.py:118` | `train_model()` is ~580 lines with ~35 parameters, 21 of them positional (called positionally from ~20 places) | `TrainConfig` dataclass; split into load / apply PEFT / build trainer |
| MEDIUM | OPEN | `training/sft.py:279,351`, `orpo.py:112`, `vision.py:109` | 4-bit `BitsAndBytesConfig` built in 3 files | One `core` helper for quantised loading |
| MEDIUM | OPEN | `cli/commands.py` (893 lines) | All commands in one module | One module per command group |
| MEDIUM | FIXED (T2/T5) | split guard, batched generation | Copied 3–4× | Shared helpers (`generate_predictions`, dataset split helpers) |
| LOW | OPEN | `export/hub.py`, `export/registry.py` | HF-token validation duplicated | `validate_hf_token()` in `core/state.py` |

### Dependencies [7/10]

| Severity | Status | Location | Issue | Recommendation / fix |
|---|---|---|---|---|
| HIGH | FIXED (T1) | ranges | Floors only → incompatible TRL/Transformers | Tested ranges, CI on floor and ceiling |
| HIGH | FIXED (T1) | torch pin | CVE-2025-32434 | torch ≥ 2.6 |
| MEDIUM | OPEN | `requirements.txt:50-51,74` | `auto-gptq`, `exllamav2`, `vllm` still hard requirements (Dockerfiles filter them out; `pip install -r` fails on CPU/macOS) | Move them to extras only (`.[quant]`, `.[vllm]`); `auto-gptq` is unmaintained |
| LOW | FIXED | PyPDF2 | Deprecated | `pypdf>=6.16.1` |
| LOW | OPEN | `config/constants.py:155` | vLLM quantisation option `"bnb"`; vLLM expects `"bitsandbytes"` | Map `bnb → bitsandbytes` |

### Observability [8/10]

| Severity | Status | Location | Issue | Recommendation / fix |
|---|---|---|---|---|
| MEDIUM | FIXED (T11) | whole codebase | No logging, `print()` diagnostics | Module loggers; `main.py` configures `LFT_LOG_LEVEL`; a test forbids `print()` |
| MEDIUM | FIXED | `raise … from e` | Lost tracebacks | ruff B904 clean |
| MEDIUM | FIXED (T6) | Heretic | Return code ignored | Checked |
| **NEW** MEDIUM | FIXED (T12) | `core/callbacks.py` `LoggingCallback` | Evaluation logs separately from training, so **eval loss never reached the loss chart** (always NaN) | Eval results attach to the record of their step; `final_train_loss()` skips eval-only records; covered by a real training test |
| LOW | OPEN | `ui/tabs/inference_tab.py:53` | Batch test returns error strings into a `gr.File` output | Separate status text component |

### Lifecycle [7/10]

| Severity | Status | Location | Issue | Recommendation / fix |
|---|---|---|---|---|
| HIGH | PARTLY FIXED | trainers | GPU memory not freed on failure | `try/finally` cleanup in SFT, reward, ORPO, KTO, GRPO; **`training/vision.py` has none** — add the same `finally` |
| MEDIUM | FIXED (T3) | UI runs | Resume impossible (fresh temp dir per run) | Persistent `<LFT_RUNS_DIR>/<run name>/` |
| MEDIUM | PARTLY FIXED | `export/hub.py:113`, `export/utils.py` | `create_repo(exist_ok=True)` added; Hub push and ZIP still include `checkpoint-*` (optimizer states) | `ignore_patterns=["checkpoint-*"]`, skip them in the ZIP |
| MEDIUM | OPEN | `config/constants.py:233,344` | Import-time `import unsloth` (patches Transformers globally) and `nltk.download("punkt")` | `find_spec` probe + lazy import at call sites (as done for Liger, lm-eval, llm-compressor) |
| LOW | OPEN | `docker-compose.yml:61-63,110-111`, Dockerfiles | `TRANSFORMERS_CACHE` deprecated; `HF_HUB_ENABLE_HF_TRANSFER` is deprecated in huggingface_hub 1.33 (hf_transfer no longer used — warns) | Drop both; keep `HF_HOME` |

### Tier 14 findings

| Severity | Status | Location | Issue | Fix |
|---|---|---|---|---|
| **NEW** HIGH | FIXED (T14) | all trainers (`lora_dropout=0.05`) | LoRA on any MoE model **failed** on Transformers 5 / PEFT 0.21: PEFT wraps the fused expert weights with a ParamWrapper that refuses dropout | `lora_dropout(model)` → 0 for MoE models; verified on tiny Qwen3-MoE with every trainer, TRL 0.29 and 1.14 |
| **NEW** MEDIUM | FIXED (T14) | trainers | MoE fine-tuning ran without the router load-balancing loss (TRL adds it only with `output_router_logits`), and LoRA trained the router | `setup_moe()` per trainer (combination verified per trainer; DPO/KTO keep TRL's handling) |
| **NEW** INFO | NOTED | accelerate on CPU | `accelerate launch` with an FSDP config sets `ACCELERATE_USE_FSDP` on CPU but trains plain DDP | `sharding_backend()` requires CUDA, so records and guards match what really runs |

### Code Quality [7/10]

| Severity | Status | Location | Issue | Recommendation / fix |
|---|---|---|---|---|
| MEDIUM | FIXED (T2) | SFT data | EOS never trained, pads masked | TRL prompt-completion format; EOS appended by SFTTrainer |
| MEDIUM | FIXED (T2) | precision | `fp16=True` forced with bf16 weights | `select_precision` / `compute_dtype` |
| MEDIUM | FIXED (T13) | `training/sft.py`, `ui/tabs/train_tab.py` | "Adapters" PEFT option always failed: PEFT has no `AdapterConfig` | Replaced with IA3 (`IA3Config`, PEFT default layers per architecture); mergeable for export; `adapter_reduction_factor` removed end to end |
| **NEW** HIGH | FIXED (T12) | `ui/handlers.py:on_file_upload` | A CSV/Excel with non-standard column names raised an error before the column-mapping dropdowns were shown, so such files could not be mapped at all (regression from the "in-memory refresh" optimisation) | Columns checked before conversion; raw rows + dropdowns shown; covered by tests |
| MEDIUM | OPEN | `data/augmentation.py:122` | Data without `text`/`instruction` (DPO) is silently duplicated N× and reported as augmented; augment/filter re-read the file and ignore the column mapping | Say "not supported for preference data"; operate on the prepared dataset |
| LOW | OPEN | `data/loader.py` JSON/JSONL branch | `column_mapping` ignored for JSON | Load through pandas like CSV |
| LOW | OPEN | `export/registry.py:150` | Versions sorted as strings (`v10` before `v2`) | `packaging.version.parse` |
| LOW | FIXED (T11) | banners | "PRODUCTION READY" / stale version strings | Removed; version from `APP_VERSION` |

### Dead Code [8/10]

| Severity | Status | Location | Issue | Recommendation / fix |
|---|---|---|---|---|
| LOW | FIXED | `load_qlora_model_v27`, duplicate handler block | Dead code | Removed |
| LOW | FIXED (T11) | `archive/`, `gradio.log`, 17 benchmark scripts, `test_batch_size.py`, verify scripts | Dead files | Removed |
| LOW | FIXED | remote branches | ~80 stale bot branches | 2 heads remain |
| LOW | FIXED (T13) | `config/constants.py` `HAS_ADAPTER_CONFIG` | Always `False` on real PEFT | Removed with the Adapters option |

---

## Advisory Findings

- **Rule 4 (cohesion):** `inference/evaluation.py` (710 LOC) and `data/preprocessing.py` (642 LOC)
  have single responsibilities and unified data models → `[High cohesion module]`, not split.
- **Rule 3 (single consumer):** HF-token validation duplication is consumed only by the Share tab —
  kept LOW.
- **mypy with the real libraries installed** reports ~190 errors, almost all library-typing noise
  (`LoraConfig(**kwargs)` overloads, PEFT model variance, guarded `try/except` imports). CI types
  torch/transformers/peft/trl as `Any`; that run is clean. The one real bug it surfaced is the
  "Adapters" finding above.
- No ADR directory exists, so no findings were downgraded under Rule 1.

---

## Audit history

| Date | Revision | Score | Notes |
|---|---|---|---|
| 2026-09-27 | `origin/main@7491ca6` | 3.8 | Baseline: app could not start, CI never ran, unauthenticated RCE |
| 2026-09-29 | Tier 12 branch | 7.3 | All CRITICAL/HIGH fixed; SOTA 7/10; open items are MEDIUM/LOW |
| 2026-09-29 | Tier 13 branch | 7.3 | Adapters → IA3 (finding fixed); distillation + adapter merging added; SOTA 7.5/10 |
| 2026-09-29 | Tier 14 branch | 7.4 | MoE LoRA fixed on Transformers 5 (was failing); MoE router loss/freeze; sharding presets; long-context options; SOTA 7.5/10 (8 once the GPU suite passes) |

Work between the two audits (all verified with the CI workflow replayed locally, floor and
ceiling library versions, live UI and Docker CPU checks before each push):
T1 entry point, security, per-session state, CI · T2 trainers on current TRL (GRPO replaces PPO,
KTO, reward model) · T3 run config, persistent runs, tracking, seeds · T4 Hub datasets, chat data ·
T5 evaluation (judge, base comparison, lm-eval benchmarks) · T6 packaging, full FT, GGUF merge,
model card · T7 all-linear LoRA, DoRA/rsLoRA, GRPO options · T8 tool calling, reasoning, vision ·
T9 multi-GPU via accelerate, GPU queue · T10 FP8/W4A16 export, serving · T11 logging, single
version, dead code · T12 mypy on all layers, coverage 76 % → 88 % with an 85 % floor, GPU test
suite + workflow, two bugs fixed (column mapping on upload, eval loss in the chart).

---

## Recommended Actions (Priority Order)

1. **[MEDIUM]** Run `.github/workflows/gpu.yml` on a GPU runner (GitHub T4 larger runner or
   self-hosted); set the `GPU_RUNNER` variable for the weekly run. Fix whatever it finds.
2. ~~**[MEDIUM]** Decide on the "Adapters" option~~ — done in Tier 13: replaced with IA3.
3. **[MEDIUM]** Path allow-list in `validate_path_traversal` (temp / upload / runs directories).
4. **[MEDIUM]** Move `vllm`, `auto-gptq`, `exllamav2` out of `requirements.txt` into extras.
5. **[MEDIUM]** Exclude `checkpoint-*` from Hub push and the download ZIP.
6. **[MEDIUM]** Lazy `unsloth` import; drop the import-time `nltk.download`; `finally` GPU cleanup in `training/vision.py`.
7. **[MEDIUM]** Split `train_model()` behind a `TrainConfig`; one quantised-loading helper; split
   `cli/commands.py`.
8. **[LOW]** Redact all UI error text; vLLM `bnb → bitsandbytes`; registry version sort; JSON
   column mapping; DPO augmentation message; drop deprecated `TRANSFORMERS_CACHE` /
   `HF_HUB_ENABLE_HF_TRANSFER`; `vllm_cache` lock.
9. **[Roadmap]** ~~Tier 13 — distillation and model merging~~ (done: GKD distillation, LoRA
   adapter merging; full-model merges wait for a mergekit release that works with current
   Transformers); ~~Tier 14 — DeepSpeed/FSDP presets, long context, MoE~~ (done; sharding and
   long-context paths still need their first GPU run).

---

## Sources Consulted

- Direct inspection of every module on the audited branch; coverage from
  `pytest --cov=.` (88 %, 498 passed, 7 GPU tests skipped on CPU); mypy 2.3 in the CI setup and with
  the real libraries installed.
- [GitHub Docs — Larger runners reference](https://docs.github.com/en/actions/reference/runners/larger-runners)
  (GPU runner: 4 vCPU, Tesla T4, 16 GB VRAM) and
  [Choosing the runner for a job](https://docs.github.com/en/actions/writing-workflows/choosing-where-your-workflow-runs/choosing-the-runner-for-a-job)
  (expressions in `runs-on`).
- [TRL releases](https://github.com/huggingface/trl/releases), [Async GRPO docs](https://huggingface.co/docs/trl/async_grpo_trainer),
  [GKD trainer](https://github.com/huggingface/trl/blob/main/docs/source/gkd_trainer.md),
  [GOLD trainer](https://huggingface.co/docs/trl/main/en/gold_trainer).
- [Unsloth vs Axolotl vs TRL vs LLaMA-Factory (MarkTechPost, Jul 2026)](https://www.marktechpost.com/2026/07/22/unsloth-vs-axolotl-vs-trl-vs-llama-factory-a-fine-tuning-framework-comparison-on-speed-vram-and-multi-gpu/),
  [Fine-Tuning in 2026 (DEV)](https://dev.to/ultraduneai/eval-003-fine-tuning-in-2026-axolotl-vs-unsloth-vs-trl-vs-llama-factory-2ohg).
- huggingface_hub 1.33 source (`constants.py`: `HF_HUB_ENABLE_HF_TRANSFER` deprecated).
- Baseline sources (2026-09-27): TRL 1.0 / Transformers 5 docs via Context7,
  [CVE-2025-32434 / GHSA-53q9-r3pm-6pq6](https://github.com/advisories/GHSA-53q9-r3pm-6pq6),
  [Gradio security advisories](https://github.com/gradio-app/gradio/security),
  [vLLM BitsAndBytes docs](https://docs.vllm.ai/en/stable/features/quantization/bnb/).
