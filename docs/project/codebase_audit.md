# Codebase Audit Report

**Date:** 2026-09-27
**Project:** llm-fine-tuner v3.2.0
**Audited revision:** `origin/main` @ `7491ca6` (latest; includes PRs up to #165)
**Language / Framework:** Python 3.10–3.12 · Gradio 5 · Transformers · TRL · PEFT · Typer
**Project Type:** Web service (Gradio UI on `0.0.0.0:7860`) + CLI — all 9 dimensions applicable
**Audit Mode:** Global (flat layered packages: `config → core → data → training → inference → export → ui/cli`)

> Line numbers below refer to `origin/main` @ `7491ca6`. The branch
> `claude/project-audit-report-qsOS1` is 84 commits behind `main` and its
> fix commit `f20f80b` was **never merged** (see §"Status of previous audit").

---

## Executive Summary

The architecture is clean and well documented, but **the shipped application cannot currently start**: `main.py` imports `app` and `commands` as top-level modules that do not exist (`ui/app.py`, `cli/commands.py`), and this has gone unnoticed because the CI file sits at the repo root instead of `.github/workflows/` and has never run. Beneath that, the RLHF layer (Reward/PPO/ORPO) and DPO are written against TRL/Transformers APIs that have since been removed, and with no upper version bounds, a fresh install gets TRL 1.x / Transformers 5.x and breaks them. Security hardening has focused heavily on string-level path checks (dozens of bot PRs), while the real exposure — an unauthenticated server bound to all interfaces that runs `trust_remote_code=True` on any user-typed Hub model ID — is unaddressed. **Top priority: fix the entry point, move CI into place, pin compatible dependency ranges, and gate remote code execution behind auth + an explicit opt-in.**

## Overall Score: 3.8 / 10

| Dimension | Score | Priority | Status |
|---|---|---|---|
| Security | 3/10 | CRITICAL | FAIL |
| Build & Types | 2/10 | CRITICAL | FAIL |
| Concurrency | 4/10 | HIGH | FAIL |
| Code Principles | 5/10 | HIGH | WARN |
| Dependencies | 3/10 | MEDIUM | FAIL |
| Observability | 3/10 | MEDIUM | FAIL |
| Lifecycle | 4/10 | MEDIUM | WARN |
| Code Quality | 5/10 | MEDIUM | WARN |
| Dead Code | 5/10 | LOW | WARN |

Raw average 3.78 → hard caps: Security CRITICAL (≤4.0), Build CRITICAL (≤5.0) → **3.8**.

---

## Strengths

1. **Clear layered architecture.** Unidirectional imports (`config → core → data → training → inference → export → ui/cli`) are respected in practice, and every Gradio event is wired in one place (`ui/app.py:build_demo`). This makes the fixes below local and low-risk.
2. **Robust archive handling.** `data/loader.py:safe_extract_zip` combines `realpath` containment (with the `os.sep` prefix-collision guard) and zip-bomb limits (file count, total size, ratio) — genuinely correct.
3. **Consistent optional-dependency gating.** `HAS_*` flags are centralised in `config/constants.py`; the Heretic check correctly uses `shutil.which` instead of spawning a subprocess.
4. **Good defensive touches in output paths.** HTML escaping in the evaluation preview (`_esc`), token redaction on Hub/registry errors (`redact_sensitive_info`), non-root `USER llmuser` in the Docker image, and the heredoc-based `HF_TOKEN` login that keeps the token off the process command line.
5. **Solid user documentation** (13 docs chapters) and vectorised data paths (PyArrow stats, pandas dedup).

---

## Findings by Dimension

### Security [3/10]

| Severity | Location | Issue | Recommendation |
|---|---|---|---|
| **CRITICAL** | `main.py:59-66`, `Dockerfile:123,171`, `docker-compose.yml` ports, 12× `trust_remote_code=True` (e.g. `inference/generate.py:90`, `training/sft.py:223,301`) | The UI binds `0.0.0.0` with **no authentication**, and any visitor can type an arbitrary Hub repo ID into "custom model" fields. Every load path passes `trust_remote_code=True`, which **executes Python code from that repo on the server** → unauthenticated remote code execution. (Latent today only because `main.py` cannot launch — it becomes live the moment the import bug is fixed.) | Default `trust_remote_code=False`; expose an explicit, off-by-default "Allow remote code" setting. Bind to `127.0.0.1` by default; wire `auth=` (env `GRADIO_AUTH`) into `demo.launch`. |
| **HIGH** | `Dockerfile` (`torch==2.5.1+cu126`), `ci.yml` (`torch==2.5.1`) + `export/utils.py:on_peft_zip_upload` → `inference/generate.py:93` | PyTorch ≤ 2.5.1 is affected by **CVE-2025-32434** (CVSS 9.3): `torch.load(weights_only=True)` can be bypassed for RCE. A user can upload a PEFT ZIP containing a crafted `adapter_model.bin`; the extracted path auto-populates `lora_path`, and `PeftModel.from_pretrained` loads it via `torch.load`. | Pin `torch>=2.6.0` everywhere (Dockerfiles, CI, requirements floor). Prefer safetensors-only adapters (reject `.bin` in uploaded ZIPs). |
| **HIGH** | `docker-entrypoint.sh:93-110` + `main.py:25` | The entrypoint's advertised `SHARE=true` and `EXTRA_ARGS="--auth user:pass"` append flags to `python3 main.py`; `main.py` routes **any** argument to the Typer CLI, which rejects them. The only documented way to add auth therefore cannot work. | Read `SHARE`/`GRADIO_AUTH` from env inside `main.py` and pass them to `demo.launch(share=..., auth=...)`; stop passing them as argv. |
| MEDIUM | `core/state.py:49-58` | `validate_path_traversal` is a string blacklist (`..`, `\`, NUL); it does not constrain *where* paths point. User-editable path fields (`gguf_tab export_model_path`, `merge_adapter_path_in`, `lora_path`) accept any absolute server path. Note: simply blocking absolute paths is **wrong** here — Gradio uploads and `tempfile.mkdtemp()` outputs are absolute. | Replace with an allow-list: resolve with `realpath` and require the result to be under the temp dir, the Gradio upload dir, or a configured workspace root. |
| MEDIUM | `constants.py` `gradio>=5.0.0`, `transformers>=4.48.0` | Version floors permit releases with published advisories (e.g. Gradio file-access/ACL-bypass advisories). | Raise floors to current patched versions and add `pip-audit` to a CI that actually runs (see Build). |
| LOW | `inference/generate.py:147`, `inference/vllm_runner.py:85,235`, training error returns | Exception text is shown raw in the UI; only Hub/registry paths use `redact_sensitive_info`. | Route all user-facing error strings through `redact_sensitive_info`. |

### Build & Types [2/10]

| Severity | Location | Issue | Recommendation |
|---|---|---|---|
| **CRITICAL** | `main.py:31,46` | `from commands import app` / `from app import build_demo` — no root-level `app.py` or `commands.py` exists (verified: `importlib.util.find_spec` returns `None` for both). Both UI mode and CLI mode raise `ModuleNotFoundError`; Docker's `exec python3 main.py` fails on start. Broken since commit `9bbf4d8`. | `from cli.commands import app as cli_app` and `from ui.app import build_demo`. Add a smoke test that imports `main` and calls `build_demo()`. |
| **CRITICAL** | `ci.yml` (repo root); no `.github/` directory | GitHub Actions only reads `.github/workflows/*.yml`. The lint/test/pip-audit/hadolint pipeline has **never executed**, which is how the entry-point break shipped. | `git mv ci.yml .github/workflows/ci.yml`. |
| **HIGH** | `training/sft.py:451,466,490`; `training/reward.py:168`; `training/orpo.py:177` | Trainers are constructed with `tokenizer=`. Current TRL (1.x) and Transformers 5 require `processing_class=` (confirmed via current docs). DPO's `except TypeError` fallback (`sft.py:457-472`) passes `tokenizer=` *and* `beta=` again, so it fails too. | Pass `processing_class=tokenizer`; drop the TypeError fallback and pin a supported TRL range. |
| **HIGH** | `training/ppo.py:148-232` | Uses the legacy PPO API (`PPOTrainer(config=...)`, `.generate()`, `.step()`) removed in TRL 0.12. In TRL 1.x, `PPOTrainer`/`ORPOTrainer` live under `trl.experimental.*`, so `HAS_PPO`/`HAS_ORPO` become `False` and PPO, ORPO **and Reward training** (which also checks `HAS_PPO`, `reward.py:66`) are disabled with a misleading "pip install trl" hint. | Rewrite PPO against the current `trl.experimental.ppo` API, or pin `trl<0.12` explicitly and document it. Import ORPO from its new location with a fallback. |
| **HIGH** | `training/ppo.py:225-226` | `AutoModelForCausalLMWithValueHead.forward` returns a tuple `(logits, loss, value)`; `outputs.values` is `tuple.values` → `AttributeError`. Reward computation cannot run on any TRL version. | Unpack `_, _, values = reward_model(**inputs)`. Better: train/load the reward model as `AutoModelForSequenceClassification(num_labels=1)`. |
| **HIGH** | `training/reward.py:98-170` | `RewardTrainer` expects a sequence-classification model whose output has `["logits"]`; it's given a value-head causal LM (tuple output). `tokenize_reward_function` also drops the `prompt`, so the reward model never sees the question. | Use `AutoModelForSequenceClassification(..., num_labels=1)`; build chosen/rejected texts as `prompt + response` (or pass raw columns to current `RewardTrainer`). |
| MEDIUM | `pyproject.toml` `[tool.setuptools.packages.find] where=["."]`, scripts `main:main` | `main.py` is a module, not a package, so it isn't installed → the `llm-finetune` console script fails on import. `find` also installs generic top-level packages named `config`, `core`, `data`, `ui`, `tests`… into site-packages (name collisions). | Move to a `src/llm_fine_tuner/` layout or declare `py-modules = ["main"]` and an explicit package include list. |
| MEDIUM | Tests | 22 of the 41 files in `tests/` are `benchmark_*`/`verify_*` scripts; no test imports `main.py`, constructs a real trainer, or exercises `ui.app.build_demo()`. | Add import/smoke tests (tiny model, `max_steps=1`) and a `build_demo()` test. |

### Concurrency [4/10]

| Severity | Location | Issue | Recommendation |
|---|---|---|---|
| **HIGH** | `core/state.py:74-125` | A single process-global `AppState` is shared by **all browser sessions**: one user's Stop button stops everyone's training; `cleanup_resource("_last_model_dir")` in user B's run deletes user A's freshly trained model, ZIP, GGUF, merged model and batch CSV. | Key state by `gr.Request.session_hash` (per-session `gr.State` for stop events and owned paths), or document single-user use and enforce `concurrency_limit=1` globally. |
| MEDIUM | `training/ppo.py` (no `stop_event.clear()`), `inference/evaluation.py:611` | `stop_event` is only cleared when SFT/Reward/ORPO start. After any Stop, PPO exits its loop immediately and reports "✅ PPO fine-tuning complete!" for an untrained model; evaluation silently produces **zero predictions** and metrics of 0.0. | Clear the event at the start of every long-running job; give evaluation its own cancel event. |
| MEDIUM | `inference/generate.py:63-111` | Loads happen outside the lock with no per-key guard: two concurrent requests (same or different models) load in parallel, briefly holding 2–3 models in VRAM → OOM. The old model is also still resident while the new one loads. | Per-key load lock with double-checked cache read; evict + `torch.cuda.empty_cache()` *before* loading a different model. |
| MEDIUM | `inference/vllm_runner.py:165-184` | `app_state.vllm_cache` is read/evicted/written with no lock; `del` on a vLLM `LLM` does not reliably release GPU memory. | Guard with a lock; on eviction call the engine's shutdown / `destroy_model_parallel` + `gc.collect()` + `empty_cache()`. |
| MEDIUM | `inference/evaluation.py:152-174` | Forces the `fork` start method inside a multi-threaded Gradio server that has initialised CUDA. Forking a threaded process can deadlock (Python 3.12+ warns about exactly this). | Use `forkserver`/`spawn` for the pool, or run BLEU/ROUGE sequentially (it's cheap relative to generation). |

### Code Principles [5/10]

| Severity | Location | Issue | Recommendation |
|---|---|---|---|
| MEDIUM | `training/sft.py:86-113` | `train_model` takes 25 positional-capable parameters and spans ~490 lines (tokenise, 3 load paths, 5 PEFT variants, 2 trainers, save, Heretic). | Introduce a `TrainConfig` dataclass; split into `load_model()`, `apply_peft()`, `build_trainer()`. |
| MEDIUM | `sft.py:212-307`, `orpo.py:90-108`, `ppo.py:99-104`, `generate.py:86-91`, `vllm_runner.py:63-68` | Model-loading/quantisation config duplicated 5×, with drift (fp16 compute in one path, bf16 in another, `trust_remote_code` inconsistent). | One `core/model_loading.py:load_causal_lm(name, quant, flash_attn, allow_remote_code)` helper. |
| MEDIUM | `sft.py:185-194`, `reward.py:124-133`, `orpo.py:121-130` | Tiny-dataset train/eval split guard copied 3×. | `data/splits.py:safe_train_eval_split(ds)`. |
| MEDIUM | `generate.py:199-224`, `evaluation.py:312-348`, `evaluation.py:610-639`, `cli/commands.py:434-452` | Batched generate + prompt-strip loop copied 4×. | Single `inference.generate.generate_batch(model, tok, prompts, **gen_kwargs)`. |
| LOW | `export/hub.py:57-67`, `export/registry.py:205-213,246-254` | HF token validation copied 3×. | `core.state.validate_hf_token()`. |

### Dependencies [3/10]

| Severity | Location | Issue | Recommendation |
|---|---|---|---|
| **HIGH** | `requirements.txt`, `pyproject.toml` | Only lower bounds (`trl>=0.8.0`, `transformers>=4.48.0`, `peft>=0.14.0`, `gradio>=5.0.0`). A fresh install resolves to TRL 1.x / Transformers 5.x, which break DPO, Reward, PPO and ORPO (see Build). | Declare tested ranges (e.g. `trl>=0.x,<0.y`) and ship a lock/constraints file; CI should test the upper bound. |
| **HIGH** | `Dockerfile`, `ci.yml` | `torch==2.5.1` pinned — vulnerable to CVE-2025-32434 (fixed in 2.6.0). Recent Transformers also refuse `.bin` loads on torch < 2.6. | `torch>=2.6` in both images and CI. |
| MEDIUM | `requirements.txt` | Declares `vllm`, `auto-gptq`, `exllamav2`, `bert-score`, `nlpaug` as **hard** requirements while the code and `pyproject.toml` treat them as optional. `vllm` drags its own torch pin; `auto-gptq` is no longer maintained and fails to build on many platforms → `pip install -r requirements.txt` fails on CPU/macOS. | Keep `requirements.txt` to core deps; reference extras (`.[vllm]`, `.[quant]`). Replace `auto-gptq` with `gptqmodel`/Transformers-native GPTQ. |
| LOW | `requirements.txt` | `PyPDF2` is deprecated in favour of `pypdf`. `vllm>=0.2.0` floor is meaningless. | Migrate to `pypdf`; set a realistic vLLM floor. |
| LOW | `inference/vllm_runner.py:177`, `constants.py:83` | UI offers quantization `"bnb"`; vLLM expects `"bitsandbytes"` → engine creation fails. | Map `"bnb" → "bitsandbytes"` or change the option label/value. |

### Observability [3/10]

| Severity | Location | Issue | Recommendation |
|---|---|---|---|
| MEDIUM | Whole codebase | Zero modules use `logging`; diagnostics are `print()` and user-facing strings. Server operators get no timestamps, levels, or tracebacks. | Add a module logger per file; `logger.exception()` in every broad `except`. |
| MEDIUM | `sft.py:573`, `loader.py:121,221`, `ppo.py:132` | `raise RuntimeError(f"...: {e}")` without `from e` discards the original traceback (ruff B904 — enabled in config, but lint never runs). | `raise RuntimeError(...) from e`. |
| MEDIUM | `training/sft.py:543-552` | Heretic subprocess return code/stderr ignored; UI reports "🔓 Heretic Mode applied!" even when the tool failed. | Check `returncode`, surface stderr. |
| LOW | `ui/handlers.py:223-229`, `inference_tab.py:46` | `batch_generate` returns error *strings* into a `gr.File` output, so users see a generic "file not found" instead of the message. | Return `(file_or_None, status_str)` to two components. |

### Lifecycle [4/10]

| Severity | Location | Issue | Recommendation |
|---|---|---|---|
| **HIGH** | `training/sft.py:511-573`, `reward.py:175-196`, `ppo.py:187-257`, `orpo.py:188-211` | GPU cleanup (`del model`, `empty_cache`, `gc.collect`) only runs on success. Any exception during training (OOM, bad data, API mismatch) leaves the model in VRAM until process restart. | `try/finally` around each training body. |
| MEDIUM | `ui/handlers.py:79-80,149`, `sft.py:497-505` | UI "Resume from checkpoint" can never work: each run uses a fresh `mkdtemp()` and the previous run's directory is deleted first. | Persist runs under a stable workspace (`/app/models/<run-name>`) and resume from there. |
| MEDIUM | `export/hub.py:75`, `export/utils.py:36` | Hub push and ZIP include `checkpoint-*` dirs (optimizer states, RNG state), multiplying upload size. `upload_folder` on a **new** repo fails — `create_repo` is never called (contrary to CLAUDE.md). | `create_repo(exist_ok=True)` first; `ignore_patterns=["checkpoint-*"]` (and skip them in the ZIP). |
| MEDIUM | `config/constants.py:147-244` | Import-time side effects: importing `unsloth` (which monkey-patches Transformers globally even when not selected), `vllm`, `exllamav2`, and a network `nltk.download("punkt")` on every cold start. `sentence_bleu` doesn't need punkt at all. | Probe with `importlib.util.find_spec` and import lazily at call sites; drop the punkt download. |
| LOW | `docker-compose.yml` | `HF_HUB_ENABLE_HF_TRANSFER=1` is set but `hf_transfer` is not installed; on `huggingface_hub` versions that honour it, downloads fail. `TRANSFORMERS_CACHE` is deprecated in favour of `HF_HOME`. | Install `hf_transfer` (or drop the flag); remove `TRANSFORMERS_CACHE`. |

### Code Quality [5/10]

| Severity | Location | Issue | Recommendation |
|---|---|---|---|
| MEDIUM | `data/preprocessing.py:216-273` + `sft.py:152,475` | `pad_token = eos_token` plus `DataCollatorForLanguageModeling` masks every pad **and EOS** position to `-100`; the non-chat path never appends EOS anyway. Fine-tuned models don't learn to stop generating. | Append `tokenizer.eos_token` to each text; use a distinct pad token or a completion-only collator. |
| MEDIUM | `sft.py:421`, `orpo.py:147`, `reward.py:150` | `fp16=True` is forced on CUDA even when weights/compute are bf16 (QLoRA Enhanced, flash-attn paths) — mixed AMP modes can error ("unscale FP16 gradients") or degrade. | `bf16=is_bf16_supported()`, `fp16=not bf16`. |
| MEDIUM | `sft.py:366-381`, `constants.py:141-145` | The UI's "Adapters" PEFT option can never work: `peft` has no `AdapterConfig` (it belongs to the separate `adapters` library), so `HAS_ADAPTER_CONFIG` is always `False`. | Remove the option or implement it with the `adapters` package. |
| MEDIUM | `data/augmentation.py:77-125`, `ui/handlers.py:97-99` | For DPO data (no `text`/`instruction` column), "augmentation" silently duplicates rows N× and reports success. Augmented/filtered datasets bypass `validate_and_clean_dataset` and the column mapping; augment and filter don't chain (each re-reads the original file). | Return an explicit "not supported for DPO" message; run validation on state datasets; chain operations on the current state. |
| LOW | `data/loader.py:154-161` | JSON/JSONL ignore `column_mapping` and don't select/enforce columns; nulls become the literal string `"None"` after `astype(str)` in cleaning. | Load via pandas for mapping parity; `fillna("")` before casting. |
| LOW | `export/registry.py:148` | Versions sorted lexicographically (`v10` before `v2`). | Sort with `packaging.version.parse`. |
| LOW | `ui/app.py:68-69`, `main.py`, `requirements.txt` header | "PRODUCTION READY" banners conflict with the current state; UI header lists changelog items instead of guidance. | Replace with version + concise status. |

### Dead Code [5/10]

| Severity | Location | Issue | Recommendation |
|---|---|---|---|
| LOW | `training/sft.py:576-620` | `load_qlora_model_v27` — self-described dead code. | Delete. |
| LOW | `ui/handlers.py:230-240` | Unreachable duplicate block after `return result`. | Delete. |
| LOW | `tests/benchmark_*.py` (19), `tests/verify_*.py` (3) | Not collected by pytest; ad-hoc scripts left by automated PRs. | Move to `benchmarks/` or delete. |
| INFO | Remote branches | ~80 stale `bolt-*`, `sentinel-*`, `palette-*` branches. | Prune merged/abandoned branches. |

---

## Advisory Findings

- **Rule 3 (single consumer):** duplicated HF-token validation in `export/registry.py` is consumed only by the Share tab — kept as LOW rather than MEDIUM.
- **Rule 4 (cohesion):** `inference/evaluation.py` (696 LOC) has a single responsibility, a unified data model (prompt/reference/prediction), and a clear domain boundary → `[High cohesion module]`; the inline CSS block could still move to `ui/css.py`.
- No ADR directory exists, so no findings were downgraded under Rule 1.

---

## Status of previous audit (branch `claude/project-audit-report-qsOS1`, commit `f20f80b`)

That commit is **not merged** and is based on a `main` that is now 84 commits old. Most of its fixes remain valid against current `main` (GPU `finally` blocks, per-key inference lock, `MAX_VLLM_ENGINES` clamp, DPO identical-pair warning, early-stop guard, progress 100 %, hyperparameter coercion, CSV quoting). **One change must not be applied:** BUG-02 (rejecting absolute paths in `validate_path_traversal`) would break every Gradio upload (`/tmp/gradio/...`), every `mkdtemp()` output directory, GGUF export and batch inference. The correct fix is the allow-list approach described under Security. Recommendation: restart the branch from current `main` and re-apply the valid subset.

---

## Recommended Actions (Priority Order)

1. **[CRITICAL]** Fix `main.py` imports (`ui.app`, `cli.commands`) and add an import smoke test.
2. **[CRITICAL]** Move `ci.yml` → `.github/workflows/ci.yml` so lint, tests and `pip-audit` actually run.
3. **[CRITICAL]** Default `trust_remote_code=False` with an explicit opt-in; bind to `127.0.0.1` by default; implement `auth`/`share` from env in `main.py` (and stop passing them as argv from the entrypoint).
4. **[HIGH]** Pin `torch>=2.6` (CVE-2025-32434) in Docker and CI; reject `.bin` files in uploaded adapter ZIPs.
5. **[HIGH]** Declare tested upper bounds for TRL/Transformers/PEFT; migrate trainer calls to `processing_class=`.
6. **[HIGH]** Rebuild the RLHF path: reward model as sequence classifier with prompt-aware inputs; PPO on the current TRL API (or pin and document the legacy range); fix `outputs.values`.
7. **[HIGH]** `try/finally` GPU cleanup in all four trainers; clear `stop_event` at the start of every job; scope state per session.
8. **[MEDIUM]** Hub push: `create_repo(exist_ok=True)` + ignore `checkpoint-*`; fix EOS/label masking; `bf16`/`fp16` selection; remove the "Adapters" option; vLLM `bitsandbytes` value.
9. **[MEDIUM]** Split `requirements.txt` into core vs extras; lazy-import heavy optional deps; replace `print` with `logging`; `raise ... from e`.
10. **[LOW]** Delete dead code and benchmark scripts from `tests/`; prune stale branches; tone down "PRODUCTION READY" banners.

---

## Sources Consulted

- Context7 — TRL v1.0.0 docs: `ORPOTrainer`/`PPOTrainer` under `trl.experimental`, `processing_class` in trainer signatures.
- Context7 — Transformers v5.0.0 migration guide: `processing_class`, `use_fast` ignored, `dtype` argument.
- [GitHub Advisory GHSA-53q9-r3pm-6pq6 / CVE-2025-32434](https://github.com/advisories/GHSA-53q9-r3pm-6pq6) — PyTorch ≤ 2.5.1 `torch.load(weights_only=True)` RCE, fixed in 2.6.0.
- [Gradio security advisories](https://github.com/gradio-app/gradio/security) and [A Security Review of Gradio 5](https://huggingface.co/blog/gradio-5-security).
- [vLLM BitsAndBytes documentation](https://docs.vllm.ai/en/stable/features/quantization/bnb/) — `quantization="bitsandbytes"`.
- Direct inspection of every non-archive Python module, Dockerfiles, entrypoint, compose file, and CI config at `origin/main@7491ca6`.
