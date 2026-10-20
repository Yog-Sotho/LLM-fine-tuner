# Changelog

All notable changes to LLM Fine-Tuner. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The version is set once, in
`config/constants.py` (`APP_VERSION`); pushing a tag `vX.Y.Z` that matches it publishes a
signed release (`.github/workflows/release.yml`) with this file's section as its notes.

## [Unreleased]

### Added

- **Container image** `ghcr.io/yog-sotho/llm-fine-tuner:{version,latest}` (built from
  `Dockerfile.cpu`) for `linux/amd64` and `linux/arm64`, published by the release workflow,
  signed with Sigstore (keyless) and with build provenance. A `Docker` workflow builds and
  smoke-tests it on both architectures for pull requests that change it. The GPU image is not
  published: its CUDA stack is too large for hosted build runners.

## [3.3.1] - 2026-10-20

### Added
- PyPI publishing via OIDC Trusted Publishing. The distribution is named
  `llm-finetuner-pro` (the bare name `llm-fine-tuner` is unavailable on PyPI);
  the `llm-finetune` command and the project brand are unchanged.

## [3.3.0] - 2026-10-18

The first published release. It covers everything since version 3.2.0 was set
(2026-09-28, Tier 11).

### Added

- **Knowledge distillation (GKD):** a small student learns from a larger teacher of the same
  family (UI **🤖 RLHF → 🎓 E. Distillation**, CLI `distill`).
- **IA3** PEFT method; it replaces the "Adapters" option, which always failed.
- **LoRA adapter merging** (TIES, DARE-TIES, DARE-linear, linear, cat, SVD): CLI
  `merge-adapters`, GGUF Export tab.
- **Training data from documents:** PDF, Word, `.txt` and `.md` files are chunked and an LLM
  (any OpenAI-compatible server or a local model) writes and rates question/answer pairs
  (Data tab, CLI `synthesize`).
- **Embedding models for search / RAG:** MNRL + Matryoshka training, optional mined hard
  negatives, full fine-tuning or merged LoRA, NDCG/MRR/Recall before and after
  (**🔎 Embeddings** tab, CLI `embed`).
- **Mixture-of-experts models:** router load-balancing loss and a frozen router by default,
  recorded in `run_config.yaml` and the model card.
- **Multi-GPU sharding presets** for `accelerate launch`: `multi_gpu`, `fsdp2`, `fsdp_qlora`,
  `deepspeed_zero2`, `deepspeed_zero3` (`configs/accelerate/`); long-sequence options.
- **Path allow-list** for the web UI (`LFT_ALLOWED_PATHS`); error text shown in the UI is
  redacted.
- GPU test suite (`tests/test_gpu.py`, `.github/workflows/gpu.yml`), mypy on every layer,
  coverage gate of 85 %.
- Documentation site: <https://yog-sotho.github.io/LLM-fine-tuner/>.
- Dependabot version updates (Python packages, GitHub Actions, Docker base images).
- This changelog and a signed release workflow (Sigstore).

### Changed

- **Dependency minimums** are the first releases without known advisories: Transformers
  5.10.1, datasets 5.0.1, torch 2.13, PEFT 0.18, Gradio 6.16, accelerate 1.15,
  huggingface_hub 1.5, sentence-transformers 5.6. Transformers 4 and datasets 4 are no longer
  supported. PyTorch 2.13+ has no CUDA 11 build: `install.sh` installs CPU wheels on CUDA 11
  drivers and says to update the driver.
- **TRL is capped below 1.15:** its fused LM head needs Triton on a GPU, so SFT/DPO/KTO/GRPO
  could no longer train on CPU.
- **BLEU is built in** (the same score as nltk's smoothing 4); nltk is no longer a direct
  dependency. Synonym augmentation needs it (`pip install nltk`; rouge-score installs it).
- The package metadata declares the licence the repository has always had, **GPL-3.0-only**
  (it said MIT).
- vLLM is the `vllm` extra (`trl[vllm]`), no longer in `requirements.txt`, so a CPU or macOS
  install works.
- Logging instead of `print()`; one version source; `train_model()` and the CLI split into
  smaller steps (behaviour unchanged, pinned by `tests/test_loading_snapshot.py`).
- `docker-compose.yml` passes `LFT_RUNS_DIR` and `LFT_ALLOWED_PATHS` through.

### Fixed

- LoRA training of mixture-of-experts models failed on Transformers 5 / PEFT 0.21
  (`lora_dropout` must be 0 there).
- Twelve findings of the codebase audit (`docs/project/codebase_audit.md`), and three bugs
  found during the Tier 18 refactor.
- The documentation: commands that failed as written (the Colab `--share` launch, PyPI
  installs of a package that isn't published), features described that don't exist (ZIP
  uploads, MPS training, several UI controls), and wrong defaults and labels.

### Removed

- `auto-gptq` and `exllamav2` (never used), the "Adapters" PEFT option (see IA3), and the
  import-time `nltk.download("punkt")`.

## [3.2.0] and earlier

Not published as releases. The version history is in `README_for_geeks.md`.
