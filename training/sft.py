"""
training/sft.py
================
Layer 3 — SFT and DPO training pipeline + QLoRA loader.
Imports: config, core, data.

Functions
---------
train_model          — unified SFT/DPO training entry point

Patch log
---------
  F-2  : ETAProgressCallback imported from core.callbacks and added to all
         trainer callback lists (SFT and DPO). The callback updates the Gradio
         progress bar with per-step ETA while training runs.
  M-4  : prompt_tuning_init_text was hardcoded to "Classify the sentiment of
         this review:" which is meaningless for non-sentiment tasks. It now
         uses the ``system_prompt`` parameter the user already configures in
         the UI ("You are a helpful, respectful and honest assistant." by
         default). This gives a sensible, task-neutral initialisation for
         Prompt Tuning in any domain.
  Fix-4 : Removed the dead ``create_model_card()`` function that was defined
          in this file.  It has a different (shorter) signature from the
          authoritative version in ``export/utils.py`` and still contained
          the old YAML-tag bug (empty string tag when heretic_mode=False)
          that was fixed in export/utils.py by M-2.  No external caller
          imports ``create_model_card`` from this module — all imports go
          through ``export.utils``.  Keeping a second, buggier version here
          created a maintenance hazard: future contributors could import
          from the wrong module and reintroduce the fixed bug.
          The module docstring Functions list has been updated accordingly.
"""

import gc
import subprocess
import threading
import time

import gradio as gr
import torch
from peft import (
    # C-2 FIX: AdapterConfig removed from unconditional top-level import.
    # It is an experimental feature absent from many peft releases. If this import
    # failed, the ENTIRE training module crashed before a single job could start.
    # AdapterConfig is now imported lazily and guarded by HAS_ADAPTER_CONFIG inside
    # the Adapters branch of train_model() below.
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)

from config.constants import (
    ALLOW_REMOTE_CODE,
    COL_IMAGES,
    COL_MESSAGES,
    DEFAULT_EVAL_SPLIT,
    DEFAULT_LORA_VARIANT,
    DEFAULT_SEED,
    HAS_ADAPTER_CONFIG,
    HAS_HERETIC,  # N-5 FIX: imported so the Heretic Mode branch can guard the subprocess call
    HAS_LIGER,
    HAS_TRL,
    HAS_UNSLOTH,
    QLORA_ENHANCED_BNB_KWARGS,
    QLORA_ENHANCED_LORA_CONFIG,
    TOKEN_STATS_SAMPLE,
    UNSLOTH_LORA_TARGETS,
)
from core.callbacks import (
    ETAProgressCallback,
    LoggingCallback,
    StopCallback,
)  # F-2: ETAProgressCallback added
from core.hardware import (
    compute_dtype,
    full_finetune_dtype,
    get_lora_targets,
    is_unsloth_supported,
    lora_variant_kwargs,
    select_precision,
)
from core.run_config import latest_checkpoint, save_run_config
from core.state import app_state, validate_path_traversal
from data.preprocessing import (
    drop_prompts_over_limit,
    format_token_report,
    to_sft_dataset,
    token_length_report,
)
from training.vision import train_vision_sft


def train_model(
    model_name,
    dataset,
    output_dir,
    hyperparams,
    device,
    peft_method,
    use_lora,
    lora_rank,
    lora_alpha,
    prefix_tuning_num_virtual_tokens,
    prefix_tuning_token_dim,
    prefix_tuning_num_layers,
    prompt_tuning_num_virtual_tokens,
    adapter_reduction_factor,
    resume_from_checkpoint,
    early_stop,
    lr_scheduler_type,
    gradient_checkpointing,
    use_unsloth,
    use_chat_template,
    system_prompt,
    training_mode="sft",
    dpo_beta=0.1,
    heretic_mode=False,
    progress=gr.Progress(),
    use_flash_attn=False,
    stop_event: threading.Event | None = None,
    seed: int = DEFAULT_SEED,
    report_to: str = "none",
    run_name: str | None = None,
    lora_variant: str = DEFAULT_LORA_VARIANT,
):
    """Unified SFT / DPO training pipeline.

    SFT and DPO share model loading, PEFT application, dataset splitting and
    TrainingArguments — they are intentionally kept in the same function.
    The DPO branch diverges only at trainer instantiation (~3 lines).

    Returns
    -------
    (summary_str, log_records_list)
    """
    # Sentinel: strip whitespace and validate against path traversal (blocking '..' and '\').
    model_name = model_name.strip() if model_name else ""
    output_dir = output_dir.strip() if output_dir else ""

    if err := (validate_path_traversal(model_name) or validate_path_traversal(output_dir)):
        raise ValueError(err)

    # v2.9 Major Fix #2: Derive QLoRA Enhanced solely from peft_method.
    use_qlora_enhanced = peft_method == "QLoRA Enhanced"
    variant_kwargs = lora_variant_kwargs(lora_variant)  # rejects unknown variants early
    # v3.0 Fix #1 (Critical): Define is_dpo here — was previously undefined.
    is_dpo = training_mode == "dpo"

    stop_event = stop_event or app_state.session().stop_event
    stop_event.clear()
    # Seed before any model/adapter is built: LoRA initialises its weights at creation,
    # before the Trainer would seed, so the seed must be set here to reproduce a run.
    set_seed(int(seed))
    log_callback = LoggingCallback()

    try:
        # ── Vision-language data: image + text chats ──────────────────────
        if COL_IMAGES in dataset.column_names:
            if is_dpo:
                raise ValueError("DPO on image + text data isn't supported; use SFT.")
            return train_vision_sft(
                model_name, dataset, output_dir, hyperparams, device, peft_method, use_lora,
                lora_rank, lora_alpha, lora_variant, gradient_checkpointing, lr_scheduler_type,
                int(early_stop), resume_from_checkpoint, int(seed), report_to, run_name,
                stop_event, progress,
            )  # fmt: skip

        # ── Tokenizer ─────────────────────────────────────────────────────
        if progress is not None:
            progress(0, desc="Loading tokenizer… ")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.eos_token is None:
            if hasattr(tokenizer, "bos_token") and tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
            elif hasattr(tokenizer, "unk_token") and tokenizer.unk_token:
                tokenizer.eos_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({"eos_token": "</s>"})
                tokenizer.eos_token = "</s>"
        tokenizer.pad_token = tokenizer.eos_token

        # ── Prepare dataset ────────────────────────────────────────────────
        # SFT: TRL prompt-completion / text format; SFTTrainer tokenises, masks the
        # prompt out of the loss and appends EOS. DPO data is already prompt/chosen/rejected.
        if progress is not None:
            progress(0.05, desc="Preparing dataset… ")
        if COL_MESSAGES in dataset.column_names and not tokenizer.chat_template:
            raise ValueError(
                f"'{model_name}' has no chat template, so chat-format ('messages') data "
                "can't be rendered. Use an instruct/chat model or instruction/output data."
            )
        dropped_long_prompts = 0
        if is_dpo:
            tokenized = dataset
        else:
            tokenized = to_sft_dataset(
                dataset,
                use_chat_template=bool(use_chat_template and tokenizer.chat_template),
                system_prompt=system_prompt,
            )
            tokenized, dropped_long_prompts = drop_prompts_over_limit(
                tokenized, tokenizer, int(hyperparams["max_length"])
            )
            if len(tokenized) == 0:
                raise ValueError(
                    f"Every prompt is at least Max Sequence Length ({hyperparams['max_length']}) "
                    "tokens, so no answer would be trained. Raise Max Sequence Length."
                )

        # ── Train / eval split ─────────────────────────────────────────────
        # v3.2 Fix #1 (High): Guard against datasets too small to split.
        # A single example produces an empty test set, crashing the Trainer.
        eval_split = float(hyperparams.get("eval_split", DEFAULT_EVAL_SPLIT))
        if not 0.0 <= eval_split < 1.0:
            raise ValueError("Eval split must be between 0 and 1 (0 = no evaluation).")
        if len(tokenized) < 2 or eval_split == 0.0:
            train_ds = tokenized
            eval_ds = None
        else:
            split = tokenized.train_test_split(test_size=eval_split, seed=seed)
            train_ds, eval_ds = split["train"], split["test"]
            # Edge case: exactly 2 examples → 10% rounds to 0; force 1 eval row.
            if len(eval_ds) == 0:
                train_ds = tokenized.select(range(len(tokenized) - 1))
                eval_ds = tokenized.select([len(tokenized) - 1])

        # Token lengths as the trainer will see them (chat template applied) — the
        # character counts shown at upload time can't tell what gets truncated.
        token_report = token_length_report(
            tokenized, tokenizer, int(hyperparams["max_length"]), TOKEN_STATS_SAMPLE
        )
        if progress is not None:
            progress(0.08, desc=format_token_report(token_report).splitlines()[0])

        # ── Model loading ──────────────────────────────────────────────────
        if progress is not None:
            progress(0.1, desc="Loading model… ")
        peft_applied = False  # prevent double PEFT application

        # ── Path A: QLoRA Enhanced (CUDA only) ────────────────────────────
        if use_qlora_enhanced and device != "cuda":
            log_callback.records.append(
                {
                    "step": 0,
                    "train_loss": 0.0,
                    "eval_loss": float("nan"),
                    "elapsed_s": 0.0,
                    "eta_s": 0.0,
                    "note": "⚠️ QLoRA Enhanced requested but CUDA unavailable — loading float32.",
                }
            )
            if progress is not None:
                progress(0.1, desc="⚠️ QLoRA Enhanced: CUDA unavailable, loading float32…")

        if use_qlora_enhanced and device == "cuda":
            if progress is not None:
                progress(0.1, desc="Loading model with QLoRA Enhanced (NF4 + double quant)… ")
            bnb_kwargs = dict(QLORA_ENHANCED_BNB_KWARGS)
            # v3.0 Fix #5: Fall back to float16 if bfloat16 unsupported.
            if not torch.cuda.is_bf16_supported():
                bnb_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            try:
                bnb = BitsAndBytesConfig(**bnb_kwargs, bnb_4bit_quant_storage=torch.bfloat16)
            except TypeError:
                bnb = BitsAndBytesConfig(**bnb_kwargs)
            model_kwargs = dict(
                quantization_config=bnb, device_map="auto", trust_remote_code=ALLOW_REMOTE_CODE
            )
            if use_flash_attn:
                # v3.1 Fix #2 (Critical): Guard bfloat16 with hardware support check.
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model_kwargs["torch_dtype"] = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=QLORA_ENHANCED_LORA_CONFIG["r"],
                lora_alpha=QLORA_ENHANCED_LORA_CONFIG["lora_alpha"],
                target_modules=QLORA_ENHANCED_LORA_CONFIG["target_modules"],
                lora_dropout=QLORA_ENHANCED_LORA_CONFIG["lora_dropout"],
                bias=QLORA_ENHANCED_LORA_CONFIG["bias"],
                **variant_kwargs,
            )
            model = get_peft_model(model, lora_cfg)
            peft_applied = True

        # ── Path B: Unsloth ────────────────────────────────────────────────
        elif (
            use_unsloth
            and HAS_UNSLOTH
            and peft_method in ["LoRA", "Auto"]
            and is_unsloth_supported(model_name)
            and lora_variant != "DoRA"  # Unsloth documents no DoRA support
        ):
            from unsloth import FastLanguageModel, is_bfloat16_supported  # lazy

            dtype = None if is_bfloat16_supported() else torch.float16
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=hyperparams["max_length"],
                dtype=dtype,
                load_in_4bit=(device == "cuda"),
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,
                target_modules=UNSLOTH_LORA_TARGETS,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                use_rslora=lora_variant == "rsLoRA",
                use_gradient_checkpointing=gradient_checkpointing,
                random_state=3407,
            )
            peft_applied = True

        # ── Path C: Standard HuggingFace load ─────────────────────────────
        else:
            # Full fine-tuning trains every weight, so the model must not be quantised:
            # Transformers refuses to train a quantised model without adapters.
            full_finetune = peft_method == "Full Fine-tuning" or (
                peft_method == "Auto" and not use_lora
            )
            if device == "cuda" and full_finetune:
                weights_dtype = full_finetune_dtype(device)
                model_kwargs = dict(torch_dtype=weights_dtype, trust_remote_code=ALLOW_REMOTE_CODE)
                if use_flash_attn and weights_dtype == torch.bfloat16:  # FA2 needs fp16/bf16
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            elif device == "cuda":
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype(device),
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs = dict(
                    quantization_config=bnb,
                    device_map="auto",
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )
                # Non-quantised tensors use the same dtype as the mixed-precision mode.
                model_kwargs["torch_dtype"] = compute_dtype(device)
                if use_flash_attn:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=ALLOW_REMOTE_CODE,
                )

        if use_unsloth and HAS_UNSLOTH and lora_variant == "DoRA":
            log_callback.records.append(
                {
                    "step": 0,
                    "train_loss": 0.0,
                    "eval_loss": float("nan"),
                    "elapsed_s": 0.0,
                    "eta_s": 0.0,
                    "note": "⚠️ Unsloth skipped: it does not support DoRA — using PEFT.",
                }
            )

        # ── Warn if Unsloth + non-LoRA PEFT ───────────────────────────────
        # v2.9 Minor Fix #8
        if use_unsloth and HAS_UNSLOTH and peft_method not in ["LoRA", "Auto"]:
            print(
                "⚠️ Warning: Unsloth is optimized for LoRA/Auto. Other PEFT methods may cause issues."
            )

        # ── Apply PEFT (if not already applied) ───────────────────────────
        # v3.1 Fix #5: Warn when Auto + use_lora=False → full fine-tune.
        if peft_method == "Auto" and not use_lora and not peft_applied:
            print(
                "⚠️ PEFT method is 'Auto' but 'Enable LoRA' is unchecked — "
                "no adapter will be applied. Training will proceed as full fine-tuning."
            )

        if peft_method != "Full Fine-tuning" and not peft_applied:
            if progress is not None:
                progress(0.15, desc=f"Applying {peft_method}… ")

            if peft_method == "LoRA" or (peft_method == "Auto" and use_lora):
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=get_lora_targets(),
                    lora_dropout=0.05,
                    bias="none",
                    **variant_kwargs,
                )
                model = get_peft_model(model, lora_cfg)

            elif peft_method == "Prefix Tuning":
                # v3.1 Fix #1 (Critical): PrefixTuningConfig uses encoder_hidden_size
                # and num_layers — NOT token_dim / num_transformer_layers.
                prefix_cfg = PrefixTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=prefix_tuning_num_virtual_tokens,
                    encoder_hidden_size=prefix_tuning_token_dim,
                    num_layers=prefix_tuning_num_layers,
                )
                model = get_peft_model(model, prefix_cfg)

            elif peft_method == "Prompt Tuning":
                # v3.1 Fix #1 (Critical): PromptTuningConfig does NOT accept
                # num_transformer_layers — removed to prevent TypeError.
                #
                # M-4 FIX: prompt_tuning_init_text was hardcoded to
                # "Classify the sentiment of this review:" which is wrong for
                # non-sentiment tasks. It now uses the ``system_prompt``
                # parameter that the user already customises in the UI,
                # giving a task-neutral initialisation for any domain.
                prompt_cfg = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=prompt_tuning_num_virtual_tokens,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    prompt_tuning_init_text=system_prompt,  # M-4 FIX
                    tokenizer_name_or_path=model_name,
                )
                model = get_peft_model(model, prompt_cfg)

            elif peft_method == "Adapters":
                # C-2 FIX: AdapterConfig is now imported lazily here, guarded by
                # HAS_ADAPTER_CONFIG. Previously this was an unconditional top-level
                # import that crashed the entire module on peft versions without it.
                if not HAS_ADAPTER_CONFIG:
                    raise ImportError(
                        "AdapterConfig requires the adapter-transformers fork of peft. "
                        "Install with: pip install adapter-transformers"
                    )
                from peft import AdapterConfig  # lazy, guarded  # noqa: PLC0415

                adapter_cfg = AdapterConfig(
                    non_linearity="relu",
                    reduction_factor=adapter_reduction_factor,
                    leave_out=[],
                )
                model = get_peft_model(model, adapter_cfg)

            elif peft_method == "QLoRA Enhanced":
                # v3.0 Fix #3 & #4: CUDA unavailable — fall back to standard LoRA.
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=get_lora_targets(),
                    lora_dropout=0.05,
                    bias="none",
                    **variant_kwargs,
                )
                model = get_peft_model(model, lora_cfg)
                print(
                    f"⚠️ QLoRA Enhanced: CUDA unavailable — NF4 quantization skipped. "
                    f"Falling back to standard LoRA (rank={lora_rank}, alpha={lora_alpha})."
                )

        # ── TrainingArguments + Trainer ────────────────────────────────────
        _eval_strategy = "no" if eval_ds is None else "steps"
        _load_best = eval_ds is not None

        base_training_args = dict(
            output_dir=output_dir,
            num_train_epochs=hyperparams["epochs"],
            per_device_train_batch_size=hyperparams["batch_size"],
            gradient_accumulation_steps=hyperparams["grad_accum"],
            learning_rate=hyperparams["learning_rate"],
            warmup_steps=hyperparams["warmup_steps"],
            logging_steps=10,
            eval_strategy=_eval_strategy,
            eval_steps=50 if eval_ds is not None else None,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=_load_best,
            metric_for_best_model="eval_loss" if _load_best else None,
            greater_is_better=False,
            # bf16 on GPUs that support it, else fp16; full precision on CPU. Always
            # explicit: TRL configs default to bf16=True, which fails on CPU.
            **select_precision(device),
            report_to=report_to,
            run_name=run_name,
            seed=seed,
            disable_tqdm=False,
            lr_scheduler_type=lr_scheduler_type,
            gradient_checkpointing=gradient_checkpointing,
            # Reentrant checkpointing gives "does not require grad" with frozen LoRA base weights.
            gradient_checkpointing_kwargs={"use_reentrant": False}
            if gradient_checkpointing
            else None,
        )

        if is_dpo:
            if not HAS_TRL:
                raise ImportError('TRL not installed. Run: pip install "trl>=0.29.1,<2"')
            from trl import DPOConfig, DPOTrainer  # lazy

            dpo_callbacks = [StopCallback(stop_event), log_callback]
            # Early stopping on < 50 train rows reacts to a 1-row eval set's noise.
            if early_stop > 0 and eval_ds is not None and len(train_ds) >= 50:
                dpo_callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(early_stop)))
            # F-2: Add ETA progress callback for DPO training
            if progress is not None:
                dpo_callbacks.append(ETAProgressCallback(gradio_progress=progress))

            dpo_config = DPOConfig(
                **base_training_args,
                remove_unused_columns=False,
                beta=dpo_beta,
                max_length=int(hyperparams["max_length"]),  # TRL otherwise uses 1024
            )
            trainer = DPOTrainer(
                model=model,
                args=dpo_config,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                processing_class=tokenizer,
                callbacks=dpo_callbacks,
            )
        else:
            from trl import SFTConfig, SFTTrainer  # lazy

            # Packing concatenates short samples; without Flash Attention the packed
            # samples attend to each other (TRL warns about cross-contamination).
            # Decided by how the model was actually loaded: some paths (e.g. fp32 full
            # fine-tuning) skip Flash Attention even when it was requested.
            uses_fa2 = getattr(model.config, "_attn_implementation", None) == "flash_attention_2"
            packing = bool(hyperparams.get("packing")) and uses_fa2 and device == "cuda"
            if hyperparams.get("packing") and not packing:
                log_callback.records.append(
                    {
                        "step": 0,
                        "train_loss": 0.0,
                        "eval_loss": float("nan"),
                        "elapsed_s": 0.0,
                        "eta_s": 0.0,
                        "note": "⚠️ Packing skipped: it needs Flash Attention 2 on a CUDA GPU.",
                    }
                )
            sft_config = SFTConfig(
                **base_training_args,
                max_length=hyperparams["max_length"],
                packing=packing,
                # Dataset prep stays in-process: forking worker processes from a process
                # that already runs threads (torch, the Gradio server) can deadlock, and
                # fast tokenizers already tokenise batches in parallel internally.
                dataset_num_proc=None,
                # Fused Triton kernels (lower memory, faster) when installed on CUDA.
                use_liger_kernel=HAS_LIGER and device == "cuda",
            )
            sft_callbacks = [StopCallback(stop_event), log_callback]
            if early_stop > 0 and eval_ds is not None and len(train_ds) >= 50:
                sft_callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(early_stop)))
            # F-2: Add ETA progress callback for SFT training
            if progress is not None:
                sft_callbacks.append(ETAProgressCallback(gradio_progress=progress))
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                processing_class=tokenizer,
                callbacks=sft_callbacks,
            )

        # ── Resume from checkpoint ─────────────────────────────────────────
        resume_path = latest_checkpoint(output_dir) if resume_from_checkpoint else None

        # ── Train ──────────────────────────────────────────────────────────
        if progress is not None:
            progress(0.3, desc="Training started… calculating ETA…")
        t0 = time.time()
        trainer.train(resume_from_checkpoint=resume_path)
        elapsed = time.time() - t0
        status = "stopped by user" if stop_event.is_set() else "complete"

        # ── Save ───────────────────────────────────────────────────────────
        if progress is not None:
            progress(0.9, desc="Saving model… ")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        save_run_config(
            output_dir,
            mode=training_mode,
            model=model_name,
            dataset=dataset,
            seed=seed,
            report_to=report_to,
            hyperparams=dict(hyperparams),
            peft={
                "method": peft_method,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_variant": lora_variant,
            },
            use_chat_template=bool(use_chat_template),
            system_prompt=system_prompt,
            dpo_beta=dpo_beta if is_dpo else None,
            use_flash_attn=bool(use_flash_attn),
            gradient_checkpointing=bool(gradient_checkpointing),
            lr_scheduler_type=lr_scheduler_type,
            early_stop=int(early_stop),
            token_stats=token_report,
            dropped_long_prompts=dropped_long_prompts,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # ── Heretic Mode ───────────────────────────────────────────────────
        if heretic_mode:
            if progress is not None:
                progress(0.95, desc="🔓 Applying Heretic… ")

            # N-5 FIX: Guard the subprocess call with HAS_HERETIC so users get a
            # clear diagnostic instead of a FileNotFoundError crash when the heretic
            # binary is not installed (it is now an optional dependency).
            if not HAS_HERETIC:
                summary = (
                    f"✅ Training {status}!\n"
                    f"⚠️ Heretic Mode skipped — binary not found.\n"
                    f"   Install with: pip install heretic-llm\n"
                    f"⏱ Elapsed: {elapsed / 60:.1f} min\n"
                    f"📁 Model saved to: {output_dir}\n"
                )
            else:
                try:
                    # heretic is interactive: no stdin, so a prompt ends it instead of
                    # blocking on the server's stdin until the timeout.
                    heretic = subprocess.run(
                        ["heretic", output_dir],
                        capture_output=True,
                        text=True,
                        timeout=600,
                        stdin=subprocess.DEVNULL,
                    )
                    if heretic.returncode == 0:
                        heretic_note = "🔓 Heretic Mode applied!"
                    else:
                        tail = (heretic.stderr or heretic.stdout or "").strip().splitlines()[-3:]
                        heretic_note = (
                            f"⚠️ Heretic failed (exit code {heretic.returncode}):\n"
                            + "\n".join(tail)
                        )
                    summary = (
                        f"✅ Training {status}!\n"
                        f"{heretic_note}\n"
                        f"⏱ Elapsed: {elapsed / 60:.1f} min\n"
                        f"📁 Model saved to: {output_dir}\n"
                    )
                except Exception as e:
                    summary = (
                        f"✅ Training {status}!\n"
                        f"⚠️ Heretic failed: {e}\n"
                        f"⏱ Elapsed: {elapsed / 60:.1f} min\n"
                        f"📁 Model saved to: {output_dir}\n"
                    )
        else:
            summary = (
                f"✅ Training {status}!\n"
                f"⏱ Elapsed: {elapsed / 60:.1f} min\n"
                f"📁 Model saved to: {output_dir}\n"
            )

        if log_callback.records:
            summary += f"📉 Final train loss: {log_callback.records[-1]['train_loss']}"
        summary += "\n" + format_token_report(token_report)
        if dropped_long_prompts:
            summary += (
                f"\n⚠️ {dropped_long_prompts} examples skipped: their prompt alone fills Max "
                "Sequence Length, so the answer would be cut off. Raise it to keep them."
            )

        if progress is not None:
            progress(1.0, desc="✅ Complete!")

        return summary, log_callback.records

    except Exception as e:
        raise RuntimeError(f"Training failed: {e}") from e
    finally:
        # Free VRAM on failure too; `model` is unbound if loading never happened
        # and already deleted on the success path.
        try:
            del model
        except (NameError, UnboundLocalError):
            pass
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
