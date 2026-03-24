"""
training/sft.py
================
Layer 3 — SFT and DPO training pipeline + model card + QLoRA loader.
Imports: config, core, data.

Functions
---------
train_model          — unified SFT/DPO training entry point
create_model_card    — generate a markdown model card string
load_qlora_model_v27 — standalone QLoRA model loader (retained, currently dead code)
"""

import gc
import glob
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from peft import (
    # C-2 FIX: AdapterConfig removed from unconditional top-level import.
    # It is an experimental feature absent from many peft releases. If this import
    # failed, the ENTIRE training module crashed before a single job could start.
    # AdapterConfig is now imported lazily and guarded by HAS_ADAPTER_CONFIG inside
    # the Adapters branch of train_model() below.
    LoraConfig,
    PeftModel,
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
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config.constants import (
    COL_INSTRUCTION,
    COL_OUTPUT,
    COL_TEXT,
    HAS_ADAPTER_CONFIG,
    HAS_HERETIC,    # N-5 FIX: imported so the Heretic Mode branch can guard the subprocess call
    HAS_TRL,
    HAS_UNSLOTH,
    QLORA_ENHANCED_BNB_KWARGS,
    QLORA_ENHANCED_LORA_CONFIG,
)
from core.callbacks import LoggingCallback, StopCallback
from core.hardware import get_lora_targets, is_unsloth_supported
from core.state import app_state
from data.preprocessing import preprocess_function


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
):
    """Unified SFT / DPO training pipeline.

    SFT and DPO share model loading, PEFT application, dataset splitting and
    TrainingArguments — they are intentionally kept in the same function.
    The DPO branch diverges only at trainer instantiation (~3 lines).

    Returns
    -------
    (summary_str, log_records_list)
    """
    # v2.9 Major Fix #2: Derive QLoRA Enhanced solely from peft_method.
    use_qlora_enhanced = (peft_method == "QLoRA Enhanced")
    # v3.0 Fix #1 (Critical): Define is_dpo here — was previously undefined.
    is_dpo = (training_mode == "dpo")

    app_state.stop_event.clear()
    log_callback = LoggingCallback()

    try:
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

        # ── Tokenise dataset ───────────────────────────────────────────────
        if progress is not None:
            progress(0.05, desc="Tokenising dataset… ")
        if is_dpo:
            tokenized = dataset
        else:
            task_type = (
                COL_INSTRUCTION
                if COL_INSTRUCTION in dataset.column_names and COL_OUTPUT in dataset.column_names
                else "lm"
            )
            tokenized = dataset.map(
                lambda x: preprocess_function(
                    x, tokenizer, hyperparams["max_length"],
                    task_type, use_chat_template, system_prompt,
                ),
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenising",
            )

        # ── Train / eval split ─────────────────────────────────────────────
        # v3.2 Fix #1 (High): Guard against datasets too small to split.
        # A single example produces an empty test set, crashing the Trainer.
        if len(tokenized) < 2:
            train_ds = tokenized
            eval_ds  = None
        else:
            split = tokenized.train_test_split(test_size=0.1, seed=42)
            train_ds, eval_ds = split["train"], split["test"]
            # Edge case: exactly 2 examples → 10% rounds to 0; force 1 eval row.
            if len(eval_ds) == 0:
                train_ds = tokenized.select(range(len(tokenized) - 1))
                eval_ds  = tokenized.select([len(tokenized) - 1])

        # ── Model loading ──────────────────────────────────────────────────
        if progress is not None:
            progress(0.1, desc="Loading model… ")
        is_unsloth  = False
        peft_applied = False  # prevent double PEFT application

        # ── Path A: QLoRA Enhanced (CUDA only) ────────────────────────────
        if use_qlora_enhanced and device != "cuda":
            log_callback.records.append({
                "step": 0, "train_loss": 0.0,
                "note": "⚠️ QLoRA Enhanced requested but CUDA unavailable — loading float32.",
            })
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
            model_kwargs = dict(quantization_config=bnb, device_map="auto", trust_remote_code=True)
            if use_flash_attn:
                # v3.1 Fix #2 (Critical): Guard bfloat16 with hardware support check.
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model_kwargs["torch_dtype"] = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            targets = (
                QLORA_ENHANCED_LORA_CONFIG["target_modules"]
                if not any(k in model_name.lower() for k in ["gpt2", "pythia", "falcon"])
                else get_lora_targets(model_name)
            )
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=QLORA_ENHANCED_LORA_CONFIG["r"],
                lora_alpha=QLORA_ENHANCED_LORA_CONFIG["lora_alpha"],
                target_modules=targets,
                lora_dropout=QLORA_ENHANCED_LORA_CONFIG["lora_dropout"],
                bias=QLORA_ENHANCED_LORA_CONFIG["bias"],
            )
            model = get_peft_model(model, lora_cfg)
            peft_applied = True

        # ── Path B: Unsloth ────────────────────────────────────────────────
        elif (
            use_unsloth
            and HAS_UNSLOTH
            and peft_method in ["LoRA", "Auto"]
            and is_unsloth_supported(model_name)
        ):
            from unsloth import FastLanguageModel, is_bfloat16_supported  # lazy

            dtype = None if is_bfloat16_supported() else torch.float16
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=hyperparams["max_length"],
                dtype=dtype,
                load_in_4bit=(device == "cuda"),
                trust_remote_code=True,
            )
            is_unsloth = True
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,
                target_modules=get_lora_targets(model_name),
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing=gradient_checkpointing,
                random_state=3407,
            )
            peft_applied = True

        # ── Path C: Standard HuggingFace load ─────────────────────────────
        else:
            if device == "cuda":
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs = dict(
                    quantization_config=bnb,
                    device_map="auto",
                    trust_remote_code=True,
                )
                # v3.2 Fix #5: Always set torch_dtype for non-quantised tensors.
                model_kwargs["torch_dtype"] = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                if use_flash_attn:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    # v3.1 Fix #2
                    model_kwargs["torch_dtype"] = (
                        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    )
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )

        # ── Warn if Unsloth + non-LoRA PEFT ───────────────────────────────
        # v2.9 Minor Fix #8
        if use_unsloth and HAS_UNSLOTH and peft_method not in ["LoRA", "Auto"]:
            print("⚠️ Warning: Unsloth is optimized for LoRA/Auto. Other PEFT methods may cause issues.")

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
                    target_modules=get_lora_targets(model_name),
                    lora_dropout=0.05,
                    bias="none",
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
                prompt_cfg = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=prompt_tuning_num_virtual_tokens,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    prompt_tuning_init_text="Classify the sentiment of this review:",
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
                targets = get_lora_targets(model_name)
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=targets,
                    lora_dropout=0.05,
                    bias="none",
                )
                model = get_peft_model(model, lora_cfg)
                print(
                    f"⚠️ QLoRA Enhanced: CUDA unavailable — NF4 quantization skipped. "
                    f"Falling back to standard LoRA (rank={lora_rank}, alpha={lora_alpha})."
                )

        # ── TrainingArguments + Trainer ────────────────────────────────────
        _eval_strategy = "no" if eval_ds is None else "steps"
        _load_best     = eval_ds is not None

        base_training_args = dict(
            output_dir=output_dir,
            overwrite_output_dir=True,
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
            fp16=(device == "cuda"),
            report_to="none",
            disable_tqdm=False,
            lr_scheduler_type=lr_scheduler_type,
            gradient_checkpointing=gradient_checkpointing,
        )

        if is_dpo:
            if not HAS_TRL:
                raise ImportError("TRL not installed. Run: pip install trl>=0.7.0")
            from trl import DPOConfig, DPOTrainer  # lazy

            dpo_callbacks = [StopCallback(), log_callback]
            if early_stop > 0 and eval_ds is not None:
                dpo_callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(early_stop)))

            # v2.9: Use DPOConfig for beta — passing beta to DPOTrainer directly
            # is deprecated in TRL >= 0.9.
            try:
                dpo_config = DPOConfig(**base_training_args, remove_unused_columns=False, beta=dpo_beta)
                trainer = DPOTrainer(
                    model=model,
                    args=dpo_config,
                    train_dataset=train_ds,
                    eval_dataset=eval_ds,
                    tokenizer=tokenizer,
                    callbacks=dpo_callbacks,
                )
            except TypeError:
                # Fallback for older TRL versions
                training_args = TrainingArguments(**base_training_args, remove_unused_columns=False)
                trainer = DPOTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_ds,
                    eval_dataset=eval_ds,
                    tokenizer=tokenizer,
                    beta=dpo_beta,
                    callbacks=dpo_callbacks,
                )
        else:
            training_args = TrainingArguments(**base_training_args)
            collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            sft_callbacks = [StopCallback(), log_callback]
            if early_stop > 0 and eval_ds is not None:
                sft_callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(early_stop)))
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=collator,
                tokenizer=tokenizer,
                callbacks=sft_callbacks,
            )

        # ── Resume from checkpoint ─────────────────────────────────────────
        resume_path = None
        if resume_from_checkpoint:
            ckpts = sorted(
                glob.glob(os.path.join(output_dir, "checkpoint-*")),
                key=lambda p: int(p.rsplit("-", 1)[-1]),
            )
            if ckpts:
                resume_path = ckpts[-1]

        # ── Train ──────────────────────────────────────────────────────────
        if progress is not None:
            progress(0.3, desc="Training started… ")
        t0 = time.time()
        trainer.train(resume_from_checkpoint=resume_path)
        elapsed = time.time() - t0
        status = "stopped by user" if app_state.stop_event.is_set() else "complete"

        # ── Save ───────────────────────────────────────────────────────────
        if progress is not None:
            progress(0.9, desc="Saving model… ")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
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
                    f"⏱ Elapsed: {elapsed/60:.1f} min\n"
                    f"📁 Model saved to: {output_dir}\n"
                )
            else:
                try:
                    subprocess.run(
                        ["heretic", output_dir],
                        capture_output=True, text=True, timeout=600,
                    )
                    summary = (
                        f"✅ Training {status}!\n"
                        f"🔓 Heretic Mode applied!\n"
                        f"⏱ Elapsed: {elapsed/60:.1f} min\n"
                        f"📁 Model saved to: {output_dir}\n"
                    )
                except Exception as e:
                    summary = (
                        f"✅ Training {status}!\n"
                        f"⚠️ Heretic failed: {e}\n"
                        f"⏱ Elapsed: {elapsed/60:.1f} min\n"
                        f"📁 Model saved to: {output_dir}\n"
                    )
        else:
            summary = (
                f"✅ Training {status}!\n"
                f"⏱ Elapsed: {elapsed/60:.1f} min\n"
                f"📁 Model saved to: {output_dir}\n"
            )

        if log_callback.records:
            summary += f"📉 Final train loss: {log_callback.records[-1]['train_loss']}"

        return summary, log_callback.records

    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")


def create_model_card(
    model_name: str,
    training_mode: str,
    peft_method: str,
    hyperparams: dict,
    output_dir: str,
) -> str:
    """Generate a markdown model card and write it to output_dir/README.md."""
    card = f"""---
base_model: {model_name}
tags:
  - fine-tuned
  - llm-fine-tuner-v3.2
  - {training_mode}
  - {peft_method.lower().replace(" ", "-")}
---

# Fine-tuned Model

**Base model:** `{model_name}`
**Training mode:** {training_mode.upper()}
**PEFT method:** {peft_method}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | {hyperparams.get("learning_rate", "N/A")} |
| Epochs | {hyperparams.get("epochs", "N/A")} |
| Batch size | {hyperparams.get("batch_size", "N/A")} |
| Max length | {hyperparams.get("max_length", "N/A")} |

*Generated by [LLM Fine-Tuner v3.2](https://github.com/Yog-Sotho/LLM-fine-tuner)*
"""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card)
    return card


def load_qlora_model_v27(model_name: str, use_flash_attn: bool = False):
    """Load a model with full QLoRA Enhanced config.

    NOTE (v3.1 Fix #7): This function is currently dead code — the equivalent
    logic is inlined inside run_ppo_v27() and train_model(). Retained for
    potential future use or external callers.
    """
    try:
        bnb_kwargs = dict(QLORA_ENHANCED_BNB_KWARGS)
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            bnb_kwargs["bnb_4bit_compute_dtype"] = torch.float16
        try:
            bnb = BitsAndBytesConfig(**bnb_kwargs, bnb_4bit_quant_storage=torch.bfloat16)
        except TypeError:
            bnb = BitsAndBytesConfig(**bnb_kwargs)
        model_kwargs = dict(quantization_config=bnb, device_map="auto", trust_remote_code=True)
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        targets = (
            QLORA_ENHANCED_LORA_CONFIG["target_modules"]
            if not any(k in model_name.lower() for k in ["gpt2", "pythia", "falcon"])
            else get_lora_targets(model_name)
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=QLORA_ENHANCED_LORA_CONFIG["r"],
            lora_alpha=QLORA_ENHANCED_LORA_CONFIG["lora_alpha"],
            target_modules=targets,
            lora_dropout=QLORA_ENHANCED_LORA_CONFIG["lora_dropout"],
            bias=QLORA_ENHANCED_LORA_CONFIG["bias"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        return model
    except Exception as e:
        raise RuntimeError(f"QLoRA Enhanced model load failed: {e}")
