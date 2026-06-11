"""
training/orpo.py
=================
Layer 3 — ORPO (Odds Ratio Preference Optimisation) training.
Imports: config, core, data.

Patch log
---------
  F-2  : ETAProgressCallback added to the ORPOTrainer callback list so the
         Gradio progress bar shows per-step ETA during ORPO training.
"""

import gc
import inspect
import os
import time

import gradio as gr
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config.constants import (
    COL_PROMPT,
    COL_CHOSEN,
    COL_REJECTED,
    HAS_ORPO,
)
from core.callbacks import ETAProgressCallback, LoggingCallback, StopCallback  # F-2: ETAProgressCallback added
from core.hardware import get_lora_targets
from core.state import app_state, validate_path_traversal
from data.loader import detect_file_type, load_dataset_from_file
from data.preprocessing import validate_and_clean_dataset


def train_orpo_v27(
    model_name: str,
    orpo_file,
    output_dir: str,
    orpo_lr: float = 1e-4,
    orpo_beta: float = 0.1,
    orpo_alpha: float = 0.1,
    orpo_epochs: int = 3,
    orpo_batch_size: int = 2,
    progress=gr.Progress(),
) -> str:
    """Train using ORPO (Odds Ratio Preference Optimisation).

    Requires trl>=0.8.0 (HAS_ORPO=True).
    Dataset must contain 'prompt', 'chosen', 'rejected' columns.

    Returns a status string for display in the UI.
    """
    # Sentinel: strip whitespace and validate against path traversal.
    model_name = model_name.strip() if model_name else ""
    output_dir = output_dir.strip() if output_dir else ""

    if err := (validate_path_traversal(model_name) or validate_path_traversal(output_dir)):
        return err

    if not HAS_ORPO:
        return "❌ ORPOTrainer not available. Install: pip install trl>=0.8.0"
    if orpo_file is None:
        return "❌ Please upload a preference dataset (prompt, chosen, rejected)."

    # Clear the stop event at the start of every ORPO training run.
    app_state.stop_event.clear()

    try:
        from trl import ORPOConfig, ORPOTrainer  # lazy

        if progress is not None:
            progress(0, desc="Loading ORPO dataset…")
        ftype = detect_file_type(orpo_file)
        ds = load_dataset_from_file(orpo_file, ftype, is_dpo=True)

        required = [COL_PROMPT, COL_CHOSEN, COL_REJECTED]
        if not all(c in ds.column_names for c in required):
            return f"❌ Dataset must contain: {required}. Found: {ds.column_names}"

        ds, _ = validate_and_clean_dataset(ds, is_dpo=True)
        if len(ds) == 0:
            return "❌ Dataset is empty after cleaning."

        if progress is not None:
            progress(0.05, desc="Loading tokenizer & model…")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if torch.cuda.is_available():
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=get_lora_targets(model_name),
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

        # v3.2 Fix #1: Guard against datasets too small to produce a non-empty eval split.
        if len(ds) < 2:
            orpo_train_ds = ds
            orpo_eval_ds  = None
        else:
            split = ds.train_test_split(test_size=0.1, seed=42)
            orpo_train_ds = split["train"]
            orpo_eval_ds  = split["test"]
            if len(orpo_eval_ds) == 0:
                orpo_train_ds = ds.select(range(len(ds) - 1))
                orpo_eval_ds  = ds.select([len(ds) - 1])

        _orpo_eval_strategy = "no" if orpo_eval_ds is None else "steps"
        _orpo_load_best     = orpo_eval_ds is not None

        orpo_config_kwargs = dict(
            output_dir=output_dir,
            learning_rate=orpo_lr,
            beta=orpo_beta,
            num_train_epochs=orpo_epochs,
            per_device_train_batch_size=orpo_batch_size,
            eval_strategy=_orpo_eval_strategy,
            eval_steps=50 if orpo_eval_ds is not None else None,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=_orpo_load_best,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        # Guard alpha — added in TRL >= 0.8.1; silently omit on older installs.
        try:
            import inspect as _inspect
            if "alpha" in _inspect.signature(ORPOConfig.__init__).parameters:
                orpo_config_kwargs["alpha"] = orpo_alpha
        except Exception:
            pass

        orpo_config = ORPOConfig(**orpo_config_kwargs)
        log_cb = LoggingCallback()

        # Build callback list: stop button, logging, and ETA progress bar
        orpo_callbacks = [StopCallback(), log_cb]
        # F-2: ETAProgressCallback wired in so users see per-step ETA.
        if progress is not None:
            orpo_callbacks.append(
                ETAProgressCallback(gradio_progress=progress, progress_start=0.3, progress_end=0.9)
            )

        orpo_kwargs = dict(
            model=model,
            args=orpo_config,
            train_dataset=orpo_train_ds,
            eval_dataset=orpo_eval_ds,
            tokenizer=tokenizer,
            callbacks=orpo_callbacks,
        )
        # BOLT OPTIMIZATION: parallelise tokenisation if trainer supports it (TRL >= 0.9.0)
        if "dataset_num_proc" in inspect.signature(ORPOTrainer.__init__).parameters:
            orpo_kwargs["dataset_num_proc"] = os.cpu_count()

        orpo_trainer = ORPOTrainer(**orpo_kwargs)

        if progress is not None:
            progress(0.3, desc="ORPO training started… calculating ETA…")
        t0 = time.time()
        orpo_trainer.train()
        elapsed = time.time() - t0

        status = "stopped by user" if app_state.stop_event.is_set() else "complete"

        if progress is not None:
            progress(0.9, desc="Saving ORPO model…")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        final_loss = log_cb.records[-1]["train_loss"] if log_cb.records else "N/A"
        return (
            f"✅ ORPO training {status}!\n"
            f"⏱ Elapsed: {elapsed/60:.1f} min\n"
            f"📉 Final train loss: {final_loss}\n"
            f"📁 Saved to: {output_dir}"
        )

    except Exception as e:
        return f"❌ ORPO training failed: {e}"
