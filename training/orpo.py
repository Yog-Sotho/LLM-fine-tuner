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
import time

import gradio as gr
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from config.constants import (
    ALLOW_REMOTE_CODE,
    COL_CHOSEN,
    COL_PROMPT,
    COL_REJECTED,
    DEFAULT_REPORT_TO,
    DEFAULT_SEED,
    HAS_ORPO,
)
from core.callbacks import (
    ETAProgressCallback,
    LoggingCallback,
    StopCallback,
    final_train_loss,
)  # F-2: ETAProgressCallback added
from core.hardware import (
    compute_dtype,
    get_lora_targets,
    is_main_process,
    lora_dropout,
    quantized_device_map,
    setup_moe,
    sharding_unsupported,
    training_device_args,
)
from core.run_config import save_run_config
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
    request: gr.Request | None = None,
) -> str:
    """Train using ORPO (Odds Ratio Preference Optimisation).

    Requires TRL with ORPO (HAS_ORPO=True).
    Dataset must contain 'prompt', 'chosen', 'rejected' columns.

    Returns a status string for display in the UI.
    """
    # Strip whitespace and validate against path traversal.
    model_name = model_name.strip() if model_name else ""
    output_dir = output_dir.strip() if output_dir else ""

    if err := (validate_path_traversal(model_name) or validate_path_traversal(output_dir)):
        return err

    if err := sharding_unsupported("ORPO"):
        return err
    if not HAS_ORPO:
        return '❌ ORPOTrainer not available. Install: pip install "trl>=0.29.1,<2"'
    if orpo_file is None:
        return "❌ Please upload a preference dataset (prompt, chosen, rejected)."

    # Clear the stop event at the start of every ORPO training run.
    stop_event = app_state.session_for(request).stop_event
    stop_event.clear()
    set_seed(DEFAULT_SEED)  # before model/LoRA creation, so runs are reproducible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        try:  # lazy; TRL 1.x moved ORPO to trl.experimental
            from trl.experimental.orpo import ORPOConfig, ORPOTrainer
        except ImportError:
            from trl import ORPOConfig, ORPOTrainer

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

        if device == "cuda":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                # Same dtype as the mixed-precision mode (bf16 where supported, else fp16).
                bnb_4bit_compute_dtype=compute_dtype(device),
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb,
                device_map=quantized_device_map(),
                trust_remote_code=ALLOW_REMOTE_CODE,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=ALLOW_REMOTE_CODE,
            )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=get_lora_targets(),
            lora_dropout=lora_dropout(model),
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

        # v3.2 Fix #1: Guard against datasets too small to produce a non-empty eval split.
        if len(ds) < 2:
            orpo_train_ds = ds
            orpo_eval_ds = None
        else:
            split = ds.train_test_split(test_size=0.1, seed=DEFAULT_SEED)
            orpo_train_ds = split["train"]
            orpo_eval_ds = split["test"]
            if len(orpo_eval_ds) == 0:
                orpo_train_ds = ds.select(range(len(ds) - 1))
                orpo_eval_ds = ds.select([len(ds) - 1])

        _orpo_eval_strategy = "no" if orpo_eval_ds is None else "steps"
        _orpo_load_best = orpo_eval_ds is not None

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
            # Explicit precision (TRL 1.x configs default to bf16=True, which fails on CPU).
            **training_device_args(device),
            report_to=DEFAULT_REPORT_TO,
            seed=DEFAULT_SEED,
        )

        # Guard alpha — added in TRL >= 0.8.1; silently omit on older installs.
        import inspect as _inspect

        try:
            if "alpha" in _inspect.signature(ORPOConfig.__init__).parameters:
                orpo_config_kwargs["alpha"] = orpo_alpha
        except Exception:
            pass

        orpo_config = ORPOConfig(**orpo_config_kwargs)
        log_cb = LoggingCallback()

        # Build callback list: stop button, logging, and ETA progress bar
        orpo_callbacks = [StopCallback(stop_event), log_cb]
        # F-2: ETAProgressCallback wired in so users see per-step ETA.
        if progress is not None:
            orpo_callbacks.append(
                ETAProgressCallback(gradio_progress=progress, progress_start=0.3, progress_end=0.9)
            )

        orpo_trainer_kwargs = {
            "model": model,
            "args": orpo_config,
            "train_dataset": orpo_train_ds,
            "eval_dataset": orpo_eval_ds,
            "processing_class": tokenizer,
            "callbacks": orpo_callbacks,
        }

        moe = setup_moe(model)  # MoE: router aux loss on, router adapter frozen
        orpo_trainer = ORPOTrainer(**orpo_trainer_kwargs)

        if progress is not None:
            progress(0.3, desc="ORPO training started… calculating ETA…")
        t0 = time.time()
        orpo_trainer.train()
        elapsed = time.time() - t0

        status = "stopped by user" if stop_event.is_set() else "complete"

        if progress is not None:
            progress(0.9, desc="Saving ORPO model…")
        # One process writes the outputs (multi-process runs: every rank holds the same weights).
        if is_main_process():
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            save_run_config(
                output_dir,
                moe=moe,
                mode="orpo",
                model=model_name,
                dataset=ds,
                seed=DEFAULT_SEED,
                report_to=DEFAULT_REPORT_TO,
                hyperparams={
                    "learning_rate": orpo_lr,
                    "beta": orpo_beta,
                    "alpha": orpo_alpha,
                    "epochs": orpo_epochs,
                    "batch_size": orpo_batch_size,
                },
                peft={"method": "LoRA", "lora_rank": 16, "lora_alpha": 32},
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        final_loss = final_train_loss(log_cb.records)
        if progress is not None:
            progress(1.0, desc="✅ Complete!")
        return (
            f"✅ ORPO training {status}!\n"
            f"⏱ Elapsed: {elapsed / 60:.1f} min\n"
            f"📉 Final train loss: {final_loss}\n"
            f"📁 Saved to: {output_dir}"
        )

    except Exception as e:
        return f"❌ ORPO training failed: {e}"
    finally:
        # Free VRAM on failure too; names are unbound if loading never happened.
        try:
            del model
        except (NameError, UnboundLocalError):
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
