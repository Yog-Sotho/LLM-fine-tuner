"""
training/reward.py
===================
Layer 3 — Reward model training with TRL's RewardTrainer.

The reward model is a sequence classifier with a single score output
(AutoModelForSequenceClassification, num_labels=1) trained on prompt / chosen /
rejected pairs, so it scores a response *in the context of its prompt*. LoRA is
trained and then merged, and the full model is saved: GRPO loads a reward-model
path directly as a sequence classifier.
"""

import gc
import time

import gradio as gr
import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from config.constants import (
    ALLOW_REMOTE_CODE,
    COL_CHOSEN,
    COL_PROMPT,
    COL_REJECTED,
    DEFAULT_REPORT_TO,
    DEFAULT_SEED,
    HAS_REWARD_TRAINER,
)
from core.callbacks import ETAProgressCallback, LoggingCallback, StopCallback
from core.hardware import compute_dtype, get_lora_targets, select_precision
from core.run_config import save_run_config
from core.state import app_state, validate_path_traversal
from data.loader import detect_file_type, load_dataset_from_file
from data.preprocessing import validate_and_clean_dataset


def train_reward_model_v27(
    model_name: str,
    reward_file,
    output_dir: str,
    rm_epochs: int = 3,
    rm_lr: float = 1e-4,
    rm_batch_size: int = 4,
    rm_eval_steps: int = 100,
    rm_max_length: int = 1024,
    progress=gr.Progress(),
    request: gr.Request | None = None,
) -> str:
    """Train a prompt-aware reward model from prompt/chosen/rejected data.

    Returns a status string for display in the UI.
    """
    model_name = model_name.strip() if model_name else ""
    output_dir = output_dir.strip() if output_dir else ""
    if err := (validate_path_traversal(model_name) or validate_path_traversal(output_dir)):
        return err

    if not HAS_REWARD_TRAINER:
        return '❌ RewardTrainer not available. Install: pip install "trl>=0.29.1,<2"'
    if reward_file is None:
        return "❌ Please upload a preference dataset (prompt, chosen, rejected)."

    stop_event = app_state.session_for(request).stop_event
    stop_event.clear()
    set_seed(DEFAULT_SEED)  # before model/LoRA creation, so runs are reproducible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from trl import RewardConfig, RewardTrainer  # lazy

        if progress is not None:
            progress(0, desc="Loading reward dataset…")
        ds = load_dataset_from_file(reward_file, detect_file_type(reward_file), is_dpo=True)
        missing = [c for c in (COL_PROMPT, COL_CHOSEN, COL_REJECTED) if c not in ds.column_names]
        if missing:
            return f"❌ Dataset is missing columns: {missing}. Needs prompt, chosen, rejected."
        ds, _ = validate_and_clean_dataset(ds, is_dpo=True)
        if len(ds) == 0:
            return "❌ Dataset is empty after cleaning."

        if progress is not None:
            progress(0.05, desc="Loading tokenizer and model…")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=compute_dtype(device),
            trust_remote_code=ALLOW_REMOTE_CODE,
        )
        # Sequence classifiers score the last non-pad token, so they need a pad id.
        model.config.pad_token_id = tokenizer.pad_token_id

        # Guard against datasets too small to produce a non-empty eval split.
        if len(ds) < 2:
            train_ds, eval_ds = ds, None
        else:
            split = ds.train_test_split(test_size=0.1, seed=DEFAULT_SEED)
            train_ds, eval_ds = split["train"], split["test"]
            if len(eval_ds) == 0:
                train_ds = ds.select(range(len(ds) - 1))
                eval_ds = ds.select([len(ds) - 1])

        config = RewardConfig(
            output_dir=output_dir,
            per_device_train_batch_size=rm_batch_size,
            num_train_epochs=rm_epochs,
            learning_rate=rm_lr,
            max_length=rm_max_length,
            eval_strategy="no" if eval_ds is None else "steps",
            eval_steps=rm_eval_steps if eval_ds is not None else None,
            save_strategy="steps",
            save_steps=rm_eval_steps * 2,
            save_total_limit=2,
            load_best_model_at_end=eval_ds is not None,
            report_to=DEFAULT_REPORT_TO,
            seed=DEFAULT_SEED,
            **select_precision(device),
        )
        # LoRA on the backbone; PEFT keeps the new score head trainable for SEQ_CLS.
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            target_modules=get_lora_targets(),
            lora_dropout=0.05,
            bias="none",
        )

        log_cb = LoggingCallback()
        callbacks = [StopCallback(stop_event), log_cb]
        if progress is not None:
            callbacks.append(
                ETAProgressCallback(gradio_progress=progress, progress_start=0.3, progress_end=0.9)
            )

        trainer = RewardTrainer(
            model=model,
            args=config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
            peft_config=peft_config,
        )

        if progress is not None:
            progress(0.3, desc="Reward model training started… calculating ETA…")
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0
        status = "stopped by user" if stop_event.is_set() else "complete"

        if progress is not None:
            progress(0.9, desc="Merging LoRA and saving reward model…")
        # Save a full sequence-classification model so GRPO can load the path directly.
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        save_run_config(
            output_dir,
            mode="reward",
            model=model_name,
            dataset=ds,
            seed=DEFAULT_SEED,
            report_to=DEFAULT_REPORT_TO,
            hyperparams={
                "epochs": rm_epochs,
                "learning_rate": rm_lr,
                "batch_size": rm_batch_size,
                "eval_steps": rm_eval_steps,
                "max_length": rm_max_length,
            },
            peft={"method": "LoRA (merged)", "lora_rank": 16, "lora_alpha": 32},
        )

        final_loss = log_cb.records[-1]["train_loss"] if log_cb.records else "N/A"
        if progress is not None:
            progress(1.0, desc="✅ Complete!")
        return (
            f"✅ Reward model training {status}!\n"
            f"⏱ Elapsed: {elapsed / 60:.1f} min\n"
            f"📉 Final train loss: {final_loss}\n"
            f"📁 Saved to: {output_dir} (use this path as the GRPO reward model)"
        )

    except Exception as e:
        return f"❌ Reward model training failed: {e}"
    finally:
        # Free VRAM on failure too; names are unbound if loading never happened.
        try:
            del merged
        except (NameError, UnboundLocalError):
            pass
        try:
            del trainer
        except (NameError, UnboundLocalError):
            pass
        try:
            del model
        except (NameError, UnboundLocalError):
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
