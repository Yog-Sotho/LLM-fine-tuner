"""
training/reward.py
===================
Layer 3 — Reward Model training via trl.RewardTrainer.
Imports: config, core, data.
"""

import gc
import time

import gradio as gr
import torch
from transformers import AutoTokenizer

from config.constants import (
    COL_CHOSEN,
    COL_REJECTED,
    HAS_PPO,
    HAS_REWARD_TRAINER,
)
from core.callbacks import LoggingCallback, StopCallback
from core.state import app_state
from data.loader import detect_file_type, load_dataset_from_file


def train_reward_model_v27(
    model_name: str,
    reward_file,
    output_dir: str,
    rm_epochs: int = 3,
    rm_lr: float = 1.4e-5,
    rm_batch_size: int = 4,
    rm_eval_steps: int = 100,
    rm_max_length: int = 1024,
    progress=gr.Progress(),
) -> str:
    """Train a Reward Model using trl.RewardTrainer.

    Requires trl>=0.7.0 (HAS_REWARD_TRAINER=True).
    Dataset must contain 'chosen' and 'rejected' columns.

    Returns a status string for display in the UI.
    """
    if not HAS_REWARD_TRAINER:
        return "❌ RewardTrainer not available. Install: pip install trl>=0.7.0"
    if reward_file is None:
        return "❌ Please upload a reward dataset (CSV/JSONL with 'chosen' & 'rejected' columns)."

    try:
        from trl import RewardConfig, RewardTrainer  # lazy

        if progress is not None:
            progress(0, desc="Loading reward dataset…")
        ftype = detect_file_type(reward_file)
        ds = load_dataset_from_file(reward_file, ftype, is_dpo=True)

        if COL_CHOSEN not in ds.column_names or COL_REJECTED not in ds.column_names:
            return f"❌ Dataset must contain '{COL_CHOSEN}' and '{COL_REJECTED}' columns."

        if progress is not None:
            progress(0.05, desc="Loading tokenizer…")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if progress is not None:
            progress(0.1, desc="Loading base model for reward training…")
        if not HAS_PPO:
            return "❌ AutoModelForCausalLMWithValueHead not available. Install: pip install trl>=0.7.0"

        from trl import AutoModelForCausalLMWithValueHead  # lazy

        # v2.9 Fix A: Load with AutoModelForCausalLMWithValueHead so the saved
        # checkpoint is directly loadable by run_ppo_v27 without architecture mismatch.
        base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        def tokenize_reward(examples):
            chosen_tok = tokenizer(
                examples[COL_CHOSEN],
                truncation=True, max_length=rm_max_length,
                padding="max_length", return_attention_mask=True,
            )
            rejected_tok = tokenizer(
                examples[COL_REJECTED],
                truncation=True, max_length=rm_max_length,
                padding="max_length", return_attention_mask=True,
            )
            return {
                "input_ids_chosen":       chosen_tok["input_ids"],
                "attention_mask_chosen":  chosen_tok["attention_mask"],
                "input_ids_rejected":     rejected_tok["input_ids"],
                "attention_mask_rejected":rejected_tok["attention_mask"],
            }

        if progress is not None:
            progress(0.15, desc="Tokenising reward pairs…")
        tokenized_ds = ds.map(tokenize_reward, batched=True, remove_columns=ds.column_names)

        # v3.2 Fix #1: Guard against datasets too small to produce a non-empty eval split.
        if len(tokenized_ds) < 2:
            rm_train_ds = tokenized_ds
            rm_eval_ds  = None
        else:
            split = tokenized_ds.train_test_split(test_size=0.1, seed=42)
            rm_train_ds = split["train"]
            rm_eval_ds  = split["test"]
            if len(rm_eval_ds) == 0:
                rm_train_ds = tokenized_ds.select(range(len(tokenized_ds) - 1))
                rm_eval_ds  = tokenized_ds.select([len(tokenized_ds) - 1])

        _rm_eval_strategy = "no" if rm_eval_ds is None else "steps"
        _rm_load_best     = rm_eval_ds is not None

        reward_config = RewardConfig(
            output_dir=output_dir,
            per_device_train_batch_size=rm_batch_size,
            num_train_epochs=rm_epochs,
            learning_rate=rm_lr,
            eval_strategy=_rm_eval_strategy,
            eval_steps=rm_eval_steps if rm_eval_ds is not None else None,
            save_strategy="steps",
            save_steps=rm_eval_steps * 2,
            save_total_limit=2,
            load_best_model_at_end=_rm_load_best,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        log_cb = LoggingCallback()
        trainer = RewardTrainer(
            model=base_model,
            args=reward_config,
            train_dataset=rm_train_ds,
            eval_dataset=rm_eval_ds,
            tokenizer=tokenizer,
            callbacks=[StopCallback(), log_cb],
        )

        if progress is not None:
            progress(0.3, desc="Reward model training started…")
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0

        base_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        final_loss = log_cb.records[-1]["train_loss"] if log_cb.records else "N/A"
        return (
            f"✅ Reward model training complete!\n"
            f"⏱ Elapsed: {elapsed/60:.1f} min\n"
            f"📉 Final train loss: {final_loss}\n"
            f"📁 Saved to: {output_dir}"
        )

    except Exception as e:
        return f"❌ Reward model training failed: {e}"
