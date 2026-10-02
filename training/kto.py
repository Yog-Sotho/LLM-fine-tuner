"""
training/kto.py
================
Layer 3 — KTO (Kahneman-Tversky Optimization) with TRL's KTOTrainer.

KTO learns from single responses labelled desirable / undesirable, so it works
with thumbs-up / thumbs-down feedback instead of ranked pairs. Accepts:
  • unpaired data: prompt, completion, label (true/false), or
  • paired data: prompt, chosen, rejected — split into one desirable and one
    undesirable row per pair (done here so behaviour is identical across TRL versions).
"""

import gc
import time

import gradio as gr
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from config.constants import (
    ALLOW_REMOTE_CODE,
    CHECKPOINT_SAVE_STEPS,
    CHECKPOINT_TOTAL_LIMIT,
    COL_CHOSEN,
    COL_COMPLETION,
    COL_LABEL,
    COL_PROMPT,
    COL_REJECTED,
    DEFAULT_REPORT_TO,
    DEFAULT_SEED,
    HAS_KTO,
)
from core.callbacks import (
    ETAProgressCallback,
    LoggingCallback,
    StopCallback,
    final_train_loss,
)
from core.hardware import (
    compute_dtype,
    get_lora_targets,
    is_main_process,
    lora_dropout,
    setup_moe,
    sharding_unsupported,
    training_device_args,
)
from core.run_config import latest_checkpoint, save_run_config
from core.state import app_state, redact_sensitive_info, validate_path_traversal
from data.loader import load_table_dataset

_TRUE = {"1", "true", "yes", "y", "desirable", "good", "👍"}
_FALSE = {"0", "false", "no", "n", "undesirable", "bad", "👎"}


def _parse_label(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"Unrecognised KTO label {value!r}; use true/false.")


def to_kto_dataset(ds: Dataset) -> Dataset:
    """Return an unpaired prompt/completion/label dataset from unpaired or paired input."""
    cols = ds.column_names
    prompts: list[str] = []
    completions: list[str] = []
    labels: list[bool] = []
    if {COL_PROMPT, COL_COMPLETION, COL_LABEL} <= set(cols):
        prompts = [str(p) for p in ds[COL_PROMPT]]
        completions = [str(c) for c in ds[COL_COMPLETION]]
        labels = [_parse_label(v) for v in ds[COL_LABEL]]
    elif {COL_PROMPT, COL_CHOSEN, COL_REJECTED} <= set(cols):
        for prompt, chosen, rejected in zip(
            ds[COL_PROMPT], ds[COL_CHOSEN], ds[COL_REJECTED], strict=True
        ):
            prompts += [str(prompt), str(prompt)]
            completions += [str(chosen), str(rejected)]
            labels += [True, False]
    else:
        raise ValueError(
            f"KTO needs prompt/completion/label or prompt/chosen/rejected columns; found {cols}"
        )
    out = Dataset.from_dict({COL_PROMPT: prompts, COL_COMPLETION: completions, COL_LABEL: labels})
    return out.filter(lambda r: r[COL_PROMPT].strip() != "" and r[COL_COMPLETION].strip() != "")


def train_kto(
    model_name: str,
    kto_file,
    output_dir: str,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    epochs: int = 1,
    batch_size: int = 4,
    max_length: int = 512,
    resume: bool = False,
    progress=gr.Progress(),
    request: gr.Request | None = None,
) -> str:
    """Train with KTO. Returns a status string for the UI."""
    model_name = model_name.strip() if model_name else ""
    output_dir = output_dir.strip() if output_dir else ""
    if err := (validate_path_traversal(model_name) or validate_path_traversal(output_dir)):
        return err
    if err := sharding_unsupported("KTO"):
        return err
    if not HAS_KTO:
        return '❌ KTOTrainer not available. Install: pip install "trl>=0.29.1,<2"'
    if kto_file is None:
        return "❌ Please upload a dataset (prompt/completion/label or prompt/chosen/rejected)."
    if int(batch_size) < 2:
        return "❌ KTO needs a batch size of at least 2 (it estimates a KL baseline per batch)."

    stop_event = app_state.session_for(request).stop_event
    stop_event.clear()
    set_seed(DEFAULT_SEED)  # before model/LoRA creation, so runs are reproducible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from trl import KTOConfig, KTOTrainer  # lazy

        if progress is not None:
            progress(0, desc="Loading KTO dataset…")
        ds = to_kto_dataset(load_table_dataset(kto_file))
        labels = ds[COL_LABEL]
        if len(ds) < 2 or all(labels) or not any(labels):
            return "❌ KTO needs at least one desirable and one undesirable example."

        if progress is not None:
            progress(0.05, desc="Loading model…")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=compute_dtype(device),
            trust_remote_code=ALLOW_REMOTE_CODE,
        )
        # LoRA applied here: the reference model is the same weights with the adapter off.
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                target_modules=get_lora_targets(),
                lora_dropout=lora_dropout(model),
                bias="none",
            ),
        )

        config = KTOConfig(
            output_dir=output_dir,
            learning_rate=learning_rate,
            beta=beta,
            num_train_epochs=epochs,
            per_device_train_batch_size=int(batch_size),
            max_length=int(max_length),
            logging_steps=1,
            save_strategy="steps",
            save_steps=CHECKPOINT_SAVE_STEPS,
            save_total_limit=CHECKPOINT_TOTAL_LIMIT,
            remove_unused_columns=False,
            report_to=DEFAULT_REPORT_TO,
            seed=DEFAULT_SEED,
            **training_device_args(device),
        )

        log_cb = LoggingCallback()
        callbacks = [StopCallback(stop_event), log_cb]
        if progress is not None:
            callbacks.append(
                ETAProgressCallback(gradio_progress=progress, progress_start=0.2, progress_end=0.9)
            )

        moe = setup_moe(model, router_aux_loss=False, freeze_router=False)  # see setup_moe
        trainer = KTOTrainer(
            model=model,
            args=config,
            train_dataset=ds,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        if progress is not None:
            progress(0.2, desc="KTO training started… calculating ETA…")
        t0 = time.time()
        trainer.train(resume_from_checkpoint=latest_checkpoint(output_dir) if resume else None)
        elapsed = time.time() - t0
        status = "stopped by user" if stop_event.is_set() else "complete"

        if progress is not None:
            progress(0.95, desc="Saving adapter…")
        # One process writes the outputs (multi-process runs: every rank holds the same weights).
        if is_main_process():
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            save_run_config(
                output_dir,
                moe=moe,
                mode="kto",
                model=model_name,
                dataset=ds,
                seed=DEFAULT_SEED,
                report_to=DEFAULT_REPORT_TO,
                hyperparams={
                    "learning_rate": learning_rate,
                    "beta": beta,
                    "epochs": epochs,
                    "batch_size": int(batch_size),
                    "max_length": int(max_length),
                },
                peft={"method": "LoRA", "lora_rank": 16, "lora_alpha": 32},
            )

        final_loss = final_train_loss(log_cb.records)
        if progress is not None:
            progress(1.0, desc="✅ Complete!")
        return (
            f"✅ KTO training {status}!\n"
            f"📊 Examples: {len(ds)} ({sum(labels)} desirable / {len(ds) - sum(labels)} undesirable)\n"
            f"⏱ Elapsed: {elapsed / 60:.1f} min\n"
            f"📉 Final train loss: {final_loss}\n"
            f"📁 Adapter saved to: {output_dir}"
        )

    except Exception as e:
        return f"❌ KTO training failed: {redact_sensitive_info(str(e))}"
    finally:
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
