"""
training/vision.py
==================
Layer 3 — supervised fine-tuning of vision-language models (image + text chats).

Used by train_model() when the chat data has an ``images`` column. TRL's SFTTrainer
does the VLM-specific work (processor, image tokens, loss on the answer only); this
module loads the model with AutoModelForImageTextToText / AutoProcessor and applies
the project's conventions: seed first, precision, LoRA on every linear layer with the
chosen variant, checkpoints/resume, stop + ETA callbacks, run_config + model card.
"""

import time

import torch
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig, EarlyStoppingCallback, set_seed

from config.constants import (
    ALLOW_REMOTE_CODE,
    DEFAULT_EVAL_SPLIT,
    HAS_TORCHVISION,
)
from core.callbacks import (
    ETAProgressCallback,
    LoggingCallback,
    StopCallback,
    final_train_loss,
)
from core.hardware import (
    compute_dtype,
    full_finetune_dtype,
    get_lora_targets,
    is_main_process,
    lora_variant_kwargs,
    quantized_device_map,
    training_device_args,
)
from core.run_config import latest_checkpoint, save_run_config
from data.preprocessing import to_sft_dataset

VISION_PEFT_METHODS = ("LoRA", "Auto", "QLoRA Enhanced", "Full Fine-tuning")


def train_vision_sft(
    model_name: str,
    dataset,
    output_dir: str,
    hyperparams: dict,
    device: str,
    peft_method: str,
    use_lora: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_variant: str,
    gradient_checkpointing: bool,
    lr_scheduler_type: str,
    early_stop: int,
    resume_from_checkpoint: bool,
    seed: int,
    report_to: str,
    run_name: str | None,
    stop_event,
    progress=None,
) -> tuple[str, list]:
    """Fine-tune a vision-language model on image + text chats. Returns (summary, logs)."""
    if not HAS_TORCHVISION:
        raise ImportError(
            "Vision-language fine-tuning needs torchvision, built for your torch version: "
            'pip install "llm-fine-tuner[vision]" (see docs/04_training.md).'
        )
    if peft_method not in VISION_PEFT_METHODS:
        raise ValueError(
            f"'{peft_method}' is not supported for vision-language models; "
            f"use one of: {', '.join(VISION_PEFT_METHODS)}."
        )
    full_finetune = peft_method == "Full Fine-tuning" or (peft_method == "Auto" and not use_lora)
    variant_kwargs = lora_variant_kwargs(lora_variant)
    set_seed(int(seed))  # before the model/LoRA are built, so runs are reproducible

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    if progress is not None:
        progress(0, desc="Loading processor… ")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=ALLOW_REMOTE_CODE)
    if not getattr(processor, "chat_template", None):
        raise ValueError(f"'{model_name}' has no chat template for image + text conversations.")

    data = to_sft_dataset(dataset, use_chat_template=True, system_prompt="")
    eval_split = float(hyperparams.get("eval_split", DEFAULT_EVAL_SPLIT))
    if not 0.0 <= eval_split < 1.0:
        raise ValueError("Eval split must be between 0 and 1 (0 = no evaluation).")
    if len(data) < 2 or eval_split == 0.0:
        train_ds, eval_ds = data, None
    else:
        split = data.train_test_split(test_size=eval_split, seed=seed)
        train_ds, eval_ds = split["train"], split["test"]
        if len(eval_ds) == 0:
            train_ds, eval_ds = data.select(range(len(data) - 1)), data.select([len(data) - 1])

    if progress is not None:
        progress(0.1, desc="Loading vision-language model… ")
    model_kwargs: dict = {"trust_remote_code": ALLOW_REMOTE_CODE}
    if full_finetune:  # never quantised: Transformers refuses to train that without adapters
        model_kwargs["torch_dtype"] = full_finetune_dtype(device)
    elif device == "cuda":  # LoRA on a 4-bit base (QLoRA), as for text models
        model_kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype(device),
                bnb_4bit_use_double_quant=True,
            ),
            device_map=quantized_device_map(),
            torch_dtype=compute_dtype(device),
        )
    else:
        model_kwargs["torch_dtype"] = torch.float32
    model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
    peft_config = None
    if not full_finetune:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            target_modules=get_lora_targets(),
            lora_dropout=0.05,
            bias="none",
            **variant_kwargs,
        )

    log_callback = LoggingCallback()
    callbacks = [StopCallback(stop_event), log_callback]
    # Early stopping on < 50 train rows reacts to a 1-row eval set's noise.
    if early_stop > 0 and eval_ds is not None and len(train_ds) >= 50:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(early_stop)))
    if progress is not None:
        callbacks.append(ETAProgressCallback(gradio_progress=progress))

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=hyperparams["epochs"],
        per_device_train_batch_size=hyperparams["batch_size"],
        gradient_accumulation_steps=hyperparams["grad_accum"],
        learning_rate=hyperparams["learning_rate"],
        warmup_steps=hyperparams["warmup_steps"],
        logging_steps=10,
        eval_strategy="no" if eval_ds is None else "steps",
        eval_steps=50 if eval_ds is not None else None,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False,
        **training_device_args(device),
        report_to=report_to,
        run_name=run_name,
        seed=seed,
        lr_scheduler_type=lr_scheduler_type,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        # Truncating could cut image tokens and break the batch (TRL's advice for VLMs).
        max_length=None,
        dataset_num_proc=None,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    if progress is not None:
        progress(0.3, desc="Training started… calculating ETA…")
    t0 = time.time()
    trainer.train(
        resume_from_checkpoint=latest_checkpoint(output_dir) if resume_from_checkpoint else None
    )
    elapsed = time.time() - t0
    status = "stopped by user" if stop_event.is_set() else "complete"

    # One process writes the outputs (multi-process runs: every rank holds the same weights).
    if is_main_process():
        trainer.model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        save_run_config(
            output_dir,
            mode="sft",
            model=model_name,
            dataset=dataset,
            seed=seed,
            report_to=report_to,
            vision=True,
            hyperparams=dict(hyperparams),
            peft={
                "method": peft_method,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_variant": lora_variant,
            },
            gradient_checkpointing=bool(gradient_checkpointing),
            lr_scheduler_type=lr_scheduler_type,
            early_stop=int(early_stop),
        )
    summary = (
        f"✅ Training {status}!\n"
        f"🖼️ Vision-language fine-tuning ({len(train_ds)} image + text examples)\n"
        f"⏱ Elapsed: {elapsed / 60:.1f} min\n"
        f"📁 Model saved to: {output_dir}\n"
    )
    if log_callback.records:
        summary += f"📉 Final train loss: {final_train_loss(log_callback.records)}"
    return summary, log_callback.records
