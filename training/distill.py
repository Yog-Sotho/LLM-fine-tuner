"""
training/distill.py
====================
Layer 3 — knowledge distillation with TRL's GKDTrainer (Generalized Knowledge
Distillation, on-policy).

A small student learns a larger teacher's next-token distribution, partly on the
student's own generations (``lmbda``), so it learns from its own mistakes. The student
is trained as a LoRA adapter; the teacher is only run forward. Both must share one
vocabulary (the same model family). Data: chats (``messages``), instruction/output or
prompt/completion — the last assistant turn is the answer the teacher grades.
"""

import gc
import time

import gradio as gr
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from config.constants import (
    ALLOW_REMOTE_CODE,
    CHECKPOINT_SAVE_STEPS,
    CHECKPOINT_TOTAL_LIMIT,
    COL_COMPLETION,
    COL_INSTRUCTION,
    COL_MESSAGES,
    COL_OUTPUT,
    COL_PROMPT,
    DEFAULT_REPORT_TO,
    DEFAULT_SEED,
    DISTILL_BETA,
    DISTILL_LMBDA,
    DISTILL_LORA_ALPHA,
    DISTILL_LORA_RANK,
    DISTILL_MAX_NEW_TOKENS,
    DISTILL_TEMPERATURE,
    HAS_GKD,
)
from core.callbacks import ETAProgressCallback, LoggingCallback, StopCallback, final_train_loss
from core.hardware import compute_dtype, get_lora_targets, is_main_process, training_device_args
from core.run_config import latest_checkpoint, save_run_config
from core.state import app_state, validate_path_traversal
from data.loader import detect_file_type, load_dataset_from_file, load_table_dataset
from data.preprocessing import chat_dataset, clean_messages


def to_distill_dataset(ds: Dataset) -> Dataset:
    """Chats ending in an assistant answer, from chat, instruction/output or prompt/completion data."""
    cols = set(ds.column_names)
    if COL_MESSAGES in cols:
        chats = [clean_messages(m) for m in ds[COL_MESSAGES]]
    elif {COL_INSTRUCTION, COL_OUTPUT} <= cols:
        chats = [[{"role": "user", "content": str(q)}, {"role": "assistant", "content": str(a)}]
                 for q, a in zip(ds[COL_INSTRUCTION], ds[COL_OUTPUT], strict=True)]  # fmt: skip
    elif {COL_PROMPT, COL_COMPLETION} <= cols:
        chats = [[{"role": "user", "content": str(q)}, {"role": "assistant", "content": str(a)}]
                 for q, a in zip(ds[COL_PROMPT], ds[COL_COMPLETION], strict=True)]  # fmt: skip
    else:
        raise ValueError(
            "Distillation needs chats (messages), instruction/output or prompt/completion; "
            f"found {sorted(cols)}"
        )
    rows = [
        {COL_MESSAGES: chat}
        for chat in chats
        if chat and len(chat) >= 2 and chat[-1].get("role") == "assistant"
        and str(chat[-1].get("content") or "").strip()
    ]  # fmt: skip
    return chat_dataset(rows)


def _load_distill_file(file) -> Dataset:
    ftype = detect_file_type(file)
    if ftype in ("json", "jsonl"):
        try:  # chats (messages) and instruction/output go through the main loader
            return load_dataset_from_file(file, ftype)
        except ValueError:
            pass
    return load_table_dataset(file)  # prompt/completion tables


def train_distill(
    student_model_name: str,
    teacher_model_name: str,
    distill_file,
    output_dir: str,
    learning_rate: float = 5e-5,
    epochs: int = 1,
    batch_size: int = 2,
    max_length: int = 512,
    lmbda: float = DISTILL_LMBDA,
    beta: float = DISTILL_BETA,
    temperature: float = DISTILL_TEMPERATURE,
    max_new_tokens: int = DISTILL_MAX_NEW_TOKENS,
    resume: bool = False,
    progress=gr.Progress(),
    request: gr.Request | None = None,
) -> str:
    """Distil ``teacher_model_name`` into a LoRA adapter on ``student_model_name``."""
    student_model_name = (student_model_name or "").strip()
    teacher_model_name = (teacher_model_name or "").strip()
    output_dir = (output_dir or "").strip()
    if err := (
        validate_path_traversal(student_model_name)
        or validate_path_traversal(teacher_model_name)
        or validate_path_traversal(output_dir)
    ):
        return err
    if not HAS_GKD:
        return '❌ GKDTrainer not available. Install: pip install "trl>=0.29.1,<2"'
    if not student_model_name or not teacher_model_name:
        return "❌ Give the student (small) and teacher (large) models."
    if distill_file is None:
        return "❌ Please upload a dataset (messages, instruction/output or prompt/completion)."
    if not 0.0 <= float(lmbda) <= 1.0 or not 0.0 <= float(beta) <= 1.0:
        return "❌ lmbda and beta must be between 0 and 1."

    stop_event = app_state.session_for(request).stop_event
    stop_event.clear()
    set_seed(DEFAULT_SEED)  # before model/LoRA creation, so runs are reproducible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from trl.experimental.gkd import GKDConfig, GKDTrainer  # lazy

        if progress is not None:
            progress(0, desc="Loading distillation dataset…")
        ds = to_distill_dataset(_load_distill_file(distill_file))
        if len(ds) == 0:
            return "❌ No usable conversations (each needs a final assistant answer)."

        if progress is not None:
            progress(0.05, desc="Loading student and teacher…")
        tokenizer = AutoTokenizer.from_pretrained(student_model_name, use_fast=True)
        if not getattr(tokenizer, "chat_template", None):
            return (
                "❌ The student's tokenizer has no chat template; distillation formats the data "
                "as chats. Use an instruct/chat model."
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kwargs = dict(torch_dtype=compute_dtype(device), trust_remote_code=ALLOW_REMOTE_CODE)
        model = AutoModelForCausalLM.from_pretrained(student_model_name, **load_kwargs)
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model_name, **load_kwargs)
        student_vocab = model.config.get_text_config().vocab_size
        teacher_vocab = teacher.config.get_text_config().vocab_size
        if student_vocab != teacher_vocab:  # GKD compares full next-token distributions
            return (
                f"❌ Student and teacher must share a vocabulary (vocab size {student_vocab} vs "
                f"{teacher_vocab}). Pick a teacher from the same model family."
            )

        config = GKDConfig(
            output_dir=output_dir,
            learning_rate=float(learning_rate),
            num_train_epochs=int(epochs),
            per_device_train_batch_size=int(batch_size),
            max_length=int(max_length),
            lmbda=float(lmbda),
            beta=float(beta),
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            logging_steps=1,
            save_strategy="steps",
            save_steps=CHECKPOINT_SAVE_STEPS,
            save_total_limit=CHECKPOINT_TOTAL_LIMIT,
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
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=DISTILL_LORA_RANK,
            lora_alpha=DISTILL_LORA_ALPHA,
            target_modules=get_lora_targets(),
            lora_dropout=0.05,
            bias="none",
        )
        trainer = GKDTrainer(
            model=model,
            teacher_model=teacher,
            args=config,
            train_dataset=ds,
            processing_class=tokenizer,
            callbacks=callbacks,
            peft_config=lora,
        )

        if progress is not None:
            progress(0.2, desc="Distillation started… calculating ETA…")
        t0 = time.time()
        trainer.train(resume_from_checkpoint=latest_checkpoint(output_dir) if resume else None)
        elapsed = time.time() - t0
        status = "stopped by user" if stop_event.is_set() else "complete"

        if progress is not None:
            progress(0.95, desc="Saving student adapter…")
        # One process writes the outputs (multi-process runs: every rank holds the same weights).
        if is_main_process():
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            save_run_config(
                output_dir,
                mode="distill",
                model=student_model_name,
                dataset=ds,
                seed=DEFAULT_SEED,
                report_to=DEFAULT_REPORT_TO,
                teacher=teacher_model_name,
                hyperparams={
                    "learning_rate": float(learning_rate),
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "max_length": int(max_length),
                    "lmbda": float(lmbda),
                    "beta": float(beta),
                    "temperature": float(temperature),
                    "max_new_tokens": int(max_new_tokens),
                },
                peft={
                    "method": "LoRA",
                    "lora_rank": DISTILL_LORA_RANK,
                    "lora_alpha": DISTILL_LORA_ALPHA,
                },  # fmt: skip
            )

        if progress is not None:
            progress(1.0, desc="✅ Complete!")
        return (
            f"✅ Distillation {status}!\n"
            f"🎓 Teacher: {teacher_model_name} → student: {student_model_name}\n"
            f"📊 Conversations: {len(ds)}\n"
            f"⏱ Elapsed: {elapsed / 60:.1f} min\n"
            f"📉 Final train loss: {final_train_loss(log_cb.records)}\n"
            f"📁 Student adapter saved to: {output_dir}"
        )

    except Exception as e:
        return f"❌ Distillation failed: {e}"
    finally:
        try:
            del trainer
        except (NameError, UnboundLocalError):
            pass
        try:
            del teacher
        except (NameError, UnboundLocalError):
            pass
        try:
            del model
        except (NameError, UnboundLocalError):
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
