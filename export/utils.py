"""
export/utils.py
================
Layer 5 — miscellaneous export / filesystem helpers.
Imports: config.constants, core.state(via data.loader), stdlib, gradio, torch.

Functions
---------
create_zip_from_folder — zip an entire model directory into a temp file
create_model_card      — generate and write a README.md model card
on_peft_zip_upload     — Gradio UI handler: extract a PEFT adapter ZIP
clear_gpu_cache        — free CUDA memory and report reserved VRAM

Patch log
---------
  M-2  : ``create_model_card()`` produced an invalid YAML front-matter entry
         ``- `` (empty string tag) when ``heretic_mode=False``.
         HuggingFace Hub rejects model cards with empty YAML list items.
         Fix: build the tags list programmatically and only include the
         "heretic" tag when heretic_mode is True.
"""

import gc
import os
import tempfile
import zipfile
from datetime import datetime

import gradio as gr
import torch

from config.constants import HF_TOKEN_PREFIX, HF_TOKEN_MIN_LEN
from data.loader import safe_extract_zip
from core.state import app_state


def validate_hf_token(token: str | None) -> str | None:
    """Standardized validation for HuggingFace write tokens.

    Returns an error message if invalid, or None if valid.
    """
    token = token.strip() if token else ""
    if (
        not token
        or not token.startswith(HF_TOKEN_PREFIX)
        or len(token) < HF_TOKEN_MIN_LEN
    ):
        return (
            "❌ Invalid Hugging Face write token.\n"
            f"Tokens start with '{HF_TOKEN_PREFIX}' and are at least {HF_TOKEN_MIN_LEN} characters long."
        )
    return None


def create_zip_from_folder(folder_path: str) -> str:
    """Zip the contents of folder_path into a temporary .zip file.

    Returns the path to the temporary ZIP archive.
    The archive uses ZIP_DEFLATED compression and preserves relative paths
    rooted at the parent of folder_path.

    M-6 FIX: The caller (ui/handlers.py → on_train_click) stores the returned
    zip_path in app_state._last_zip_path and deletes the previous zip on the
    next training run, preventing indefinite accumulation of large ZIP files in
    the OS temp directory.
    """
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(folder_path):
                for fname in files:
                    fpath    = os.path.join(root, fname)
                    arc_name = os.path.relpath(fpath, start=os.path.dirname(folder_path))
                    zf.write(fpath, arc_name)
    return zip_path


def create_model_card(
    model_name: str,
    dataset_info: dict,
    hyperparams: dict,
    output_dir: str,
    peft_method: str,
    training_mode: str = "sft",
    heretic_mode: bool = False,
) -> None:
    """Generate a HuggingFace-compatible README.md model card and write it to output_dir.

    Parameters
    ----------
    model_name   : base HF model identifier
    dataset_info : dict with keys 'num_examples' and 'avg_length'
    hyperparams  : dict with training hyperparameters
    output_dir   : directory to write README.md into
    peft_method  : PEFT method string (e.g. 'LoRA', 'Full Fine-tuning')
    training_mode: 'sft' or 'dpo'
    heretic_mode : whether Heretic Mode was applied

    M-2 FIX: ``tag_heretic = "" if not heretic_mode`` produced an empty YAML
    list entry ``- `` (literal empty string) which HuggingFace Hub rejects with
    a 400 error when pushing.  Tags are now built as a Python list and rendered
    cleanly — the "heretic" tag is only included when heretic_mode is True.
    """
    mode          = peft_method if peft_method != "Full Fine-tuning" else "full fine-tune"
    training_type = "DPO Alignment" if training_mode == "dpo" else "Supervised Fine-Tuning"

    tag_peft = (
        "lora"           if peft_method in ["LoRA", "QLoRA Enhanced"]
        else "peft"      if peft_method != "Full Fine-tuning"
        else "full-finetune"
    )
    tag_train = "dpo" if training_mode == "dpo" else "sft"

    # M-2 FIX: Build the tags list conditionally so no empty string is ever
    # serialised as a YAML list item.  The previous code always included an
    # empty-string tag entry when heretic_mode was False.
    tags: list[str] = ["fine-tuned", tag_peft, "causal-lm", tag_train, "gguf-ready"]
    if heretic_mode:
        tags.append("heretic")

    # Render as YAML list: "- tag\n- tag\n..."
    tags_yaml = "\n".join(f"- {t}" for t in tags)

    card = f"""---
language: en
tags:
{tags_yaml}
datasets:
- custom
---
# {training_type} Model Card
This model is a {mode} of `{model_name}` trained with **{training_type}**.
{"**🔓 Heretic Mode applied** — safety restrictions removed." if heretic_mode else ""}
## Training Data
- Examples: {dataset_info.get("num_examples", "N/A")}
- Average length: {dataset_info.get("avg_length", 0):.0f} chars
## Hyperparameters
| Param | Value |
| --- | --- |
| Learning rate | {hyperparams.get("learning_rate")} |
| Epochs | {hyperparams.get("epochs")} |
| Batch size | {hyperparams.get("batch_size")} |
| Max length | {hyperparams.get("max_length")} |
| PEFT Method | {peft_method} |
"""
    if training_mode == "dpo":
        card += f"| DPO Beta | {hyperparams.get('dpo_beta', 0.1)} |\n"
    if peft_method in ["LoRA", "QLoRA Enhanced"]:
        card += f"| LoRA rank | {hyperparams.get('lora_rank', 'N/A')} |\n"
        card += f"| LoRA alpha | {hyperparams.get('lora_alpha', 'N/A')} |\n"
    card += (
        f"| LR scheduler | {hyperparams.get('lr_scheduler', 'linear')} |\n"
        f"Trained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"GGUF & Heretic ready for maximum potential."
    )

    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)


def on_peft_zip_upload(zip_file) -> tuple:
    """Gradio UI handler: extract an uploaded PEFT adapter ZIP archive.

    Walks the extracted tree looking for adapter_config.json or known weight
    files to determine the real adapter root (ZIP may contain a top-level folder).

    Returns (adapter_dir_str, status_str, adapter_dir_str) — the path is
    returned twice so it can update both a text box and a state component.
    """
    if zip_file is None:
        return " ", "No file uploaded.", " "

    # Sentinel: Clean up the previous PEFT extraction directory to prevent disk exhaustion (DoS).
    app_state.cleanup_resource("_last_peft_dir")

    try:
        extract_dir = tempfile.mkdtemp(prefix="peft_zip_")
        # Sentinel: Track the new PEFT directory for future cleanup.
        app_state._last_peft_dir = extract_dir
        safe_extract_zip(zip_file.name, extract_dir)

        # Walk the extracted tree to find the actual adapter root.
        adapter_dir = extract_dir
        for root, dirs, files in os.walk(extract_dir):
            if (
                "adapter_config.json" in files
                or "adapter_model.bin" in files
                or "pytorch_model.bin" in files
            ):
                adapter_dir = root
                break

        return (
            adapter_dir,
            f"✅ PEFT adapter extracted to: `{adapter_dir}` ",
            adapter_dir,
        )
    except Exception as e:
        return " ", f"❌ Failed to extract ZIP: {e} ", " "


def clear_gpu_cache() -> str:
    """Free CUDA memory cache and run Python GC.

    Returns a status string reporting post-clear reserved VRAM,
    or an info message when no GPU is detected.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        free = torch.cuda.memory_reserved(0) / 1e9
        return f"🧹 GPU cache cleared. Reserved: {free:.2f} GB"
    return "ℹ️ No GPU detected."
