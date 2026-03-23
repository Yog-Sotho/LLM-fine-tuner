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
"""

import gc
import os
import tempfile
import zipfile
from datetime import datetime

import gradio as gr
import torch

from data.loader import safe_extract_zip


def create_zip_from_folder(folder_path: str) -> str:
    """Zip the contents of folder_path into a temporary .zip file.

    Returns the path to the temporary ZIP archive.
    The archive uses ZIP_DEFLATED compression and preserves relative paths
    rooted at the parent of folder_path.

    M-6 FIX: The caller (ui/handlers.py → on_train_click) now stores the
    returned zip_path in app_state._last_zip_path and deletes the previous
    zip on the next training run, preventing indefinite accumulation of
    large ZIP files in the OS temp directory.
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
    """
    mode          = peft_method if peft_method != "Full Fine-tuning" else "full fine-tune"
    training_type = "DPO Alignment" if training_mode == "dpo" else "Supervised Fine-Tuning"

    tag_peft = (
        "lora"         if peft_method in ["LoRA", "QLoRA Enhanced"]
        else "peft"    if peft_method != "Full Fine-tuning"
        else "full-finetune"
    )
    tag_train  = "dpo" if training_mode == "dpo" else "sft"
    tag_heretic = "heretic" if heretic_mode else ""

    card = f"""---
language: en
tags:
- fine-tuned
- {tag_peft}
- causal-lm
- {tag_train}
- {tag_heretic}
- gguf-ready
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

    try:
        extract_dir = tempfile.mkdtemp(prefix="peft_zip_")
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
