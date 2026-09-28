"""
export/utils.py
================
Layer 5 — miscellaneous export / filesystem helpers.
Imports: config.constants, core.state(via data.loader), stdlib, gradio, torch.

Functions
---------
create_zip_from_folder — zip an entire model directory into a temp file
on_peft_zip_upload     — Gradio UI handler: extract a PEFT adapter ZIP
clear_gpu_cache        — free CUDA memory and report reserved VRAM

The README.md model card is written by core.model_card when each trainer saves.
"""

import gc
import os
import tempfile
import zipfile

import gradio as gr
import torch

from core.state import PICKLE_WEIGHT_SUFFIXES, app_state, validate_adapter_dir
from data.loader import safe_extract_zip


def create_zip_from_folder(folder_path: str) -> str:
    """Zip the contents of folder_path into a temporary .zip file.

    Returns the path to the temporary ZIP archive.
    The archive uses ZIP_DEFLATED compression and preserves relative paths
    rooted at the parent of folder_path.

    The caller (ui/handlers.py → on_train_click) tracks the returned path in the
    session state, which deletes it on the session's next run or when the tab closes.
    """
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(folder_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arc_name = os.path.relpath(fpath, start=os.path.dirname(folder_path))
                    zf.write(fpath, arc_name)
    return zip_path


def on_peft_zip_upload(zip_file, request: gr.Request | None = None) -> tuple:
    """Gradio UI handler: extract an uploaded PEFT adapter ZIP archive.

    Only safetensors adapters are accepted: an archive containing any pickle-based
    weight file (.bin/.pt/...) is rejected, since loading one can execute code.

    Returns (adapter_dir_str, status_str, adapter_dir_str) — the path is
    returned twice so it can update both a text box and a state component.
    """
    if zip_file is None:
        return " ", "No file uploaded.", " "

    if hasattr(zip_file, "name") and zip_file.name:
        from core.state import validate_path_traversal

        if err := validate_path_traversal(zip_file.name):
            return " ", err, " "

    session = app_state.session_for(request)
    session.release("peft_dir")
    extract_dir = tempfile.mkdtemp(prefix="peft_zip_")
    session.track("peft_dir", extract_dir)

    try:
        safe_extract_zip(zip_file.name, extract_dir)

        adapter_dir = None
        for root, _, files in os.walk(extract_dir):
            if any(f.lower().endswith(PICKLE_WEIGHT_SUFFIXES) for f in files):
                session.release("peft_dir")
                return (
                    " ",
                    "❌ Rejected: the ZIP contains pickle-based weights (.bin/.pt). "
                    "Upload an adapter saved as adapter_model.safetensors.",
                    " ",
                )
            if adapter_dir is None and "adapter_config.json" in files:
                adapter_dir = root

        if adapter_dir is None or (err := validate_adapter_dir(adapter_dir)):
            session.release("peft_dir")
            return " ", err if adapter_dir else "❌ No adapter_config.json found in the ZIP.", " "

        return (
            adapter_dir,
            f"✅ PEFT adapter extracted to: `{adapter_dir}` ",
            adapter_dir,
        )
    except Exception as e:
        session.release("peft_dir")
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
