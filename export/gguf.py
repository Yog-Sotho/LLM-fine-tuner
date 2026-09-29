"""
export/gguf.py
===============
Layer 5 — GGUF export via Unsloth (preferred) or llama.cpp fallback.
Imports: config.constants, core.state, inference.vllm_runner, stdlib, subprocess.

Functions
---------
export_to_gguf   — export a trained model to GGUF with optional quantisation
on_export_gguf   — Gradio UI handler for the GGUF Export button
"""

import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

import gradio as gr

from config.constants import HAS_UNSLOTH, MERGEABLE_PEFT_TYPES
from core.state import app_state, validate_path_traversal
from inference.vllm_runner import merge_adapter_for_inference

logger = logging.getLogger(__name__)


def merge_adapter_to_temp(adapter_dir: str) -> tuple[str | None, str]:
    """Merge a LoRA / IA3 adapter into its base model in a temporary folder.

    Exporters (llama.cpp GGUF, llm-compressor) need a full model; an adapter folder
    has no config.json or base weights. Returns (merged_dir, "") or (None, error);
    the caller deletes merged_dir.
    """
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as f:
        adapter_config = json.load(f)
    if str(adapter_config.get("peft_type", "")).upper() not in MERGEABLE_PEFT_TYPES:
        return None, (
            f"❌ Export needs a LoRA or IA3 adapter, or a full model; "
            f"{adapter_config.get('peft_type')} adapters cannot be merged into the weights."
        )
    base_model = str(adapter_config.get("base_model_name_or_path") or "").strip()
    if not base_model:
        return None, "❌ adapter_config.json does not name the base model to merge into."
    merged_dir = tempfile.mkdtemp(prefix="gguf_merge_")
    status = merge_adapter_for_inference(base_model, adapter_dir, merged_dir)
    if not status.startswith("✅"):
        shutil.rmtree(merged_dir, ignore_errors=True)
        return None, status
    return merged_dir, ""


def export_to_gguf(model_path: str, output_dir: str, quantization: str = "q6_k") -> str:
    """Export a HuggingFace model directory to GGUF format.

    Strategy:
    1. Unsloth (preferred) — fastest, no external tools required.
    2. llama.cpp fallback  — uses convert_hf_to_gguf.py + llama-quantize.
       Both tools must be in PATH or ~/llama.cpp/. llama.cpp converts full models
       only, so a LoRA adapter folder is first merged into its base model.

    Parameters
    ----------
    model_path    : path to the saved HF model directory
    output_dir    : destination directory for the GGUF file(s)
    quantization  : llama.cpp quantisation type string (e.g. 'q4_k_m', 'q6_k')

    Returns a status string for display in the UI.
    """
    # Strip whitespace and validate inputs for defense-in-depth API level security
    model_path = model_path.strip() if model_path else ""
    output_dir = output_dir.strip() if output_dir else ""
    quantization = quantization.strip() if quantization else ""

    if err := (validate_path_traversal(model_path) or validate_path_traversal(output_dir)):
        return f"❌ {err.lstrip('❌ ')}"

    from core.state import validate_identifier

    if err := validate_identifier(quantization):
        return f"❌ {err.lstrip('❌ ')}"

    try:
        os.makedirs(output_dir, exist_ok=True)

        # ── Path A: Unsloth ───────────────────────────────────────────────
        if HAS_UNSLOTH:
            try:
                from unsloth import FastLanguageModel  # lazy

                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_path,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=False,
                )
                model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization)
                gguf_files = glob.glob(os.path.join(output_dir, "*.gguf"))
                if gguf_files:
                    size_gb = os.path.getsize(gguf_files[0]) / 1e9
                    return (
                        f"✅ GGUF exported via Unsloth ({quantization.upper()}).\n"
                        f"📦 Size: {size_gb:.2f} GB\n"
                        f"📁 Path: {gguf_files[0]}"
                    )
            except Exception as unsloth_err:
                # H-6 FIX: Log the Unsloth failure before falling through.
                # Previously `except Exception: pass` silently swallowed CUDA OOM,
                # disk-full, and corrupt-model errors, making diagnosis impossible.
                logger.warning(
                    "Unsloth GGUF export failed (%r), trying the llama.cpp fallback", unsloth_err
                )

        # ── Path B: llama.cpp ─────────────────────────────────────────────
        convert_script = shutil.which("convert_hf_to_gguf.py")
        if convert_script is None:
            candidate = os.path.join(os.path.expanduser("~"), "llama.cpp", "convert_hf_to_gguf.py")
            if os.path.isfile(candidate):
                convert_script = candidate

        if convert_script is None:
            return (
                "❌ GGUF export requires either:\n"
                "1. Unsloth library (pip install unsloth)\n"
                "2. llama.cpp: git clone https://github.com/ggml-org/llama.cpp && "
                "cmake -S llama.cpp -B llama.cpp/build && "
                "cmake --build llama.cpp/build --target llama-quantize\n"
                "   Then put llama.cpp/ (convert_hf_to_gguf.py) and llama.cpp/build/bin "
                "(llama-quantize) on PATH, or re-run install.sh"
            )

        merged_dir = None
        if os.path.isfile(os.path.join(model_path, "adapter_config.json")):
            merged_dir, error = merge_adapter_to_temp(model_path)
            if merged_dir is None:
                return error
        fp16_path = os.path.join(output_dir, "model_fp16.gguf")
        try:
            # sys.executable: the converter needs this environment's torch/transformers,
            # not whatever "python" happens to be first on PATH.
            result = subprocess.run(
                [sys.executable, convert_script, merged_dir or model_path,
                 "--outtype", "f16", "--outfile", fp16_path],
                capture_output=True,
                text=True,
                timeout=900,
            )  # fmt: skip
        finally:
            if merged_dir:
                shutil.rmtree(merged_dir, ignore_errors=True)
        if result.returncode != 0:
            return (
                f"❌ llama.cpp conversion failed:\n{result.stderr}\n"
                f"Ensure llama.cpp is built and tools are in PATH"
            )

        quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
        if quantize_bin:
            gguf_out = os.path.join(output_dir, f"model_{quantization}.gguf")
            # v2.9 Minor Fix #6: Pass quantisation string in its original case.
            result2 = subprocess.run(
                [quantize_bin, fp16_path, gguf_out, quantization],
                capture_output=True,
                text=True,
                timeout=900,
            )
            if result2.returncode == 0:
                os.remove(fp16_path)
                size_gb = os.path.getsize(gguf_out) / 1e9
                return (
                    f"✅ GGUF exported & quantized ({quantization}).\n"
                    f"📦 Size: {size_gb:.2f} GB\n"
                    f"📁 Path: {gguf_out}"
                )
            else:
                return f"⚠️ Quantization failed. Using FP16 version.\n{result2.stderr}"

        size_gb = os.path.getsize(fp16_path) / 1e9
        return (
            f"✅ GGUF exported (FP16 only).\n"
            f"📦 Size: {size_gb:.2f} GB\n"
            f"📁 Path: {fp16_path}\n"
            f"⚠️ Install llama.cpp quantize tool for quantization"
        )

    except subprocess.TimeoutExpired:
        return (
            "❌ GGUF conversion timed out after 15 minutes.\n"
            "The model may be too large or disk I/O is slow."
        )
    except Exception as e:
        return f"❌ GGUF export error: {e}\nEnsure dependencies are installed correctly"


def on_export_gguf(model_path: str, quantization: str, request: gr.Request | None = None):
    """Gradio UI handler for the GGUF Export button.

    Returns (status_str, gguf_file_path_or_None).
    """
    # Strip whitespace and validate against path traversal (blocking '..' and '\').
    model_path = model_path.strip() if model_path else ""
    quantization = quantization.strip() if quantization else ""

    from core.state import validate_identifier

    if err := validate_path_traversal(model_path):
        return err, None
    if err := validate_identifier(quantization):
        return err, None

    if not model_path or not os.path.isdir(model_path):
        return "❌ No trained model found. Train first.", None

    session = app_state.session_for(request)
    session.release("gguf_dir")
    gguf_dir = tempfile.mkdtemp(prefix="gguf_")
    session.track("gguf_dir", gguf_dir)

    result = export_to_gguf(model_path, gguf_dir, quantization)
    gguf_files = glob.glob(os.path.join(gguf_dir, "*.gguf"))
    return result, gguf_files[0] if gguf_files else None
