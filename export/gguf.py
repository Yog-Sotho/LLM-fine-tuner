"""
export/gguf.py
===============
Layer 5 — GGUF export via Unsloth (preferred) or llama.cpp fallback.
Imports: config.constants, stdlib, subprocess, shutil, glob.

Functions
---------
export_to_gguf   — export a trained model to GGUF with optional quantisation
on_export_gguf   — Gradio UI handler for the GGUF Export button
"""

import glob
import os
import shutil
import subprocess
import tempfile

from config.constants import HAS_UNSLOTH
from core.state import app_state


def export_to_gguf(model_path: str, output_dir: str, quantization: str = "q6_k") -> str:
    """Export a HuggingFace model directory to GGUF format.

    Strategy:
    1. Unsloth (preferred) — fastest, no external tools required.
    2. llama.cpp fallback  — uses convert_hf_to_gguf.py + llama-quantize.
       Both tools must be in PATH or ~/llama.cpp/.

    Parameters
    ----------
    model_path    : path to the saved HF model directory
    output_dir    : destination directory for the GGUF file(s)
    quantization  : llama.cpp quantisation type string (e.g. 'q4_k_m', 'q6_k')

    Returns a status string for display in the UI.
    """
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
                print(
                    f"⚠️ Unsloth GGUF export failed ({unsloth_err!r}), "
                    f"trying llama.cpp fallback..."
                )

        # ── Path B: llama.cpp ─────────────────────────────────────────────
        convert_script = shutil.which("convert_hf_to_gguf.py")
        if convert_script is None:
            candidate = os.path.join(
                os.path.expanduser("~"), "llama.cpp", "convert_hf_to_gguf.py"
            )
            if os.path.isfile(candidate):
                convert_script = candidate

        if convert_script is None:
            return (
                "❌ GGUF export requires either:\n"
                "1. Unsloth library (pip install unsloth)\n"
                "2. llama.cpp tools: git clone https://github.com/ggerganov/llama.cpp "
                "&& cd llama.cpp && make\n"
                "   Then ensure convert_hf_to_gguf.py and llama-quantize are in PATH"
            )

        fp16_path = os.path.join(output_dir, "model_fp16.gguf")
        result = subprocess.run(
            ["python", convert_script, model_path, "--outtype", "f16", "--outfile", fp16_path],
            capture_output=True, text=True, timeout=900,
        )
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
                capture_output=True, text=True, timeout=900,
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

    except Exception as e:
        return (
            f"❌ GGUF export error: {e}\n"
            f"Ensure dependencies are installed correctly"
        )


def on_export_gguf(model_path: str, quantization: str):
    """Gradio UI handler for the GGUF Export button.

    Returns (status_str, gguf_file_path_or_None).
    """
    # Sentinel: strip whitespace and validate against path traversal (blocking '..' and '\').
    model_path   = model_path.strip()   if model_path   else ""
    quantization = quantization.strip() if quantization else ""

    from core.state import validate_path_traversal
    if err := (validate_path_traversal(model_path) or validate_path_traversal(quantization)):
        return err, None

    if not model_path or not os.path.isdir(model_path):
        return "❌ No trained model found. Train first.", None

    # Sentinel: Clean up the previous GGUF directory to prevent disk exhaustion (DoS).
    app_state.cleanup_resource("_last_gguf_dir")

    gguf_dir = tempfile.mkdtemp(prefix="gguf_")
    # Sentinel: Track the new GGUF directory for future cleanup.
    app_state._last_gguf_dir = gguf_dir

    result = export_to_gguf(model_path, gguf_dir, quantization)
    gguf_files = glob.glob(os.path.join(gguf_dir, "*.gguf"))
    return result, gguf_files[0] if gguf_files else None
