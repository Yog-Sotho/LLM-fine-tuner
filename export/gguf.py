"""
export/gguf.py
===============
Layer 5 — GGUF export via Unsloth (preferred) or llama.cpp fallback.
Imports: config.constants, stdlib, subprocess, shutil, glob.

Functions
---------
export_to_gguf   — export a trained model to GGUF with optional quantisation
on_export_gguf   — Gradio UI handler for the GGUF Export button

Fix log
-------
  H7 (High): model_path and quantization were passed directly to subprocess
     without validation. A user-controlled quantization string could
     theoretically contain an unexpected value causing cryptic llama.cpp
     errors. model_path was never verified to be a real directory before
     being passed as a subprocess argument. Added:
       1. `ALLOWED_QUANTIZATIONS` whitelist — only known-safe presets accepted.
       2. `model_path` realpath validation — must be an existing directory.
     Both checks return a clear human-readable error before any subprocess
     call is made.
  L6 (Low): file-size reporting changed from `/ 1e9` (decimal GB) to
     `/ (1024 ** 3)` (binary GiB) which matches how GPUs and storage
     report capacity. Display label updated from "GB" to "GiB" accordingly.
"""

import glob
import os
import shutil
import subprocess
import tempfile

from config.constants import HAS_UNSLOTH

# H7 FIX: Whitelist of allowed quantization values.
# Only these strings are passed to the llama.cpp quantize binary.
ALLOWED_QUANTIZATIONS: frozenset[str] = frozenset({
    "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k", "f16", "f32",
})


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
    # H7 FIX: validate model_path is a real existing directory.
    model_path_real = os.path.realpath(model_path)
    if not os.path.isdir(model_path_real):
        return (
            f"❌ model_path is not a valid directory: {model_path!r}\n"
            f"Please train a model first or provide a correct path."
        )

    # H7 FIX: validate quantization against the known-safe whitelist.
    if quantization not in ALLOWED_QUANTIZATIONS:
        return (
            f"❌ Invalid quantization preset: {quantization!r}\n"
            f"Allowed values: {sorted(ALLOWED_QUANTIZATIONS)}"
        )

    try:
        os.makedirs(output_dir, exist_ok=True)

        # ── Path A: Unsloth ───────────────────────────────────────────────
        if HAS_UNSLOTH:
            try:
                from unsloth import FastLanguageModel  # lazy

                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_path_real,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=False,
                )
                model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization)
                gguf_files = glob.glob(os.path.join(output_dir, "*.gguf"))
                if gguf_files:
                    # L6 FIX: use binary GiB (1024**3), not decimal GB (1e9).
                    size_gib = os.path.getsize(gguf_files[0]) / (1024 ** 3)
                    return (
                        f"✅ GGUF exported via Unsloth ({quantization.upper()}).\n"
                        f"📦 Size: {size_gib:.2f} GiB\n"
                        f"📁 Path: {gguf_files[0]}"
                    )
            except Exception:
                pass  # Fall through to llama.cpp path

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
        # H7 FIX: use validated model_path_real (realpath-resolved).
        result = subprocess.run(
            ["python", convert_script, model_path_real, "--outtype", "f16", "--outfile", fp16_path],
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
            # H7 FIX: quantization is already whitelist-validated above.
            result2 = subprocess.run(
                [quantize_bin, fp16_path, gguf_out, quantization],
                capture_output=True, text=True, timeout=900,
            )
            if result2.returncode == 0:
                os.remove(fp16_path)
                # L6 FIX: binary GiB.
                size_gib = os.path.getsize(gguf_out) / (1024 ** 3)
                return (
                    f"✅ GGUF exported & quantized ({quantization}).\n"
                    f"📦 Size: {size_gib:.2f} GiB\n"
                    f"📁 Path: {gguf_out}"
                )
            else:
                return f"⚠️ Quantization failed. Using FP16 version.\n{result2.stderr}"

        # L6 FIX: binary GiB.
        size_gib = os.path.getsize(fp16_path) / (1024 ** 3)
        return (
            f"✅ GGUF exported (FP16 only).\n"
            f"📦 Size: {size_gib:.2f} GiB\n"
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
    if not model_path or not os.path.isdir(model_path):
        return "❌ No trained model found. Train first.", None

    gguf_dir = tempfile.mkdtemp(prefix="gguf_")
    result = export_to_gguf(model_path, gguf_dir, quantization)
    gguf_files = glob.glob(os.path.join(gguf_dir, "*.gguf"))
    return result, gguf_files[0] if gguf_files else None
