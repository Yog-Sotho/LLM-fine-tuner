"""
export/quantize.py
==================
Layer 5 — quantized export with llm-compressor, for serving with vLLM.

Writes a compressed-tensors safetensors model (loadable by vLLM and Transformers):
  • fp8   — FP8 weights + dynamic FP8 activations; data-free (no calibration).
  • w4a16 — 4-bit GPTQ weights, 16-bit activations; calibrated on your own text.
A LoRA adapter is first merged into its base model (in a temporary folder).
"""

import os
import shutil

import gradio as gr

from config.constants import (
    ALLOW_REMOTE_CODE,
    COL_INSTRUCTION,
    COL_MESSAGES,
    COL_OUTPUT,
    COL_TEXT,
    HAS_LLMCOMPRESSOR,
    QUANT_CALIBRATION_MAX_LENGTH,
    QUANT_CALIBRATION_SAMPLES,
    QUANT_EXPORT_FORMATS,
    RUN_CONFIG_FILENAME,
)
from core.model_card import write_model_card
from core.run_config import load_run_config
from core.state import redact_sensitive_info, validate_path_traversal
from data.preprocessing import content_text
from export.gguf import merge_adapter_to_temp


def calibration_texts(dataset, tokenizer, limit: int = QUANT_CALIBRATION_SAMPLES) -> list[str]:
    """Up to ``limit`` training-like texts from a dataset (chats use the chat template)."""
    texts: list[str] = []
    for row in dataset.select(range(min(limit, len(dataset)))):
        if row.get(COL_MESSAGES):
            conv = row[COL_MESSAGES]
            if getattr(tokenizer, "chat_template", None):
                texts.append(tokenizer.apply_chat_template(conv, tokenize=False))
            else:
                texts.append("\n".join(content_text(t.get("content")) for t in conv))
        elif row.get(COL_INSTRUCTION) is not None and row.get(COL_OUTPUT) is not None:
            texts.append(f"{row[COL_INSTRUCTION]}\n{row[COL_OUTPUT]}")
        elif row.get(COL_TEXT):
            texts.append(row[COL_TEXT])
    return [t for t in texts if t.strip()]


def quantize_model(
    model_path: str,
    output_dir: str,
    fmt: str,
    texts: list[str] | None = None,
    max_length: int = QUANT_CALIBRATION_MAX_LENGTH,
) -> str:
    """Quantize a model (or LoRA adapter, merged first) into ``output_dir``. Returns status."""
    model_path, output_dir = (model_path or "").strip(), (output_dir or "").strip()
    if err := (validate_path_traversal(model_path) or validate_path_traversal(output_dir)):
        return err
    if fmt not in QUANT_EXPORT_FORMATS:
        return f"❌ Format must be one of: {', '.join(QUANT_EXPORT_FORMATS)}"
    if not HAS_LLMCOMPRESSOR:
        return '❌ Quantized export needs llm-compressor: pip install "llm-fine-tuner[compress]"'
    if not model_path or not output_dir:
        return "❌ Give the model path and an output folder."
    if fmt == "w4a16" and not texts:
        return "❌ W4A16 (GPTQ) needs calibration text: load your training data first."

    merged_dir = None
    try:
        source = model_path
        if os.path.isfile(os.path.join(model_path, "adapter_config.json")):
            merged_dir, error = merge_adapter_to_temp(model_path)
            if merged_dir is None:
                return error
            source = merged_dir

        from datasets import Dataset
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=ALLOW_REMOTE_CODE)
        model = AutoModelForCausalLM.from_pretrained(
            source, dtype="auto", trust_remote_code=ALLOW_REMOTE_CODE
        )
        scheme = QUANT_EXPORT_FORMATS[fmt]
        if fmt == "fp8":
            recipe = QuantizationModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])
            oneshot(model=model, recipe=recipe)
        else:
            calib = Dataset.from_dict({"text": list(texts or [])}).map(
                lambda row: tokenizer(row["text"], truncation=True, max_length=max_length),
                remove_columns=["text"],
            )
            recipe = GPTQModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])
            oneshot(
                model=model,
                dataset=calib,
                recipe=recipe,
                max_seq_length=max_length,
                num_calibration_samples=len(calib),
            )
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir, save_compressed=True)
        tokenizer.save_pretrained(output_dir)

        record_path = os.path.join(model_path, RUN_CONFIG_FILENAME)
        if os.path.isfile(record_path):  # carry the run record; card tagged with the format
            record = {**load_run_config(record_path), "quantization": fmt}
            write_model_card(output_dir, record)
        size_gb = sum(
            os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir)
        ) / (1024**3)
        return (
            f"✅ Exported {fmt.upper()} model ({size_gb:.2f} GiB)\n"
            f"📁 {output_dir}\n"
            f"🚀 Serve it (vLLM, CUDA): python main.py serve --model {output_dir}"
        )
    except RuntimeError as e:
        if "unflatten" in str(e):  # group size (128) doesn't divide the layer width
            return "❌ W4A16 needs layer widths divisible by 128 — this model's aren't. Try FP8."
        return f"❌ Quantized export failed: {redact_sensitive_info(str(e))}"
    except Exception as e:  # model download/load errors
        return f"❌ Quantized export failed: {redact_sensitive_info(str(e))}"
    finally:
        if merged_dir:
            shutil.rmtree(merged_dir, ignore_errors=True)


def on_quantize_click(model_path: str, fmt: str, dataset, progress=gr.Progress()) -> str:
    """Export tab handler. Output goes next to the model (``<model>-<fmt>``).

    ``dataset`` is the data loaded in the Data tab (calibration for W4A16).
    """
    model_path = (model_path or "").strip().rstrip("/")
    if not model_path or not os.path.isdir(model_path):
        return "❌ No trained model found. Train first, or enter a model folder."
    texts = None
    if fmt == "w4a16":
        if dataset is None or len(dataset) == 0:
            return "❌ W4A16 needs calibration text: load your training data in 📂 Data first."
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            return f"❌ Cannot load the tokenizer: {redact_sensitive_info(str(e))}"
        texts = calibration_texts(dataset, tokenizer)
    progress(0.1, desc=f"Quantizing to {fmt.upper()}…")
    return quantize_model(model_path, f"{model_path}-{fmt}", fmt, texts)
