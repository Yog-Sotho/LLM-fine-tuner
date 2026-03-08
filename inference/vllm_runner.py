"""
inference/vllm_runner.py
=========================
Layer 4 — LoRA adapter merge + vLLM high-throughput inference.
Imports: config.constants, core.state, stdlib, transformers, peft, gradio, torch.

Functions
---------
merge_adapter_for_inference  — merge LoRA adapter into base model (required before vLLM)
on_merge_adapter_click       — Gradio UI handler for the Merge Adapter button
vllm_generate_v27            — batched vLLM generation with engine caching
on_vllm_generate             — Gradio UI handler for the vLLM Generate button
"""

import gc
import os

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.constants import HAS_VLLM
from core.state import app_state


def merge_adapter_for_inference(
    base_model_name: str,
    adapter_path: str,
    merged_output_dir: str,
) -> str:
    """Merge a PEFT LoRA adapter into the base model weights and save a full model.

    vLLM cannot load PEFT adapters directly — this merge step is required before
    passing the fine-tuned model to vllm_generate_v27().

    v2.9 Fix G: This function was added to make the vLLM inference path usable
    end-to-end without manual post-processing.

    Returns a status string for display in the UI.
    """
    if not base_model_name or not base_model_name.strip():
        return "❌ Please provide the base model ID used during training."
    if not adapter_path or not os.path.isdir(adapter_path):
        return "❌ Adapter path is invalid or does not exist. Provide the training output directory."

    try:
        os.makedirs(merged_output_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base, adapter_path)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)

        del merged_model, peft_model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return (
            f"✅ Adapter merged successfully!\n"
            f"📁 Merged model saved to: {merged_output_dir}\n"
            f"⚡ You can now use this path with vLLM inference."
        )
    except Exception as e:
        return f"❌ Adapter merge failed: {e}"


def on_merge_adapter_click(
    base_model_name: str,
    adapter_path: str,
    model_path_state: str,
):
    """Gradio UI handler for the Merge Adapter button.

    Falls back to model_path_state when adapter_path is empty.
    On success, also updates the merged model path state component.
    """
    base    = base_model_name.strip() if base_model_name and base_model_name.strip() else ""
    adapter = adapter_path.strip() if adapter_path and adapter_path.strip() else (model_path_state or "")

    if not adapter or not os.path.isdir(str(adapter)):
        return (
            "❌ No valid adapter/model path. Train a model first or enter a path.",
            gr.update(),
        )

    merged_dir = adapter.rstrip("/\\") + "_merged"
    result = merge_adapter_for_inference(base, adapter, merged_dir)

    if "✅" in result:
        return result, gr.update(value=merged_dir)
    return result, gr.update()


def vllm_generate_v27(
    model_path: str,
    prompts: list[str],
    vllm_quantization: str = "none",
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    tensor_parallel_size: int = 1,
) -> list[str]:
    """Run batched inference using a cached vLLM engine.

    The engine is instantiated once per (model_path, vllm_quantization,
    tensor_parallel_size) tuple and stored in app_state.vllm_cache.
    Subsequent calls with the same key reuse the engine without reloading.

    FIX 3e (v3.x): Cache avoids the ~30s engine startup cost on every call.

    Raises ImportError when vLLM is not installed.
    """
    if not HAS_VLLM:
        raise ImportError("vLLM not installed. Run: pip install vllm>=0.2.0")

    from vllm import LLM, SamplingParams  # lazy — only when vLLM is available

    cache_key = (model_path, vllm_quantization, tensor_parallel_size)
    if cache_key in app_state.vllm_cache:
        llm = app_state.vllm_cache[cache_key]
    else:
        quant = None if vllm_quantization == "none" else vllm_quantization
        llm = LLM(
            model=model_path,
            quantization=quant,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        app_state.vllm_cache[cache_key] = llm

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


def on_vllm_generate(
    model_path_state: str,
    vllm_prompt: str,
    vllm_quant: str,
    vllm_max_tokens: int,
    vllm_temp: float,
    vllm_top_p: float,
) -> str:
    """Gradio UI handler for the vLLM Generate button."""
    if not HAS_VLLM:
        return (
            "❌ vLLM not installed. Run: pip install vllm>=0.2.0\n"
            "Falling back to standard inference is not supported here."
        )
    if not model_path_state or not os.path.isdir(model_path_state):
        return "❌ No trained model path found. Train a model first or enter a model path."
    if not vllm_prompt.strip():
        return "❌ Please enter a prompt."

    try:
        results = vllm_generate_v27(
            model_path=model_path_state,
            prompts=[vllm_prompt.strip()],
            vllm_quantization=vllm_quant,
            max_tokens=int(vllm_max_tokens),
            temperature=vllm_temp,
            top_p=vllm_top_p,
        )
        return results[0] if results else "No output generated."
    except Exception as e:
        return f"❌ vLLM inference failed: {e}"
