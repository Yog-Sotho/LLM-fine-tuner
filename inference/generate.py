"""
inference/generate.py
======================
Layer 4 — model loading cache, single-prompt and batch generation.
Imports: config.constants, core.state, stdlib, transformers, peft, torch, pandas.

Functions
---------
_load_for_inference  — load (or return cached) model + tokenizer pair
generate_text        — single-prompt greedy / sampling generation
batch_generate       — batch generation from CSV / txt file; returns CSV path
"""

import tempfile

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.constants import FILE_EXT_CSV
from core.state import app_state


def _load_for_inference(model_name: str, lora_path: str | None):
    """Return a (model, tokenizer) pair, loading and caching on first access.

    Cache key is (model_name, lora_path).
    When a *new* key is requested the existing cache entry is evicted first
    to avoid holding two large models in VRAM simultaneously.

    FIX 3a (v3.x): Cache is only cleared when loading a different model,
    not on every call — prevents redundant reloads on repeated requests.
    """
    key = (model_name, lora_path)
    if key not in app_state.inference_cache:
        # Evict the current cached model before loading a new one.
        if app_state.inference_cache:
            app_state.inference_cache.clear()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure eos/pad tokens are set
        if tokenizer.eos_token is None:
            if hasattr(tokenizer, "bos_token") and tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
            elif hasattr(tokenizer, "unk_token") and tokenizer.unk_token:
                tokenizer.eos_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({"eos_token": "</s>"})
                tokenizer.eos_token = "</s>"
        tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, lora_path) if (lora_path and __import__("os").path.isdir(lora_path)) else base
        model.eval()
        app_state.inference_cache[key] = (model, tokenizer)

    return app_state.inference_cache[key]


def generate_text(
    model_name: str,
    lora_path: str | None,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a single response from a prompt.

    Returns the generated text only (prompt is stripped from the output).
    Returns an error string prefixed with '❌' on failure.
    """
    try:
        model, tokenizer = _load_for_inference(model_name, lora_path)
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["attention_mask"].sum(dim=-1)[0].item()
        return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    except Exception as e:
        return f"❌ Generation failed: {e}"


def batch_generate(
    model_name: str,
    lora_path: str | None,
    prompts_file,
    max_new_tokens: int = 150,
) -> str:
    """Run batched generation over a CSV ('prompt' column) or plain-text file.

    Returns the path to a temporary CSV file with columns [prompt, response],
    or an error string on failure.
    """
    try:
        if prompts_file.name.endswith(FILE_EXT_CSV):
            df = pd.read_csv(prompts_file.name)
            if "prompt" not in df.columns:
                return "CSV must have a 'prompt' column."
            prompts = df["prompt"].tolist()
        else:
            with open(prompts_file.name, "r", encoding="utf-8") as f:
                prompts = [ln.strip() for ln in f if ln.strip()]

        batch_size = min(8, len(prompts))
        all_responses = []
        model, tokenizer = _load_for_inference(model_name, lora_path)

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i: i + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_responses.extend(responses)

        result_df = pd.DataFrame({"prompt": prompts, "response": all_responses})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            result_df.to_csv(tmp.name, index=False)
        return tmp.name

    except Exception as e:
        return str(e)
