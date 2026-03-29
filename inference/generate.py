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

import atexit
import os
import tempfile
import threading

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.constants import FILE_EXT_CSV
from core.state import app_state

# H-8 FIX: Thread-safe model cache. Gradio runs in multi-threaded mode; without
# this lock, concurrent requests could simultaneously evict the cache and trigger
# multiple redundant model downloads, or read a half-loaded cache entry.
_cache_lock = threading.Lock()


def _load_for_inference(model_name: str, lora_path: str | None):
    """Return a (model, tokenizer) pair, loading and caching on first access.

    Cache key is (model_name, lora_path).
    When a *new* key is requested the existing cache entry is evicted first
    to avoid holding two large models in VRAM simultaneously.

    FIX 3a (v3.x): Cache is only cleared when loading a different model,
    not on every call — prevents redundant reloads on repeated requests.

    H-8 FIX: Access to the cache dict is guarded by _cache_lock to prevent
    race conditions in Gradio's multi-threaded request handling.

    N-1 FIX: Eliminated the TOCTOU race where the cached entry could be evicted
    by another thread between the write-back lock release and the final
    `return app_state.inference_cache[key]` read, causing a KeyError crash.
    The fix returns the locally-held (model, tokenizer) tuple directly instead
    of re-reading from the shared dict after releasing the lock.
    """
    key = (model_name, lora_path)

    # Fast path: return cached entry under lock
    with _cache_lock:
        if key in app_state.inference_cache:
            return app_state.inference_cache[key]

    # Slow path: load model outside the lock to avoid blocking other threads
    # during the (potentially long) download/load.
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
    model = (
        PeftModel.from_pretrained(base, lora_path)
        if (lora_path and os.path.isdir(lora_path))
        else base
    )
    model.eval()

    # N-1 FIX: Write back under lock, then return the LOCAL (model, tokenizer)
    # tuple — NOT app_state.inference_cache[key].
    # The previous code returned `app_state.inference_cache[key]` after releasing
    # the lock, creating a TOCTOU window: a concurrent thread could evict the
    # entry between the lock release and the dict read, raising a KeyError.
    with _cache_lock:
        if key not in app_state.inference_cache:
            # Evict before adding to keep at most one model resident
            if app_state.inference_cache:
                app_state.inference_cache.clear()
            app_state.inference_cache[key] = (model, tokenizer)

    return model, tokenizer


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

    C-4 FIX: The previous implementation called tokenizer.batch_decode(outputs)
    on the full generated sequences, which included the prompt prepended to every
    response. Every CSV row contained the prompt repeated verbatim before the actual
    generated text. Fixed by applying the same attention-mask-based prompt stripping
    that is already used in generate_text() and the evaluation module.

    N-2 FIX: When prompts_file is empty (empty CSV or blank text file), len(prompts)
    is 0, causing `min(8, 0) = 0` which makes `range(0, 0, 0)` raise ValueError.
    Added an early-exit guard that returns a clear user-facing error message.
    """
    try:
        if prompts_file.name.endswith(FILE_EXT_CSV):
            df = pd.read_csv(prompts_file.name)
            if "prompt" not in df.columns:
                return "❌ CSV must have a 'prompt' column."
            prompts = df["prompt"].tolist()
        else:
            with open(prompts_file.name, "r", encoding="utf-8") as f:
                prompts = [ln.strip() for ln in f if ln.strip()]

        # N-2 FIX: Guard against empty input before computing batch_size.
        # Previously `min(8, len(prompts))` returned 0 for an empty file, and
        # `range(0, 0, 0)` raised ValueError with no user-friendly context.
        if not prompts:
            return "❌ No prompts found in the file. Please check that the file is not empty and contains a 'prompt' column."

        batch_size = min(8, len(prompts))
        all_responses: list[str] = []
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
            # C-4 FIX: Strip prompt tokens using attention_mask lengths.
            # Previously batch_decode(outputs) returned full sequences including
            # the prompt prefix, so every response started with the original prompt.
            input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            for idx, gen_ids in enumerate(outputs):
                input_len = int(input_lengths[idx])
                response_ids = gen_ids[input_len:] if input_len < gen_ids.shape[0] else gen_ids
                all_responses.append(tokenizer.decode(response_ids, skip_special_tokens=True))

        result_df = pd.DataFrame({"prompt": prompts, "response": all_responses})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            result_df.to_csv(tmp_path, index=False)
        # M-6 FIX: Register cleanup so the temp CSV is deleted on process exit.
        atexit.register(lambda p=tmp_path: os.unlink(p) if os.path.exists(p) else None)
        return tmp_path

    except Exception as e:
        return str(e)
