"""
core/state.py
==============
Application-wide mutable state. Pure stdlib — no internal project imports.

The module-level singleton `app_state` is the single source of truth for:
  • stop_event       — set by on_stop(), polled by StopCallback every step
  • inference_cache  — (model_name, lora_path) → (model, tokenizer)
  • vllm_cache       — (model_path, quant, tensor_parallel) → LLM engine
  • _last_zip_path   — path of last training ZIP for cleanup (M-6 fix)
  • _last_model_dir  — path of last training output directory (Sentinel fix)
  • _last_gguf_dir   — path of last GGUF export directory (Sentinel fix)
  • _last_peft_dir   — path of last PEFT extraction directory (Sentinel fix)
  • _last_batch_path — path of last batch inference CSV (Sentinel fix)
  • _last_merged_dir — path of last merged model directory (Sentinel fix)
"""

import os
import re
import shutil
import threading


def redact_sensitive_info(text: str | None) -> str:
    """Scan error/exception messages for sensitive inputs (like API keys, tokens, or passwords)
    and replace them with [REDACTED] before presenting or logging the error.

    Specifically redacts:
    - Hugging Face API write tokens (using regex hf_[a-zA-Z0-9_]{30,})
    - Any environment variable HF_TOKEN, HF_TOKEN_WRITE, etc.
    """
    if not text:
        return ""

    # Match standard HF token format: hf_ followed by at least 30 alphanumeric/underscore characters.
    # Pattern uses a word boundary or start of string/word non-greedy boundary.
    pattern = r"hf_[a-zA-Z0-9_]{30,}"
    text = re.sub(pattern, "[REDACTED]", text)

    # Redact environment HF_TOKEN if present
    for env_var in ["HF_TOKEN", "HF_TOKEN_WRITE"]:
        val = os.environ.get(env_var)
        if val and len(val) >= 8 and val in text:
            text = text.replace(val, "[REDACTED]")

    return text


def validate_path_traversal(path: str | None) -> str | None:
    """Check if the path contains '..' or '\\' or null bytes (unsafe).

    Returns a standardized error message if unsafe, or None if safe.
    """
    if not path:
        return None
    if ".." in path or "\\" in path or "\0" in path:
        return "❌ Path traversal attempt detected."
    return None


def validate_identifier(identifier: str | None) -> str | None:
    """Check if an identifier (like a version or tag) contains path-like characters.

    Blocks '..', '\\', '/', and null bytes.
    Returns a standardized error message if unsafe, or None if safe.
    """
    if not identifier:
        return None
    if "/" in identifier or ".." in identifier or "\\" in identifier or "\0" in identifier:
        return "❌ Path traversal attempt detected."
    return None


class AppState:
    def __init__(self) -> None:
        # Training stop signal
        self.stop_event: threading.Event = threading.Event()

        # Inference model cache — avoids reloading the same model on every call.
        # v3.1 Fix: cleared only when a *different* model is requested.
        self.inference_cache: dict = {}

        # vLLM engine cache — avoids expensive engine re-initialisation.
        # v2.9 Fix 3e: keyed by (model_path, quantization, tensor_parallel_size).
        # H-9 FIX: Cache is now bounded by MAX_VLLM_ENGINES (default 1).
        # Previously unbounded, causing memory exhaustion when multiple models
        # were loaded in the same session. Set MAX_VLLM_ENGINES=2 in the
        # environment to allow more concurrent engines (requires proportionally
        # more GPU VRAM per additional engine).
        self.vllm_cache: dict = {}
        self.max_vllm_engines: int = int(os.environ.get("MAX_VLLM_ENGINES", "1"))

        # ── Resource tracking for DoS prevention (Sentinel) ────────────────
        # Many operations create temporary files or directories that are
        # never deleted. We track the most recent ones to ensure we don't
        # exhaust disk space in long-running sessions.
        self._last_zip_path: str | None = None
        self._last_model_dir: str | None = None
        self._last_gguf_dir: str | None = None
        self._last_peft_dir: str | None = None
        self._last_batch_path: str | None = None
        self._last_merged_dir: str | None = None

    def cleanup_resource(self, attr_name: str, new_value: str | None = None) -> None:
        """Safely delete a file or directory tracked by an attribute and update it.

        Sentinel: Centralised cleanup to prevent disk exhaustion (DoS prevention).
        If the path exists, it is removed (shutil.rmtree for dirs, os.unlink for files).
        The attribute is then updated to `new_value` (default None).
        """
        old_path = getattr(self, attr_name, None)
        if old_path and os.path.exists(old_path):
            try:
                if os.path.isdir(old_path):
                    shutil.rmtree(old_path)
                else:
                    os.unlink(old_path)
            except OSError:
                pass  # Best effort cleanup — avoid crashing on permission errors
        setattr(self, attr_name, new_value)


# Module-level singleton — import this everywhere:
#   from core.state import app_state
app_state: AppState = AppState()


def validate_path_traversal(path: str) -> str | None:
    """Standardized validation to block path traversal and malformed inputs.

    Returns an error message starting with '❌' if '..' or '\' is detected,
    otherwise returns None.
    """
    if not path:
        return None
    if ".." in path or "\\" in path:
        return "❌ Path traversal attempt detected (contains '..' or '\\')."
    return None
