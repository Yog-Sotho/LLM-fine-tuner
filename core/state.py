"""
core/state.py
==============
Application-wide mutable state. Pure stdlib — no internal project imports.

The module-level singleton `app_state` is the single source of truth for:
  • stop_event       — set by on_stop(), polled by StopCallback every step
  • inference_cache  — (model_name, lora_path) → (model, tokenizer)
  • vllm_cache       — (model_path, quant, tensor_parallel) → LLM engine
  • _last_zip_path   — path of last training ZIP for cleanup (M-6 fix)
"""

import os
import threading


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

        # M-6 FIX: Track last ZIP path for cleanup on next training run.
        # create_zip_from_folder() creates NamedTemporaryFile(delete=False)
        # which is never deleted without explicit cleanup. On_train_click
        # stores the current zip path here and removes the previous one.
        self._last_zip_path: str | None = None


# Module-level singleton — import this everywhere:
#   from core.state import app_state
app_state: AppState = AppState()
