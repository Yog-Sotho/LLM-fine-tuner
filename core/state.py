"""
core/state.py
==============
Application-wide mutable state. Pure stdlib — no internal project imports.

The module-level singleton `app_state` is the single source of truth for:
  • stop_event       — set by on_stop(), polled by StopCallback every step
  • inference_cache  — (model_name, lora_path) → (model, tokenizer)
  • vllm_cache       — (model_path, quant, tensor_parallel) → LLM engine
"""

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
        self.vllm_cache: dict = {}


# Module-level singleton — import this everywhere:
#   from core.state import app_state
app_state: AppState = AppState()
