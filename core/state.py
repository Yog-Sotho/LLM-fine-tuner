"""
core/state.py
==============
Application-wide mutable state. Pure stdlib — no internal project imports.

The module-level singleton `app_state` is the single source of truth for:
  • stop_event       — set by on_stop(), polled by StopCallback every step
  • inference_cache  — (model_name, lora_path) → (model, tokenizer)
  • vllm_cache       — (model_path, quant, tensor_parallel) → LLM engine
  • _cache_lock      — mutex protecting inference_cache read-modify-write ops
  • _vllm_lock       — mutex protecting vllm_cache read-modify-write ops

Fix log
-------
  H1 (High): Gradio dispatches event handlers on multiple threads. The
     check-then-write pattern in _load_for_inference was a TOCTOU race:
     two concurrent inference requests with different model IDs could
     clear the cache at the same time and corrupt state. Added
     `_cache_lock` (threading.Lock) to guard the inference_cache and
     `_vllm_lock` to guard the vllm_cache. Both locks are used in
     generate.py and vllm_runner.py respectively.
"""

import threading


class AppState:
    def __init__(self) -> None:
        # Training stop signal
        self.stop_event: threading.Event = threading.Event()

        # Inference model cache — avoids reloading the same model on every call.
        # v3.1 Fix: cleared only when a *different* model is requested.
        self.inference_cache: dict = {}

        # H1 FIX: Mutex for inference_cache. Acquire before any check-then-write
        # on inference_cache to prevent TOCTOU races in multi-threaded Gradio.
        self._cache_lock: threading.Lock = threading.Lock()

        # vLLM engine cache — avoids expensive engine re-initialisation.
        # v2.9 Fix 3e: keyed by (model_path, quantization, tensor_parallel_size).
        self.vllm_cache: dict = {}

        # H5 FIX: Mutex for vllm_cache. Same rationale as _cache_lock above.
        self._vllm_lock: threading.Lock = threading.Lock()


# Module-level singleton — import this everywhere:
#   from core.state import app_state
app_state: AppState = AppState()
