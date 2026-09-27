"""
core/state.py
==============
Application-wide mutable state. Pure stdlib — no internal project imports.

The module-level singleton `app_state` holds:
  • per-session state   — app_state.session_for(request) → SessionState with the
                          session's stop_event and the temp files/dirs it owns
                          (training output, ZIP, GGUF, PEFT upload, batch CSV,
                          merged model); freed when the browser tab closes
  • inference_cache     — (model_name, lora_path) → (model, tokenizer), shared
  • vllm_cache          — (model_path, quant, tensor_parallel) → LLM engine, shared
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


def _remove_path(path: str | None) -> None:
    """Best-effort delete of a tracked file or directory."""
    if not path or not os.path.exists(path):
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)
    except OSError:
        pass  # Best effort — never crash a request over cleanup


# Pickle-based weight formats execute code on load (torch.load), so uploaded or
# user-selected adapters must ship safetensors weights instead.
PICKLE_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt")
SAFE_ADAPTER_WEIGHTS = "adapter_model.safetensors"


def validate_adapter_dir(path: str) -> str | None:
    """Return an error message unless ``path`` is a PEFT adapter with safetensors weights."""
    if not os.path.isfile(os.path.join(path, "adapter_config.json")):
        return "❌ Not a PEFT adapter directory (adapter_config.json missing)."
    if not os.path.isfile(os.path.join(path, SAFE_ADAPTER_WEIGHTS)):
        return (
            f"❌ Adapter must contain {SAFE_ADAPTER_WEIGHTS}. Pickle-based weights "
            "(.bin/.pt) are rejected because loading them can execute code."
        )
    return None


class SessionState:
    """State owned by one browser session (or the CLI's default session).

    Keeps one session's Stop button from stopping another session's job, and one
    session's new run from deleting another session's results.
    """

    def __init__(self) -> None:
        self.stop_event: threading.Event = threading.Event()
        self._paths: dict[str, str] = {}
        self._closed = False
        self._lock = threading.Lock()

    def release(self, slot: str) -> None:
        """Delete the file/dir currently tracked under ``slot`` (e.g. the previous run's output)."""
        with self._lock:
            old = self._paths.pop(slot, None)
        _remove_path(old)

    def track(self, slot: str, path: str) -> None:
        """Track ``path`` under ``slot`` so it is deleted on the next release or session close."""
        with self._lock:
            if not self._closed:
                old = self._paths.get(slot)
                self._paths[slot] = path
                path = old  # delete whatever it replaced, outside the lock
        _remove_path(path)

    def tracked(self, slot: str) -> str | None:
        with self._lock:
            return self._paths.get(slot)

    def close(self) -> None:
        """Stop the session's running job and delete everything it tracked."""
        self.stop_event.set()
        with self._lock:
            self._closed = True
            paths, self._paths = list(self._paths.values()), {}
        for p in paths:
            _remove_path(p)


DEFAULT_SESSION = "default"


class AppState:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._sessions_lock = threading.Lock()

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
        # Clamped to [1, 8]: an unbounded value would let one env var exhaust VRAM.
        self.max_vllm_engines: int = min(max(1, int(os.environ.get("MAX_VLLM_ENGINES", "1"))), 8)

    def session(self, session_id: str | None = None) -> SessionState:
        """Return the state for ``session_id`` (the CLI uses the default session)."""
        key = session_id or DEFAULT_SESSION
        with self._sessions_lock:
            state = self._sessions.get(key)
            if state is None:
                state = self._sessions[key] = SessionState()
            return state

    def session_for(self, request: object | None) -> SessionState:
        """Return the state for a Gradio request (``None`` → default session)."""
        return self.session(getattr(request, "session_hash", None))

    def close_session(self, session_id: str | None) -> None:
        """Drop a session when its browser tab closes, stopping its job and freeing its files."""
        if not session_id:
            return
        with self._sessions_lock:
            state = self._sessions.pop(session_id, None)
        if state is not None:
            state.close()


# Module-level singleton — import this everywhere:
#   from core.state import app_state
app_state: AppState = AppState()
