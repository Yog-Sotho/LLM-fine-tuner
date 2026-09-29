"""
export/serve.py
===============
Layer 5 — serve a trained model behind an OpenAI-compatible API.

  • A ``.gguf`` file      → llama.cpp's ``llama-server`` (CPU or GPU).
  • A model folder / Hub id → ``vllm serve`` (CUDA). A LoRA adapter folder is served
    on top of its base model (``--enable-lora --lora-modules``).
Both expose ``/v1/chat/completions`` and ``/v1/models``. The API key goes through the
environment (VLLM_API_KEY / LLAMA_API_KEY), never the command line (visible in ``ps``).
Commands are argument lists — no shell.
"""

import json
import logging
import os
import re
import shutil
import subprocess

from config.constants import HAS_VLLM, HUB_MODEL_ID_PATTERN
from core.state import validate_adapter_dir, validate_path_traversal

logger = logging.getLogger(__name__)

SERVED_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}"


def build_serve_command(
    model: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str = "",
    name: str = "model",
) -> tuple[list[str], dict[str, str]]:
    """(command, extra environment) to serve ``model``; raises ValueError on bad input."""
    model, host, name = (model or "").strip(), (host or "").strip(), (name or "").strip()
    if not model or validate_path_traversal(model):
        raise ValueError("Give a model folder, a .gguf file or a Hub model id.")
    if not re.fullmatch(r"[A-Za-z0-9.:\[\]-]+", host):
        raise ValueError(f"Invalid host: {host!r}")
    if not 1 <= int(port) <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    if not re.fullmatch(SERVED_NAME_PATTERN, name):
        raise ValueError("Served model name: letters, digits, '.', '_' or '-' (max 64).")

    if model.endswith(".gguf"):
        if not os.path.isfile(model):
            raise ValueError(f"GGUF file not found: {model}")
        server = shutil.which("llama-server")
        if server is None:
            raise ValueError(
                "llama-server not found: build llama.cpp (see docs/08_export_and_deploy.md) "
                "and put llama.cpp/build/bin on PATH."
            )
        cmd = [server, "-m", model, "--host", host, "--port", str(port), "--alias", name]
        return cmd, ({"LLAMA_API_KEY": api_key} if api_key else {})

    if not HAS_VLLM:
        raise ValueError('vLLM is not installed (CUDA only): pip install "trl[vllm]".')
    vllm = shutil.which("vllm")
    if vllm is None:
        raise ValueError("The `vllm` command is not on PATH.")
    if os.path.isdir(model) and os.path.isfile(os.path.join(model, "adapter_config.json")):
        if err := validate_adapter_dir(model):
            raise ValueError(err)
        with open(os.path.join(model, "adapter_config.json"), encoding="utf-8") as f:
            adapter_config = json.load(f)
        if str(adapter_config.get("peft_type", "")).upper() != "LORA":
            raise ValueError(
                "vLLM serves LoRA adapters only. Merge this adapter first: "
                f"python main.py merge --adapter {model} --output <folder>"
            )
        base = str(adapter_config.get("base_model_name_or_path") or "").strip()
        if not base or validate_path_traversal(base):
            raise ValueError("adapter_config.json does not name a usable base model.")
        cmd = [vllm, "serve", base, "--host", host, "--port", str(port),
               "--served-model-name", f"{name}-base",
               "--enable-lora", "--lora-modules", f"{name}={model}"]  # fmt: skip
    elif os.path.isdir(model) or re.fullmatch(HUB_MODEL_ID_PATTERN, model):
        cmd = [vllm, "serve", model, "--host", host, "--port", str(port),
               "--served-model-name", name]  # fmt: skip
    else:
        raise ValueError(f"Model not found: {model}")
    return cmd, ({"VLLM_API_KEY": api_key} if api_key else {})


def serve(model: str, host: str, port: int, api_key: str = "", name: str = "model") -> int:
    """Run the server in the foreground (Ctrl+C stops it). Returns its exit code."""
    cmd, env = build_serve_command(model, host, port, api_key, name)
    logger.info("🚀 %s", " ".join(cmd))
    logger.info("OpenAI-compatible API: http://%s:%s/v1  (model: %s)", host, port, name)
    return subprocess.run(cmd, env={**os.environ, **env}).returncode
