"""
inference/benchmarks.py
========================
Layer 4 — standard benchmarks through EleutherAI's lm-evaluation-harness.

The harness model is built as an object (``HFLM(...)``) rather than from a
``"pretrained=...,peft=..."`` string: user input in such a string could inject
extra options (e.g. enabling ``trust_remote_code``). Adapters must be safetensors
and remote code follows ALLOW_REMOTE_CODE, as everywhere else.
"""

import math
import numbers
import os

import gradio as gr
import pandas as pd
import torch

from config.constants import (
    ALLOW_REMOTE_CODE,
    BENCHMARK_MAX_LIMIT,
    BENCHMARK_TASKS,
    HAS_LM_EVAL,
)
from core.state import redact_sensitive_info, validate_adapter_dir, validate_path_traversal


def _task_metrics(result: dict) -> dict[str, float]:
    """Scores of one task, without stderr/sample counts ("acc,none" → "acc")."""
    out = {}
    for key, value in result.items():
        if "stderr" in key or key.startswith("sample_len"):
            continue
        if not isinstance(value, numbers.Real) or isinstance(value, bool):  # numpy floats too
            continue
        out[key.replace(",none", "")] = round(float(value), 4)
    return out


def run_benchmarks(
    model_name: str,
    lora_path: str | None,
    tasks: list[str],
    limit: int,
    compare_base: bool = False,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Run the selected benchmarks; one row per (task, metric).

    Columns: task, metric, model score — or fine-tuned, base and Δ when
    ``compare_base`` is set (the base model is evaluated without the adapter).
    """
    if not HAS_LM_EVAL:
        raise ImportError(
            'lm-evaluation-harness not installed. Run: pip install "lm-eval>=0.4.13,<0.5"'
        )
    model_name = (model_name or "").strip()
    lora_path = (lora_path or "").strip() or None
    if not model_name:
        raise ValueError("Choose a model.")
    if err := (validate_path_traversal(model_name) or validate_path_traversal(lora_path)):
        raise ValueError(err)
    unknown = [t for t in tasks if t not in BENCHMARK_TASKS]
    if not tasks or unknown:
        raise ValueError(f"Choose benchmarks from: {', '.join(BENCHMARK_TASKS)}")
    limit = int(limit)
    if not 1 <= limit <= BENCHMARK_MAX_LIMIT:
        raise ValueError(f"Examples per task must be between 1 and {BENCHMARK_MAX_LIMIT}.")
    if lora_path:
        if not os.path.isdir(lora_path):
            raise ValueError(f"Adapter folder not found: {lora_path}")
        if err := validate_adapter_dir(lora_path):
            raise ValueError(err)
    if compare_base and not lora_path:
        raise ValueError("Comparing with the base model needs a LoRA adapter path.")

    from lm_eval import simple_evaluate  # lazy: imports the whole task registry
    from lm_eval.models.huggingface import HFLM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def evaluate(peft: str | None) -> dict:
        lm = HFLM(
            pretrained=model_name,
            peft=peft,
            batch_size=batch_size,
            device=device,
            trust_remote_code=ALLOW_REMOTE_CODE,
        )
        try:
            # Never pass use_cache: lm-eval's response cache is a pickle-based sqlitedict
            # file (CVE-2024-35515), waived in CI's pip-audit only because it is unused.
            return simple_evaluate(model=lm, tasks=list(tasks), limit=limit)["results"]
        finally:
            del lm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    tuned = evaluate(lora_path)
    base = evaluate(None) if compare_base else {}

    rows = []
    for task in tasks:
        scores = _task_metrics(tuned.get(task, {}))
        base_scores = _task_metrics(base.get(task, {}))
        for metric, value in scores.items():
            if compare_base:
                b = base_scores.get(metric, math.nan)
                rows.append(
                    {"task": task, "metric": metric, "fine-tuned": value, "base": b,
                     "Δ": round(value - b, 4)}
                )  # fmt: skip
            else:
                rows.append({"task": task, "metric": metric, "score": value})
    return pd.DataFrame(rows)


def on_benchmark_click(
    model_choice: str,
    custom_model: str,
    lora_path: str,
    tasks: list[str],
    limit: int,
    compare_base: bool,
    progress=gr.Progress(),
) -> tuple[str, pd.DataFrame]:
    """Evaluation tab handler for the Run Benchmarks button."""
    model_name = (custom_model or "").strip() or model_choice
    progress(0.05, desc="Running benchmarks (lm-evaluation-harness)…")
    try:
        table = run_benchmarks(model_name, lora_path, tasks or [], limit, compare_base)
    except Exception as e:  # missing package, bad input, download or model errors
        return f"❌ Benchmarks failed: {redact_sensitive_info(str(e))}", pd.DataFrame()
    progress(1.0, desc="Done!")
    return (
        f"✅ {len(tasks)} benchmark(s), up to {int(limit)} examples each. "
        "Small limits give rough estimates; scores are fractions (1.0 = 100%).",
        table,
    )
