"""
export/merge.py
===============
Layer 5 — combine several LoRA adapters trained on the same base model into one.

Uses PEFT's ``add_weighted_adapter`` (no extra dependency):
  • ties / dare_ties / dare_linear — keep the ``density`` share of each adapter's
    changes, resolve sign conflicts (TIES) or drop and rescale at random (DARE);
  • linear — weighted sum; cat — exact concatenation (rank = sum of ranks);
  • svd — exact sum compressed back to the largest rank.
The result is a normal LoRA adapter (safetensors) with its own run record and card.
DoRA adapters are refused: PEFT's merge drops their magnitude vectors.
"""

import json
import logging
import os
import shutil
import tempfile

import gradio as gr

from config.constants import (
    ADAPTER_MERGE_DENSITY_METHODS,
    ADAPTER_MERGE_METHODS,
    ALLOW_REMOTE_CODE,
    DEFAULT_ADAPTER_MERGE_DENSITY,
    DEFAULT_ADAPTER_MERGE_METHOD,
)
from core.run_config import save_run_config
from core.state import redact_sensitive_info, validate_adapter_dir, validate_path_traversal

logger = logging.getLogger(__name__)

_TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
    "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
)  # fmt: skip


def _adapter_config(path: str) -> dict:
    with open(os.path.join(path, "adapter_config.json"), encoding="utf-8") as f:
        return json.load(f)


def merge_lora_adapters(
    adapter_dirs: list[str],
    output_dir: str,
    weights: list[float] | None = None,
    method: str = DEFAULT_ADAPTER_MERGE_METHOD,
    density: float = DEFAULT_ADAPTER_MERGE_DENSITY,
) -> str:
    """Merge LoRA adapters of one base model into ``output_dir``. Returns a status string."""
    adapter_dirs = [(d or "").strip().rstrip("/") for d in adapter_dirs if (d or "").strip()]
    output_dir = (output_dir or "").strip()
    for path in [*adapter_dirs, output_dir]:
        if err := validate_path_traversal(path):
            return err
    if len(adapter_dirs) < 2:
        return "❌ Give at least two adapter folders to merge."
    if not output_dir:
        return "❌ Give an output folder."
    if method not in ADAPTER_MERGE_METHODS:
        return f"❌ Method must be one of: {', '.join(ADAPTER_MERGE_METHODS)}"
    weights = [1.0] * len(adapter_dirs) if not weights else [float(w) for w in weights]
    if len(weights) != len(adapter_dirs):
        return f"❌ Give one weight per adapter ({len(adapter_dirs)}), or none for equal weights."
    if method in ADAPTER_MERGE_DENSITY_METHODS and not 0.0 < float(density) <= 1.0:
        return "❌ Density must be above 0 and at most 1 (the share of each adapter kept)."

    bases, ranks = set(), []
    for path in adapter_dirs:
        if not os.path.isdir(path):
            return f"❌ Adapter folder not found: {path}"
        if err := validate_adapter_dir(path):
            return f"{err} ({path})"
        config = _adapter_config(path)
        if str(config.get("peft_type", "")).upper() != "LORA":
            return f"❌ Only LoRA adapters can be merged; {path} is {config.get('peft_type')}."
        if config.get("use_dora"):
            return f"❌ DoRA adapters can't be merged (their magnitude vectors are lost): {path}"
        bases.add(str(config.get("base_model_name_or_path") or "").strip())
        ranks.append(config.get("r"))
    if method not in ("cat", "svd") and len(set(ranks)) > 1:
        return (
            f"❌ {method} needs adapters of the same rank (found {ranks}); "
            "use cat or svd to merge different ranks."
        )
    if len(bases) != 1 or "" in bases:
        return f"❌ All adapters must be trained on the same base model; found {sorted(bases)}."
    (base,) = bases
    if err := validate_path_traversal(base):
        return err

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        # fp32 on the CPU: merging is a one-off, and TIES/SVD need the precision.
        model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=ALLOW_REMOTE_CODE)
        names = [f"adapter_{i}" for i in range(len(adapter_dirs))]
        model = PeftModel.from_pretrained(model, adapter_dirs[0], adapter_name=names[0])
        for name, path in zip(names[1:], adapter_dirs[1:], strict=True):
            model.load_adapter(path, adapter_name=name)
        model.add_weighted_adapter(
            names, weights, "merged", combination_type=method,
            density=float(density) if method in ADAPTER_MERGE_DENSITY_METHODS else None,
        )  # fmt: skip
        rank = model.peft_config["merged"].r
        # Older PEFT leaves SVD results as views; safetensors only saves contiguous tensors.
        for name, param in model.named_parameters():
            if ".merged." in name and not param.data.is_contiguous():
                param.data = param.data.contiguous()

        os.makedirs(output_dir, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            # PEFT saves a named (non-default) adapter in a sub-folder of that name.
            model.save_pretrained(tmp, selected_adapters=["merged"])
            for name in os.listdir(os.path.join(tmp, "merged")):
                shutil.move(os.path.join(tmp, "merged", name), os.path.join(output_dir, name))
        # The tokenizer the adapters were trained with (chat template, pad token).
        for name in _TOKENIZER_FILES:
            if os.path.isfile(os.path.join(adapter_dirs[0], name)):
                shutil.copy2(os.path.join(adapter_dirs[0], name), output_dir)

        save_run_config(
            output_dir,
            mode="merge",
            model=base,
            dataset=None,
            sources=[{"adapter": d, "weight": w} for d, w in zip(adapter_dirs, weights, strict=True)],
            merge={"method": method,
                   "density": float(density) if method in ADAPTER_MERGE_DENSITY_METHODS else None},
            peft={"method": "LoRA", "lora_rank": rank},
        )  # fmt: skip
        return (
            f"✅ Merged {len(adapter_dirs)} adapters with {method} (rank {rank})\n"
            f"🧬 Base model: {base}\n"
            f"📁 {output_dir}"
        )
    except Exception as e:  # load errors, mismatched layers, rank limits
        logger.warning("Adapter merge failed: %s", e)
        return f"❌ Merge failed: {redact_sensitive_info(str(e))}"


def on_merge_adapters_click(adapters_text: str, weights_text: str, method: str, density: float,
                            output_dir: str, progress=gr.Progress()) -> str:  # fmt: skip
    """Export tab handler: one adapter folder per line; weights comma-separated (optional)."""
    adapters = [line for line in (adapters_text or "").splitlines() if line.strip()]
    try:
        weights = [float(w) for w in (weights_text or "").replace(";", ",").split(",") if w.strip()]
    except ValueError:
        return "❌ Weights must be numbers separated by commas, e.g. 1, 0.5"
    if progress is not None:
        progress(0.1, desc=f"Merging {len(adapters)} adapters ({method})…")
    return merge_lora_adapters(adapters, output_dir, weights or None, method, density)
