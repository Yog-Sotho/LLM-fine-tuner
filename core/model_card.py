"""
core/model_card.py
==================
Layer 1 — README.md model card written next to every trained model.

Built from the run record (run_config.yaml), so every trainer — UI or CLI, SFT to
GRPO — gets the same card. The Hub reads the YAML header: ``base_model`` links the
model to its base, ``library_name: peft`` marks an adapter, ``datasets`` links the
training data. The header is rendered by huggingface_hub, so it is always valid YAML.
"""

import os
import re

from huggingface_hub import ModelCard, ModelCardData

from config.constants import HUB_MODEL_ID_PATTERN

_MODE_NAMES = {
    "sft": "supervised fine-tuning (SFT)",
    "dpo": "Direct Preference Optimization (DPO)",
    "orpo": "Odds Ratio Preference Optimization (ORPO)",
    "kto": "Kahneman-Tversky Optimization (KTO)",
    "grpo": "Group Relative Policy Optimization (GRPO)",
    "reward": "reward modelling",
    "distill": "knowledge distillation (GKD)",
    "merge": "adapter merging",
}


def hub_model_id(model: str) -> str | None:
    """``model`` if it can be a Hub model id; None for local paths.

    The Hub rejects a card whose ``base_model`` is not a Hub id (e.g. a local path).
    """
    model = (model or "").strip()
    if not model or os.path.exists(model) or not re.fullmatch(HUB_MODEL_ID_PATTERN, model):
        return None
    return model


def _method_tag(method: str) -> str:
    if "lora" in method.lower():
        return "lora"
    return "full-finetune" if method == "Full Fine-tuning" else "peft"


def build_model_card(record: dict, is_adapter: bool) -> ModelCard:
    """Model card for a run record (the contents of run_config.yaml)."""
    mode = record.get("mode", "sft")
    model = record.get("model", "")
    base_model = hub_model_id(model)
    method = (record.get("peft") or {}).get("method", "")
    dataset = record.get("dataset") or {}
    hub_dataset = dataset.get("hub_id")

    data = ModelCardData(
        base_model=base_model,
        library_name="peft" if is_adapter else "transformers",
        pipeline_tag="text-classification"
        if mode == "reward"
        else "image-text-to-text"
        if record.get("vision")
        else "text-generation",
        datasets=[hub_dataset] if hub_dataset else None,
        tags=[
            "llm-fine-tuner",
            "trl",
            mode,
            *([_method_tag(method)] if method else []),
            *(["vision"] if record.get("vision") else []),
            *([record["quantization"]] if record.get("quantization") else []),
        ],
    )

    if record.get("vision"):
        usage = (
            "from transformers import AutoModelForImageTextToText, AutoProcessor\n\n"
            'processor = AutoProcessor.from_pretrained("<this repo>")\n'
            + (
                "from peft import PeftModel\n\n"
                f'base = AutoModelForImageTextToText.from_pretrained("{model}")\n'
                'model = PeftModel.from_pretrained(base, "<this repo>")'
                if is_adapter
                else 'model = AutoModelForImageTextToText.from_pretrained("<this repo>")'
            )
        )
    elif mode == "reward":
        usage = (
            "from transformers import AutoModelForSequenceClassification\n\n"
            'model = AutoModelForSequenceClassification.from_pretrained("<this repo>")'
        )
    elif is_adapter:
        usage = (
            "from peft import AutoPeftModelForCausalLM\n\n"
            'model = AutoPeftModelForCausalLM.from_pretrained("<this repo>")'
        )
    else:
        usage = (
            "from transformers import AutoModelForCausalLM\n\n"
            'model = AutoModelForCausalLM.from_pretrained("<this repo>")'
        )

    settings = {**(record.get("hyperparams") or {}), **(record.get("peft") or {})}
    rows = "\n".join(f"| {k} | {v} |" for k, v in settings.items())
    source = f"[{hub_dataset}](https://huggingface.co/datasets/{hub_dataset})" if hub_dataset else (
        "a local file"
    )  # fmt: skip
    libraries = ", ".join(f"{k} {v}" for k, v in (record.get("libraries") or {}).items())
    teacher_line = f"- **Teacher:** `{record['teacher']}`\n" if record.get("teacher") else ""
    data_line = (
        f"- **Data:** {dataset.get('rows', '?')} examples from {source} "
        f"(SHA-256 `{dataset.get('sha256', '?')}`)\n"
        if dataset
        else ""
    )
    if record.get("sources"):  # adapter merge
        merged = ", ".join(f"`{s['adapter']}` × {s['weight']}" for s in record["sources"])
        data_line = f"- **Merged from:** {merged} ({(record.get('merge') or {}).get('method')})\n"
    body = f"""
# {model.rstrip("/").split("/")[-1] or "Model"} — {mode.upper()}

Trained from `{model}` with {_MODE_NAMES.get(mode, mode)} using
[LLM Fine-Tuner](https://github.com/Yog-Sotho/LLM-fine-tuner){" as a LoRA adapter" if is_adapter else ""}.

## Usage

```python
{usage}
```

## Training

| Setting | Value |
| --- | --- |
{rows}
| seed | {record.get("seed")} |

{data_line}{teacher_line}- **Libraries:** {libraries}
- **Trained:** {record.get("created_at", "?")}

`run_config.yaml` in this repository records every setting needed to repeat the run.

## License

This model inherits the license and usage terms of the base model `{model}`.
"""
    return ModelCard(f"---\n{data.to_yaml()}\n---\n{body}")


def write_model_card(output_dir: str, record: dict) -> str:
    """Write README.md for the run in ``output_dir`` and return its path."""
    is_adapter = os.path.isfile(os.path.join(output_dir, "adapter_config.json"))
    path = os.path.join(output_dir, "README.md")
    build_model_card(record, is_adapter).save(path)
    return path
