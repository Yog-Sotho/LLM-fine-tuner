"""
cli/commands.py
================
v3.2 fully-functional Typer CLI — all five commands implemented (not stubs).
Imports: config.constants, data.loader, data.preprocessing, training.*,
         inference.evaluation, stdlib, torch, typer.

Commands
--------
train     — headless SFT / DPO training
reward    — train a reward model from preference data
orpo      — ORPO alignment training
ppo       — PPO fine-tuning with a trained reward model
evaluate  — batched BLEU / ROUGE / BERTScore evaluation

Fix history preserved inline:
  Minor Fix 1   : --qlora-enhanced overrides --peft correctly
  v3.2 Fix #3   : ALL sys.argv > 1 cases delegated to Typer (--help fixed)
  v2.9-E        : data parameter name consistent across all commands
  v2.9 Minor #7 : ORPO --alpha option present
  FIX 2b        : Batched generation in evaluate (attention-mask strip)
  FIX 2c        : reward --max-length exposed
"""

import os
import sys
from datetime import datetime
from typing import Optional

import torch
import typer

from config.constants import (
    COL_CHOSEN, COL_REJECTED,
    COL_PROMPT, COL_TEXT, COL_INSTRUCTION,
    HAS_REWARD_TRAINER, HAS_PPO, HAS_ORPO,
)
from data.loader import load_dataset_from_file
from data.preprocessing import validate_and_clean_dataset
from inference.evaluation import compute_bleu_rouge, compute_bertscore_metric
from inference.generate import _load_for_inference
from training.orpo import train_orpo_v27
from training.ppo import run_ppo_v27
from training.reward import train_reward_model_v27
from training.sft import train_model


app = typer.Typer(
    name="llm-fine-tuner",
    help="🧠 LLM Fine-Tuner v3.2 — headless CLI for every training mode.",
    add_completion=False,
)


class DummyFile:
    """Minimal file-like proxy so core functions that expect gr.File work in CLI context."""
    def __init__(self, name: str):
        self.name = name


# ── train ──────────────────────────────────────────────────────────────────

@app.command()
def train(
    model: str = typer.Option(..., "--model", help="Base model ID or local path"),
    data: str = typer.Option(..., "--data", help="Dataset file (.csv or .jsonl)"),
    output: str = typer.Option("./output", "--output", help="Output directory"),
    epochs: int = typer.Option(3, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(2, "--batch-size", help="Per-device batch size"),
    max_length: int = typer.Option(256, "--max-length", help="Maximum sequence length"),
    learning_rate: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    peft_method: str = typer.Option("LoRA", "--peft",
                                     help="PEFT method: LoRA | QLoRA Enhanced | Full Fine-tuning | Auto"),
    lora_rank: int = typer.Option(8, "--lora-rank", help="LoRA rank"),
    use_qlora_enhanced: bool = typer.Option(
        False, "--qlora-enhanced",
        help="Activate QLoRA Enhanced (NF4 + double quant). Overrides --peft.",
    ),
    use_flash_attn: bool = typer.Option(False, "--flash-attn", help="Enable Flash Attention 2"),
):
    """Headless SFT training — reuses the same pipeline as the Gradio UI."""
    # Minor Fix 1: --qlora-enhanced actually overrides --peft instead of being ignored.
    if use_qlora_enhanced:
        if peft_method != "QLoRA Enhanced":
            typer.echo(f"⚠️  --qlora-enhanced overrides --peft '{peft_method}' → 'QLoRA Enhanced'")
        peft_method = "QLoRA Enhanced"

    typer.echo(f"🚀 Starting training: {model} | PEFT: {peft_method} | Data: {data} | Output: {output}")

    if not os.path.exists(data):
        typer.echo(f"❌ Dataset not found: {data}", err=True)
        raise typer.Exit(code=1)

    ftype = _infer_ftype(data)
    if ftype is None:
        typer.echo("❌ Unsupported format. Use .csv or .jsonl", err=True)
        raise typer.Exit(code=1)

    try:
        ds = load_dataset_from_file(DummyFile(data), ftype)
        ds, issues = validate_and_clean_dataset(ds)
        if len(ds) == 0:
            typer.echo("❌ Dataset empty after validation", err=True)
            raise typer.Exit(code=1)
        _print_issues(issues)

        hyperparams = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": 4,
            "max_length": max_length,
            "warmup_steps": 100,
            "lora_rank": lora_rank,
            "lora_alpha": lora_rank * 2,
            "lr_scheduler": "cosine",
        }
        msg, _ = train_model(
            model_name=model, dataset=ds, output_dir=output,
            hyperparams=hyperparams,
            device="cuda" if torch.cuda.is_available() else "cpu",
            peft_method=peft_method,
            use_lora=True, lora_rank=lora_rank, lora_alpha=lora_rank * 2,
            prefix_tuning_num_virtual_tokens=30, prefix_tuning_token_dim=512,
            prefix_tuning_num_layers=2, prompt_tuning_num_virtual_tokens=20,
            adapter_reduction_factor=16,
            resume_from_checkpoint=False, early_stop=3,
            lr_scheduler_type="cosine", gradient_checkpointing=True,
            use_unsloth=False, use_chat_template=False,
            system_prompt="You are a helpful assistant.",
            training_mode="sft", dpo_beta=0.1, heretic_mode=False,
            progress=None,
            use_flash_attn=use_flash_attn,
        )
        typer.echo(f"\n✅ {msg}")
        typer.echo(f"📁 Model saved to: {os.path.abspath(output)}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"\n❌ Training failed: {e}", err=True)
        raise typer.Exit(code=1)


# ── reward ─────────────────────────────────────────────────────────────────

@app.command()
def reward(
    model: str = typer.Option(..., "--model", help="Base model ID for reward training"),
    data: str = typer.Option(..., "--data", help="Preference dataset (chosen/rejected columns)"),
    output: str = typer.Option("./reward_model", "--output", help="Output directory"),
    epochs: int = typer.Option(3, "--epochs"),
    lr: float = typer.Option(1.4e-5, "--lr"),
    max_length: int = typer.Option(1024, "--max-length", help="Max sequence length (FIX 2c)"),
    batch_size: int = typer.Option(4, "--batch-size"),
):
    """Train a Reward Model from preference data (FIX 3c: full implementation)."""
    typer.echo(f"🎖️  Training reward model: {model} | Max Length: {max_length}")

    if not HAS_REWARD_TRAINER:
        typer.echo("❌ Install: pip install trl>=0.7.0", err=True)
        raise typer.Exit(code=1)

    ftype = _infer_ftype(data)
    try:
        ds = load_dataset_from_file(DummyFile(data), ftype, is_dpo=True)
        if not (COL_CHOSEN in ds.column_names and COL_REJECTED in ds.column_names):
            typer.echo(
                f"❌ Dataset requires '{COL_CHOSEN}' and '{COL_REJECTED}' columns", err=True
            )
            raise typer.Exit(code=1)

        result = train_reward_model_v27(
            model_name=model,
            reward_file=DummyFile(data),
            output_dir=output,
            rm_epochs=epochs,
            rm_lr=lr,
            rm_batch_size=batch_size,
            rm_eval_steps=100,
            rm_max_length=max_length,
            progress=None,
        )
        if "✅" in result:
            typer.echo(f"\n{result}")
            typer.echo(f"📁 Reward model saved to: {os.path.abspath(output)}")
        else:
            typer.echo(result, err=True)
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"\n❌ Reward training failed: {e}", err=True)
        raise typer.Exit(code=1)


# ── orpo ───────────────────────────────────────────────────────────────────

@app.command()
def orpo(
    model: str = typer.Option(..., "--model", help="Base model ID"),
    data: str = typer.Option(..., "--data",
                              help="Preference dataset (prompt / chosen / rejected)"),
    output: str = typer.Option("./orpo_model", "--output", help="Output directory"),
    epochs: int = typer.Option(3, "--epochs"),
    lr: float = typer.Option(1e-4, "--lr"),
    beta: float = typer.Option(0.1, "--beta"),
    alpha: float = typer.Option(0.1, "--alpha", help="ORPO alpha parameter"),  # v2.9 Minor Fix #7
    batch_size: int = typer.Option(2, "--batch-size"),
):
    """ORPO alignment training (FIX 3c: full implementation)."""
    typer.echo(f"🌀 ORPO training: {model} | Beta: {beta} | Alpha: {alpha}")

    if not HAS_ORPO:
        typer.echo("❌ Install: pip install trl>=0.8.0", err=True)
        raise typer.Exit(code=1)

    ftype = _infer_ftype(data)
    try:
        ds = load_dataset_from_file(DummyFile(data), ftype, is_dpo=True)
        required = [COL_PROMPT, COL_CHOSEN, COL_REJECTED]
        if not all(c in ds.column_names for c in required):
            typer.echo(f"❌ Dataset requires columns: {required}", err=True)
            raise typer.Exit(code=1)

        result = train_orpo_v27(
            model_name=model,
            orpo_file=DummyFile(data),
            output_dir=output,
            orpo_lr=lr, orpo_beta=beta, orpo_alpha=alpha,
            orpo_epochs=epochs, orpo_batch_size=batch_size,
            progress=None,
        )
        if "✅" in result:
            typer.echo(f"\n{result}")
            typer.echo(f"📁 ORPO model saved to: {os.path.abspath(output)}")
        else:
            typer.echo(result, err=True)
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"\n❌ ORPO training failed: {e}", err=True)
        raise typer.Exit(code=1)


# ── ppo ────────────────────────────────────────────────────────────────────

@app.command()
def ppo(
    policy_model: str = typer.Option(..., "--policy-model", help="Policy model ID"),
    reward_model: str = typer.Option(..., "--reward-model", help="Trained reward model path"),
    data: str = typer.Option(..., "--data", help="Prompts dataset (prompt column)"),
    output: str = typer.Option("./ppo_model", "--output", help="Output directory"),
    epochs: int = typer.Option(1, "--epochs", help="PPO epochs"),
    lr: float = typer.Option(1.4e-5, "--lr"),
    batch_size: int = typer.Option(1, "--batch-size"),
    mini_batch_size: int = typer.Option(1, "--mini-batch-size"),
    max_new_tokens: int = typer.Option(128, "--max-new-tokens",
                                        help="Max tokens per generated response"),
):
    """PPO fine-tuning with a trained reward model (FIX 3c: full implementation)."""
    typer.echo(f"🔁 PPO: Policy={policy_model} | Reward={reward_model}")

    if not HAS_PPO:
        typer.echo("❌ Install: pip install trl>=0.7.0", err=True)
        raise typer.Exit(code=1)
    if not os.path.isdir(reward_model):
        typer.echo(f"❌ Reward model path invalid: {reward_model}", err=True)
        raise typer.Exit(code=1)

    ftype = _infer_ftype(data)
    try:
        ds = load_dataset_from_file(DummyFile(data), ftype)
        # Normalise column name to COL_PROMPT
        if COL_PROMPT not in ds.column_names:
            if COL_TEXT in ds.column_names:
                ds = ds.rename_column(COL_TEXT, COL_PROMPT)
            elif COL_INSTRUCTION in ds.column_names:
                ds = ds.rename_column(COL_INSTRUCTION, COL_PROMPT)
            else:
                typer.echo(
                    f"❌ Dataset requires 'prompt' column. Found: {ds.column_names}", err=True
                )
                raise typer.Exit(code=1)

        result = run_ppo_v27(
            policy_model_name=policy_model,
            reward_model_path=reward_model,
            ppo_file=DummyFile(data),
            output_dir=output,
            ppo_lr=lr, ppo_batch_size=batch_size,
            ppo_mini_batch_size=mini_batch_size,
            ppo_epochs=epochs,
            ppo_max_new_tokens=max_new_tokens,
            progress=None,
        )
        if "✅" in result:
            typer.echo(f"\n{result}")
            typer.echo(f"📁 PPO model saved to: {os.path.abspath(output)}")
        else:
            typer.echo(result, err=True)
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"\n❌ PPO training failed: {e}", err=True)
        raise typer.Exit(code=1)


# ── evaluate ───────────────────────────────────────────────────────────────

@app.command()
def evaluate(
    model: str = typer.Option(..., "--model", help="Model ID or local path"),
    data: str = typer.Option(..., "--data", help="Test dataset (prompt / reference columns)"),
    lora: Optional[str] = typer.Option(None, "--lora", help="PEFT adapter path"),
    bertscore: bool = typer.Option(False, "--bertscore", help="Compute BERTScore"),
    batch_size: int = typer.Option(4, "--batch-size", help="Generation batch size (FIX 2b)"),
    max_new_tokens: int = typer.Option(150, "--max-new-tokens", help="Tokens to generate"),
):
    """Batched BLEU / ROUGE / BERTScore evaluation suite (FIX 2b: batched generation)."""
    typer.echo(f"🧪 Evaluating {model} on {data} (batch_size={batch_size})")

    if not os.path.isfile(data):
        typer.echo(f"❌ Dataset not found: {data}", err=True)
        raise typer.Exit(code=1)

    try:
        import pandas as pd

        df = pd.read_csv(data) if data.endswith(".csv") else pd.read_json(data, lines=True)
        if "prompt" not in df.columns:
            typer.echo("❌ Dataset requires 'prompt' column", err=True)
            raise typer.Exit(code=1)

        prompts    = df["prompt"].astype(str).tolist()
        references = df["reference"].astype(str).tolist() if "reference" in df.columns else []

        # FIX 2b: batched generation with attention-mask-based prompt stripping
        model_obj, tokenizer = _load_for_inference(model, lora)
        predictions: list[str] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model_obj.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # CRITICAL FIX #3 (evaluate): token-based prompt stripping via attention mask
            input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            for idx, gen_ids in enumerate(outputs):
                input_len    = input_lengths[idx]
                response_ids = gen_ids[input_len:] if input_len < gen_ids.shape[0] else gen_ids
                predictions.append(tokenizer.decode(response_ids, skip_special_tokens=True))

        metrics: dict = {}
        if references:
            metrics.update(compute_bleu_rouge(predictions, references))
            if bertscore:
                metrics.update(compute_bertscore_metric(predictions, references))

        typer.echo("\n📊 EVALUATION RESULTS")
        typer.echo("=" * 50)
        if metrics:
            for k, v in metrics.items():
                typer.echo(f"{k:15s}: {v}")
        else:
            typer.echo("ℹ️  No reference column — automatic metrics skipped.")

        import pandas as _pd
        result_df = _pd.DataFrame({"prompt": prompts, "prediction": predictions})
        if references:
            result_df["reference"] = references
        out_file = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result_df.to_csv(out_file, index=False)
        typer.echo(f"\n✅ Evaluation complete — {len(predictions)} examples")
        typer.echo(f"💾 Saved to: {out_file}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"\n❌ Evaluation failed: {e}", err=True)
        raise typer.Exit(code=1)


# ── helpers ────────────────────────────────────────────────────────────────

def _infer_ftype(path: str) -> Optional[str]:
    """Return 'csv' or 'jsonl' based on file extension, or None if unsupported."""
    if path.endswith(".csv"):
        return "csv"
    if path.endswith(".jsonl") or path.endswith(".json"):
        return "jsonl"
    return None


def _print_issues(issues: list[str]) -> None:
    if issues:
        typer.echo("\n⚠️  Data warnings:")
        for issue in issues:
            typer.echo(f"  • {issue}")
