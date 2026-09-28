"""
training/grpo.py
=================
Layer 3 — GRPO (Group Relative Policy Optimization) with TRL's GRPOTrainer.

GRPO samples several completions per prompt and uses each group's mean reward as
the baseline, so it needs no value model. Rewards come from:
  • a reward model trained in the Reward Model tab (sequence classifier path), and/or
  • built-in verifiable rewards (GRPO_REWARDS): reference match and maths answer
    (need a ``reference`` column), <think> format, valid JSON, regex match.
Maths and think-format use TRL's own reward functions. Generation can run on vLLM
(colocated on the training GPU). Replaces the legacy PPO pipeline.
"""

import gc
import json
import os
import re
import time

import gradio as gr
import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from config.constants import (
    ALLOW_REMOTE_CODE,
    CHECKPOINT_SAVE_STEPS,
    CHECKPOINT_TOTAL_LIMIT,
    COL_INSTRUCTION,
    COL_PROMPT,
    COL_REFERENCE,
    COL_TEXT,
    DEFAULT_GRPO_LOSS_TYPE,
    DEFAULT_LORA_VARIANT,
    DEFAULT_REPORT_TO,
    DEFAULT_SEED,
    GRPO_LORA_ALPHA,
    GRPO_LORA_RANK,
    GRPO_LOSS_TYPES,
    GRPO_REWARDS,
    GRPO_REWARDS_NEEDING_REFERENCE,
    GRPO_VLLM_GPU_MEMORY,
    HAS_GRPO,
    HAS_MATH_VERIFY,
    HAS_VLLM,
)
from core.callbacks import ETAProgressCallback, LoggingCallback, StopCallback
from core.hardware import compute_dtype, get_lora_targets, lora_variant_kwargs, select_precision
from core.run_config import latest_checkpoint, save_run_config
from core.state import app_state, validate_path_traversal
from data.loader import load_table_dataset


def _completion_text(completion) -> str:
    """Completions are strings, or message lists for conversational prompts."""
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"] if completion else ""


def _as_messages(completions) -> list:
    """TRL's reward functions read ``completion[0]["content"]``; wrap plain strings."""
    return [[{"role": "assistant", "content": c}] if isinstance(c, str) else c for c in completions]


def reference_match_reward(completions, reference=None, **kwargs) -> list[float]:
    """1.0 when the expected answer appears in the completion (case-insensitive), else 0.0."""
    if reference is None:
        return [0.0] * len(completions)
    return [
        1.0 if str(ref).strip() and str(ref).strip().lower() in _completion_text(c).lower() else 0.0
        for c, ref in zip(completions, reference, strict=True)
    ]


def math_answer_reward(completions, reference=None, **kwargs) -> list[float | None]:
    """1.0 when the answer equals the reference mathematically (TRL + math-verify).

    The answer must be LaTeX (e.g. ``\\boxed{42}``). None — ignored by GRPO — when the
    reference itself cannot be parsed.
    """
    from trl.rewards import accuracy_reward  # lazy: needs math-verify

    if reference is None:
        return [0.0] * len(completions)
    return accuracy_reward(_as_messages(completions), solution=[str(r) for r in reference])


def think_format_reward(completions, **kwargs) -> list[float]:
    """1.0 for ``<think>reasoning</think>`` followed by the answer (TRL's check)."""
    from trl.rewards import think_format_reward as trl_think_format

    return trl_think_format(_as_messages(completions))


_JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def json_reward(completions, **kwargs) -> list[float]:
    """1.0 when the whole completion is valid JSON (one ```json fence allowed)."""
    rewards = []
    for completion in completions:
        text = _completion_text(completion).strip()
        if fence := _JSON_FENCE.match(text):
            text = fence.group(1)
        try:
            json.loads(text)
            rewards.append(1.0)
        except ValueError:
            rewards.append(0.0)
    return rewards


def make_regex_reward(pattern: str):
    """Reward function: 1.0 when the whole completion (stripped) matches ``pattern``."""
    try:
        compiled = re.compile(pattern, re.DOTALL)
    except re.error as e:
        raise ValueError(f"Invalid regular expression: {e}") from e

    def regex_reward(completions, **kwargs) -> list[float]:
        return [
            1.0 if compiled.fullmatch(_completion_text(c).strip()) else 0.0 for c in completions
        ]

    return regex_reward


def build_reward_funcs(rewards, has_reference: bool, regex_pattern: str = "") -> list:
    """Reward functions for the chosen built-in rewards; raises ValueError on bad choices."""
    unknown = [r for r in rewards if r not in GRPO_REWARDS]
    if unknown:
        raise ValueError(f"Unknown reward(s) {unknown}. Choose from: {list(GRPO_REWARDS)}")
    needs_reference = [r for r in rewards if r in GRPO_REWARDS_NEEDING_REFERENCE]
    if needs_reference and not has_reference:
        raise ValueError(
            f"The {', '.join(needs_reference)} reward needs a 'reference' column in the dataset."
        )
    if "math" in rewards and not HAS_MATH_VERIFY:
        raise ValueError('The maths reward needs math-verify: pip install "math-verify>=0.5.2"')
    if "regex" in rewards and not (regex_pattern or "").strip():
        raise ValueError("The regex reward needs a regular expression.")
    funcs = {
        "reference": reference_match_reward,
        "math": math_answer_reward,
        "think_format": think_format_reward,
        "json": json_reward,
    }
    return [make_regex_reward(regex_pattern.strip()) if r == "regex" else funcs[r] for r in rewards]


def train_grpo(
    policy_model_name: str,
    reward_model_path: str,
    prompts_file,
    output_dir: str,
    learning_rate: float = 1e-5,
    epochs: int = 1,
    num_generations: int = 4,
    prompts_per_step: int = 1,
    max_completion_length: int = 128,
    beta: float = 0.0,
    resume: bool = False,
    loss_type: str = DEFAULT_GRPO_LOSS_TYPE,
    rewards: list[str] | None = None,
    regex_pattern: str = "",
    lora_rank: int = GRPO_LORA_RANK,
    lora_alpha: int = GRPO_LORA_ALPHA,
    lora_variant: str = DEFAULT_LORA_VARIANT,
    use_vllm: bool = False,
    progress=gr.Progress(),
    request: gr.Request | None = None,
) -> str:
    """Fine-tune a policy with GRPO. Returns a status string for the UI."""
    policy_model_name = policy_model_name.strip() if policy_model_name else ""
    reward_model_path = reward_model_path.strip() if reward_model_path else ""
    output_dir = output_dir.strip() if output_dir else ""
    if err := (
        validate_path_traversal(policy_model_name)
        or validate_path_traversal(reward_model_path)
        or validate_path_traversal(output_dir)
    ):
        return err

    if not HAS_GRPO:
        return '❌ GRPOTrainer not available. Install: pip install "trl>=0.29.1,<2"'
    if prompts_file is None:
        return "❌ Please upload a dataset with a 'prompt' column (optionally 'reference')."
    if reward_model_path and not os.path.isfile(os.path.join(reward_model_path, "config.json")):
        return "❌ Reward model path must be a saved model directory (train one in step A)."
    num_generations = int(num_generations)
    if num_generations < 2:
        return "❌ GRPO needs at least 2 generations per prompt to compute a group baseline."
    if loss_type not in GRPO_LOSS_TYPES:
        return f"❌ Loss type must be one of: {', '.join(GRPO_LOSS_TYPES)}"
    try:
        variant_kwargs = lora_variant_kwargs(lora_variant)
    except ValueError as e:
        return f"❌ {e}"
    if use_vllm and not (HAS_VLLM and torch.cuda.is_available()):
        return (
            "❌ vLLM generation needs a CUDA GPU and vLLM built for your TRL version: "
            'pip install "trl[vllm]"'
        )

    stop_event = app_state.session_for(request).stop_event
    stop_event.clear()
    set_seed(DEFAULT_SEED)  # before model/LoRA creation, so runs are reproducible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from trl import GRPOConfig, GRPOTrainer  # lazy

        if progress is not None:
            progress(0, desc="Loading prompts…")
        ds = load_table_dataset(prompts_file)
        if COL_PROMPT not in ds.column_names:
            for alt in (COL_TEXT, COL_INSTRUCTION):
                if alt in ds.column_names:
                    ds = ds.rename_column(alt, COL_PROMPT)
                    break
            else:
                return f"❌ Dataset needs a 'prompt' column. Found: {ds.column_names}"
        keep = [c for c in (COL_PROMPT, COL_REFERENCE) if c in ds.column_names]
        ds = ds.select_columns(keep).filter(lambda row: str(row[COL_PROMPT]).strip() != "")
        if len(ds) == 0:
            return "❌ No non-empty prompts found."

        has_reference = COL_REFERENCE in ds.column_names
        if rewards is None:  # callers that predate reward selection: reference if available
            rewards = ["reference"] if has_reference else []
        rewards = list(rewards)
        try:
            reward_funcs: list = build_reward_funcs(rewards, has_reference, regex_pattern)
        except ValueError as e:
            return f"❌ {e}"
        if reward_model_path:
            reward_funcs.insert(0, reward_model_path)
        if not reward_funcs:
            return (
                "❌ GRPO needs a reward: give a reward model path (step A) and/or choose "
                "built-in rewards (reference answer and maths need a 'reference' column)."
            )
        # Described now: GRPOTrainer replaces reward-model paths in the list with loaded models.
        reward_desc = " + ".join(
            (["reward model"] if reward_model_path else [])
            + ["reference match" if r == "reference" else r.replace("_", " ") for r in rewards]
        )

        if progress is not None:
            progress(0.05, desc="Loading policy model…")
        tokenizer = AutoTokenizer.from_pretrained(policy_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # generation needs left padding
        model = AutoModelForCausalLM.from_pretrained(
            policy_model_name,
            torch_dtype=compute_dtype(device),
            trust_remote_code=ALLOW_REMOTE_CODE,
        )

        config = GRPOConfig(
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            # One step processes `prompts_per_step` prompts × `num_generations` completions,
            # which keeps the generation batch divisible by num_generations.
            per_device_train_batch_size=num_generations * int(prompts_per_step),
            gradient_accumulation_steps=1,
            num_generations=num_generations,
            max_completion_length=int(max_completion_length),
            beta=beta,
            loss_type=loss_type,
            logging_steps=1,
            save_strategy="steps",
            save_steps=CHECKPOINT_SAVE_STEPS,
            save_total_limit=CHECKPOINT_TOTAL_LIMIT,
            report_to=DEFAULT_REPORT_TO,
            seed=DEFAULT_SEED,
            **select_precision(device),
            # vllm_mode is set explicitly: its default differs between TRL versions.
            **(
                {"use_vllm": True, "vllm_mode": "colocate",
                 "vllm_gpu_memory_utilization": GRPO_VLLM_GPU_MEMORY}
                if use_vllm else {}
            ),
        )  # fmt: skip
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            target_modules=get_lora_targets(),
            lora_dropout=0.05,
            bias="none",
            **variant_kwargs,
        )

        log_cb = LoggingCallback()
        callbacks = [StopCallback(stop_event), log_cb]
        if progress is not None:
            callbacks.append(
                ETAProgressCallback(gradio_progress=progress, progress_start=0.2, progress_end=0.9)
            )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=list(reward_funcs),
            args=config,
            train_dataset=ds,
            processing_class=tokenizer,
            callbacks=callbacks,
            peft_config=peft_config,
        )

        if progress is not None:
            progress(0.2, desc="GRPO training started… calculating ETA…")
        t0 = time.time()
        trainer.train(resume_from_checkpoint=latest_checkpoint(output_dir) if resume else None)
        elapsed = time.time() - t0
        status = "stopped by user" if stop_event.is_set() else "complete"

        if progress is not None:
            progress(0.95, desc="Saving policy adapter…")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        save_run_config(
            output_dir,
            mode="grpo",
            model=policy_model_name,
            dataset=ds,
            seed=DEFAULT_SEED,
            report_to=DEFAULT_REPORT_TO,
            reward=reward_desc,
            reward_model=reward_model_path or None,
            rewards=rewards,
            regex_pattern=regex_pattern.strip() if "regex" in rewards else None,
            loss_type=loss_type,
            use_vllm=bool(use_vllm),
            hyperparams={
                "learning_rate": learning_rate,
                "epochs": epochs,
                "num_generations": num_generations,
                "prompts_per_step": int(prompts_per_step),
                "max_completion_length": int(max_completion_length),
                "beta": beta,
            },
            peft={
                "method": "LoRA",
                "lora_rank": int(lora_rank),
                "lora_alpha": int(lora_alpha),
                "lora_variant": lora_variant,
            },
        )

        if progress is not None:
            progress(1.0, desc="✅ Complete!")
        return (
            f"✅ GRPO training {status}!\n"
            f"🎯 Reward: {reward_desc}\n"
            f"⏱ Elapsed: {elapsed / 60:.1f} min\n"
            f"📁 Adapter saved to: {output_dir}"
        )

    except Exception as e:
        return f"❌ GRPO training failed: {e}"
    finally:
        try:
            del trainer
        except (NameError, UnboundLocalError):
            pass
        try:
            del model
        except (NameError, UnboundLocalError):
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
