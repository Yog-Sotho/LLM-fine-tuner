"""
training/ppo.py
================
Layer 3 — PPO fine-tuning via trl.PPOTrainer.
Imports: config, core, data.
"""

import gc
import os  # H-7 FIX: was previously called via __import__("os") inside a conditional
import time

import gradio as gr
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.constants import (
    COL_PROMPT,
    COL_TEXT,
    COL_INSTRUCTION,
    HAS_PPO,
    QLORA_ENHANCED_LORA_CONFIG,
)
from core.state import app_state
from core.hardware import get_lora_targets
from data.loader import detect_file_type, load_dataset_from_file


def run_ppo_v27(
    policy_model_name: str,
    reward_model_path: str,
    ppo_file,
    output_dir: str,
    ppo_lr: float = 1.4e-5,
    ppo_batch_size: int = 1,
    ppo_mini_batch_size: int = 1,
    ppo_epochs: int = 1,
    ppo_max_new_tokens: int = 128,
    progress=gr.Progress(),
) -> str:
    """Run PPO fine-tuning using trl.PPOTrainer.

    Requires trl>=0.7.0 (HAS_PPO=True).
    Dataset must contain a 'prompt' column.
    Reward model must have been saved with AutoModelForCausalLMWithValueHead.

    Returns a status string for display in the UI.
    """
    if not HAS_PPO:
        return "❌ PPOTrainer not available. Install: pip install trl>=0.7.0"
    if ppo_file is None:
        return "❌ Please upload a dataset with a 'prompt' column."
    # H-7 FIX: os is now a proper top-level import, not __import__("os") inline.
    if not reward_model_path or not os.path.isdir(reward_model_path):
        return "❌ Reward model path is invalid or does not exist. Train a reward model first."

    try:
        from trl import (  # lazy
            AutoModelForCausalLMWithValueHead,
            PPOConfig,
            PPOTrainer,
        )

        if progress is not None:
            progress(0, desc="Loading PPO dataset…")
        ftype = detect_file_type(ppo_file)
        ds = load_dataset_from_file(ppo_file, ftype)

        # Normalise to 'prompt' column
        if COL_PROMPT not in ds.column_names:
            if COL_TEXT in ds.column_names:
                ds = ds.rename_column(COL_TEXT, COL_PROMPT)
            elif COL_INSTRUCTION in ds.column_names:
                ds = ds.rename_column(COL_INSTRUCTION, COL_PROMPT)
            else:
                return f"❌ Dataset must contain a 'prompt' column. Available: {ds.column_names}"

        if progress is not None:
            progress(0.05, desc="Loading tokenizers and models…")
        tokenizer = AutoTokenizer.from_pretrained(policy_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # v3.0 Fix #2 (Critical): PPOTrainer requires the policy model to have a value head.
        # load_qlora_model_v27 returns a plain PeftModel without a value head, causing
        # PPOTrainer to fail. Fix: Load with AutoModelForCausalLMWithValueHead then apply LoRA.
        base_policy = AutoModelForCausalLMWithValueHead.from_pretrained(
            policy_model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

        ppo_targets = (
            QLORA_ENHANCED_LORA_CONFIG["target_modules"]
            if not any(k in policy_model_name.lower() for k in ["gpt2", "pythia", "falcon"])
            else get_lora_targets(policy_model_name)
        )
        lora_cfg_ppo = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=QLORA_ENHANCED_LORA_CONFIG["r"],
            lora_alpha=QLORA_ENHANCED_LORA_CONFIG["lora_alpha"],
            target_modules=ppo_targets,
            lora_dropout=QLORA_ENHANCED_LORA_CONFIG["lora_dropout"],
            bias=QLORA_ENHANCED_LORA_CONFIG["bias"],
        )
        policy_model = get_peft_model(base_policy, lora_cfg_ppo)

        # v2.9 Fix A: Reward model was saved with AutoModelForCausalLMWithValueHead.
        try:
            reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                reward_model_path,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            reward_model.eval()
            for param in reward_model.parameters():
                param.requires_grad = False
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Reward Model. Ensure it was saved with a ValueHead "
                f"(train_reward_model_v27). Error: {e}"
            )

        # v2.9 Fix F: Reference model loaded silently (no debug prints).
        ref_model = AutoModelForCausalLM.from_pretrained(
            policy_model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        # CRITICAL Fix #4: Set ppo_epochs=1 in config; outer loop controls epochs.
        ppo_config = PPOConfig(
            output_dir=output_dir,
            learning_rate=ppo_lr,
            mini_batch_size=ppo_mini_batch_size,
            batch_size=ppo_batch_size,
            ppo_epochs=1,
            report_to="none",
        )
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=policy_model,
            ref_model=ref_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
        )

        if progress is not None:
            progress(0.2, desc="Running PPO training loop…")
        t0 = time.time()
        prompts = ds[COL_PROMPT]

        for epoch in range(ppo_epochs):
            if app_state.stop_event.is_set():
                break
            for batch_idx in range(0, len(prompts), ppo_batch_size):
                if app_state.stop_event.is_set():
                    break
                batch_prompts = prompts[batch_idx: batch_idx + ppo_batch_size]
                query_tensors = [
                    tokenizer.encode(p, return_tensors="pt").squeeze(0)
                    for p in batch_prompts
                ]

                _ppo_gen_result = ppo_trainer.generate(
                    query_tensors,
                    max_new_tokens=ppo_max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
                # Guard: newer TRL versions may return (response_tensors, logprobs).
                response_tensors = (
                    _ppo_gen_result[0]
                    if isinstance(_ppo_gen_result, tuple)
                    else _ppo_gen_result
                )
                decoded_responses = [
                    tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors
                ]

                # v3.2 Fix #2 (Medium): reward_val is already a Python float from .item().
                # Wrapping in torch.tensor() creates a 0-D tensor which causes type
                # errors in some TRL versions. Append the float directly.
                rewards = []
                with torch.no_grad():
                    for prompt, response in zip(batch_prompts, decoded_responses):
                        full_text = prompt + response
                        inputs = tokenizer(
                            full_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=1024,
                            padding=True,
                            return_attention_mask=True,
                        ).to(reward_model.device)
                        outputs = reward_model(**inputs)
                        values = outputs.values
                        last_token_index = inputs["attention_mask"][0].sum().item() - 1
                        reward_val = values[0, last_token_index].item()
                        rewards.append(reward_val)

                # M-3 FIX: Validate lengths match before calling step to surface mismatches.
                if len(rewards) != len(response_tensors):
                    raise RuntimeError(
                        f"Reward count ({len(rewards)}) != response count "
                        f"({len(response_tensors)}). Batch generation produced "
                        "mismatched results."
                    )
                ppo_trainer.step(query_tensors, response_tensors, rewards)
                done = min(batch_idx + ppo_batch_size, len(prompts))
                if progress is not None:
                    progress(
                        0.2 + 0.7 * done / len(prompts),
                        desc=f"PPO Epoch {epoch+1} Step {done}/{len(prompts)}…",
                    )

        elapsed = time.time() - t0
        if progress is not None:
            progress(0.95, desc="Saving PPO model…")
        policy_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        del policy_model, reward_model, ref_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return (
            f"✅ PPO fine-tuning complete!\n"
            f"⏱ Elapsed: {elapsed/60:.1f} min\n"
            f"📁 Saved to: {output_dir}"
        )

    except Exception as e:
        return f"❌ PPO training failed: {e}"
