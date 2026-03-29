"""ui/tabs/rlhf_tab.py — Reward model, PPO, ORPO sub-tabs."""
import gradio as gr

from config.constants import HAS_ORPO, HAS_PPO, HAS_REWARD_TRAINER
from core.hardware import auto_recommend_model


def build_rlhf_tab() -> dict:
    recommended_model = auto_recommend_model()

    with gr.Tab("🤖 RLHF Pipeline"):
        gr.HTML(
            '<div id="rlhf-banner">'
            '<h3 style="color:#34d399;margin:0">'
            "🤖 v2.7 RLHF Pipeline — Reward Model · PPO · ORPO (ALL FIXES APPLIED)"
            "</h3></div>"
        )
        gr.Markdown(
            f"Dependencies: RewardTrainer {'✅' if HAS_REWARD_TRAINER else '❌'} | "
            f"PPO {'✅ (ValueHead Rewards Fixed)' if HAS_PPO else '❌'} | "
            f"ORPO {'✅' if HAS_ORPO else '❌'}\n"
            "_Install all: `pip install trl>=0.8.0`_"
        )

        with gr.Tabs():
            # ── A. Reward Model ──────────────────────────────────────────
            with gr.Tab("🎖️ A. Reward Model"):
                gr.Markdown(
                    "Train a **Reward Model** from preference data.\n"
                    "Dataset needs `chosen` and `rejected` text columns."
                )
                with gr.Row():
                    with gr.Column():
                        # L-2 FIX: Use Dropdown with preset options, matching Training tab UX.
                        rm_model_choice = gr.Dropdown(
                            choices=[
                                "gpt2", "distilgpt2", "facebook/opt-125m",
                                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                "mistralai/Mistral-7B-v0.1",
                            ],
                            value=recommended_model,
                            label="Base Model ID",
                            allow_custom_value=True,
                        )
                        rm_file = gr.File(
                            label="Preference Dataset (CSV/JSONL with chosen & rejected)",
                            file_types=[".csv", ".jsonl"],
                        )
                        rm_output_dir = gr.Textbox(label="Output Directory", value="./reward_model")
                        with gr.Row():
                            rm_epochs    = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                            rm_lr        = gr.Number(value=1.4e-5, label="Learning Rate", precision=8)
                            rm_batch     = gr.Slider(1, 16, value=4, step=1, label="Batch Size")
                        with gr.Row():
                            rm_eval_steps = gr.Slider(10, 500, value=100, step=10, label="Eval Steps")
                            rm_max_length = gr.Slider(128, 4096, value=1024, step=128,
                                                       label="Max Length")
                        rm_train_btn = gr.Button("🎖️ Train Reward Model", variant="primary")
                    with gr.Column():
                        rm_status = gr.Textbox(
                            label="Reward Model Training Status", lines=12, interactive=False
                        )

            # ── B. PPO ───────────────────────────────────────────────────
            with gr.Tab("🔁 B. PPO Fine-Tuning"):
                gr.Markdown(
                    "Fine-tune a policy model using **PPO** with your trained reward model.\n"
                    "Dataset needs a `prompt` column."
                )
                with gr.Row():
                    with gr.Column():
                        # L-2 FIX: Use Dropdown to match Training tab UX.
                        ppo_policy_model = gr.Dropdown(
                            choices=[
                                "gpt2", "distilgpt2", "facebook/opt-125m",
                                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                "mistralai/Mistral-7B-v0.1",
                            ],
                            value=recommended_model,
                            label="Policy Model ID",
                            allow_custom_value=True,
                        )
                        ppo_reward_path  = gr.Textbox(
                            label="Reward Model Path (from step A)",
                            placeholder="./reward_model",
                        )
                        ppo_file = gr.File(
                            label="Prompts Dataset (CSV/JSONL with 'prompt' column)",
                            file_types=[".csv", ".jsonl"],
                        )
                        ppo_output_dir = gr.Textbox(label="Output Directory", value="./ppo_model")
                        with gr.Row():
                            ppo_lr         = gr.Number(value=1.4e-5, label="Learning Rate",
                                                        precision=8)
                            ppo_batch      = gr.Slider(1, 8, value=1, step=1, label="Batch Size")
                            ppo_mini_batch = gr.Slider(1, 8, value=1, step=1,
                                                        label="Mini Batch Size")
                        with gr.Row():
                            ppo_epochs         = gr.Slider(1, 5, value=1, step=1, label="PPO Epochs")
                            ppo_max_new_tokens = gr.Slider(32, 512, value=128, step=16,
                                                            label="Max New Tokens (per response)")
                        ppo_train_btn = gr.Button("🔁 Run PPO Fine-Tuning", variant="primary")
                    with gr.Column():
                        ppo_status = gr.Textbox(
                            label="PPO Training Status", lines=12, interactive=False
                        )

            # ── C. ORPO ──────────────────────────────────────────────────
            with gr.Tab("🌀 C. ORPO / ARPO"):
                gr.Markdown(
                    "Train with **ORPO** (Odds Ratio Preference Optimization) — "
                    "a modern, reference-free DPO alternative.\n"
                    "Dataset needs `prompt`, `chosen`, `rejected` columns."
                )
                with gr.Row():
                    with gr.Column():
                        # L-2 FIX: Use Dropdown to match Training tab UX.
                        orpo_model_choice = gr.Dropdown(
                            choices=[
                                "gpt2", "distilgpt2", "facebook/opt-125m",
                                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                "mistralai/Mistral-7B-v0.1",
                            ],
                            value=recommended_model,
                            label="Base Model ID",
                            allow_custom_value=True,
                        )
                        orpo_file = gr.File(
                            label="Preference Dataset (prompt, chosen, rejected)",
                            file_types=[".csv", ".jsonl"],
                        )
                        orpo_output_dir = gr.Textbox(label="Output Directory",
                                                      value="./orpo_model")
                        with gr.Row():
                            orpo_lr    = gr.Number(value=1e-4, label="Learning Rate", precision=8)
                            orpo_beta  = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Beta")
                            orpo_alpha = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Alpha")
                        with gr.Row():
                            orpo_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                            orpo_batch  = gr.Slider(1, 16, value=2, step=1, label="Batch Size")
                        orpo_train_btn = gr.Button("🌀 Run ORPO Training", variant="primary")
                    with gr.Column():
                        orpo_status = gr.Textbox(
                            label="ORPO Training Status", lines=12, interactive=False
                        )

    return dict(
        rm_model_choice=rm_model_choice, rm_file=rm_file, rm_output_dir=rm_output_dir,
        rm_epochs=rm_epochs, rm_lr=rm_lr, rm_batch=rm_batch,
        rm_eval_steps=rm_eval_steps, rm_max_length=rm_max_length,
        rm_train_btn=rm_train_btn, rm_status=rm_status,
        ppo_policy_model=ppo_policy_model, ppo_reward_path=ppo_reward_path,
        ppo_file=ppo_file, ppo_output_dir=ppo_output_dir,
        ppo_lr=ppo_lr, ppo_batch=ppo_batch, ppo_mini_batch=ppo_mini_batch,
        ppo_epochs=ppo_epochs, ppo_max_new_tokens=ppo_max_new_tokens,
        ppo_train_btn=ppo_train_btn, ppo_status=ppo_status,
        orpo_model_choice=orpo_model_choice, orpo_file=orpo_file,
        orpo_output_dir=orpo_output_dir,
        orpo_lr=orpo_lr, orpo_beta=orpo_beta, orpo_alpha=orpo_alpha,
        orpo_epochs=orpo_epochs, orpo_batch=orpo_batch,
        orpo_train_btn=orpo_train_btn, orpo_status=orpo_status,
    )
