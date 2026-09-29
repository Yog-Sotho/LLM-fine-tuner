"""ui/tabs/rlhf_tab.py — Reward model, GRPO, ORPO, KTO and distillation sub-tabs (layout only)."""

import gradio as gr

from config.constants import (
    DEFAULT_GRPO_LOSS_TYPE,
    DEFAULT_GRPO_REWARDS,
    DEFAULT_LORA_VARIANT,
    DISTILL_BETA,
    DISTILL_LMBDA,
    DISTILL_MAX_NEW_TOKENS,
    DISTILL_TEMPERATURE,
    GRPO_LORA_ALPHA,
    GRPO_LORA_RANK,
    GRPO_LOSS_TYPES,
    GRPO_REWARDS,
    HAS_GKD,
    HAS_GRPO,
    HAS_KTO,
    HAS_MATH_VERIFY,
    HAS_ORPO,
    HAS_REWARD_TRAINER,
    HAS_VLLM,
    LORA_VARIANTS,
)
from core.hardware import auto_recommend_model


def _ok(flag: bool) -> str:
    return "✅" if flag else "❌"


def build_rlhf_tab() -> dict:
    recommended_model = auto_recommend_model()

    with gr.Tab("🤖 RLHF Pipeline"):
        gr.HTML(
            '<div id="rlhf-banner">'
            '<h3 style="color:#34d399;margin:0">'
            "🤖 Alignment — Reward Model · GRPO · ORPO · KTO · Distillation"
            "</h3></div>"
        )
        gr.Markdown(
            f"Reward model {_ok(HAS_REWARD_TRAINER)} | GRPO {_ok(HAS_GRPO)} | "
            f"ORPO {_ok(HAS_ORPO)} | KTO {_ok(HAS_KTO)} | Distillation {_ok(HAS_GKD)}\n"
            '_All five use TRL. If one shows ❌, install: `pip install "trl>=0.29.1,<2"`._'
        )

        with gr.Tabs():
            # ── A. Reward Model ──────────────────────────────────────────
            with gr.Tab("🎖️ A. Reward Model"):
                gr.Markdown(
                    "Train a **reward model** that scores a response *for its prompt*.\n"
                    "Dataset needs `prompt`, `chosen` and `rejected` columns. The saved "
                    "model can be used directly as the GRPO reward in step B."
                )
                with gr.Row():
                    with gr.Column():
                        rm_model_choice = gr.Textbox(
                            label="Base Model ID",
                            value=recommended_model,
                            placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct",
                            max_length=512,
                        )
                        rm_file = gr.File(
                            label="Preference Dataset (CSV/JSONL: prompt, chosen, rejected)",
                            file_types=[".csv", ".jsonl"],
                        )
                        rm_output_dir = gr.Textbox(
                            label="Output Directory", value="./reward_model", max_length=512
                        )
                        with gr.Row():
                            rm_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                            rm_lr = gr.Number(value=1e-4, label="Learning Rate", precision=8)
                            rm_batch = gr.Slider(1, 16, value=4, step=1, label="Batch Size")
                        with gr.Row():
                            rm_eval_steps = gr.Slider(
                                10, 500, value=100, step=10, label="Eval Steps"
                            )
                            rm_max_length = gr.Slider(
                                128, 4096, value=1024, step=128, label="Max Length"
                            )
                        rm_train_btn = gr.Button("🎖️ Train Reward Model", variant="primary")
                    with gr.Column():
                        rm_status = gr.Textbox(
                            label="Reward Model Training Status", lines=12, interactive=False
                        )

            # ── B. GRPO ──────────────────────────────────────────────────
            with gr.Tab("🎯 B. GRPO"):
                gr.Markdown(
                    "**GRPO** samples several answers per prompt and reinforces the ones that "
                    "score above their group's average — no value model needed.\n"
                    "Dataset needs a `prompt` column. Rewards come from the reward model in "
                    "step A and/or the built-in rewards below; *reference answer* and *maths* "
                    "need a `reference` column with the expected answer."
                )
                with gr.Row():
                    with gr.Column():
                        grpo_policy_model = gr.Textbox(
                            label="Policy Model ID", value=recommended_model, max_length=512
                        )
                        grpo_reward_path = gr.Textbox(
                            label="Reward Model Path (from step A, optional)",
                            placeholder="./reward_model",
                            max_length=512,
                        )
                        grpo_file = gr.File(
                            label="Prompts Dataset (CSV/JSONL: prompt [, reference])",
                            file_types=[".csv", ".jsonl"],
                        )
                        grpo_output_dir = gr.Textbox(
                            label="Output Directory", value="./grpo_model", max_length=512
                        )
                        with gr.Row():
                            grpo_lr = gr.Number(value=1e-5, label="Learning Rate", precision=8)
                            grpo_epochs = gr.Slider(1, 5, value=1, step=1, label="Epochs")
                            grpo_beta = gr.Slider(
                                0.0, 0.5, value=0.0, step=0.01, label="KL Beta (0 = off)"
                            )
                        with gr.Row():
                            grpo_num_generations = gr.Slider(
                                2, 16, value=4, step=1, label="Completions per Prompt"
                            )
                            grpo_prompts_per_step = gr.Slider(
                                1, 8, value=1, step=1, label="Prompts per Step"
                            )
                            grpo_max_completion = gr.Slider(
                                16, 1024, value=128, step=16, label="Max Completion Tokens"
                            )
                        grpo_rewards = gr.CheckboxGroup(
                            choices=[(label, key) for key, label in GRPO_REWARDS.items()],
                            value=DEFAULT_GRPO_REWARDS,
                            label="Built-in rewards (added together)",
                            info=None
                            if HAS_MATH_VERIFY
                            else "Maths needs: pip install math-verify",
                        )
                        grpo_regex = gr.Textbox(
                            label="Regex (for the regex reward)",
                            placeholder=r"e.g. Answer: \d+",
                            max_length=512,
                        )
                        with gr.Row():
                            grpo_loss_type = gr.Dropdown(
                                GRPO_LOSS_TYPES,
                                value=DEFAULT_GRPO_LOSS_TYPE,
                                label="Loss",
                                info="dapo / dr_grpo avoid favouring short answers.",
                            )
                            grpo_lora_variant = gr.Radio(
                                list(LORA_VARIANTS),
                                value=DEFAULT_LORA_VARIANT,
                                label="LoRA variant",
                            )
                        with gr.Row():
                            grpo_lora_rank = gr.Slider(
                                4, 128, value=GRPO_LORA_RANK, step=4, label="LoRA Rank"
                            )
                            grpo_lora_alpha = gr.Slider(
                                4, 256, value=GRPO_LORA_ALPHA, step=4, label="LoRA Alpha"
                            )
                        grpo_use_vllm = gr.Checkbox(
                            label="Generate with vLLM (CUDA, faster)",
                            value=False,
                            interactive=HAS_VLLM,
                            info="Shares the training GPU."
                            if HAS_VLLM
                            else 'Not installed: pip install "trl[vllm]" (CUDA GPU needed).',
                        )
                        grpo_resume = gr.Checkbox(
                            label="Resume from last checkpoint",
                            value=False,
                            info="Continue from the newest checkpoint in the output directory.",
                        )
                        grpo_train_btn = gr.Button("🎯 Run GRPO", variant="primary")
                    with gr.Column():
                        grpo_status = gr.Textbox(
                            label="GRPO Training Status", lines=12, interactive=False
                        )

            # ── C. ORPO ──────────────────────────────────────────────────
            with gr.Tab("🌀 C. ORPO"):
                gr.Markdown(
                    "Train with **ORPO** (Odds Ratio Preference Optimization) — "
                    "a reference-free DPO alternative.\n"
                    "Dataset needs `prompt`, `chosen`, `rejected` columns."
                )
                with gr.Row():
                    with gr.Column():
                        orpo_model_choice = gr.Textbox(
                            label="Base Model ID", value=recommended_model, max_length=512
                        )
                        orpo_file = gr.File(
                            label="Preference Dataset (prompt, chosen, rejected)",
                            file_types=[".csv", ".jsonl"],
                        )
                        orpo_output_dir = gr.Textbox(
                            label="Output Directory", value="./orpo_model", max_length=512
                        )
                        with gr.Row():
                            orpo_lr = gr.Number(value=1e-4, label="Learning Rate", precision=8)
                            orpo_beta = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Beta")
                            orpo_alpha = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Alpha")
                        with gr.Row():
                            orpo_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                            orpo_batch = gr.Slider(1, 16, value=2, step=1, label="Batch Size")
                        orpo_train_btn = gr.Button("🌀 Run ORPO Training", variant="primary")
                    with gr.Column():
                        orpo_status = gr.Textbox(
                            label="ORPO Training Status", lines=12, interactive=False
                        )

            # ── D. KTO ───────────────────────────────────────────────────
            with gr.Tab("👍 D. KTO"):
                gr.Markdown(
                    "**KTO** learns from single responses marked good or bad "
                    "(thumbs-up / thumbs-down) — no ranked pairs needed.\n"
                    "Dataset: `prompt`, `completion`, `label` (true/false), or "
                    "`prompt`, `chosen`, `rejected` pairs."
                )
                with gr.Row():
                    with gr.Column():
                        kto_model_choice = gr.Textbox(
                            label="Base Model ID", value=recommended_model, max_length=512
                        )
                        kto_file = gr.File(
                            label="Feedback Dataset (CSV/JSONL)",
                            file_types=[".csv", ".jsonl"],
                        )
                        kto_output_dir = gr.Textbox(
                            label="Output Directory", value="./kto_model", max_length=512
                        )
                        with gr.Row():
                            kto_lr = gr.Number(value=5e-5, label="Learning Rate", precision=8)
                            kto_beta = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Beta")
                            kto_epochs = gr.Slider(1, 10, value=1, step=1, label="Epochs")
                        with gr.Row():
                            kto_batch = gr.Slider(2, 16, value=4, step=1, label="Batch Size")
                            kto_max_length = gr.Slider(
                                64, 4096, value=512, step=64, label="Max Length"
                            )
                        kto_resume = gr.Checkbox(
                            label="Resume from last checkpoint",
                            value=False,
                            info="Continue from the newest checkpoint in the output directory.",
                        )
                        kto_train_btn = gr.Button("👍 Run KTO Training", variant="primary")
                    with gr.Column():
                        kto_status = gr.Textbox(
                            label="KTO Training Status", lines=12, interactive=False
                        )

            # ── E. Distillation ──────────────────────────────────────────
            with gr.Tab("🎓 E. Distillation"):
                gr.Markdown(
                    "**Knowledge distillation (GKD)**: a small *student* learns the next-token "
                    "distribution of a larger *teacher*, partly on its own answers, so it "
                    "learns from its own mistakes. Both must be from the same model family "
                    "(shared vocabulary), e.g. Qwen3-0.6B ← Qwen3-8B. The student is trained "
                    "as a LoRA adapter.\n"
                    "Dataset: chats (`messages`), `instruction`/`output` or "
                    "`prompt`/`completion` — the last answer is what the teacher grades."
                )
                with gr.Row():
                    with gr.Column():
                        distill_student = gr.Textbox(
                            label="Student model (small)", value=recommended_model, max_length=512
                        )
                        distill_teacher = gr.Textbox(
                            label="Teacher model (larger, same family)", max_length=512
                        )
                        distill_file = gr.File(
                            label="Dataset (CSV/JSONL)", file_types=[".csv", ".jsonl", ".json"]
                        )
                        distill_output_dir = gr.Textbox(
                            label="Output Directory", value="./distilled_model", max_length=512
                        )
                        with gr.Row():
                            distill_lr = gr.Number(value=5e-5, label="Learning Rate", precision=8)
                            distill_epochs = gr.Slider(1, 10, value=1, step=1, label="Epochs")
                            distill_batch = gr.Slider(1, 16, value=2, step=1, label="Batch Size")
                        with gr.Row():
                            distill_max_length = gr.Slider(
                                64, 4096, value=512, step=64, label="Max Length"
                            )
                            distill_max_new_tokens = gr.Slider(
                                8, 1024, value=DISTILL_MAX_NEW_TOKENS, step=8,
                                label="Student answer length",
                            )  # fmt: skip
                        with gr.Row():
                            distill_lmbda = gr.Slider(
                                0.0, 1.0, value=DISTILL_LMBDA, step=0.05, label="On-policy share",
                                info="0 = learn on the dataset answers, 1 = only on its own",
                            )  # fmt: skip
                            distill_beta = gr.Slider(
                                0.0, 1.0, value=DISTILL_BETA, step=0.05, label="Divergence (β)",
                                info="0 ≈ forward KL (cover the teacher), 1 ≈ reverse KL (focus)",
                            )  # fmt: skip
                            distill_temperature = gr.Slider(
                                0.1, 2.0, value=DISTILL_TEMPERATURE, step=0.05, label="Temperature"
                            )
                        distill_resume = gr.Checkbox(
                            label="Resume from last checkpoint",
                            value=False,
                            info="Continue from the newest checkpoint in the output directory.",
                        )
                        distill_train_btn = gr.Button("🎓 Run Distillation", variant="primary")
                    with gr.Column():
                        distill_status = gr.Textbox(
                            label="Distillation Status", lines=12, interactive=False
                        )

    return dict(
        rm_model_choice=rm_model_choice,
        rm_file=rm_file,
        rm_output_dir=rm_output_dir,
        rm_epochs=rm_epochs,
        rm_lr=rm_lr,
        rm_batch=rm_batch,
        rm_eval_steps=rm_eval_steps,
        rm_max_length=rm_max_length,
        rm_train_btn=rm_train_btn,
        rm_status=rm_status,
        grpo_policy_model=grpo_policy_model,
        grpo_reward_path=grpo_reward_path,
        grpo_file=grpo_file,
        grpo_output_dir=grpo_output_dir,
        grpo_lr=grpo_lr,
        grpo_epochs=grpo_epochs,
        grpo_beta=grpo_beta,
        grpo_num_generations=grpo_num_generations,
        grpo_prompts_per_step=grpo_prompts_per_step,
        grpo_max_completion=grpo_max_completion,
        grpo_resume=grpo_resume,
        grpo_loss_type=grpo_loss_type,
        grpo_rewards=grpo_rewards,
        grpo_regex=grpo_regex,
        grpo_lora_rank=grpo_lora_rank,
        grpo_lora_alpha=grpo_lora_alpha,
        grpo_lora_variant=grpo_lora_variant,
        grpo_use_vllm=grpo_use_vllm,
        grpo_train_btn=grpo_train_btn,
        grpo_status=grpo_status,
        orpo_model_choice=orpo_model_choice,
        orpo_file=orpo_file,
        orpo_output_dir=orpo_output_dir,
        orpo_lr=orpo_lr,
        orpo_beta=orpo_beta,
        orpo_alpha=orpo_alpha,
        orpo_epochs=orpo_epochs,
        orpo_batch=orpo_batch,
        orpo_train_btn=orpo_train_btn,
        orpo_status=orpo_status,
        kto_model_choice=kto_model_choice,
        kto_file=kto_file,
        kto_output_dir=kto_output_dir,
        kto_lr=kto_lr,
        kto_beta=kto_beta,
        kto_epochs=kto_epochs,
        kto_batch=kto_batch,
        kto_max_length=kto_max_length,
        kto_resume=kto_resume,
        kto_train_btn=kto_train_btn,
        kto_status=kto_status,
        distill_student=distill_student,
        distill_teacher=distill_teacher,
        distill_file=distill_file,
        distill_output_dir=distill_output_dir,
        distill_lr=distill_lr,
        distill_epochs=distill_epochs,
        distill_batch=distill_batch,
        distill_max_length=distill_max_length,
        distill_lmbda=distill_lmbda,
        distill_beta=distill_beta,
        distill_temperature=distill_temperature,
        distill_max_new_tokens=distill_max_new_tokens,
        distill_resume=distill_resume,
        distill_train_btn=distill_train_btn,
        distill_status=distill_status,
    )
