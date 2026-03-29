"""ui/tabs/train_tab.py — Model selection, PEFT, hyperparams, training controls.

Patch log
---------
  Fix-1 (High) : ``loss_df`` gr.Dataframe previously declared
                 ``headers=["Step", "Train Loss", "Eval Loss"]`` and
                 ``datatype=["number", "number", "number"]``.  When
                 ``build_loss_chart()`` returns a 4-column DataFrame that
                 includes the new "ETA" string column, Gradio 5 tries to
                 coerce the 4th column to ``number``, producing a column
                 of NaN values — making the ETA feature invisible to users.
                 Fix: remove both constraints so Gradio infers column names
                 and types directly from the DataFrame at render time.
                 This is the correct Gradio 5 pattern for dynamically-shaped
                 DataFrames and is fully backwards-compatible with the 3-column
                 output produced by old-format log records.
"""
import gradio as gr

from config.constants import HAS_UNSLOTH, HAS_VLLM
from core.hardware import auto_recommend_model, get_model_info


def build_train_tab() -> dict:
    recommended_model = auto_recommend_model()

    with gr.Tab("🚀 Training"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Model selection")
                model_choice = gr.Dropdown(
                    choices=[
                        "gpt2", "distilgpt2",
                        "facebook/opt-125m", "facebook/opt-350m",
                        "EleutherAI/pythia-70m", "EleutherAI/pythia-160m",
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        "mistralai/Mistral-7B-v0.1",
                    ],
                    value=recommended_model,
                    label="Base Model",
                )
                custom_model  = gr.Textbox(
                    label="Or enter any HuggingFace model ID",
                    placeholder="e.g., meta-llama/Llama-2-7b-hf",
                )
                model_info_md = gr.Markdown(get_model_info(recommended_model))

                gr.Markdown("### Training Mode")
                training_mode = gr.Radio(
                    choices=["SFT (Supervised Fine-Tuning)", "DPO (Alignment)"],
                    value="SFT (Supervised Fine-Tuning)",
                    label="",
                )
                dpo_beta = gr.Slider(0.01, 1.0, value=0.1, step=0.01,
                                     label="DPO Beta (used only in DPO mode)")

                with gr.Row():
                    use_unsloth        = gr.Checkbox(label="🚀 Use Unsloth (2-5× faster)",
                                                     value=False, interactive=HAS_UNSLOTH)
                    use_chat_template  = gr.Checkbox(label="💬 Use Smart Chat Template", value=True)
                    heretic_mode       = gr.Checkbox(
                        label="🔓 Heretic Mode (remove restrictions — use responsibly)", value=False
                    )
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are a helpful, respectful and honest assistant.",
                    lines=2,
                )

                with gr.Row():
                    use_flash_attn = gr.Checkbox(
                        label="⚡ Flash Attention 2 (CUDA + bfloat16 required)",
                        value=False,
                        info="Significantly reduces VRAM and speeds up attention computation.",
                    )
                    # Minor Fix 5: QLoRA Enhanced checkbox removed from UI — driven by peft_method radio.
                    # gr.State preserves event-wiring arity.
                    use_qlora_enhanced = gr.State(False)

                gr.Markdown("### Parameter-efficient method")
                gr.Markdown(
                    "_💡 Select **QLoRA Enhanced** in the radio below to enable NF4 quantisation "
                    "(≈70 % VRAM reduction). All other modes use standard precision._"
                )
                peft_method = gr.Radio(
                    choices=["Full Fine-tuning", "Auto", "LoRA", "QLoRA Enhanced",
                             "Prefix Tuning", "Prompt Tuning", "Adapters"],
                    value="Auto",
                    label="",
                )

                gr.Markdown("### Training preset")
                training_preset = gr.Radio(
                    choices=["Quick (1 epoch)", "Balanced (3 epochs)",
                             "Accurate (5 epochs)", "Advanced"],
                    value="Balanced (3 epochs)",
                    label=" ",
                )

                with gr.Accordion("⚙️ Advanced hyperparameters", open=False):
                    with gr.Group():
                        gr.Markdown("### PEFT Method Settings")
                        with gr.Tab("LoRA"):
                            use_lora   = gr.Checkbox(label="Enable LoRA", value=True)
                            lora_rank  = gr.Slider(1, 64, value=8, step=1, label="LoRA Rank")
                            lora_alpha = gr.Slider(1, 128, value=16, step=1, label="LoRA Alpha")
                        with gr.Tab("Prefix Tuning"):
                            prefix_tuning_num_virtual_tokens = gr.Slider(10, 100, value=30, step=5,
                                                                          label="Virtual Tokens")
                            prefix_tuning_token_dim          = gr.Slider(100, 1024, value=512, step=64,
                                                                          label="Token Dimension")
                            prefix_tuning_num_layers         = gr.Slider(1, 32, value=2, step=1,
                                                                          label="Layers")
                        with gr.Tab("Prompt Tuning"):
                            prompt_tuning_num_virtual_tokens = gr.Slider(10, 100, value=20, step=5,
                                                                          label="Virtual Tokens")
                            # Minor Fix 2: PromptTuningConfig has no num_layers — use gr.State.
                            prompt_tuning_num_layers = gr.State(None)
                        with gr.Tab("Adapters"):
                            adapter_reduction_factor = gr.Slider(2, 64, value=16, step=2,
                                                                   label="Reduction Factor")
                    lr         = gr.Number(value=2e-4, label="Learning Rate", precision=6)
                    epochs     = gr.Slider(1, 20, value=3, step=1, label="Epochs")
                    bs         = gr.Slider(1, 16, value=2, step=1, label="Batch Size")
                    grad_accum = gr.Slider(1, 16, value=4, step=1, label="Gradient Accumulation Steps")
                    max_len    = gr.Slider(64, 2048, value=256, step=64, label="Max Sequence Length")
                    warmup     = gr.Slider(0, 500, value=100, step=10, label="Warmup Steps")
                    early_stop = gr.Slider(0, 10, value=3, step=1,
                                           label="Early Stopping Patience (0 = off)")
                    lr_sched   = gr.Dropdown(
                        choices=["linear", "cosine", "cosine_with_restarts", "constant"],
                        value="cosine", label="LR Scheduler",
                    )
                    grad_ckpt  = gr.Checkbox(
                        label="Gradient Checkpointing (saves VRAM, ~20% slower)", value=False
                    )
                    resume_ckpt = gr.Checkbox(label="Resume from last checkpoint", value=False)

                with gr.Row():
                    train_btn = gr.Button("▶  Start Training", variant="primary", scale=3)
                    stop_btn  = gr.Button("⏹  Stop", variant="stop", scale=1)

            with gr.Column(scale=3):
                gr.Markdown("### Training log")
                log_output = gr.Textbox(
                    label=" ", lines=14, interactive=False,
                    placeholder="Training output will appear here…",
                )
                with gr.Column(elem_id="loss-chart-wrap"):
                    gr.Markdown("### 📉 Loss Curve")
                    # Fix-1: headers and datatype constraints removed.
                    # build_loss_chart() returns either 3 columns (Step, Train Loss,
                    # Eval Loss) or 4 columns (+ ETA string) depending on whether
                    # timing data is present.  Declaring a fixed 3-column schema with
                    # datatype=["number","number","number"] caused Gradio 5 to coerce
                    # the ETA string column to NaN, making it invisible.
                    # Gradio infers the correct schema from the DataFrame at render time.
                    loss_df = gr.Dataframe(
                        label=" ", interactive=False,
                    )
                clear_gpu_btn = gr.Button("🧹 Clear GPU Cache", variant="secondary")

        model_path_state    = gr.State()
        log_records_state   = gr.State([])

    return dict(
        model_choice=model_choice, custom_model=custom_model, model_info_md=model_info_md,
        training_mode=training_mode, dpo_beta=dpo_beta,
        use_unsloth=use_unsloth, use_chat_template=use_chat_template,
        heretic_mode=heretic_mode, system_prompt=system_prompt,
        use_flash_attn=use_flash_attn, use_qlora_enhanced=use_qlora_enhanced,
        peft_method=peft_method, training_preset=training_preset,
        use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha,
        prefix_tuning_num_virtual_tokens=prefix_tuning_num_virtual_tokens,
        prefix_tuning_token_dim=prefix_tuning_token_dim,
        prefix_tuning_num_layers=prefix_tuning_num_layers,
        prompt_tuning_num_virtual_tokens=prompt_tuning_num_virtual_tokens,
        prompt_tuning_num_layers=prompt_tuning_num_layers,
        adapter_reduction_factor=adapter_reduction_factor,
        lr=lr, epochs=epochs, bs=bs, grad_accum=grad_accum,
        max_len=max_len, warmup=warmup, early_stop=early_stop,
        lr_sched=lr_sched, grad_ckpt=grad_ckpt, resume_ckpt=resume_ckpt,
        train_btn=train_btn, stop_btn=stop_btn,
        log_output=log_output, loss_df=loss_df, clear_gpu_btn=clear_gpu_btn,
        model_path_state=model_path_state, log_records_state=log_records_state,
    )
