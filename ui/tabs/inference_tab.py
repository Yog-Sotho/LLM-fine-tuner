"""ui/tabs/inference_tab.py — Standard inference, batch, PEFT zip restore, vLLM."""
import gradio as gr

from config.constants import HAS_VLLM, VLLM_QUANT_OPTIONS


def build_inference_tab() -> dict:
    with gr.Tab("💬 Inference"):
        gr.Markdown("### Test your fine-tuned model")
        with gr.Row():
            with gr.Column():
                infer_model = gr.Dropdown(
                    choices=[
                        "gpt2", "distilgpt2", "facebook/opt-125m",
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        "mistralai/Mistral-7B-v0.1",
                    ],
                    value="gpt2", label="Model",
                )
                infer_custom = gr.Textbox(label="Or custom model ID",
                                          placeholder="username/my-model",
                                          max_length=512)
                lora_path    = gr.Textbox(
                    label="PEFT adapter path (auto-filled after training)",
                    interactive=True,
                    placeholder="./output or leave blank to use base model only",
                    info="You can manually enter or paste a custom PEFT adapter path here.",
                )
                prompt_in    = gr.Textbox(
                    label="Prompt", lines=4,
                    placeholder="Enter your prompt here…",
                    max_length=4000,
                )
                with gr.Row():
                    max_tok = gr.Slider(10, 500, value=200, step=10, label="Max new tokens")
                    temp    = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                    top_p   = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                gen_btn = gr.Button("Generate ✨", variant="primary")
            with gr.Column():
                gen_out = gr.Textbox(label="Model response", lines=12, interactive=False)

        gr.Markdown("### Batch inference")
        with gr.Row():
            batch_file = gr.File(label="Upload prompts (CSV with 'prompt' col, or .txt one per line)")
            batch_btn  = gr.Button("Run Batch", variant="secondary")
            batch_out  = gr.File(label="Download responses CSV")

        gr.Markdown("### Load a saved PEFT adapter")
        with gr.Row():
            lora_zip_upload    = gr.File(
                label="Upload PEFT ZIP (previously downloaded model)",
                file_types=[".zip"],
            )
            lora_zip_status    = gr.Markdown("_Upload a ZIP to restore a fine-tuned adapter._")
            lora_zip_dir_state = gr.State(" ")

        gr.Markdown("---")
        gr.Markdown("### ⚡ v2.7 vLLM Production Inference")
        gr.Markdown(
            f"_vLLM available: {'✅ (cached)' if HAS_VLLM else '❌ (pip install vllm>=0.2.0)'}_"
        )
        gr.Markdown(
            "⚠️ **v2.9 Note:** vLLM requires a **merged full model**, not a PEFT adapter "
            "directory. Use the Merge Adapter tool below before running vLLM inference."
        )
        gr.Markdown("#### 🔗 Step 1 — Merge LoRA Adapter (required before vLLM)")
        with gr.Row():
            with gr.Column():
                merge_base_model_in   = gr.Textbox(
                    label="Base Model ID (used during training)",
                    placeholder="e.g. mistralai/Mistral-7B-v0.1 or gpt2",
                    max_length=512,
                )
                merge_adapter_path_in = gr.Textbox(
                    label="Adapter / Model Path (auto-filled after training)",
                    interactive=True,
                    placeholder="./output or leave blank to use last trained model path",
                    max_length=512,
                )
                merge_btn = gr.Button("🔗 Merge Adapter into Full Model", variant="secondary")
            with gr.Column():
                merge_status_out = gr.Textbox(label="Merge Status", lines=5, interactive=False)

        merged_model_path_state = gr.State("")

        gr.Markdown("#### ⚡ Step 2 — vLLM Inference (use merged model path above)")
        with gr.Row():
            with gr.Column():
                vllm_prompt_in  = gr.Textbox(
                    label="vLLM Prompt", lines=4,
                    placeholder="Enter prompt for high-throughput vLLM inference…",
                    max_length=4000,
                )
                with gr.Row():
                    vllm_quant_select = gr.Dropdown(
                        choices=VLLM_QUANT_OPTIONS, value="none", label="vLLM Quantization"
                    )
                    vllm_max_tokens   = gr.Slider(64, 2048, value=512, step=64, label="Max Tokens")
                with gr.Row():
                    vllm_temp_sl  = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                    vllm_top_p_sl = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                vllm_gen_btn = gr.Button(
                    "⚡ Generate with vLLM (merged model)",
                    variant="primary",
                    interactive=HAS_VLLM,
                )
            with gr.Column():
                vllm_gen_out = gr.Textbox(label="vLLM Response", lines=10, interactive=False)

    return dict(
        infer_model=infer_model, infer_custom=infer_custom,
        lora_path=lora_path, prompt_in=prompt_in,
        max_tok=max_tok, temp=temp, top_p=top_p, gen_btn=gen_btn, gen_out=gen_out,
        batch_file=batch_file, batch_btn=batch_btn, batch_out=batch_out,
        lora_zip_upload=lora_zip_upload, lora_zip_status=lora_zip_status,
        lora_zip_dir_state=lora_zip_dir_state,
        merge_base_model_in=merge_base_model_in,
        merge_adapter_path_in=merge_adapter_path_in,
        merge_btn=merge_btn, merge_status_out=merge_status_out,
        merged_model_path_state=merged_model_path_state,
        vllm_prompt_in=vllm_prompt_in, vllm_quant_select=vllm_quant_select,
        vllm_max_tokens=vllm_max_tokens, vllm_temp_sl=vllm_temp_sl,
        vllm_top_p_sl=vllm_top_p_sl, vllm_gen_btn=vllm_gen_btn, vllm_gen_out=vllm_gen_out,
    )
