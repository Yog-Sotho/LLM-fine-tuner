"""ui/tabs/gguf_tab.py — GGUF export tab."""
import gradio as gr

from config.constants import GGUF_QUANT_PRESETS


def build_gguf_tab() -> dict:
    with gr.Tab("📦 GGUF Export"):
        gr.Markdown("### Export to GGUF for Ollama / LM Studio / llama.cpp")
        with gr.Row():
            with gr.Column():
                export_model_path = gr.Textbox(
                    label="Model Path",
                    placeholder="e.g. ./output (auto-filled after training, or enter custom path)",
                    interactive=True,
                    max_length=512,
                )
                quantization = gr.Dropdown(
                    choices=list(GGUF_QUANT_PRESETS.keys()),
                    value="q6_k",
                    label="Quantization",
                )
                export_btn = gr.Button("🔄 Export to GGUF", variant="primary")
            with gr.Column():
                export_status = gr.Textbox(label="Status", lines=6, interactive=False)
                gguf_file     = gr.File(label="Download GGUF")

    return dict(
        export_model_path=export_model_path,
        quantization=quantization,
        export_btn=export_btn,
        export_status=export_status,
        gguf_file=gguf_file,
    )
