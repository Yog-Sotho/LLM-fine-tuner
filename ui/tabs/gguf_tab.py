"""ui/tabs/gguf_tab.py — GGUF export tab."""

import gradio as gr

from config.constants import GGUF_QUANT_PRESETS, HAS_LLMCOMPRESSOR


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
                gguf_file = gr.File(label="Download GGUF")

        gr.Markdown(
            "### Quantized safetensors for vLLM (llm-compressor)\n"
            "**FP8** — no calibration needed; runs natively on Hopper/Ada GPUs. "
            "**W4A16** — 4-bit GPTQ, calibrated on the data loaded in 📂 Data. "
            "Saved next to the model as `<model>-fp8` / `<model>-w4a16`; serve with "
            "`python main.py serve --model <folder>`."
            + ("" if HAS_LLMCOMPRESSOR else
               '\n\n❌ Not installed: `pip install "llm-fine-tuner[compress]"`')
        )  # fmt: skip
        with gr.Row():
            with gr.Column():
                quant_format = gr.Radio(["fp8", "w4a16"], value="fp8", label="Format")
                quant_btn = gr.Button("🗜️ Export quantized", variant="primary",
                                      interactive=HAS_LLMCOMPRESSOR)  # fmt: skip
            with gr.Column():
                quant_status = gr.Textbox(label="Status", lines=5, interactive=False)

    return dict(
        export_model_path=export_model_path,
        quantization=quantization,
        export_btn=export_btn,
        export_status=export_status,
        gguf_file=gguf_file,
        quant_format=quant_format,
        quant_btn=quant_btn,
        quant_status=quant_status,
    )
