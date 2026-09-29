"""ui/tabs/gguf_tab.py — export tab: GGUF, quantized safetensors, adapter merging."""

import gradio as gr

from config.constants import (
    ADAPTER_MERGE_METHODS,
    DEFAULT_ADAPTER_MERGE_DENSITY,
    DEFAULT_ADAPTER_MERGE_METHOD,
    GGUF_QUANT_PRESETS,
    HAS_LLMCOMPRESSOR,
)


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

        gr.Markdown(
            "### 🧬 Merge LoRA adapters\n"
            "Combine adapters trained on the **same base model** (e.g. one per skill) into one "
            "adapter. TIES / DARE keep each adapter's strongest changes and resolve conflicts."
        )
        with gr.Row():
            with gr.Column():
                merge_adapters = gr.Textbox(
                    label="Adapter folders (one per line)", lines=3, max_length=4096,
                    placeholder="runs/math\nruns/code",
                )  # fmt: skip
                merge_weights = gr.Textbox(
                    label="Weights (optional, comma-separated)", placeholder="1, 1", max_length=256
                )
                merge_method = gr.Dropdown(
                    choices=[(f"{name} — {text}", name) for name, text in ADAPTER_MERGE_METHODS.items()],
                    value=DEFAULT_ADAPTER_MERGE_METHOD, label="Method",
                )  # fmt: skip
                merge_density = gr.Slider(
                    0.05, 1.0, value=DEFAULT_ADAPTER_MERGE_DENSITY, step=0.05, label="Density",
                    info="TIES / DARE: share of each adapter's changes kept",
                )  # fmt: skip
                merge_output = gr.Textbox(
                    label="Output folder", value="./merged_adapter", max_length=512
                )
                merge_btn = gr.Button("🧬 Merge adapters", variant="primary")
            with gr.Column():
                merge_status = gr.Textbox(label="Status", lines=5, interactive=False)

    return dict(
        export_model_path=export_model_path,
        quantization=quantization,
        export_btn=export_btn,
        export_status=export_status,
        gguf_file=gguf_file,
        quant_format=quant_format,
        quant_btn=quant_btn,
        quant_status=quant_status,
        merge_adapters=merge_adapters,
        merge_weights=merge_weights,
        merge_method=merge_method,
        merge_density=merge_density,
        merge_output=merge_output,
        merge_btn=merge_btn,
        merge_status=merge_status,
    )
