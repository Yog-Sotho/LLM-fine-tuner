"""ui/tabs/embedding_tab.py — embedding-model fine-tuning for search / RAG (layout only)."""

import gradio as gr

from config.constants import (
    DEFAULT_EVAL_SPLIT,
    EMBED_MAX_SEQ_LENGTH,
    EMBED_METHODS,
    EMBED_MODEL_SUGGESTIONS,
    HAS_SENTENCE_TRANSFORMERS,
)


def build_embedding_tab() -> dict:
    with gr.Tab("🔎 Embeddings"):
        gr.Markdown(
            "### Fine-tune an embedding model for search and RAG\n"
            "Teaches a retrieval model which passages answer which questions in **your** "
            "domain. Data: CSV / JSON / JSONL with `anchor` + `positive` (optional "
            "`negative`), the JSONL from **📂 Data → create training data from your documents** "
            "(question ↔ source passage), `instruction` + `output`, `prompt` + `completion`, or "
            "chats. Retrieval quality is measured on held-out pairs before and after training."
            + ("" if HAS_SENTENCE_TRANSFORMERS else
               '\n\n❌ Not installed: `pip install "llm-fine-tuner[embedding]"`')
        )  # fmt: skip
        with gr.Row():
            with gr.Column():
                embed_model = gr.Dropdown(
                    choices=list(EMBED_MODEL_SUGGESTIONS),
                    value=EMBED_MODEL_SUGGESTIONS[0],
                    allow_custom_value=True,
                    label="Embedding model (Hub id or local folder)",
                )
                embed_file = gr.File(
                    label="Training pairs (CSV / JSON / JSONL)",
                    file_types=[".csv", ".json", ".jsonl"],
                )
                embed_run_name = gr.Textbox(
                    label="Run name (saved under the runs folder; empty = automatic)",
                    max_length=128,
                )
                embed_method = gr.Radio(
                    list(EMBED_METHODS), value=EMBED_METHODS[0], label="Method",
                    info="LoRA trains fewer weights (large models); it is merged on save.",
                )  # fmt: skip
                with gr.Row():
                    embed_lr = gr.Number(value=2e-5, label="Learning Rate", precision=8)
                    embed_epochs = gr.Slider(1, 10, value=1, step=1, label="Epochs")
                with gr.Row():
                    embed_batch = gr.Slider(
                        2, 256, value=32, step=1, label="Batch Size",
                        info="Bigger is better: the other passages in a batch are the negatives.",
                    )  # fmt: skip
                    embed_max_len = gr.Slider(
                        32, 2048, value=EMBED_MAX_SEQ_LENGTH, step=32, label="Max tokens"
                    )
                with gr.Row():
                    embed_matryoshka = gr.Checkbox(
                        value=True, label="Matryoshka (truncatable embeddings)"
                    )
                    embed_hard_negatives = gr.Checkbox(value=False, label="Mine hard negatives")
                embed_eval_split = gr.Slider(
                    0.0, 0.5, value=DEFAULT_EVAL_SPLIT, step=0.05, label="Held-out share"
                )
                embed_query_prompt = gr.Textbox(
                    label="Query prompt (optional, e.g. 'query: ' for E5; empty = the model's own)",
                    max_length=256,
                )
                with gr.Row():
                    embed_train_btn = gr.Button(
                        "🔎 Train embedding model", variant="primary",
                        interactive=HAS_SENTENCE_TRANSFORMERS,
                    )  # fmt: skip
                    embed_stop_btn = gr.Button("⏹  Stop", variant="stop")
            with gr.Column():
                embed_status = gr.Textbox(label="Status", lines=16, interactive=False)

    return dict(
        embed_model=embed_model,
        embed_file=embed_file,
        embed_run_name=embed_run_name,
        embed_method=embed_method,
        embed_lr=embed_lr,
        embed_epochs=embed_epochs,
        embed_batch=embed_batch,
        embed_max_len=embed_max_len,
        embed_matryoshka=embed_matryoshka,
        embed_hard_negatives=embed_hard_negatives,
        embed_eval_split=embed_eval_split,
        embed_query_prompt=embed_query_prompt,
        embed_train_btn=embed_train_btn,
        embed_stop_btn=embed_stop_btn,
        embed_status=embed_status,
    )
