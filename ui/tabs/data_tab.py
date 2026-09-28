"""ui/tabs/data_tab.py — Data upload, preview, augmentation, quality filter tab."""

import gradio as gr

from config.constants import HUB_DEFAULT_MAX_ROWS, HUB_MAX_ROWS_LIMIT


def build_data_tab() -> dict:
    with gr.Tab("📂 Data"):
        gr.Markdown("### Upload your training data")
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="Upload File",
                    file_types=[".csv", ".jsonl", ".json", ".txt", ".xlsx", ".pdf"],
                )
                file_status = gr.Markdown("_No file loaded yet._")
                with gr.Accordion("…or load from the Hugging Face Hub", open=False):
                    hub_dataset_id = gr.Textbox(
                        label="Dataset ID",
                        placeholder="owner/name, e.g. trl-lib/Capybara",
                        max_length=200,
                    )
                    with gr.Row():
                        hub_config = gr.Textbox(label="Config (optional)", max_length=64)
                        hub_split = gr.Textbox(label="Split", value="train", max_length=64)
                        hub_max_rows = gr.Number(
                            label="Max rows",
                            value=HUB_DEFAULT_MAX_ROWS,
                            precision=0,
                            minimum=1,
                            maximum=HUB_MAX_ROWS_LIMIT,
                        )
                    hub_load_btn = gr.Button("⬇️ Load from Hub", variant="secondary")
                    gr.Markdown(
                        "_Layouts: `messages` (chat), `text`, `instruction`+`output`, or "
                        "`prompt`+`chosen`+`rejected` for DPO. Rows are streamed, so only "
                        "what you load is downloaded. Private datasets need `HF_TOKEN`._"
                    )
            with gr.Column(scale=3):
                with gr.Row():
                    col_inst = gr.Dropdown(
                        label="→ Prompt/Instruction", visible=False, interactive=True
                    )
                    col_out = gr.Dropdown(label="→ Chosen/Output", visible=False, interactive=True)
                    col_text = gr.Dropdown(label="→ Rejected/Text", visible=False, interactive=True)
                refresh_preview_btn = gr.Button(
                    "🔄 Apply Mapping & Refresh Preview",
                    variant="primary",
                    elem_id="refresh-preview-btn",
                )
                preview_box = gr.DataFrame(
                    label="Dataset Preview (first 10 rows)", interactive=False
                )
                stats_box = gr.Markdown("_Statistics will appear here._")

        raw_df_state = gr.State(None)
        file_type_state = gr.State(None)

        # C-5 FIX: New gr.State that holds the augmented/filtered Dataset object.
        # When the user clicks "Augment" or "Quality Filter", the resulting Dataset
        # is stored here. on_train_click in handlers.py reads this state and uses it
        # instead of re-loading from the raw file, so training actually uses the
        # augmented/filtered data rather than the original.
        # Reset to None whenever a new file is uploaded (see app.py event wiring).
        augmented_ds_state = gr.State(None)

        gr.Markdown("---")
        gr.Markdown("### 🔧 v2.7 Dataset Enhancement")
        # C-5 FIX: Added info banner so non-technical users understand the workflow.
        gr.Markdown(
            "_💡 After augmenting or filtering, click **▶ Start Training** — "
            "the enhanced dataset will be used automatically._"
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📈 Data Augmentation")
                aug_factor = gr.Slider(2, 5, value=2, step=1, label="Augmentation Factor (×)")
                aug_type = gr.Dropdown(
                    choices=["synonym", "random_word", "spelling"],
                    value="synonym",
                    label="Augmentation Type",
                )
                aug_btn = gr.Button("🔀 Augment Dataset", variant="secondary")
                aug_status = gr.Textbox(label="Augmentation Status", lines=4, interactive=False)
            with gr.Column():
                gr.Markdown("#### 🔍 Quality Filtering")
                qf_min_len = gr.Slider(10, 500, value=50, step=10, label="Min Character Length")
                qf_max_len = gr.Slider(
                    256, 8192, value=2048, step=256, label="Max Character Length"
                )
                qf_btn = gr.Button("✅ Apply Quality Filter", variant="secondary")
                qf_status = gr.Textbox(label="Filter Status", lines=4, interactive=False)

        aug_preview = gr.DataFrame(
            label="Preview after Enhancement", interactive=False, visible=False
        )
        aug_stats = gr.Markdown(visible=False)

    return dict(
        file_input=file_input,
        file_status=file_status,
        hub_dataset_id=hub_dataset_id,
        hub_config=hub_config,
        hub_split=hub_split,
        hub_max_rows=hub_max_rows,
        hub_load_btn=hub_load_btn,
        col_inst=col_inst,
        col_out=col_out,
        col_text=col_text,
        refresh_preview_btn=refresh_preview_btn,
        preview_box=preview_box,
        stats_box=stats_box,
        raw_df_state=raw_df_state,
        file_type_state=file_type_state,
        # C-5 FIX: New state component exposed to app.py and handlers.py
        augmented_ds_state=augmented_ds_state,
        aug_factor=aug_factor,
        aug_type=aug_type,
        aug_btn=aug_btn,
        aug_status=aug_status,
        qf_min_len=qf_min_len,
        qf_max_len=qf_max_len,
        qf_btn=qf_btn,
        qf_status=qf_status,
        aug_preview=aug_preview,
        aug_stats=aug_stats,
    )
