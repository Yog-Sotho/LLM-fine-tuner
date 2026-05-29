"""ui/tabs/share_tab.py — Download ZIP, push to Hub, versioned registry."""
import gradio as gr


def build_share_tab() -> dict:
    with gr.Tab("📤 Share"):
        gr.Markdown("### Download your model")
        download_btn = gr.File(label="Model ZIP (available after training)", visible=True)

        gr.Markdown("### Push to Hugging Face Hub")
        with gr.Row():
            repo_id    = gr.Textbox(
                label="Repo ID",
                placeholder="username/my-finetuned-model",
                max_length=512,
            )
            hf_token   = gr.Textbox(
                label="HF Token (write access)",
                type="password",
                max_length=512,
            )
            push_btn   = gr.Button("🚀 Push to Hub", variant="primary")
            push_status = gr.Markdown(" ")

        gr.Markdown("---")
        gr.Markdown("### 📊 v2.7 Model Registry & Versioning (Base Model Auto-Filled)")
        gr.Markdown(
            "_Upload versioned model snapshots with metadata to the Hugging Face Hub._"
        )
        with gr.Row():
            with gr.Column():
                registry_repo_id  = gr.Textbox(
                    label="Registry Repo ID",
                    placeholder="username/my-model-registry",
                    max_length=512,
                )
                registry_token    = gr.Textbox(
                    label="HF Token (write access)",
                    type="password",
                    max_length=512,
                )
                registry_version  = gr.Textbox(
                    label="Version Tag",
                    placeholder="e.g. 1.0, 2.0.1, beta-1",
                    max_length=100,
                )
                registry_notes    = gr.Textbox(
                    label="Notes / Changelog",
                    placeholder="What changed in this version?",
                    lines=3,
                    max_length=4096,
                )
                with gr.Row():
                    registry_upload_btn = gr.Button("📤 Upload Versioned Model", variant="primary")
                    registry_list_btn   = gr.Button("📋 List Versions", variant="secondary")
            with gr.Column():
                registry_status = gr.Textbox(
                    label="Registry Status", lines=10, interactive=False
                )

    return dict(
        download_btn=download_btn,
        repo_id=repo_id, hf_token=hf_token,
        push_btn=push_btn, push_status=push_status,
        registry_repo_id=registry_repo_id, registry_token=registry_token,
        registry_version=registry_version, registry_notes=registry_notes,
        registry_upload_btn=registry_upload_btn, registry_list_btn=registry_list_btn,
        registry_status=registry_status,
    )
