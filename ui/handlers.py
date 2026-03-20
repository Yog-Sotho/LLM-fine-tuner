"""
ui/handlers.py
===============
Thin Gradio glue handlers — bridge between UI components and the core layers.
All heavy logic lives in training/, inference/, export/, data/.

Functions
---------
on_train_click        — pre-process inputs, call train_model, zip output
on_stop               — set the global stop event
on_generate           — single-prompt inference handler
on_batch_test         — batch inference handler
on_push               — push model to HF Hub
on_file_upload        — load/validate dataset on file upload
on_refresh_preview    — re-preview dataset after column mapping change
build_loss_chart      — turn log records into a display DataFrame

Fix log
-------
  H6 (High): on_refresh_preview created a NamedTemporaryFile and immediately
     wrote to it by passing its .name path to pandas, while the file handle
     was still open. On Windows this raises PermissionError because
     NamedTemporaryFile holds an exclusive lock. On Linux it works but
     leaks the file handle if an exception fires before unlink. Fixed by
     using the context manager to close the handle first, then writing to
     the path, then cleaning up in a finally block.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import gradio as gr
import torch

from config.constants import (
    COL_INSTRUCTION, COL_OUTPUT, COL_TEXT,
    COL_PROMPT, COL_CHOSEN, COL_REJECTED,
)
from core.state import app_state
from data.loader import detect_file_type, load_dataset_from_file
from data.preprocessing import validate_and_clean_dataset, preview_dataset
from export.hub import push_to_hub
from export.utils import create_zip_from_folder, create_model_card
from inference.generate import generate_text, batch_generate
from training.sft import train_model


# ── Training ───────────────────────────────────────────────────────────────

def on_train_click(
    file, model_choice, custom_model, training_preset, peft_method,
    use_lora, lora_rank, lora_alpha,
    prefix_tuning_num_virtual_tokens, prefix_tuning_token_dim, prefix_tuning_num_layers,
    prompt_tuning_num_virtual_tokens,
    adapter_reduction_factor,
    lr, epochs, bs, grad_accum, max_len, warmup,
    early_stop, lr_sched, grad_ckpt, resume,
    col_inst, col_out, col_text,
    use_unsloth, use_chat_template, system_prompt,
    training_mode, dpo_beta, heretic_mode,
    use_flash_attn=False,
    use_qlora_enhanced=False,   # kept for UI arity — ignored; peft_method drives QLoRA
    progress=gr.Progress(),
):
    """Handler for the Start Training button.

    Orchestrates: file load → validate → preset apply → train → card → zip.
    Returns (log_str, zip_file_path, model_dir_path, log_records).
    """
    app_state.stop_event.clear()
    training_mode = "dpo" if "dpo" in training_mode.lower() else "sft"

    if file is None:
        return "❌ Please upload a data file first.", None, None, []

    model_name = custom_model.strip() if custom_model.strip() else model_choice
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    ftype      = detect_file_type(file)
    is_dpo     = training_mode == "dpo"

    # Build column mapping from UI dropdowns
    col_map = {}
    if is_dpo:
        if col_inst and col_out and col_text:
            col_map[col_inst] = COL_PROMPT
            col_map[col_out]  = COL_CHOSEN
            col_map[col_text] = COL_REJECTED
    else:
        if col_inst and col_out:
            col_map[col_inst] = COL_INSTRUCTION
            col_map[col_out]  = COL_OUTPUT
        elif col_text:
            col_map[col_text] = COL_TEXT

    try:
        ds = load_dataset_from_file(file, ftype, col_map, is_dpo=is_dpo)
    except Exception as e:
        return str(e), None, None, []

    ds, issues = validate_and_clean_dataset(ds, is_dpo=is_dpo)
    if len(ds) == 0:
        return "❌ Dataset is empty after cleaning.", None, None, []
    issues_str = "\n".join(issues) if issues else "✅ No data issues."

    # Apply training presets (override lr / epochs)
    if training_preset == "Quick (1 epoch)":
        epochs, lr = 1, 5e-4
    elif training_preset == "Balanced (3 epochs)":
        epochs, lr = 3, 2e-4
    elif training_preset == "Accurate (5 epochs)":
        epochs, lr = 5, 1e-4

    hyperparams = dict(
        learning_rate=lr, epochs=epochs, batch_size=bs,
        grad_accum=grad_accum, max_length=max_len,
        warmup_steps=warmup, lora_rank=lora_rank,
        lora_alpha=lora_alpha, lr_scheduler=lr_sched,
        prefix_tuning_num_virtual_tokens=prefix_tuning_num_virtual_tokens,
        prefix_tuning_token_dim=prefix_tuning_token_dim,
        prefix_tuning_num_layers=prefix_tuning_num_layers,
        prompt_tuning_num_virtual_tokens=prompt_tuning_num_virtual_tokens,
        adapter_reduction_factor=adapter_reduction_factor,
        dpo_beta=dpo_beta,
    )
    output_dir = tempfile.mkdtemp()

    # Compute dataset stats for model card
    if is_dpo:
        _lengths = [
            len(str(p)) + len(str(c)) + len(str(r))
            for p, c, r in zip(ds[COL_PROMPT], ds[COL_CHOSEN], ds[COL_REJECTED])
        ]
    elif COL_TEXT in ds.column_names:
        _lengths = [len(str(t)) for t in ds[COL_TEXT]]
    else:
        _lengths = [len(str(i)) + len(str(o)) for i, o in zip(ds[COL_INSTRUCTION], ds[COL_OUTPUT])]

    dataset_info = {
        "num_examples": len(ds),
        "avg_length": float(np.mean(_lengths)) if _lengths else 0.0,
    }

    try:
        # CRITICAL FIX #1: use_qlora_enhanced is NOT passed to train_model;
        # peft_method already encodes that choice.
        msg, log_records = train_model(
            model_name, ds, output_dir, hyperparams,
            device, peft_method, use_lora, lora_rank, lora_alpha,
            prefix_tuning_num_virtual_tokens, prefix_tuning_token_dim, prefix_tuning_num_layers,
            prompt_tuning_num_virtual_tokens,
            adapter_reduction_factor,
            resume, early_stop, lr_sched, grad_ckpt,
            use_unsloth, use_chat_template, system_prompt,
            training_mode=training_mode, dpo_beta=dpo_beta, heretic_mode=heretic_mode,
            progress=progress,
            use_flash_attn=use_flash_attn,
        )
        create_model_card(
            model_name, dataset_info, hyperparams,
            output_dir, peft_method,
            training_mode=training_mode, heretic_mode=heretic_mode,
        )
        zip_path  = create_zip_from_folder(output_dir)
        full_msg  = msg + "\n" + issues_str
        return full_msg, zip_path, output_dir, log_records

    except Exception as e:
        return f"❌ Training failed: {e}\n{issues_str}", None, None, []


def on_stop() -> str:
    """Signal the training loop to halt after the current step."""
    app_state.stop_event.set()
    return "🛑 Stop signal sent — will halt after the current step."


# ── Inference ──────────────────────────────────────────────────────────────

def on_generate(prompt, model_choice, custom_model, lora_path, max_tok, temp, top_p) -> str:
    model_name = custom_model.strip() if custom_model.strip() else model_choice
    return generate_text(model_name, lora_path, prompt, int(max_tok), temp, top_p)


def on_batch_test(f, model_choice, custom_model, lora_path) -> str:
    model_name = custom_model.strip() if custom_model.strip() else model_choice
    return batch_generate(model_name, lora_path, f)


# ── Hub ────────────────────────────────────────────────────────────────────

def on_push(model_path: str, repo_id: str, token: str) -> str:
    return push_to_hub(model_path, repo_id, token)


# ── Data tab helpers ───────────────────────────────────────────────────────

def on_file_upload(file, training_mode="sft"):
    """Load, validate, and preview a dataset on file upload.

    Returns 8 values matching the Data tab output list:
    (status_str, col_inst_update, col_out_update, col_text_update,
     preview_df, stats_str, raw_df_state, file_type_state)
    """
    training_mode = "dpo" if "dpo" in training_mode.lower() else "sft"
    is_dpo        = training_mode == "dpo"

    if file is None:
        return (
            "No file uploaded.",
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            pd.DataFrame(), " ", None, None,
        )

    ftype = detect_file_type(file)
    if ftype is None:
        return (
            "⚠️ Unsupported file type.",
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            pd.DataFrame(), " ", None, None,
        )

    try:
        ds = load_dataset_from_file(file, ftype, is_dpo=is_dpo)
        ds, issues = validate_and_clean_dataset(ds, is_dpo=is_dpo)
        preview_df  = preview_dataset(ds, is_dpo=is_dpo)
        issues_txt  = "\n".join(issues) if issues else "✅ No issues."
        raw_df      = None

        if ftype in ("csv", "excel"):
            import pandas as _pd
            raw_df = _pd.read_csv(file.name) if ftype == "csv" else _pd.read_excel(file.name)
            cols   = list(raw_df.columns)

            if is_dpo:
                need_map = not all(c in cols for c in [COL_PROMPT, COL_CHOSEN, COL_REJECTED])
            else:
                need_map = not ((COL_INSTRUCTION in cols and COL_OUTPUT in cols) or COL_TEXT in cols)

            if need_map:
                stats = f"**Total examples:** {len(ds)}\n**Preview ready**"
                return (
                    f"⚠️ Map columns below ({cols}). ",
                    gr.update(visible=True, choices=cols),
                    gr.update(visible=True, choices=cols),
                    gr.update(visible=True, choices=cols),
                    preview_df, stats + "\n" + issues_txt, raw_df, ftype,
                )

        stats = f"**Total examples:** {len(ds)}"
        return (
            f"✅ Loaded {len(ds)} examples. ",
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            preview_df, stats + "\n" + issues_txt, raw_df, ftype,
        )

    except Exception as e:
        return (
            f"❌ Error: {e}",
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            pd.DataFrame(), " ", None, None,
        )


def on_refresh_preview(file, training_mode, col_inst, col_out, col_text, raw_df_state, file_type_state):
    """Re-build dataset preview after the user changes column mapping dropdowns.

    FIX 2e: Refresh preview button.

    H6 FIX: The previous implementation passed `tmp.name` to pandas while
    the NamedTemporaryFile handle was still open, causing PermissionError
    on Windows. Now uses the context manager to close the handle before
    writing to the path, and guarantees cleanup in a finally block.
    """
    if file is None or raw_df_state is None or file_type_state is None:
        return pd.DataFrame(), "⚠️ No dataset loaded."

    training_mode = "dpo" if "dpo" in str(training_mode).lower() else "sft"
    is_dpo        = training_mode == "dpo"

    col_map = {}
    if is_dpo:
        if col_inst and col_out and col_text:
            col_map[col_inst] = COL_PROMPT
            col_map[col_out]  = COL_CHOSEN
            col_map[col_text] = COL_REJECTED
    else:
        if col_inst and col_out:
            col_map[col_inst] = COL_INSTRUCTION
            col_map[col_out]  = COL_OUTPUT
        elif col_text:
            col_map[col_text] = COL_TEXT

    try:
        if file_type_state in ("csv", "excel"):
            # H6 FIX: Use context manager so the file handle is closed before
            # pandas writes to the path. This is required on Windows and is
            # cleaner on all platforms — the finally block guarantees cleanup.
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{file_type_state}",
            ) as tmp:
                tmp_path = tmp.name
            # Handle is now closed; safe to write on all platforms.
            try:
                if file_type_state == "csv":
                    raw_df_state.to_csv(tmp_path, index=False)
                else:
                    raw_df_state.to_excel(tmp_path, index=False)
                dummy = type("_F", (), {"name": tmp_path})()
                ds = load_dataset_from_file(dummy, file_type_state, col_map, is_dpo=is_dpo)
            finally:
                os.unlink(tmp_path)
        else:
            ds = load_dataset_from_file(file, file_type_state, col_map, is_dpo=is_dpo)

        ds, issues = validate_and_clean_dataset(ds, is_dpo=is_dpo)
        preview_df = preview_dataset(ds, is_dpo=is_dpo)
        issues_txt = "\n".join(issues) if issues else "✅ No issues."
        stats      = f"**Total examples:** {len(ds)}\n{issues_txt}"
        return preview_df, stats

    except Exception as e:
        return pd.DataFrame(), f"❌ Preview refresh failed: {e}"


# ── Loss chart ─────────────────────────────────────────────────────────────

def build_loss_chart(log_records: list) -> pd.DataFrame:
    """Convert a list of LoggingCallback records into a display DataFrame.

    Replaces NaN eval_loss values (produced when eval_strategy='no') with
    the string 'N/A' so the Gradio table shows a meaningful label instead
    of a raw NaN to non-technical users.
    """
    if not log_records:
        return pd.DataFrame(columns=["Step", "Train Loss", "Eval Loss"])

    import math

    def _fmt_eval(v):
        if isinstance(v, float) and math.isnan(v):
            return "N/A"
        return v

    return pd.DataFrame({
        "Step":       [r["step"]            for r in log_records],
        "Train Loss": [r["train_loss"]       for r in log_records],
        "Eval Loss":  [_fmt_eval(r["eval_loss"]) for r in log_records],
    })
