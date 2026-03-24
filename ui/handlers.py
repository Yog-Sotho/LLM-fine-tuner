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
    augmented_ds=None,          # C-5 FIX: augmented/filtered dataset from gr.State
    progress=gr.Progress(),
):
    """Handler for the Start Training button.

    Orchestrates: file load → validate → preset apply → train → card → zip.
    Returns (log_str, zip_file_path, model_dir_path, log_records).

    C-5 FIX: When the user has augmented or filtered the dataset, `augmented_ds`
    will be a non-None HuggingFace Dataset object. In that case we skip
    loading from the raw file and use the pre-processed dataset directly,
    ensuring the button-click result is what actually gets trained on.

    M-6 FIX: The previous training run's ZIP file is deleted at the start of
    each new run to prevent the OS temp directory from filling up over time.

    N-3 FIX: Computing `_lengths` for the model card previously trusted `is_dpo`
    (derived from the current UI training mode selector) while `augmented_ds` may
    have been produced under a different mode. If a DPO-structured dataset was
    augmented then the user switched to SFT mode before training, the `else` branch
    tried ds[COL_INSTRUCTION] on a dataset that only has prompt/chosen/rejected
    columns — raising a KeyError.  The fix inspects the *actual* dataset columns
    instead of relying on the UI mode flag.
    """
    app_state.stop_event.clear()

    # M-6 FIX: Clean up the previous training ZIP file before creating a new one.
    if app_state._last_zip_path and os.path.isfile(app_state._last_zip_path):
        try:
            os.unlink(app_state._last_zip_path)
        except OSError:
            pass  # Non-fatal — best effort cleanup
    app_state._last_zip_path = None

    training_mode = "dpo" if "dpo" in training_mode.lower() else "sft"

    if file is None and augmented_ds is None:
        return "❌ Please upload a data file first.", None, None, []

    model_name = custom_model.strip() if custom_model.strip() else model_choice
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    is_dpo     = training_mode == "dpo"

    # C-5 FIX: Use the augmented/filtered dataset from state when available.
    # This ensures "Augment → Train" actually trains on the augmented data
    # rather than re-loading the original file.
    if augmented_ds is not None:
        ds = augmented_ds
        issues_str = "✅ Using augmented/filtered dataset from Data Enhancement step."
    else:
        if file is None:
            return "❌ Please upload a data file first.", None, None, []

        ftype = detect_file_type(file)

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

    # N-3 FIX: Compute dataset stats by inspecting ACTUAL column names on `ds`,
    # not by trusting the UI `is_dpo` flag.  When `augmented_ds` comes from a
    # previous augmentation run under a different mode the columns may not match
    # what the current training_mode flag implies, causing a KeyError on the old
    # conditional branches.
    try:
        if COL_PROMPT in ds.column_names and COL_CHOSEN in ds.column_names:
            # DPO-structured dataset (regardless of current UI mode)
            _lengths = [
                len(str(p)) + len(str(c)) + len(str(r))
                for p, c, r in zip(ds[COL_PROMPT], ds[COL_CHOSEN], ds[COL_REJECTED])
            ]
        elif COL_TEXT in ds.column_names:
            _lengths = [len(str(t)) for t in ds[COL_TEXT]]
        elif COL_INSTRUCTION in ds.column_names and COL_OUTPUT in ds.column_names:
            _lengths = [
                len(str(i)) + len(str(o))
                for i, o in zip(ds[COL_INSTRUCTION], ds[COL_OUTPUT])
            ]
        else:
            # Fallback: estimate from the first available column so model card
            # generation never crashes even on unexpected column structures.
            first_col = ds.column_names[0] if ds.column_names else None
            _lengths = [len(str(v)) for v in ds[first_col]] if first_col else []
    except Exception:
        # Ultimate safety net — wrong column types, Arrow errors, etc.
        _lengths = [100] * len(ds)

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
        zip_path = create_zip_from_folder(output_dir)

        # M-6 FIX: Store the new ZIP path so the next call can clean it up.
        app_state._last_zip_path = zip_path

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
        import tempfile as _tmp, os as _os

        if file_type_state in ("csv", "excel"):
            tmp = _tmp.NamedTemporaryFile(delete=False, suffix=f".{file_type_state}")
            try:
                if file_type_state == "csv":
                    raw_df_state.to_csv(tmp.name, index=False)
                else:
                    raw_df_state.to_excel(tmp.name, index=False)
                # Create a minimal file-like proxy so load_dataset_from_file works
                dummy = type("_F", (), {"name": tmp.name})()
                ds = load_dataset_from_file(dummy, file_type_state, col_map, is_dpo=is_dpo)
            finally:
                _os.unlink(tmp.name)
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
    """Convert a list of LoggingCallback records into a display DataFrame."""
    if not log_records:
        return pd.DataFrame(columns=["Step", "Train Loss", "Eval Loss"])
    return pd.DataFrame({
        "Step":       [r["step"]       for r in log_records],
        "Train Loss": [r["train_loss"] for r in log_records],
        "Eval Loss":  [r["eval_loss"]  for r in log_records],
    })
