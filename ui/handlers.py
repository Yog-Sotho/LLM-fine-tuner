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

Patch log
---------
  F-2  : ``build_loss_chart()`` now includes an "ETA" column whenever the
         log records contain ``eta_s`` data (populated by the updated
         ``LoggingCallback`` in core/callbacks.py).  Old records that
         predate this field (e.g. from CLI runs or pre-patch log replay)
         are handled gracefully — the ETA column is simply omitted.
         The ETA value is formatted as a human-readable string
         ("2m 30s", "45s", etc.) so non-technical users can read it
         directly in the Loss Curve table without conversion.
"""

import glob
import os
import shutil

import gradio as gr
import pandas as pd
import torch

from config.constants import (
    COL_CHOSEN,
    COL_INSTRUCTION,
    COL_OUTPUT,
    COL_PROMPT,
    COL_REJECTED,
    COL_TEXT,
    DEFAULT_EVAL_SPLIT,
    DEFAULT_LORA_VARIANT,
    DEFAULT_REPORT_TO,
    DEFAULT_SEED,
    RUNS_DIR,
)
from core.run_config import new_run_name, resolve_report_to, run_dir_for
from core.state import app_state, redact_sensitive_info, validate_path_traversal
from data.documents import document_chunks
from data.loader import detect_file_type, load_dataset_from_file, load_hub_dataset
from data.preprocessing import column_mapping, preview_dataset, validate_and_clean_dataset
from export.hub import push_to_hub
from export.quantize import on_quantize_click
from export.utils import create_zip_from_folder
from inference.generate import batch_generate, generate_text
from inference.remote import remote_chat
from inference.synthesize import format_stats, local_writer, remote_writer, synthesize_pairs
from training.embedding import train_embedding
from training.sft import train_model

# ── Training ───────────────────────────────────────────────────────────────


def on_train_click(
    file,
    model_choice,
    custom_model,
    training_preset,
    peft_method,
    use_lora,
    lora_rank,
    lora_alpha,
    prefix_tuning_num_virtual_tokens,
    prefix_tuning_token_dim,
    prefix_tuning_num_layers,
    prompt_tuning_num_virtual_tokens,
    lr,
    epochs,
    bs,
    grad_accum,
    max_len,
    warmup,
    early_stop,
    lr_sched,
    grad_ckpt,
    resume,
    col_inst,
    col_out,
    col_text,
    use_unsloth,
    use_chat_template,
    system_prompt,
    training_mode,
    dpo_beta,
    heretic_mode,
    use_flash_attn=False,
    use_qlora_enhanced=False,  # kept for UI arity — ignored; peft_method drives QLoRA
    augmented_ds=None,  # C-5 FIX: augmented/filtered dataset from gr.State
    packing=False,
    run_name="",
    seed=DEFAULT_SEED,
    report_to=DEFAULT_REPORT_TO,
    eval_split=DEFAULT_EVAL_SPLIT,
    lora_variant=DEFAULT_LORA_VARIANT,
    activation_offloading=False,
    padding_free=False,
    progress=gr.Progress(),
    request: gr.Request | None = None,
):
    """Handler for the Start Training button.

    Orchestrates: file load → validate → preset apply → train → card → zip.
    Returns (log_str, zip_file_path, model_dir_path, log_records).
    """
    session = app_state.session_for(request)
    session.stop_event.clear()

    # Free this session's previous ZIP. Run folders persist under RUNS_DIR.
    session.release("zip")

    training_mode = "dpo" if "dpo" in training_mode.lower() else "sft"

    # Runs live in <RUNS_DIR>/<run name>/ so they survive restarts and can be resumed.
    try:
        report_to = resolve_report_to(report_to)
        run_name = (run_name or "").strip() or new_run_name(training_mode)
        output_dir = run_dir_for(run_name)
    except ValueError as e:
        return f"❌ {redact_sensitive_info(str(e))}", None, None, []
    run_exists = os.path.isdir(output_dir)
    if run_exists and not resume:
        return (
            f"❌ Run '{run_name}' already exists. Tick 'Resume from last checkpoint' to "
            "continue it, or choose another run name.",
            None,
            None,
            [],
        )
    if resume and not run_exists:
        return f"❌ Nothing to resume: run '{run_name}' not found in {RUNS_DIR}.", None, None, []

    if file is None and augmented_ds is None:
        return "❌ Please upload a data file first.", None, None, []

    # Strip whitespace and validate against path traversal.
    custom_model = custom_model.strip() if custom_model else ""
    if err := validate_path_traversal(custom_model):
        return err, None, None, []

    model_name = custom_model if custom_model else model_choice
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_dpo = training_mode == "dpo"

    # C-5 FIX: Use the augmented/filtered dataset from state when available.
    if augmented_ds is not None:
        ds = augmented_ds
        issues_str = (
            "✅ Using the dataset prepared in the Data tab"
            " (Hub load, documents, augmentation or filter)."
        )
    else:
        if file is None:
            return "❌ Please upload a data file first.", None, None, []

        ftype = detect_file_type(file)

        col_map = column_mapping(is_dpo, col_inst, col_out, col_text)

        try:
            ds = load_dataset_from_file(file, ftype, col_map, is_dpo=is_dpo)
        except Exception as e:
            return redact_sensitive_info(str(e)), None, None, []

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

    # Gradio can deliver ints as floats (e.g. 4.0); DataLoader rejects float batch sizes.
    hyperparams = dict(
        learning_rate=float(lr),
        epochs=int(epochs),
        batch_size=int(bs),
        grad_accum=int(grad_accum),
        max_length=int(max_len),
        warmup_steps=int(warmup),
        lora_rank=int(lora_rank),
        lora_alpha=int(lora_alpha),
        lr_scheduler=str(lr_sched),
        prefix_tuning_num_virtual_tokens=int(prefix_tuning_num_virtual_tokens),
        prefix_tuning_token_dim=int(prefix_tuning_token_dim),
        prefix_tuning_num_layers=int(prefix_tuning_num_layers),
        prompt_tuning_num_virtual_tokens=int(prompt_tuning_num_virtual_tokens),
        dpo_beta=float(dpo_beta),
        packing=bool(packing),
        activation_offloading=bool(activation_offloading),
        padding_free=bool(padding_free),
        eval_split=float(eval_split),
    )
    os.makedirs(output_dir, exist_ok=True)

    try:
        msg, log_records = train_model(
            model_name,
            ds,
            output_dir,
            hyperparams,
            device,
            peft_method,
            use_lora,
            lora_rank,
            lora_alpha,
            prefix_tuning_num_virtual_tokens,
            prefix_tuning_token_dim,
            prefix_tuning_num_layers,
            prompt_tuning_num_virtual_tokens,
            resume,
            early_stop,
            lr_sched,
            grad_ckpt,
            use_unsloth,
            use_chat_template,
            system_prompt,
            training_mode=training_mode,
            dpo_beta=dpo_beta,
            heretic_mode=heretic_mode,
            progress=progress,
            use_flash_attn=use_flash_attn,
            stop_event=session.stop_event,
            seed=int(seed),
            report_to=report_to,
            run_name=run_name,
            lora_variant=lora_variant,
        )
        zip_path = create_zip_from_folder(output_dir)

        session.track("zip", zip_path)

        full_msg = msg + "\n" + issues_str
        return full_msg, zip_path, output_dir, log_records

    except Exception as e:
        # Remove a run folder this attempt created, unless it saved checkpoints to resume from.
        if not run_exists and not glob.glob(os.path.join(output_dir, "checkpoint-*")):
            shutil.rmtree(output_dir, ignore_errors=True)
        return f"❌ Training failed: {redact_sensitive_info(str(e))}\n{issues_str}", None, None, []


def on_embed_click(
    model_name,
    file,
    run_name,
    method,
    learning_rate,
    epochs,
    batch_size,
    max_seq_length,
    matryoshka,
    hard_negatives,
    eval_split,
    query_prompt,
    progress=gr.Progress(),
    request: gr.Request | None = None,
) -> str:
    """Embeddings tab: train into <runs folder>/<run name>/ (never a raw user path)."""
    try:
        run_name = (run_name or "").strip() or new_run_name("embedding")
        output_dir = run_dir_for(run_name)
    except ValueError as e:
        return f"❌ {redact_sensitive_info(str(e))}"
    if os.path.isdir(output_dir):
        return f"❌ Run '{run_name}' already exists. Choose another run name."
    return train_embedding(
        model_name, file, output_dir, method, learning_rate, epochs, batch_size, max_seq_length,
        matryoshka, hard_negatives, eval_split, query_prompt, progress=progress, request=request,
    )  # fmt: skip


def on_stop(request: gr.Request | None = None) -> str:
    """Signal this session's running job to halt after the current step."""
    app_state.session_for(request).stop_event.set()
    return "🛑 Stop signal sent — will halt after the current step."


# ── Inference ──────────────────────────────────────────────────────────────


def on_generate(prompt, model_choice, custom_model, lora_path, max_tok, temp, top_p) -> str:
    # Strip whitespace and validate against path traversal.
    custom_model = custom_model.strip() if custom_model else ""
    lora_path = lora_path.strip() if lora_path else ""
    if err := (validate_path_traversal(custom_model) or validate_path_traversal(lora_path)):
        return err

    model_name = custom_model if custom_model else model_choice
    return generate_text(model_name, lora_path, prompt, int(max_tok), temp, top_p)


def on_batch_test(
    f, model_choice, custom_model, lora_path, request: gr.Request | None = None
) -> tuple[str, str | None]:
    """Batch inference → (status, responses CSV or None).

    Errors go to the status box: returned into the gr.File output, Gradio would try to
    open the message as a file path and show a generic error instead.
    """
    # Strip whitespace and validate against path traversal.
    custom_model = custom_model.strip() if custom_model else ""
    lora_path = lora_path.strip() if lora_path else ""
    if err := (validate_path_traversal(custom_model) or validate_path_traversal(lora_path)):
        return err, None

    if f and hasattr(f, "name") and f.name:
        if err := validate_path_traversal(f.name):
            return err, None

    session = app_state.session_for(request)
    session.release("batch")

    model_name = custom_model if custom_model else model_choice
    result = batch_generate(model_name, lora_path, f)
    if not os.path.isfile(result):  # an error message
        return result, None
    session.track("batch", result)
    return f"✅ {len(pd.read_csv(result))} responses — download the CSV.", result


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
    is_dpo = training_mode == "dpo"

    if file is None:
        return (
            "No file uploaded.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            pd.DataFrame(),
            " ",
            None,
            None,
        )

    # Validate path traversal on file upload
    if file and hasattr(file, "name") and file.name:
        if err := validate_path_traversal(file.name):
            return (
                f"❌ {err}",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                pd.DataFrame(),
                " ",
                None,
                None,
            )

    ftype = detect_file_type(file)
    if ftype is None:
        return (
            "⚠️ Unsupported file type.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            pd.DataFrame(),
            " ",
            None,
            None,
        )

    try:
        # CSV / Excel are read once; the DataFrame is kept for re-mapping columns.
        raw_df = None
        if ftype in ("csv", "excel"):
            from data.loader import load_dataset_from_dataframe

            raw_df = (
                pd.read_csv(file.name)
                if ftype == "csv"
                else pd.read_excel(file.name, engine="openpyxl")
            )
            cols = list(raw_df.columns)
            if is_dpo:
                need_map = not all(c in cols for c in [COL_PROMPT, COL_CHOSEN, COL_REJECTED])
            else:
                need_map = not (
                    (COL_INSTRUCTION in cols and COL_OUTPUT in cols) or COL_TEXT in cols
                )
            if need_map:
                # The columns can't be read as training data yet: show the raw rows and
                # the mapping dropdowns (Refresh preview applies the mapping).
                return (
                    f"⚠️ Map columns below ({cols}). ",
                    gr.update(visible=True, choices=cols),
                    gr.update(visible=True, choices=cols),
                    gr.update(visible=True, choices=cols),
                    raw_df.head(10),
                    f"**Rows in file:** {len(raw_df)}\n**Map the columns, then 🔄 Apply Mapping.**",
                    raw_df,
                    ftype,
                )
            ds = load_dataset_from_dataframe(raw_df, is_dpo=is_dpo)
        else:
            ds = load_dataset_from_file(file, ftype, is_dpo=is_dpo)

        ds, issues = validate_and_clean_dataset(ds, is_dpo=is_dpo)
        preview_df = preview_dataset(ds, is_dpo=is_dpo)
        issues_txt = "\n".join(issues) if issues else "✅ No issues."

        stats = f"**Total examples:** {len(ds)}"
        return (
            f"✅ Loaded {len(ds)} examples. ",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            preview_df,
            stats + "\n" + issues_txt,
            raw_df,
            ftype,
        )

    except Exception as e:
        return (
            f"❌ Error: {redact_sensitive_info(str(e))}",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            pd.DataFrame(),
            " ",
            None,
            None,
        )


def on_hub_load(repo_id, config, split, max_rows, training_mode="sft"):
    """Load, clean and preview a Hub dataset; the result is used by Start Training.

    Returns (status_md, preview_df, stats_md, dataset_or_None).
    """
    is_dpo = "dpo" in str(training_mode).lower()
    try:
        ds = load_hub_dataset(repo_id, split=split, config=config, max_rows=max_rows, is_dpo=is_dpo)
        ds, issues = validate_and_clean_dataset(ds, is_dpo=is_dpo)
    except Exception as e:  # network, missing dataset, bad layout — shown to the user
        return f"❌ {redact_sensitive_info(str(e))}", pd.DataFrame(), " ", None
    if len(ds) == 0:
        return "❌ No usable rows after cleaning.", pd.DataFrame(), "\n".join(issues), None
    issues_txt = "\n".join(issues) if issues else "✅ No issues."
    return (
        f"✅ Loaded {len(ds)} rows from `{repo_id.strip()}` — used by **▶ Start Training**.",
        preview_dataset(ds, is_dpo=is_dpo),
        f"**Total examples:** {len(ds)}\n{issues_txt}",
        ds,
    )


def on_refresh_preview(
    file, training_mode, col_inst, col_out, col_text, raw_df_state, file_type_state
):
    """Re-build dataset preview after the user changes column mapping dropdowns."""
    if file is None or raw_df_state is None or file_type_state is None:
        return pd.DataFrame(), "⚠️ No dataset loaded."

    training_mode = "dpo" if "dpo" in str(training_mode).lower() else "sft"
    is_dpo = training_mode == "dpo"

    from data.loader import load_dataset_from_dataframe

    col_map = column_mapping(is_dpo, col_inst, col_out, col_text)

    try:
        # Bypassing I/O by loading directly from raw_df_state
        # when available, avoiding redundant temporary file creation.
        if raw_df_state is not None:
            ds = load_dataset_from_dataframe(raw_df_state, col_map, is_dpo=is_dpo)
        else:
            ds = load_dataset_from_file(file, file_type_state, col_map, is_dpo=is_dpo)

        ds, issues = validate_and_clean_dataset(ds, is_dpo=is_dpo)
        preview_df = preview_dataset(ds, is_dpo=is_dpo)
        issues_txt = "\n".join(issues) if issues else "✅ No issues."
        stats = f"**Total examples:** {len(ds)}\n{issues_txt}"
        return preview_df, stats

    except Exception as e:
        return pd.DataFrame(), f"❌ Preview refresh failed: {redact_sensitive_info(str(e))}"


# ── Loss chart ─────────────────────────────────────────────────────────────


def _fmt_eta(eta_s: float) -> str:
    """Format ETA seconds into a human-readable string.

    Examples: 0 → "—", 45.0 → "45s", 125.0 → "2m 05s", 3720.0 → "1h 02m"
    """
    eta_s = max(0.0, eta_s)
    if eta_s < 1:
        return "—"
    if eta_s < 60:
        return f"{int(eta_s)}s"
    if eta_s < 3600:
        m, s = divmod(int(eta_s), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(eta_s), 3600)
    m = rem // 60
    return f"{h}h {m:02d}m"


def build_loss_chart(log_records: list) -> pd.DataFrame:
    """Convert a list of LoggingCallback records into a display DataFrame.

    F-2 FIX: When records contain ``eta_s`` data (added by the updated
    LoggingCallback in core/callbacks.py), an "ETA" column with human-readable
    strings is included.  Old records that lack ``eta_s`` (CLI runs, pre-patch
    replays) produce a DataFrame without the ETA column — fully backwards
    compatible with callers that only look at "Step", "Train Loss", "Eval Loss".
    """
    if not log_records:
        return pd.DataFrame(columns=["Step", "Train Loss", "Eval Loss"])

    data: dict = {
        "Step": [r["step"] for r in log_records],
        "Train Loss": [None if pd.isna(r["train_loss"]) else r["train_loss"] for r in log_records],
        # NaN (no eval split) renders as a gap rather than a "NaN" cell.
        "Eval Loss": [None if pd.isna(r["eval_loss"]) else r["eval_loss"] for r in log_records],
    }

    # F-2: Include ETA column only when timing data is actually present.
    # Using .get() with a sentinel avoids KeyError on old-format records.
    _MISSING = object()
    first_eta = log_records[0].get("eta_s", _MISSING)
    if first_eta is not _MISSING:
        data["ETA"] = [_fmt_eta(r.get("eta_s", 0.0)) for r in log_records]

    return pd.DataFrame(data)


# ── Deployment: quantized export and remote endpoints ──────────────────────


def on_quantize_export(model_path, fmt, file, augmented_ds, progress=gr.Progress()):
    """Export tab: FP8 / W4A16 safetensors for vLLM (W4A16 calibrates on the Data tab's data)."""
    dataset = augmented_ds
    if dataset is None and file is not None and fmt == "w4a16":
        try:
            ds = load_dataset_from_file(file, detect_file_type(file))
            dataset, _ = validate_and_clean_dataset(ds)
        except Exception as e:
            return (
                f"❌ Cannot read the training data for calibration: {redact_sensitive_info(str(e))}"
            )
    return on_quantize_click(model_path, fmt, dataset, progress)


def on_remote_chat(url, model, api_key, system_prompt, prompt, max_tokens, temperature):
    """Inference tab: one chat turn with an OpenAI-compatible server."""
    try:
        return remote_chat(url, prompt, model, api_key, system_prompt,
                           int(max_tokens), float(temperature))  # fmt: skip
    except ValueError as e:
        return f"❌ {redact_sensitive_info(str(e))}"


# ── Training data from documents ───────────────────────────────────────────


def on_synthesize(files, writer, url, server_model, api_key, local_model, pairs_per_chunk,
                  threshold, max_chunks, progress=gr.Progress(),
                  request: gr.Request | None = None):  # fmt: skip
    """Data tab: documents → question/answer pairs (used by Start Training) + a JSONL file.

    Returns (status, preview, stats, dataset_or_None, jsonl_path_or_None).
    """
    import json
    import tempfile

    from datasets import Dataset

    from config.constants import SYNTH_WRITERS

    fail = lambda msg: (msg, pd.DataFrame(), " ", None, None)  # noqa: E731
    paths = [f.name if hasattr(f, "name") else str(f) for f in (files or [])]
    if not paths:
        return fail("❌ Upload one or more documents (PDF, Word, text or Markdown).")
    try:
        chunks = document_chunks(paths)
    except ValueError as e:
        return fail(f"❌ {redact_sensitive_info(str(e))}")
    if not chunks:
        return fail("❌ No text found in the documents (scanned PDFs need OCR first).")
    chunks = chunks[: int(max_chunks)]
    if writer == SYNTH_WRITERS[1]:
        local_model = (local_model or "").strip()
        if not local_model:
            return fail("❌ Enter the local model (Hub id or folder).")
        if err := validate_path_traversal(local_model):
            return fail(err)
        ask = local_writer(local_model)
    else:
        if not (url or "").strip():
            return fail("❌ Enter the server URL, e.g. http://127.0.0.1:8000")
        ask = remote_writer(url, server_model or "", api_key or "")

    session = app_state.session_for(request)
    session.stop_event.clear()

    def on_progress(i, total):
        if progress is not None:
            progress(i / total, desc=f"Writing pairs: chunk {i + 1} of {total}")

    rows, stats = synthesize_pairs(chunks, ask, int(pairs_per_chunk), float(threshold),
                                   on_progress, session.stop_event.is_set)  # fmt: skip
    status = format_stats(stats)
    if not rows:
        return fail(status)
    ds = Dataset.from_dict({COL_INSTRUCTION: [r[COL_INSTRUCTION] for r in rows],
                            COL_OUTPUT: [r[COL_OUTPUT] for r in rows]})  # fmt: skip
    session.release("synth")
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    session.track("synth", f.name)
    return (
        status + "\nUsed by **▶ Start Training** (or download the JSONL).",
        preview_dataset(ds),
        f"**Total examples:** {len(ds)}",
        ds,
        f.name,
    )
