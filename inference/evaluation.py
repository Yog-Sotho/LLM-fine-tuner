"""
inference/evaluation.py
========================
Layer 4 — automated evaluation: BLEU, ROUGE, BERTScore, LLM-as-Judge.
Imports: config.constants, core.state, inference.generate, stdlib, pandas, gradio.

Functions
---------
compute_bleu_rouge            — compute BLEU-1 and ROUGE-1/2/L scores
compute_bertscore_metric      — compute BERTScore precision / recall / F1
llm_judge_evaluate            — score responses with a local LLM judge
build_prediction_preview_html — build styled HTML showing N random example rows (F-6)
on_evaluate_click             — Gradio UI handler for the Evaluation tab Run button

Patch log
---------
  M-5    : ``import numpy as np`` was inside the body of ``compute_bleu_rouge()``.
           Moved to the top-level import block where it belongs. Python caches
           module imports so this has zero runtime cost; it improves readability
           and static analysis coverage.
  F-6    : New ``build_prediction_preview_html()`` helper.  After evaluation,
           3 random examples are rendered as a styled HTML table showing the
           prompt, reference, prediction, and per-row ROUGE-L score (when
           rouge-score is installed and references are available). The table
           makes quality tangible for non-technical users who find aggregate
           BLEU/ROUGE numbers hard to interpret.
           ``on_evaluate_click()`` now returns a third value — the HTML preview
           string — so the evaluation tab can display it without a separate
           button click.
  Fix-2  : ``_esc()`` helper was defined inside the ``for`` loop in
           ``build_prediction_preview_html()``.  Python creates a new function
           object on every iteration (Ruff B023 / Pylint W0640).  Hoisted to
           module scope so it is defined exactly once and reused for all rows.
  Fix-3  : Misleading docstring comment claimed ``random.sample`` used a
           "fixed seed" — no seed was ever set.  The behaviour (different
           examples on each click) is actually desirable for variety; corrected
           the comment to accurately describe what the code does.
"""

import random

import gradio as gr
import numpy as np  # M-5 FIX: moved from inside compute_bleu_rouge() to module top-level
import pandas as pd
import torch

from config.constants import (
    HAS_BERTSCORE,
    HAS_NLTK,
    HAS_ROUGE,
)
from core.state import app_state
from inference.generate import _load_for_inference

# ── HTML escaping helper ───────────────────────────────────────────────────
# Fix-2: Defined at module scope, not inside the rendering loop.
# Escapes the four characters that are meaningful inside HTML attribute values
# and element text, preventing XSS from model outputs or user prompts.

def _esc(s: str) -> str:
    """Escape HTML special characters in ``s`` for safe inline rendering.

    Sentinel: Enhanced to include single quote escaping (&#x27;) as per
    OWASP recommendations for robust XSS prevention.
    """
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#x27;")
    )


# ── BLEU + ROUGE ───────────────────────────────────────────────────────────

def compute_bleu_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute BLEU-1, ROUGE-1, ROUGE-2, ROUGE-L over paired lists.

    Falls back to string placeholders when optional deps are missing.
    """
    # numpy is imported at the top of the module (M-5 FIX)
    results = {}

    if HAS_NLTK and predictions and references:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # lazy

        smoothing = SmoothingFunction().method4
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens  = [ref.split()]
            if pred_tokens:
                try:
                    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                    bleu_scores.append(score)
                except Exception:
                    bleu_scores.append(0.0)
        results["BLEU-1"] = round(float(np.mean(bleu_scores)), 4) if bleu_scores else 0.0
    else:
        results["BLEU-1"] = "nltk not installed"

    if HAS_ROUGE and predictions and references:
        from rouge_score import rouge_scorer as rouge_scorer_lib  # lazy

        scorer = rouge_scorer_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1_scores, r2_scores, rl_scores = [], [], []
        for pred, ref in zip(predictions, references):
            try:
                scores = scorer.score(ref, pred)
                r1_scores.append(scores["rouge1"].fmeasure)
                r2_scores.append(scores["rouge2"].fmeasure)
                rl_scores.append(scores["rougeL"].fmeasure)
            except Exception:
                r1_scores.append(0.0)
                r2_scores.append(0.0)
                rl_scores.append(0.0)
        results["ROUGE-1"] = round(float(np.mean(r1_scores)), 4)
        results["ROUGE-2"] = round(float(np.mean(r2_scores)), 4)
        results["ROUGE-L"] = round(float(np.mean(rl_scores)), 4)
    else:
        results["ROUGE-1"] = results["ROUGE-2"] = results["ROUGE-L"] = "rouge_score not installed"

    return results


# ── BERTScore ──────────────────────────────────────────────────────────────

def compute_bertscore_metric(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
) -> dict:
    """Compute BERTScore Precision, Recall, and F1.

    Returns placeholder strings when bert_score is not installed.
    """
    if not HAS_BERTSCORE:
        return {
            "BERTScore-P":  "bert_score not installed",
            "BERTScore-R":  "N/A",
            "BERTScore-F1": "N/A",
        }
    if not predictions or not references:
        return {"BERTScore-P": 0.0, "BERTScore-R": 0.0, "BERTScore-F1": 0.0}

    try:
        from bert_score import score as bert_score_fn  # lazy

        P, R, F1 = bert_score_fn(predictions, references, lang=lang, verbose=False)
        return {
            "BERTScore-P":  round(float(P.mean()),  4),
            "BERTScore-R":  round(float(R.mean()),  4),
            "BERTScore-F1": round(float(F1.mean()), 4),
        }
    except Exception as e:
        return {
            "BERTScore-P":  f"Error: {e}",
            "BERTScore-R":  "N/A",
            "BERTScore-F1": "N/A",
        }


# ── LLM-as-Judge ──────────────────────────────────────────────────────────

def llm_judge_evaluate(
    prompts: list[str],
    responses: list[str],
    criteria: str,
    judge_model_name: str,
    judge_lora_path: str | None = None,
    max_new_tokens: int = 128,
) -> list[dict]:
    """Use a local LLM to score each (prompt, response) pair.

    Constructs a structured eval prompt and asks the judge model for a
    1-10 score with brief reasoning. Returns a list of result dicts.

    H5 FIX: Previously swallowed all exceptions and appended a phantom row
    with ``"prompt": "ERROR"`` — a broken judge silently contaminated the
    output DataFrame. Now raises RuntimeError so the caller can report
    the failure cleanly instead of mixing error rows with real results.

    Raises
    ------
    RuntimeError — if the judge model cannot be loaded or inference fails.
    """
    # Load model outside the per-example loop so a load failure raises
    # immediately rather than producing a partial + error result.
    model, tokenizer = _load_for_inference(judge_model_name, judge_lora_path)

    results = []
    # BOLT OPTIMIZATION: Process judge evaluations in batches (default size 8)
    # to utilize GPU parallelism, significantly speeding up large evaluations.
    batch_size = 8
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        batch_responses = responses[i: i + batch_size]

        eval_texts = [
            f"Evaluate the following response based on: {criteria}\n"
            f"Prompt: {p}\nResponse: {r}\n"
            f"Score (1-10) and brief reasoning:"
            for p, r in zip(batch_prompts, batch_responses)
        ]

        inputs = tokenizer(
            eval_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Simplified prompt stripping: left-padding ensures all responses start
        # at the same relative offset (input_ids.shape[1]).
        # BOLT OPTIMIZATION: Using batch_decode instead of serial decode for faster processing.
        input_len = inputs["input_ids"].shape[1]
        judgments = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

        for p, r, judgment in zip(batch_prompts, batch_responses, judgments):
            results.append({"prompt": p, "response": r, "judgment": judgment.strip()})

    return results


# ── F-6: Per-example prediction preview ───────────────────────────────────

def build_prediction_preview_html(
    prompts: list[str],
    predictions: list[str],
    references: list[str],
    n: int = 3,
) -> str:
    """Build a styled HTML table showing up to ``n`` random prediction examples.

    F-6: Makes evaluation quality tangible for non-technical users by showing
    actual model outputs side-by-side with the prompt and reference answer.
    Aggregate numbers like ROUGE-1=0.32 are hard to interpret; seeing three
    concrete examples immediately reveals whether the model is on-topic.

    Parameters
    ----------
    prompts     : list of prompt strings (required)
    predictions : list of generated prediction strings (required, same length)
    references  : list of reference strings (may be empty — column is hidden)
    n           : number of random examples to display (default 3)

    Returns
    -------
    HTML string ready for a ``gr.HTML`` component. Returns an empty string
    when there are no predictions to show.
    """
    if not predictions:
        return ""

    # Clamp n to the actual number of examples available
    n = min(n, len(predictions))

    # Fix-3: The previous comment claimed a fixed seed was used for stability.
    # No seed was ever set — random.sample() uses the global random state,
    # which produces different rows on each call.  This is intentional: showing
    # different examples on repeated clicks gives the user more variety and a
    # better sense of overall model quality.
    indices = random.sample(range(len(predictions)), n)

    # Optionally compute per-row ROUGE-L when rouge-score is installed and
    # references are available.
    per_row_rouge: dict[int, float] = {}
    if HAS_ROUGE and references and len(references) == len(predictions):
        try:
            from rouge_score import rouge_scorer as _rs_lib  # lazy

            _scorer = _rs_lib.RougeScorer(["rougeL"], use_stemmer=True)
            for idx in indices:
                try:
                    score = _scorer.score(references[idx], predictions[idx])
                    per_row_rouge[idx] = round(score["rougeL"].fmeasure, 3)
                except Exception:
                    per_row_rouge[idx] = 0.0
        except Exception:
            pass  # rouge import failed at runtime — skip scores silently

    has_references = bool(references and len(references) == len(predictions))
    has_rouge = bool(per_row_rouge)

    # ── CSS ──────────────────────────────────────────────────────────────
    css = """
    <style>
      .pred-preview-wrap {
        background: #1a1a2e;
        border: 1px solid #7c3aed;
        border-radius: 10px;
        padding: 16px;
        margin-top: 12px;
        font-family: 'Inter', system-ui, sans-serif;
      }
      .pred-preview-wrap h4 {
        color: #a78bfa;
        margin: 0 0 12px 0;
        font-size: 0.95rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }
      .pred-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
      }
      .pred-table th {
        background: #4c1d95;
        color: #e2e8f0;
        padding: 8px 10px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #7c3aed;
      }
      .pred-table td {
        padding: 8px 10px;
        vertical-align: top;
        color: #cbd5e1;
        border-bottom: 1px solid #334155;
        line-height: 1.45;
        word-break: break-word;
      }
      .pred-table tr:last-child td { border-bottom: none; }
      .pred-table tr:nth-child(even) td { background: #16213e; }
      .cell-prompt  { color: #93c5fd; font-style: italic; }
      .cell-ref     { color: #6ee7b7; }
      .cell-pred    { color: #fde68a; }
      .cell-rouge   { color: #f9a8d4; text-align: center; font-weight: 600; }
      .badge-idx    {
        display: inline-block;
        background: #7c3aed;
        color: white;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.75rem;
        margin-right: 6px;
        vertical-align: middle;
      }
    </style>
    """

    # ── Table header ──────────────────────────────────────────────────────
    header_cells = "<th>#</th><th>Prompt</th>"
    if has_references:
        header_cells += "<th>Reference</th>"
    header_cells += "<th>Prediction</th>"
    if has_rouge:
        header_cells += "<th>ROUGE-L</th>"

    # Fix-2: _esc() is now defined at module scope (see top of file).
    # It is no longer redefined on every loop iteration.
    rows_html = ""
    for rank, idx in enumerate(indices, start=1):
        # Truncate very long strings to keep the table readable
        prompt_txt = (prompts[idx][:280] + "…") if len(prompts[idx]) > 280 else prompts[idx]
        pred_txt   = (predictions[idx][:380] + "…") if len(predictions[idx]) > 380 else predictions[idx]

        row = (
            f'<td><span class="badge-idx">{rank}</span></td>'
            f'<td class="cell-prompt">{_esc(prompt_txt)}</td>'
        )
        if has_references:
            ref_txt = (references[idx][:280] + "…") if len(references[idx]) > 280 else references[idx]
            row += f'<td class="cell-ref">{_esc(ref_txt)}</td>'
        row += f'<td class="cell-pred">{_esc(pred_txt)}</td>'
        if has_rouge:
            rouge_val = per_row_rouge.get(idx, "—")
            row += f'<td class="cell-rouge">{rouge_val}</td>'

        rows_html += f"<tr>{row}</tr>\n"

    rouge_note = " · ROUGE-L per row" if has_rouge else ""
    table_html = f"""
    <div class="pred-preview-wrap">
      <h4>🔍 Prediction Preview — {n} random examples{rouge_note}</h4>
      <table class="pred-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """
    return css + table_html


# ── Gradio UI handler ──────────────────────────────────────────────────────

def on_evaluate_click(
    eval_model_name: str,
    eval_custom_model: str,
    eval_lora_path: str,
    eval_file,
    eval_run_bertscore: bool,
    eval_use_judge: bool,
    judge_model_name: str,
    judge_criteria: str,
    eval_max_new_tokens: int = 150,
    progress=gr.Progress(),
):
    """Handler for the Evaluation tab Run button.

    Loads the evaluation dataset (CSV or JSONL with 'prompt' / 'reference'),
    generates predictions in batches, then computes all requested metrics.

    F-6: Now returns a third value — an HTML preview string from
    ``build_prediction_preview_html()`` — so the evaluation tab can display
    concrete examples alongside the aggregate metric scores.

    Returns (metrics_str, result_dataframe, preview_html).
    """
    # Sentinel: strip whitespace and validate against path traversal.
    eval_custom_model = eval_custom_model.strip() if eval_custom_model else ""
    eval_lora_path    = eval_lora_path.strip()    if eval_lora_path    else ""
    judge_model_name  = judge_model_name.strip()  if judge_model_name  else ""

    if ".." in eval_custom_model or "\\" in eval_custom_model or \
       ".." in eval_lora_path    or "\\" in eval_lora_path    or \
       ".." in judge_model_name  or "\\" in judge_model_name:
        return "❌ Path traversal attempt detected.", pd.DataFrame(), ""

    model_name = eval_custom_model if eval_custom_model else eval_model_name
    if not model_name:
        return "❌ Please select a model.", pd.DataFrame(), ""
    if eval_file is None:
        return (
            "❌ Please upload a test dataset (CSV with 'prompt' and 'reference' columns).",
            pd.DataFrame(),
            "",
        )

    try:
        progress(0, desc="Loading evaluation dataset…")
        if eval_file.name.endswith(".csv"):
            eval_df = pd.read_csv(eval_file.name)
        elif eval_file.name.endswith(".jsonl"):
            eval_df = pd.read_json(eval_file.name, lines=True)
        else:
            return (
                "❌ Evaluation dataset must be CSV or JSONL with 'prompt' and 'reference' columns.",
                pd.DataFrame(),
                "",
            )

        if "prompt" not in eval_df.columns:
            return (
                f"❌ Dataset must have a 'prompt' column. Found: {list(eval_df.columns)}",
                pd.DataFrame(),
                "",
            )

        prompts    = eval_df["prompt"].astype(str).tolist()
        references = eval_df["reference"].astype(str).tolist() if "reference" in eval_df.columns else []

        # ── Batched generation ─────────────────────────────────────────────
        progress(0.1, desc="Generating predictions (Batched)…")
        predictions: list[str] = []
        model, tokenizer = _load_for_inference(
            model_name,
            eval_lora_path if eval_lora_path else None,
        )
        # BOLT OPTIMIZATION: Increased batch size to 8 to better utilize GPU
        # parallelism, matching the judge evaluation component.
        batch_size = 8

        for i in range(0, len(prompts), batch_size):
            if app_state.stop_event.is_set():
                break
            batch_prompts = prompts[i: i + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=int(eval_max_new_tokens),
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # BOLT OPTIMIZATION: Left-padding simplifies prompt stripping by
            # ensuring all responses start at input_ids.shape[1].
            # Using batch_decode instead of serial decode for faster processing.
            input_len = inputs["input_ids"].shape[1]
            batch_responses = tokenizer.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True
            )
            predictions.extend(batch_responses)

        # ── Metrics ────────────────────────────────────────────────────────
        metrics: dict = {}
        if references:
            progress(0.5, desc="Computing BLEU & ROUGE…")
            metrics.update(compute_bleu_rouge(predictions, references))
            if eval_run_bertscore:
                progress(0.65, desc="Computing BERTScore…")
                metrics.update(compute_bertscore_metric(predictions, references))

        # ── LLM-as-Judge ───────────────────────────────────────────────────
        judge_results: list[dict] = []
        if eval_use_judge and judge_model_name:
            progress(0.75, desc="Running LLM-as-Judge…")
            try:
                judge_results = llm_judge_evaluate(
                    prompts, predictions, judge_criteria, judge_model_name
                )
            except RuntimeError as judge_err:
                # H5 FIX: surface the judge failure as a clear warning in the
                # metrics string rather than silently polluting the result DataFrame.
                metrics["LLM-Judge-Error"] = str(judge_err)

        progress(1.0, desc="Done!")

        metrics_str = (
            "\n".join(f"**{k}:** {v}" for k, v in metrics.items())
            if metrics
            else "No reference data — skipped automatic metrics."
        )
        if judge_results:
            metrics_str += f"\n**LLM-as-Judge:** {len(judge_results)} examples evaluated."

        result_data: dict = {
            "prompt": prompts[:len(predictions)],
            "prediction": predictions,
        }
        if references:
            result_data["reference"] = references[:len(predictions)]
        if judge_results:
            result_data["judgment"] = [r["judgment"] for r in judge_results[:len(predictions)]]

        # F-6: Build the per-example HTML preview (safe — never raises)
        try:
            preview_html = build_prediction_preview_html(
                prompts=prompts[:len(predictions)],
                predictions=predictions,
                references=references[:len(predictions)] if references else [],
                n=3,
            )
        except Exception:
            preview_html = ""

        return metrics_str, pd.DataFrame(result_data), preview_html

    except Exception as e:
        return f"❌ Evaluation failed: {e}", pd.DataFrame(), ""
