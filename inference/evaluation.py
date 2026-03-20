"""
inference/evaluation.py
========================
Layer 4 — automated evaluation: BLEU, ROUGE, BERTScore, LLM-as-Judge.
Imports: config.constants, core.state, inference.generate, stdlib, pandas, gradio.

Functions
---------
compute_bleu_rouge         — compute BLEU-1 and ROUGE-1/2/L scores
compute_bertscore_metric   — compute BERTScore precision / recall / F1
llm_judge_evaluate         — score responses with a local LLM judge
on_evaluate_click          — Gradio UI handler for the Evaluation tab Run button

Fix log
-------
  H5 (High): llm_judge_evaluate previously caught any exception and appended
     a dict with `"prompt": "ERROR"` to the results list. A completely broken
     judge (wrong path, OOM, CUDA error) silently contaminated the output
     DataFrame with a phantom row indistinguishable from real results.
     The function now raises RuntimeError on failure so on_evaluate_click
     can surface a clear error message to the user.
"""

import pandas as pd
import gradio as gr
import torch

from config.constants import (
    HAS_NLTK,
    HAS_ROUGE,
    HAS_BERTSCORE,
    LLM_JUDGE_CRITERIA,
)
from core.state import app_state
from inference.generate import _load_for_inference


# ── BLEU + ROUGE ───────────────────────────────────────────────────────────

def compute_bleu_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute BLEU-1, ROUGE-1, ROUGE-2, ROUGE-L over paired lists.

    Falls back to string placeholders when optional deps are missing.
    """
    import numpy as np  # lazy — available wherever torch is

    results = {}

    if HAS_NLTK and predictions and references:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # lazy

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
    with `"prompt": "ERROR"` — a broken judge silently contaminated the
    output DataFrame. Now raises RuntimeError so the caller can report
    the failure cleanly instead of mixing error rows with real results.

    Raises
    ------
    RuntimeError — if the judge model cannot be loaded or inference fails.
    """
    # H5 FIX: load the model outside the per-example loop so a load failure
    # raises immediately rather than producing a partial + error result.
    model, tokenizer = _load_for_inference(judge_model_name, judge_lora_path)

    results = []
    for prompt, response in zip(prompts, responses):
        eval_prompt = (
            f"Evaluate the following response based on: {criteria}\n"
            f"Prompt: {prompt}\nResponse: {response}\n"
            f"Score (1-10) and brief reasoning:"
        )
        inputs = tokenizer(
            eval_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        judgment = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        results.append({"prompt": prompt, "response": response, "judgment": judgment})

    return results


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
    eval_max_new_tokens: int = 150,  # Minor Fix 6: was hardcoded
    progress=gr.Progress(),
):
    """Handler for the Evaluation tab Run button.

    Loads the evaluation dataset (CSV or JSONL with 'prompt' / 'reference'),
    generates predictions in batches, then computes all requested metrics.

    Returns (metrics_str, result_dataframe).
    """
    model_name = eval_custom_model.strip() if eval_custom_model.strip() else eval_model_name
    if not model_name:
        return "❌ Please select a model.", pd.DataFrame()
    if eval_file is None:
        return (
            "❌ Please upload a test dataset (CSV with 'prompt' and 'reference' columns).",
            pd.DataFrame(),
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
            )

        if "prompt" not in eval_df.columns:
            return (
                f"❌ Dataset must have a 'prompt' column. Found: {list(eval_df.columns)}",
                pd.DataFrame(),
            )

        prompts    = eval_df["prompt"].astype(str).tolist()
        references = eval_df["reference"].astype(str).tolist() if "reference" in eval_df.columns else []

        # ── Batched generation ─────────────────────────────────────────────
        progress(0.1, desc="Generating predictions (Batched)…")
        predictions = []
        model, tokenizer = _load_for_inference(
            model_name,
            eval_lora_path if eval_lora_path else None,
        )
        batch_size = 4

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
            # CRITICAL FIX #3: Strip prompt tokens via attention mask lengths,
            # not by slicing at a fixed offset.
            input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            for idx, gen_ids in enumerate(outputs):
                gen_len   = gen_ids.shape[0]
                input_len = input_lengths[idx]
                response_ids = gen_ids[input_len:] if input_len < gen_len else gen_ids
                predictions.append(
                    tokenizer.decode(response_ids, skip_special_tokens=True)
                )

        # ── Metrics ────────────────────────────────────────────────────────
        metrics = {}
        if references:
            progress(0.5, desc="Computing BLEU & ROUGE…")
            metrics.update(compute_bleu_rouge(predictions, references))
            if eval_run_bertscore:
                progress(0.65, desc="Computing BERTScore…")
                metrics.update(compute_bertscore_metric(predictions, references))

        # ── LLM-as-Judge ───────────────────────────────────────────────────
        judge_results = []
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

        return metrics_str, pd.DataFrame(result_data)

    except Exception as e:
        return f"❌ Evaluation failed: {e}", pd.DataFrame()
