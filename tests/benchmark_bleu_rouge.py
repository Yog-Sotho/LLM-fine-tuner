"""
tests/benchmark_bleu_rouge.py
==============================
Measures performance and verifies correctness of multiprocessing vs sequential
BLEU and ROUGE evaluations.
"""

import time
import numpy as np
import os
from inference.evaluation import compute_bleu_rouge

def generate_dummy_data(n: int = 2000):
    """Generates synthetic predictions and references for evaluation."""
    base_refs = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "A journey of a thousand miles begins with a single step.",
        "All that glitters is not gold.",
        "Where there is a will, there is a way."
    ]
    base_preds = [
        "The fast brown fox leaped over the sleeping dog.",
        "To exist or not to exist, that is the query.",
        "A long trip of thousands of miles starts with one step.",
        "Everything that shines is not gold.",
        "If you have the will, you can find a way."
    ]

    predictions = []
    references = []
    for i in range(n):
        idx = i % len(base_refs)
        predictions.append(base_preds[idx])
        references.append(base_refs[idx])
    return predictions, references

def run_sequential_only(predictions: list[str], references: list[str]) -> dict:
    """Helper that runs sequential execution logic exactly as in evaluation.py."""
    results = {}
    from config.constants import HAS_NLTK, HAS_ROUGE

    has_nltk = bool(HAS_NLTK)
    has_rouge = bool(HAS_ROUGE)

    if has_nltk:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
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
            else:
                bleu_scores.append(0.0)
        results["BLEU-1"] = round(float(np.mean(bleu_scores)), 4) if bleu_scores else 0.0
    else:
        results["BLEU-1"] = "nltk not installed"

    if has_rouge:
        from rouge_score import rouge_scorer as rouge_scorer_lib
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

def test_multiprocessing_vs_sequential():
    """Verify correctness and performance."""
    print("Generating synthetic dataset (2000 items)...")
    preds, refs = generate_dummy_data(2000)

    print("Running sequential evaluation...")
    t0 = time.time()
    seq_results = run_sequential_only(preds, refs)
    t_seq = time.time() - t0
    print(f"Sequential took: {t_seq:.4f} seconds")
    print(f"Sequential results: {seq_results}")

    print("\\nRunning multiprocessing evaluation...")
    t0 = time.time()
    mp_results = compute_bleu_rouge(preds, refs)
    t_mp = time.time() - t0
    print(f"Multiprocessing took: {t_mp:.4f} seconds")
    print(f"Multiprocessing results: {mp_results}")

    # Assert correctness/equivalence
    for metric in ["BLEU-1", "ROUGE-1", "ROUGE-2", "ROUGE-L"]:
        if isinstance(seq_results[metric], float):
            assert seq_results[metric] == mp_results[metric], f"Mismatch for {metric}: {seq_results[metric]} vs {mp_results[metric]}"

    print("\\n✅ Correctness check PASSED: Sequential and Multiprocessing results are mathematically identical!")

    speedup = t_seq / t_mp if t_mp > 0 else 1.0
    print(f"⚡ Performance gain: {speedup:.2f}x speedup on {os.cpu_count()} cores.")

if __name__ == "__main__":
    test_multiprocessing_vs_sequential()
