import time
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer as rouge_scorer_lib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def _compute_bleu_rouge_chunk(preds_chunk, refs_chunk, has_nltk, has_rouge):
    bleu_scores = []
    r1_scores = []
    r2_scores = []
    rl_scores = []

    if has_nltk and preds_chunk and refs_chunk:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # lazy
        smoothing = SmoothingFunction().method4
        for pred, ref in zip(preds_chunk, refs_chunk):
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]
            if pred_tokens:
                try:
                    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                    bleu_scores.append(score)
                except Exception:
                    bleu_scores.append(0.0)

    if has_rouge and preds_chunk and refs_chunk:
        from rouge_score import rouge_scorer as rouge_scorer_lib  # lazy
        scorer = rouge_scorer_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        for pred, ref in zip(preds_chunk, refs_chunk):
            try:
                scores = scorer.score(ref, pred)
                r1_scores.append(scores["rouge1"].fmeasure)
                r2_scores.append(scores["rouge2"].fmeasure)
                rl_scores.append(scores["rougeL"].fmeasure)
            except Exception:
                r1_scores.append(0.0)
                r2_scores.append(0.0)
                rl_scores.append(0.0)

    return bleu_scores, r1_scores, r2_scores, rl_scores

def compute_mp(predictions, references):
    num_workers = multiprocessing.cpu_count()
    chunk_size = max(1, len(predictions) // num_workers)

    chunks = []
    for i in range(0, len(predictions), chunk_size):
        chunks.append((predictions[i:i+chunk_size], references[i:i+chunk_size]))

    all_bleu, all_r1, all_r2, all_rl = [], [], [], []

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(mp_context=ctx, max_workers=num_workers) as executor:
        futures = [executor.submit(_compute_bleu_rouge_chunk, preds, refs, True, True) for preds, refs in chunks]
        for fut in futures:
            b, r1, r2, rl = fut.result()
            all_bleu.extend(b)
            all_r1.extend(r1)
            all_r2.extend(r2)
            all_rl.extend(rl)

    return {
        "BLEU-1": round(float(np.mean(all_bleu)), 4) if all_bleu else 0.0,
        "ROUGE-1": round(float(np.mean(all_r1)), 4) if all_r1 else 0.0,
        "ROUGE-2": round(float(np.mean(all_r2)), 4) if all_r2 else 0.0,
        "ROUGE-L": round(float(np.mean(all_rl)), 4) if all_rl else 0.0
    }

def compute_seq(predictions, references):
    results = {}
    smoothing = SmoothingFunction().method4
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = [ref.split()]
        if pred_tokens:
            try:
                score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(score)
            except Exception:
                bleu_scores.append(0.0)
    results["BLEU-1"] = round(float(np.mean(bleu_scores)), 4) if bleu_scores else 0.0

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

    results["ROUGE-1"] = round(float(np.mean(r1_scores)), 4) if r1_scores else 0.0
    results["ROUGE-2"] = round(float(np.mean(r2_scores)), 4) if r2_scores else 0.0
    results["ROUGE-L"] = round(float(np.mean(rl_scores)), 4) if rl_scores else 0.0
    return results

if __name__ == "__main__":
    # Generate dummy predictions and references with some empty rows!
    preds = [
        "this is a sample prediction sentence which we will use to test bleu and rouge performance " * 5
        for _ in range(1000)
    ]
    refs = [
        "this is a sample reference sentence which we will use to test bleu and rouge score " * 5
        for _ in range(1000)
    ]

    # Add empty rows
    preds[10] = ""
    preds[20] = "   "
    preds[30] = ""

    print("Benchmarking Sequential on 1000 items (with empty rows)...")
    t0 = time.time()
    res_seq = compute_seq(preds, refs)
    seq_time = time.time() - t0
    print(f"Sequential took: {seq_time:.4f}s")
    print("Seq Results:", res_seq)

    print("\nBenchmarking Multiprocessed on 1000 items (with empty rows)...")
    t0 = time.time()
    res_mp = compute_mp(preds, refs)
    mp_time = time.time() - t0
    print(f"Multiprocessed took: {mp_time:.4f}s")
    print("MP Results:", res_mp)

    print(f"\nSpeedup: {seq_time / mp_time:.2f}x")
    assert res_seq == res_mp, "Results differ!"
    print("Exact parity verified with empty rows!")
