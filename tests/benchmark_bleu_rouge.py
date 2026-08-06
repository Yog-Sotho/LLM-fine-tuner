import time
import os
from inference.evaluation import compute_bleu_rouge
from config.constants import HAS_NLTK, HAS_ROUGE

def test_benchmark_bleu_rouge_speed_and_correctness():
    # Prepare dummy predictions and references
    # 1000 items to show real multiprocessing speedup vs sequential
    predictions = [
        f"This is a dummy model generated text for sample {i}. It is supposed to contain several words and long sentences to simulate real LLM evaluation."
        for i in range(1000)
    ]
    references = [
        f"This is a dummy model reference text for sample {i}. It is supposed to contain several words and long sentences to simulate real LLM evaluation."
        for i in range(1000)
    ]

    print("\n" + "="*60)
    print("BLEU/ROUGE MULTIPROCESSING BENCHMARK")
    print("="*60)

    # 1. Warmup / Initial run (also checks correctness)
    # Run on a small slice to check sequential behavior (< 100 items)
    small_preds = predictions[:50]
    small_refs = references[:50]

    t0 = time.time()
    results_small = compute_bleu_rouge(small_preds, small_refs)
    t1 = time.time()
    print(f"Sequential small (50 items) time: {t1 - t0:.4f}s")
    print(f"Sequential small results: {results_small}")

    # 2. Large sequential run (bypass multiprocessing by mocking os.cpu_count to return 1)
    original_cpu_count = os.cpu_count
    os.cpu_count = lambda: 1

    t0 = time.time()
    seq_results = compute_bleu_rouge(predictions, references)
    t1 = time.time()
    seq_time = t1 - t0
    print(f"Sequential full (1000 items) time: {seq_time:.4f}s")
    print(f"Sequential results: {seq_results}")

    # Restore cpu_count to test multiprocessing
    os.cpu_count = original_cpu_count

    # 3. Multiprocessing run on the full 1000 items
    t0 = time.time()
    mp_results = compute_bleu_rouge(predictions, references)
    t1 = time.time()
    mp_time = t1 - t0
    print(f"Multiprocessing full (1000 items) time: {mp_time:.4f}s")
    print(f"Multiprocessing results: {mp_results}")

    # Calculate and report speedup if multiple cores are available
    cores = os.cpu_count() or 1
    if cores > 1:
        speedup = seq_time / mp_time if mp_time > 0 else 1.0
        print(f"Speedup achieved: {speedup:.2f}x (Cores: {cores})")
    else:
        print("Single core machine - sequential execution fallback was used.")

    # 4. Correctness Check
    # Verify that the metrics are exactly identical
    assert seq_results["BLEU-1"] == mp_results["BLEU-1"], "BLEU-1 scores differ!"
    assert seq_results["ROUGE-1"] == mp_results["ROUGE-1"], "ROUGE-1 scores differ!"
    assert seq_results["ROUGE-2"] == mp_results["ROUGE-2"], "ROUGE-2 scores differ!"
    assert seq_results["ROUGE-L"] == mp_results["ROUGE-L"], "ROUGE-L scores differ!"
    print("✅ Correctness check PASSED: sequential and multiprocessing outputs are identical.")
    print("="*60)

if __name__ == "__main__":
    test_benchmark_bleu_rouge_speed_and_correctness()
