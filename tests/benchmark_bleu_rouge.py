import os
import time
from unittest.mock import patch
from inference.evaluation import compute_bleu_rouge, HAS_NLTK, HAS_ROUGE

def run_benchmark():
    # 1. Create dummy data
    N = 1000
    print(f"Creating {N} dummy prediction-reference pairs...")

    # Let's generate repetitive but valid sentences
    predictions = [f"This is a dummy generated text prediction sentence number {i} with some words." for i in range(N)]
    references = [f"This is a dummy generated text reference sentence number {i} with some other words." for i in range(N)]

    print(f"NLTK Installed: {HAS_NLTK}")
    print(f"ROUGE Installed: {HAS_ROUGE}")
    print(f"Available CPU Cores: {os.cpu_count()}")

    # 2. Sequential Run (Force num_cores = 1 via mock)
    print("\nRunning BLEU/ROUGE calculation sequentially...")
    t0 = time.time()
    with patch("os.cpu_count", return_value=1):
        res_seq = compute_bleu_rouge(predictions, references)
    t_seq = time.time() - t0
    print(f"Sequential took: {t_seq:.4f}s")
    print(f"Sequential Results: {res_seq}")

    # 3. Multiprocessing Run
    print("\nRunning BLEU/ROUGE calculation with multiprocessing...")
    t0 = time.time()
    res_mp = compute_bleu_rouge(predictions, references)
    t_mp = time.time() - t0
    print(f"Multiprocessing took: {t_mp:.4f}s")
    print(f"Multiprocessing Results: {res_mp}")

    # 4. Verify parity
    print("\nVerifying results parity...")
    assert res_seq == res_mp, f"Parity mismatch! Sequential: {res_seq}, MP: {res_mp}"
    print("✅ Parity verified! Exactly identical results achieved.")

    if t_mp > 0:
        speedup = t_seq / t_mp
        print(f"\n⚡ Speedup: {speedup:.2f}x")
    else:
        print("\n⚡ Speedup: N/A")

if __name__ == "__main__":
    run_benchmark()
