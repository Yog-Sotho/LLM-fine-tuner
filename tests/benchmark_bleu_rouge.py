import time
import numpy as np
from inference.evaluation import compute_bleu_rouge, _evaluate_chunk

def main():
    # Setup dummy data of size 500
    N = 500
    print(f"Generating {N} dummy prediction-reference pairs...")

    # Simple sentences that have varying overlaps to get meaningful scores
    predictions = [
        f"the quick brown fox jumps over the lazy dog {i}" for i in range(N)
    ]
    references = [
        f"the quick brown fox jumps over the lazy dog {i}" if i % 2 == 0
        else f"a fast brown fox jumped over a lazy dog {i}" for i in range(N)
    ]

    print("Benchmarking sequential vs parallel evaluation...")

    # 1. Benchmark Sequential path using _evaluate_chunk directly
    start_seq = time.time()
    b_seq, r1_seq, r2_seq, rl_seq = _evaluate_chunk(predictions, references, True, True)

    # Format sequential results identically
    seq_res = {
        "BLEU-1": round(float(np.mean(b_seq)), 4) if b_seq else 0.0,
        "ROUGE-1": round(float(np.mean(r1_seq)), 4) if r1_seq else 0.0,
        "ROUGE-2": round(float(np.mean(r2_seq)), 4) if r2_seq else 0.0,
        "ROUGE-L": round(float(np.mean(rl_seq)), 4) if rl_seq else 0.0,
    }
    seq_time = time.time() - start_seq
    print(f"Sequential evaluation completed in {seq_time:.4f}s")
    print("Sequential results:", seq_res)

    # 2. Benchmark Parallel path (using compute_bleu_rouge which triggers ProcessPoolExecutor since N >= 100)
    start_par = time.time()
    par_res = compute_bleu_rouge(predictions, references)
    par_time = time.time() - start_par
    print(f"Parallel evaluation completed in {par_time:.4f}s")
    print("Parallel results:  ", par_res)

    # 3. Assert correctness parity
    for key in seq_res:
        assert seq_res[key] == par_res[key], f"Parity mismatch on {key}: Sequential={seq_res[key]}, Parallel={par_res[key]}"

    print("\nParity verification PASSED! Scores are exactly identical.")
    if par_time < seq_time:
        speedup = seq_time / par_time
        print(f"Multiprocessing Speedup: {speedup:.2f}x faster!")
    else:
        print("Note: On small environments or systems with high spawn overhead, multiprocessing overhead may exceed execution gains.")

if __name__ == "__main__":
    main()
