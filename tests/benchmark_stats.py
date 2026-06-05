
import time
import numpy as np
import pandas as pd
from datasets import Dataset
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from data.preprocessing import get_dataset_stats
from config.constants import COL_INSTRUCTION, COL_OUTPUT, COL_PROMPT, COL_CHOSEN, COL_REJECTED, COL_TEXT

def get_stats_slow(ds):
    try:
        if COL_PROMPT in ds.column_names and COL_CHOSEN in ds.column_names:
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
            first_col = ds.column_names[0] if ds.column_names else None
            _lengths = [len(str(v)) for v in ds[first_col]] if first_col else []
    except Exception:
        _lengths = [100] * len(ds)
    return float(np.mean(_lengths)) if _lengths else 0.0

def benchmark():
    n = 100000
    print(f"Generating {n} rows of dummy data...")
    data = {
        COL_INSTRUCTION: ["Instruction " + str(i) for i in range(n)],
        COL_OUTPUT: ["Output " + str(i) for i in range(n)]
    }
    ds = Dataset.from_dict(data)

    print("Running slow stats...")
    start = time.time()
    avg_slow = get_stats_slow(ds)
    end = time.time()
    slow_time = end - start
    print(f"Slow time: {slow_time:.4f}s, Result: {avg_slow}")

    print("Running optimized get_dataset_stats...")
    start = time.time()
    res = get_dataset_stats(ds)
    avg_fast = res["avg_length"]
    end = time.time()
    fast_time = end - start
    print(f"Fast time: {fast_time:.4f}s, Result: {avg_fast}")

    print(f"Speedup: {slow_time / fast_time:.1f}x")

    assert abs(avg_slow - avg_fast) < 1e-6
    print("Verification successful: results match!")

if __name__ == "__main__":
    benchmark()
