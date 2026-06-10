import time
import pandas as pd
import numpy as np
from datasets import Dataset
import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.constants import COL_PROMPT, COL_CHOSEN, COL_REJECTED, COL_TEXT, COL_INSTRUCTION, COL_OUTPUT
from data.preprocessing import get_dataset_stats

def slow_stats(ds):
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

    return {
        "num_examples": len(ds),
        "avg_length": float(np.mean(_lengths)) if _lengths else 0.0,
    }

def fast_stats(ds, is_dpo=False):
    return get_dataset_stats(ds, is_dpo=is_dpo)

def main():
    n = 100_000
    print(f"Generating synthetic dataset with {n} rows...")
    data = {
        COL_INSTRUCTION: ["Instruction " + str(i) for i in range(n)],
        COL_OUTPUT: ["Output " + str(i) for i in range(n)]
    }
    ds = Dataset.from_dict(data)

    print("Running slow_stats...")
    start = time.time()
    res_slow = slow_stats(ds)
    end = time.time()
    slow_time = end - start
    print(f"Slow stats took {slow_time:.4f}s: {res_slow}")

    print("Running fast_stats...")
    start = time.time()
    res_fast = fast_stats(ds)
    end = time.time()
    fast_time = end - start
    print(f"Fast stats took {fast_time:.4f}s: {res_fast}")

    print(f"Speedup: {slow_time / fast_time:.2f}x")

if __name__ == "__main__":
    main()
