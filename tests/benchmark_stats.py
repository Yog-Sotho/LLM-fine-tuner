import time
import numpy as np
import pandas as pd
from datasets import Dataset
import sys
import os

# Mock constants
COL_PROMPT = "prompt"
COL_CHOSEN = "chosen"
COL_REJECTED = "rejected"
COL_TEXT = "text"
COL_INSTRUCTION = "instruction"
COL_OUTPUT = "output"

def original_stats(ds):
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
    except Exception as e:
        print(f"Error in original: {e}")
        _lengths = [100] * len(ds)
    return _lengths

def optimized_stats(ds):
    try:
        df_stats = ds.to_pandas()
        if COL_PROMPT in df_stats.columns and COL_CHOSEN in df_stats.columns:
            _lengths = (
                df_stats[COL_PROMPT].astype(str).str.len() +
                df_stats[COL_CHOSEN].astype(str).str.len() +
                df_stats[COL_REJECTED].astype(str).str.len()
            ).tolist()
        elif COL_TEXT in df_stats.columns:
            _lengths = df_stats[COL_TEXT].astype(str).str.len().tolist()
        elif COL_INSTRUCTION in df_stats.columns and COL_OUTPUT in df_stats.columns:
            _lengths = (
                df_stats[COL_INSTRUCTION].astype(str).str.len() +
                df_stats[COL_OUTPUT].astype(str).str.len()
            ).tolist()
        else:
            first_col = df_stats.columns[0] if not df_stats.empty else None
            _lengths = df_stats[first_col].astype(str).str.len().tolist() if first_col else []
    except Exception as e:
        print(f"Error in optimized: {e}")
        _lengths = [100] * len(ds)
    return _lengths

def run_benchmark(n_rows=100000):
    print(f"Generating dataset with {n_rows} rows...")
    data = {
        COL_PROMPT: ["Instruction " * 10] * n_rows,
        COL_CHOSEN: ["Response chosen " * 20] * n_rows,
        COL_REJECTED: ["Response rejected " * 20] * n_rows,
    }
    ds = Dataset.from_dict(data)

    print("Benchmarking original loop-based stats...")
    start = time.time()
    res1 = original_stats(ds)
    end = time.time()
    time1 = end - start
    print(f"Original took: {time1:.4f}s")

    print("Benchmarking optimized Pandas-based stats...")
    start = time.time()
    res2 = optimized_stats(ds)
    end = time.time()
    time2 = end - start
    print(f"Optimized took: {time2:.4f}s")

    print(f"Speedup: {time1/time2:.2f}x")

    assert len(res1) == len(res2)
    # Check a few values
    for i in range(min(5, n_rows)):
        assert res1[i] == res2[i], f"Mismatch at index {i}: {res1[i]} != {res2[i]}"
    print("Verification successful!")

if __name__ == "__main__":
    n = 100000
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    run_benchmark(n)
