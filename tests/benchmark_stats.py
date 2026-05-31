
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

def benchmark_stats():
    N = 100000
    print(f"Generating {N} rows of dummy data...")
    data = {
        COL_INSTRUCTION: ["Instruction " + str(i) for i in range(N)],
        COL_OUTPUT: ["Output " + str(i) for i in range(N)]
    }
    ds = Dataset.from_dict(data)

    print("Benchmarking original loop-based stats calculation...")
    start = time.time()
    # Original logic from ui/handlers.py
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
        print(f"Error: {e}")
        _lengths = [100] * len(ds)

    avg_len = float(np.mean(_lengths)) if _lengths else 0.0
    end = time.time()
    print(f"Original time: {end - start:.4f}s, Avg len: {avg_len}")

    print("Benchmarking optimized Pandas-based stats calculation...")
    start = time.time()
    # Optimized logic
    df = ds.to_pandas()
    if COL_PROMPT in df.columns and COL_CHOSEN in df.columns:
        _lengths_vec = (
            df[COL_PROMPT].astype(str).str.len() +
            df[COL_CHOSEN].astype(str).str.len() +
            df[COL_REJECTED].astype(str).str.len()
        )
    elif COL_TEXT in df.columns:
        _lengths_vec = df[COL_TEXT].astype(str).str.len()
    elif COL_INSTRUCTION in df.columns and COL_OUTPUT in df.columns:
        _lengths_vec = (
            df[COL_INSTRUCTION].astype(str).str.len() +
            df[COL_OUTPUT].astype(str).str.len()
        )
    else:
        first_col = df.columns[0] if not df.empty else None
        _lengths_vec = df[first_col].astype(str).str.len() if first_col else pd.Series(dtype=int)

    avg_len_vec = float(_lengths_vec.mean()) if not _lengths_vec.empty else 0.0
    end = time.time()
    print(f"Optimized time: {end - start:.4f}s, Avg len: {avg_len_vec}")

    assert np.isclose(avg_len, avg_len_vec)
    print("Success! Logic is identical.")

if __name__ == "__main__":
    benchmark_stats()
