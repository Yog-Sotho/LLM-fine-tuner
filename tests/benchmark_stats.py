import time
import numpy as np
import pandas as pd
from datasets import Dataset

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
    except Exception:
        _lengths = [100] * len(ds)
    return float(np.mean(_lengths)) if _lengths else 0.0

def optimized_stats(ds):
    try:
        df = ds.to_pandas()
        if COL_PROMPT in df.columns and COL_CHOSEN in df.columns:
            _lengths = df[COL_PROMPT].astype(str).str.len() + \
                       df[COL_CHOSEN].astype(str).str.len() + \
                       df[COL_REJECTED].astype(str).str.len()
        elif COL_TEXT in df.columns:
            _lengths = df[COL_TEXT].astype(str).str.len()
        elif COL_INSTRUCTION in df.columns and COL_OUTPUT in df.columns:
            _lengths = df[COL_INSTRUCTION].astype(str).str.len() + \
                       df[COL_OUTPUT].astype(str).str.len()
        else:
            first_col = df.columns[0] if not df.empty else None
            _lengths = df[first_col].astype(str).str.len() if first_col else pd.Series(dtype=float)
    except Exception:
        return 0.0
    return float(_lengths.mean()) if not _lengths.empty else 0.0

# Setup dummy data
N = 100000
data = {
    COL_PROMPT: ["prompt " * 10] * N,
    COL_CHOSEN: ["chosen " * 20] * N,
    COL_REJECTED: ["rejected " * 15] * N
}
ds = Dataset.from_dict(data)

print(f"Benchmarking with {N} examples (DPO format)...")

start = time.time()
avg1 = original_stats(ds)
end = time.time()
print(f"Original: {end - start:.4f}s, avg_len={avg1}")

start = time.time()
avg2 = optimized_stats(ds)
end = time.time()
print(f"Optimized: {end - start:.4f}s, avg_len={avg2}")

# SFT format
data_sft = {
    COL_INSTRUCTION: ["instruction " * 10] * N,
    COL_OUTPUT: ["output " * 20] * N
}
ds_sft = Dataset.from_dict(data_sft)

print(f"\nBenchmarking with {N} examples (SFT format)...")

start = time.time()
avg1 = original_stats(ds_sft)
end = time.time()
print(f"Original: {end - start:.4f}s, avg_len={avg1}")

start = time.time()
avg2 = optimized_stats(ds_sft)
end = time.time()
print(f"Optimized: {end - start:.4f}s, avg_len={avg2}")
