import time
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset

# Mock constants
COL_PROMPT = "prompt"
COL_CHOSEN = "chosen"
COL_REJECTED = "rejected"
COL_TEXT = "text"
COL_INSTRUCTION = "instruction"
COL_OUTPUT = "output"

def get_stats_loop(ds):
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

def get_stats_vectorized(ds):
    df = ds.to_pandas()
    if COL_PROMPT in df.columns and COL_CHOSEN in df.columns:
        # Sum of lengths
        lengths = df[COL_PROMPT].astype(str).str.len() + \
                  df[COL_CHOSEN].astype(str).str.len() + \
                  df[COL_REJECTED].astype(str).str.len()
    elif COL_TEXT in df.columns:
        lengths = df[COL_TEXT].astype(str).str.len()
    elif COL_INSTRUCTION in df.columns and COL_OUTPUT in df.columns:
        lengths = df[COL_INSTRUCTION].astype(str).str.len() + \
                  df[COL_OUTPUT].astype(str).str.len()
    else:
        first_col = df.columns[0] if not df.empty else None
        lengths = df[first_col].astype(str).str.len() if first_col else pd.Series(dtype=float)

    return float(lengths.mean()) if not lengths.empty else 0.0

def get_stats_arrow(ds):
    table = ds.data
    col_names = ds.column_names
    if (COL_PROMPT in col_names and COL_CHOSEN in col_names):
        p_len = pc.fill_null(pc.utf8_length(pc.cast(table[COL_PROMPT], pa.string())), 0)
        c_len = pc.fill_null(pc.utf8_length(pc.cast(table[COL_CHOSEN], pa.string())), 0)
        r_len = pc.fill_null(pc.utf8_length(pc.cast(table[COL_REJECTED], pa.string())), 0)
        lengths = pc.add(pc.add(p_len, c_len), r_len)
    elif COL_TEXT in col_names:
        lengths = pc.fill_null(pc.utf8_length(pc.cast(table[COL_TEXT], pa.string())), 0)
    elif COL_INSTRUCTION in col_names and COL_OUTPUT in col_names:
        i_len = pc.fill_null(pc.utf8_length(pc.cast(table[COL_INSTRUCTION], pa.string())), 0)
        o_len = pc.fill_null(pc.utf8_length(pc.cast(table[COL_OUTPUT], pa.string())), 0)
        lengths = pc.add(i_len, o_len)
    else:
        first_col = col_names[0] if col_names else None
        if first_col:
            lengths = pc.fill_null(pc.utf8_length(pc.cast(table[first_col], pa.string())), 0)
        else:
            return 0.0
    mean_length = pc.mean(lengths).as_py()
    return float(mean_length) if mean_length is not None else 0.0

if __name__ == "__main__":
    # Create a large dataset
    N = 100_000
    print(f"Benchmarking with {N} rows...")
    data = {
        COL_PROMPT: ["prompt " * 10] * N,
        COL_CHOSEN: ["chosen " * 20] * N,
        COL_REJECTED: ["rejected " * 15] * N,
    }
    ds = Dataset.from_dict(data)

    t0 = time.time()
    avg_loop = get_stats_loop(ds)
    t1 = time.time()
    loop_time = t1 - t0
    print(f"Loop took: {loop_time:.4f}s, avg: {avg_loop}")

    t0 = time.time()
    avg_vec = get_stats_vectorized(ds)
    t1 = time.time()
    vec_time = t1 - t0
    print(f"Vectorized Pandas took: {vec_time:.4f}s, avg: {avg_vec}")

    t0 = time.time()
    avg_arrow = get_stats_arrow(ds)
    t1 = time.time()
    arrow_time = t1 - t0
    print(f"Vectorized Arrow took: {arrow_time:.4f}s, avg: {avg_arrow}")

    print(f"Arrow Speedup over Loop: {loop_time/arrow_time:.2f}x")
    print(f"Arrow Speedup over Pandas: {vec_time/arrow_time:.2f}x")
