
import time
import numpy as np
import pandas as pd
from datasets import Dataset

# Constants from the app
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
        if COL_PROMPT in df.columns and COL_CHOSEN in df.columns and COL_REJECTED in df.columns:
            _lengths = df[COL_PROMPT].astype(str).str.len() + \
                       df[COL_CHOSEN].astype(str).str.len() + \
                       df[COL_REJECTED].astype(str).str.len()
        elif COL_TEXT in df.columns:
            _lengths = df[COL_TEXT].astype(str).str.len()
        elif COL_INSTRUCTION in df.columns and COL_OUTPUT in df.columns:
            _lengths = df[COL_INSTRUCTION].astype(str).str.len() + \
                       df[COL_OUTPUT].astype(str).str.len()
        else:
            first_col = ds.column_names[0] if ds.column_names else None
            _lengths = df[first_col].astype(str).str.len() if (first_col and first_col in df.columns) else pd.Series(dtype=int)
    except Exception as e:
        print(f"Error in optimized_stats: {e}")
        _lengths = pd.Series([100] * len(ds))

    return float(_lengths.mean()) if not _lengths.empty else 0.0

def run_benchmark(name, ds):
    print(f"--- Benchmarking {name} on {len(ds)} rows ---")
    t0 = time.time()
    avg_orig = original_stats(ds)
    t1 = time.time()
    print(f"Original: {t1 - t0:.4f}s (avg: {avg_orig:.2f})")

    t0 = time.time()
    avg_opt = optimized_stats(ds)
    t1 = time.time()
    print(f"Optimized: {t1 - t0:.4f}s (avg: {avg_opt:.2f})")

    assert abs(avg_orig - avg_opt) < 1e-5, f"Mismatch in {name}: {avg_orig} vs {avg_opt}"

n_rows = 100_000

# Case 1: SFT (Instruction/Output)
ds_sft = Dataset.from_dict({
    COL_INSTRUCTION: ["Tell me a story about a " + str(i) for i in range(n_rows)],
    COL_OUTPUT: ["Once upon a time, there was a " + str(i) for i in range(n_rows)]
})
run_benchmark("SFT", ds_sft)

# Case 2: DPO (Prompt/Chosen/Rejected)
ds_dpo = Dataset.from_dict({
    COL_PROMPT: ["Question " + str(i) for i in range(n_rows)],
    COL_CHOSEN: ["Good answer " + str(i) for i in range(n_rows)],
    COL_REJECTED: ["Bad answer " + str(i) for i in range(n_rows)]
})
run_benchmark("DPO", ds_dpo)

# Case 3: Text only
ds_text = Dataset.from_dict({
    COL_TEXT: ["Just some text " + str(i) for i in range(n_rows)]
})
run_benchmark("Text", ds_text)

# Case 4: Custom column
ds_custom = Dataset.from_dict({
    "my_col": ["Custom content " + str(i) for i in range(n_rows)]
})
run_benchmark("Custom", ds_custom)
