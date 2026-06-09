import time
import pandas as pd
from datasets import Dataset
import os
import sys
import numpy as np

# Add current directory to path so we can import data.preprocessing
sys.path.append(os.getcwd())

from data.preprocessing import get_dataset_stats
from config.constants import COL_INSTRUCTION, COL_OUTPUT

def benchmark():
    num_rows = 100000
    print(f"Generating {num_rows} rows of dummy data...")
    data = {
        COL_INSTRUCTION: ["Instruction " + str(i) for i in range(num_rows)],
        COL_OUTPUT: ["Output " + str(i) for i in range(num_rows)]
    }
    ds = Dataset.from_dict(data)

    print(f"Running NEW vectorized get_dataset_stats on {len(ds)} rows...")
    start_time = time.time()
    stats = get_dataset_stats(ds)
    new_time = time.time() - start_time
    print(f"New time: {new_time:.4f} seconds")
    print(f"Stats: {stats}")

    print(f"Running OLD sequential loop on {len(ds)} rows...")
    start_time = time.time()
    try:
        if COL_INSTRUCTION in ds.column_names and COL_OUTPUT in ds.column_names:
            _lengths = [
                len(str(i)) + len(str(o))
                for i, o in zip(ds[COL_INSTRUCTION], ds[COL_OUTPUT])
            ]
        else:
            first_col = ds.column_names[0] if ds.column_names else None
            _lengths = [len(str(v)) for v in ds[first_col]] if first_col else []
    except Exception:
        _lengths = [100] * len(ds)

    old_stats = {
        "num_examples": len(ds),
        "avg_length": float(np.mean(_lengths)) if _lengths else 0.0,
    }
    old_time = time.time() - start_time
    print(f"Old time: {old_time:.4f} seconds")
    print(f"Old Stats: {old_stats}")

    print(f"\nSpeedup: {old_time / new_time:.1f}x")

if __name__ == "__main__":
    benchmark()
