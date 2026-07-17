
import time
import pandas as pd
import tempfile
import os
import sys
from datasets import Dataset

# Add current directory to path
sys.path.append(os.getcwd())

from data.loader import load_dataset_from_dataframe, load_dataset_from_file

def benchmark_refresh_bolt(n=100000):
    print(f"⚡ Bolt: Benchmarking refresh optimization for {n} rows...")
    df = pd.DataFrame({"text": ["hello world " * 10] * n})

    # Old method (simulated): write to CSV then read back
    t0 = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        dummy = type("_F", (), {"name": tmp.name})()
        ds_old = load_dataset_from_file(dummy, "csv")
        os.unlink(tmp.name)
    t1 = time.time()
    old_time = t1 - t0
    print(f"❌ Old I/O-based method: {old_time:.4f}s")

    # New optimized method: load directly from DataFrame
    t0 = time.time()
    ds_new = load_dataset_from_dataframe(df)
    t1 = time.time()
    new_time = t1 - t0
    print(f"⚡ New direct method: {new_time:.4f}s")

    speedup = old_time / new_time if new_time > 0 else float('inf')
    print(f"🚀 Speedup: {speedup:.2f}x")

    # Verify correctness
    assert len(ds_old) == len(ds_new)
    assert ds_old.column_names == ds_new.column_names
    print("✅ Correctness verified.")

if __name__ == "__main__":
    benchmark_refresh_bolt()
