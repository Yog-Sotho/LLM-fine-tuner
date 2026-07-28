import time
import pandas as pd
import tempfile
import os
import sys
from datasets import Dataset

# Add current directory to path
sys.path.append(os.getcwd())

from data.loader import load_dataset_from_dataframe, load_dataset_from_file

def benchmark_file_upload_bolt(n=100000):
    print(f"⚡ Bolt: Benchmarking file upload optimization for {n} rows...")

    # Create dummy DataFrame
    df = pd.DataFrame({
        "instruction": ["What is the capital of France? " * 3] * n,
        "output": ["The capital is Paris. " * 3] * n
    })

    # Prepare file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        file_path = tmp.name

    dummy_file = type("_F", (), {"name": file_path})()

    # 1. Old method: Load via load_dataset_from_file, then read_csv AGAIN from file
    t0 = time.time()
    ds_old = load_dataset_from_file(dummy_file, "csv")
    raw_df_old = pd.read_csv(dummy_file.name)
    t1 = time.time()
    old_time = t1 - t0
    print(f"❌ Old redundant read method: {old_time:.4f}s")

    # 2. Optimized method: Read CSV once, and load dataset from the loaded DataFrame
    t0 = time.time()
    raw_df_new = pd.read_csv(dummy_file.name)
    ds_new = load_dataset_from_dataframe(raw_df_new)
    t1 = time.time()
    new_time = t1 - t0
    print(f"⚡ New single-pass read method: {new_time:.4f}s")

    speedup = old_time / new_time if new_time > 0 else float('inf')
    print(f"🚀 Speedup: {speedup:.2f}x")

    # Clean up
    os.unlink(file_path)

    # Verify correctness
    assert len(ds_old) == len(ds_new)
    assert ds_old.column_names == ds_new.column_names
    assert len(raw_df_old) == len(raw_df_new)
    print("✅ Correctness and logical parity verified.")

if __name__ == "__main__":
    benchmark_file_upload_bolt()
