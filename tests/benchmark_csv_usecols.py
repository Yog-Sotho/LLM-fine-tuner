import time
import pandas as pd
import tempfile
import os

def benchmark_csv_usecols(n=100000):
    print(f"⚡ Benchmarking CSV loading with usecols vs full read on {n} rows...")

    # Create a dummy DataFrame with many columns, some of which are very large
    data = {
        "prompt": ["This is a test prompt number " + str(i) for i in range(n)],
        "reference": ["This is a reference answer " + str(i) for i in range(n)],
        "large_col1": ["Very large text content " * 50 for _ in range(n)],
        "large_col2": ["Another large text content " * 50 for _ in range(n)],
        "unused_col": [i for i in range(n)]
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        file_path = tmp.name

    try:
        # Method 1: Full CSV read (current)
        t0 = time.perf_counter()
        full_df = pd.read_csv(file_path)
        _ = full_df["prompt"].astype(str).tolist()
        _ = full_df["reference"].astype(str).tolist() if "reference" in full_df.columns else []
        t1 = time.perf_counter()
        full_read_time = t1 - t0
        print(f"❌ Full read took: {full_read_time:.4f}s")

        # Method 2: Header first + usecols read (optimized)
        t0 = time.perf_counter()
        header_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
        usecols = ["prompt"]
        if "reference" in header_cols:
            usecols.append("reference")
        opt_df = pd.read_csv(file_path, usecols=usecols)
        _ = opt_df["prompt"].astype(str).tolist()
        _ = opt_df["reference"].astype(str).tolist() if "reference" in opt_df.columns else []
        t1 = time.perf_counter()
        opt_read_time = t1 - t0
        print(f"⚡ Optimized read took: {opt_read_time:.4f}s")

        speedup = full_read_time / opt_read_time if opt_read_time > 0 else float('inf')
        print(f"🚀 Speedup: {speedup:.2f}x")

    finally:
        os.unlink(file_path)

if __name__ == "__main__":
    benchmark_csv_usecols()
