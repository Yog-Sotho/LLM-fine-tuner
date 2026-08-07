
import time
from data.augmentation import quality_filter_v27
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset
import numpy as np

# Simulate a large dataset
N = 1_000_000
data = {
    "prompt": ["Prompt " + str(i) for i in range(N)],
    "chosen": ["Chosen " + str(i) for i in range(N)],
    "rejected": ["Rejected " + str(i) for i in range(N)]
}
dataset = Dataset.from_dict(data)

min_length = 5
max_length = 2048

def filter_batched_original(ds):
    def filter_dpo(batch):
        col_p = batch.get("prompt", [""] * len(next(iter(batch.values()))))
        col_c = batch.get("chosen", [""] * len(col_p))
        col_r = batch.get("rejected", [""] * len(col_p))
        return [
            (min_length <= len(str(p)) <= max_length
             and min_length <= len(str(c)) <= max_length
             and min_length <= len(str(r)) <= max_length)
            for p, c, r in zip(col_p, col_c, col_r)
        ]
    return ds.filter(filter_dpo, batched=True)

def filter_batched_vectorized(ds):
    def filter_dpo(batch):
        # We use pd.Series for vectorization but keep it within the batch
        # to remain memory safe (only one batch at a time).
        p_len = pd.Series(batch.get("prompt", [])).astype(str).str.len()
        c_len = pd.Series(batch.get("chosen", [])).astype(str).str.len()
        r_len = pd.Series(batch.get("rejected", [])).astype(str).str.len()

        # Handle cases where columns might be missing
        if p_len.empty: p_len = pd.Series([min_length] * len(next(iter(batch.values()))))
        if c_len.empty: c_len = pd.Series([min_length] * len(p_len))
        if r_len.empty: r_len = pd.Series([min_length] * len(p_len))

        mask = (p_len >= min_length) & (p_len <= max_length) & \
               (c_len >= min_length) & (c_len <= max_length) & \
               (r_len >= min_length) & (r_len <= max_length)
        return mask.tolist()
    return ds.filter(filter_dpo, batched=True)

def filter_pandas_full(ds):
    df = ds.to_pandas()
    mask = (
        (df["prompt"].astype(str).str.len() >= min_length) & (df["prompt"].astype(str).str.len() <= max_length) &
        (df["chosen"].astype(str).str.len() >= min_length) & (df["chosen"].astype(str).str.len() <= max_length) &
        (df["rejected"].astype(str).str.len() >= min_length) & (df["rejected"].astype(str).str.len() <= max_length)
    )
    df_filtered = df[mask].reset_index(drop=True)
    return Dataset.from_pandas(df_filtered, preserve_index=False)

def filter_pyarrow_native(ds):
    filtered_ds, _ = quality_filter_v27(ds, min_length=min_length, max_length=max_length, is_dpo=True)
    return filtered_ds

print(f"Benchmarking filtering on {N} rows...")

start = time.time()
res1 = filter_batched_original(dataset)
end = time.time()
time_orig = end - start
print(f"Original Batched filter: {time_orig:.4f}s")

start = time.time()
res2 = filter_batched_vectorized(dataset)
end = time.time()
time_vec_batch = end - start
print(f"Vectorized Batched filter: {time_vec_batch:.4f}s")

start = time.time()
res3 = filter_pandas_full(dataset)
end = time.time()
time_full_pd = end - start
print(f"Full Pandas (OOM risk): {time_full_pd:.4f}s")

start = time.time()
res4 = filter_pyarrow_native(dataset)
end = time.time()
time_pyarrow = end - start
print(f"Native PyArrow filter (Our Optimization): {time_pyarrow:.4f}s")

print(f"PyArrow Speedup over Original Batched: {time_orig / time_pyarrow:.2f}x")
print(f"PyArrow Speedup over Vectorized Batched: {time_vec_batch / time_pyarrow:.2f}x")
print(f"PyArrow Speedup over Full Pandas: {time_full_pd / time_pyarrow:.2f}x")

assert len(res1) == len(res2) == len(res3) == len(res4)
print("Success!")
