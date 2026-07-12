
import time
import pandas as pd
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

def filter_batched(ds):
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

def filter_pandas(ds):
    df = ds.to_pandas()
    mask = (
        (df["prompt"].astype(str).str.len() >= min_length) & (df["prompt"].astype(str).str.len() <= max_length) &
        (df["chosen"].astype(str).str.len() >= min_length) & (df["chosen"].astype(str).str.len() <= max_length) &
        (df["rejected"].astype(str).str.len() >= min_length) & (df["rejected"].astype(str).str.len() <= max_length)
    )
    df_filtered = df[mask].reset_index(drop=True)
    return Dataset.from_pandas(df_filtered, preserve_index=False)

print(f"Benchmarking filtering on {N} rows...")

start = time.time()
res1 = filter_batched(dataset)
end = time.time()
time_batched = end - start
print(f"Batched filter (dataset.filter): {time_batched:.4f}s")

start = time.time()
res2 = filter_pandas(dataset)
end = time.time()
time_pandas = end - start
print(f"Vectorized Pandas filter: {time_pandas:.4f}s")

print(f"Speedup: {time_batched / time_pandas:.2f}x")

assert len(res1) == len(res2)
print("Success!")
