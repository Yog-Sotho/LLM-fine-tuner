
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

def filter_batched_vectorized_large(ds):
    def filter_dpo(batch):
        p_len = pd.Series(batch.get("prompt", [])).astype(str).str.len()
        c_len = pd.Series(batch.get("chosen", [])).astype(str).str.len()
        r_len = pd.Series(batch.get("rejected", [])).astype(str).str.len()

        mask = (p_len >= min_length) & (p_len <= max_length) & \
               (c_len >= min_length) & (c_len <= max_length) & \
               (r_len >= min_length) & (r_len <= max_length)
        return mask.tolist()
    return ds.filter(filter_dpo, batched=True, batch_size=100_000)

print(f"Benchmarking filtering on {N} rows...")

start = time.time()
res = filter_batched_vectorized_large(dataset)
end = time.time()
print(f"Vectorized Batched (batch_size=100k): {end - start:.4f}s")
