import time
import numpy as np
from datasets import Dataset

# Create a dummy dataset with 100k rows
data = {"text": ["This is a test sentence number " + str(i) for i in range(100000)]}
ds = Dataset.from_dict(data)

print(f"Dataset size: {len(ds)}")

# Method 1: Row-by-row iteration
t0 = time.time()
texts_1 = [str(x["text"]) for x in ds]
t1 = time.time()
print(f"Row-by-row iteration: {t1 - t0:.4f}s")

# Method 2: Column access (optimized)
t0 = time.time()
texts_2 = [str(t) for t in ds["text"]]
t1 = time.time()
print(f"Column access: {t1 - t0:.4f}s")

# Method 3: Direct column access (even faster if we don't need str())
t0 = time.time()
texts_3 = ds["text"]
t1 = time.time()
print(f"Direct column access: {t1 - t0:.4f}s")

# Method 4: Pandas vectorized (for stats)
t0 = time.time()
import pandas as pd
df = ds.to_pandas()
lengths = df["text"].astype(str).str.len()
avg_len = lengths.mean()
t1 = time.time()
print(f"Pandas vectorized stats: {t1 - t0:.4f}s (avg_len={avg_len})")

# Method 5: Python loop for stats (current)
t0 = time.time()
_lengths = [len(str(t)) for t in ds["text"]]
avg_len_2 = np.mean(_lengths)
t1 = time.time()
print(f"Python loop for stats: {t1 - t0:.4f}s (avg_len={avg_len_2})")

assert texts_1 == texts_2
