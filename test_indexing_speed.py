
import time
from datasets import Dataset
import numpy as np

# Create a large-ish dataset (100k rows)
n = 100000
ds = Dataset.from_dict({"text": ["some text " + str(i) for i in range(n)]})

print(f"Dataset size: {len(ds)}")

# Method 1: column then slice
t0 = time.time()
for _ in range(10):
    _ = ds["text"][:5]
t1 = time.time()
print(f"Method 1 (ds['text'][:5]): {(t1-t0)/10:.6f}s")

# Method 2: slice then column
t0 = time.time()
for _ in range(10):
    _ = ds[:5]["text"]
t1 = time.time()
print(f"Method 2 (ds[:5]['text']): {(t1-t0)/10:.6f}s")
