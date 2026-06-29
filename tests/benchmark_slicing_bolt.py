import time
import pandas as pd
from datasets import Dataset

# Create a large dummy dataset
num_rows = 1_000_000
data = {"text": ["This is a test sentence."] * num_rows}
ds = Dataset.from_dict(data)

def test_inefficient():
    start = time.time()
    _ = ds["text"][:10]
    return time.time() - start

def test_efficient():
    start = time.time()
    _ = ds[:10]["text"]
    return time.time() - start

# Warmup
test_inefficient()
test_efficient()

t_inefficient = sum(test_inefficient() for _ in range(10)) / 10
t_efficient = sum(test_efficient() for _ in range(10)) / 10

print(f"Inefficient (ds[COL][:10]): {t_inefficient:.6f}s")
print(f"Efficient (ds[:10][COL]):   {t_efficient:.6f}s")
print(f"Speedup: {t_inefficient / t_efficient:.2f}x")
