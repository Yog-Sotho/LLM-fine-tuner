import time
import pandas as pd
from datasets import Dataset

# Create a large dataset
n = 1000000
data = {"text": ["This is a test sentence."] * n}
ds = Dataset.from_dict(data)

print(f"Dataset size: {len(ds)}")

# Measure inefficient way
start = time.time()
_ = ds["text"][:5]
end = time.time()
print(f"Inefficient way (ds['text'][:5]): {end - start:.6f}s")

# Measure efficient way
start = time.time()
_ = ds[:5]["text"]
end = time.time()
print(f"Efficient way (ds[:5]['text']): {end - start:.6f}s")
