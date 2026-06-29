import time
from datasets import Dataset

# Create a dataset
n = 100000
data = {"text": ["This is a test sentence. " * 10] * n}
ds = Dataset.from_dict(data)

print(f"Dataset size: {len(ds)}")

# Measure row-wise access
start = time.time()
_ = [str(x["text"]) for x in ds]
end = time.time()
print(f"Row-wise access ([str(x['text']) for x in ds]): {end - start:.6f}s")

# Measure columnar access
start = time.time()
_ = list(ds["text"])
end = time.time()
print(f"Columnar access (list(ds['text'])): {end - start:.6f}s")
