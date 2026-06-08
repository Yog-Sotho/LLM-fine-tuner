import time
from datasets import Dataset

# Create a dummy dataset with 100k rows
data = {"text": ["This is a test sentence number " + str(i) for i in range(100000)]}
ds = Dataset.from_dict(data)

# Method 1: Row-by-row iteration (current)
t0 = time.time()
texts_1 = [str(x["text"]) for x in ds]
t1 = time.time()
print(f"Row-by-row iteration: {t1 - t0:.4f}s")

# Method 2: Column access (optimized)
t0 = time.time()
texts_2 = [str(t) for t in ds["text"]]
t1 = time.time()
print(f"Column access: {t1 - t0:.4f}s")

# Verify they are the same
assert texts_1 == texts_2
