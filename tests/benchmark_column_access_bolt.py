import time
from datasets import Dataset

num_rows = 100_000
ds = Dataset.from_dict({"text": ["hello world"] * num_rows})

def test_slow():
    start = time.time()
    _ = [str(x["text"]) for x in ds]
    return time.time() - start

def test_fast():
    start = time.time()
    _ = ds["text"]
    return time.time() - start

t_slow = test_slow()
t_fast = test_fast()

print(f"Slow: {t_slow:.4f}s")
print(f"Fast: {t_fast:.4f}s")
print(f"Speedup: {t_slow / t_fast:.2f}x")
