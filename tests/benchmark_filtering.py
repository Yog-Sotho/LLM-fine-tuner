
import time
import os
from datasets import Dataset

# Simulate a large dataset
N = 100_000
data = {
    "instruction": ["Instruction " + str(i) for i in range(N)],
    "output": ["Output " + str(i) for i in range(N)]
}
# Add some empty rows
data["instruction"][500] = ""
data["output"][1000] = "  "

dataset = Dataset.from_dict(data)

def original_filter(ds):
    return ds.filter(
        lambda x: (
            len(str(x["instruction"]).strip()) > 0
            and len(str(x["output"]).strip()) > 0
        )
    )

def batched_filter(ds):
    def filter_fn(batch):
        return [
            len(str(i).strip()) > 0 and len(str(o).strip()) > 0
            for i, o in zip(batch["instruction"], batch["output"])
        ]
    return ds.filter(filter_fn, batched=True)

print(f"Benchmarking filtering on {N} rows...")

start = time.time()
res1 = original_filter(dataset)
end = time.time()
print(f"Original filter: {end - start:.4f}s")

start = time.time()
res2 = batched_filter(dataset)
end = time.time()
print(f"Batched filter: {end - start:.4f}s")

assert len(res1) == len(res2)
print("Success!")
