
import time
import pandas as pd
from datasets import Dataset
import numpy as np

# Mock constants
COL_TEXT = "text"
COL_INSTRUCTION = "instruction"
COL_OUTPUT = "output"
COL_PROMPT = "prompt"
COL_CHOSEN = "chosen"
COL_REJECTED = "rejected"

def current_impl(dataset, is_dpo=False):
    issues = []
    valid_indices = []
    lengths_after_empty_removal = []

    if is_dpo:
        col_p = [str(x).strip() for x in dataset[COL_PROMPT]]
        col_c = [str(x).strip() for x in dataset[COL_CHOSEN]]
        col_r = [str(x).strip() for x in dataset[COL_REJECTED]]
        for idx, (p, c, r) in enumerate(zip(col_p, col_c, col_r)):
            if p and c and r:
                valid_indices.append(idx)
                lengths_after_empty_removal.append(len(p) + len(c) + len(r))
    elif COL_TEXT in dataset.column_names:
        col_t = [str(x).strip() for x in dataset[COL_TEXT]]
        for idx, t in enumerate(col_t):
            if t:
                valid_indices.append(idx)
                lengths_after_empty_removal.append(len(t))

    empty = len(dataset) - len(valid_indices)
    if empty:
        dataset = dataset.select(valid_indices)

    if COL_TEXT in dataset.column_names:
        texts = [str(t) for t in dataset[COL_TEXT]]
        seen = set()
        unique_indices = []
        for idx, text in enumerate(texts):
            if text not in seen:
                seen.add(text)
                unique_indices.append(idx)
        n_dups = len(texts) - len(unique_indices)
        if n_dups > 0:
            dataset = dataset.select(unique_indices)

    return dataset

def bolt_impl(dataset, is_dpo=False):
    df = dataset.to_pandas()
    if is_dpo:
        cols = [COL_PROMPT, COL_CHOSEN, COL_REJECTED]
        for col in cols:
            df[col] = df[col].astype(str).str.strip()
        mask = (df[COL_PROMPT] != "") & (df[COL_CHOSEN] != "") & (df[COL_REJECTED] != "")
        df = df[mask]
        df = df.drop_duplicates(subset=cols)
    elif COL_TEXT in dataset.column_names:
        df[COL_TEXT] = df[COL_TEXT].astype(str).str.strip()
        df = df[df[COL_TEXT] != ""]
        df = df.drop_duplicates(subset=[COL_TEXT])

    return Dataset.from_pandas(df, preserve_index=False)

# Test with 100k rows
n = 100_000
data = {
    COL_TEXT: ["  some text  "] * (n // 2) + [""] * (n // 4) + ["unique " + str(i) for i in range(n // 4)]
}
ds = Dataset.from_dict(data)

print(f"Testing with {n} rows...")

t0 = time.time()
_ = current_impl(ds)
print(f"Current implementation: {time.time() - t0:.4f}s")

t0 = time.time()
_ = bolt_impl(ds)
print(f"Bolt implementation (Pandas): {time.time() - t0:.4f}s")
