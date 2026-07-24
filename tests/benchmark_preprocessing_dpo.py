import time
import pandas as pd
from datasets import Dataset
import os
import sys

# Add current directory to path so we can import data/preprocessing
sys.path.append(os.getcwd())

from data.preprocessing import validate_and_clean_dataset
from config.constants import COL_PROMPT, COL_CHOSEN, COL_REJECTED

def benchmark_dpo_dedup():
    print("Generating 100,000 rows of dummy DPO/preference data with duplicates...")
    data = {
        COL_PROMPT: ["Prompt " + str(i % 50000) for i in range(100000)],
        COL_CHOSEN: ["Chosen response " + str(i % 50000) for i in range(100000)],
        COL_REJECTED: ["Rejected response " + str(i % 50000) for i in range(100000)]
    }

    # The dataset has 50,000 unique rows, and each row is duplicated exactly once (total 100,000 rows)
    ds = Dataset.from_dict(data)

    print(f"Running validate_and_clean_dataset with DPO deduplication on {len(ds)} rows...")
    start_time = time.time()
    cleaned_ds, issues = validate_and_clean_dataset(ds, is_dpo=True)
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"Time taken to clean and deduplicate DPO dataset: {elapsed:.4f} seconds")
    print(f"Issues found: {issues}")
    print(f"Cleaned dataset size: {len(cleaned_ds)} (expected 50,000 rows)")

    # Assert correctness of our vectorized O(N) deduplication
    assert len(cleaned_ds) == 50000, f"Expected 50000 rows, got {len(cleaned_ds)}"
    print("Correctness check passed!")

if __name__ == "__main__":
    benchmark_dpo_dedup()
