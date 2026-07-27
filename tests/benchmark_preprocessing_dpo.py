import time
import pandas as pd
from datasets import Dataset
import os
import sys

# Add current directory to path so we can import data.preprocessing
sys.path.append(os.getcwd())

from data.preprocessing import validate_and_clean_dataset
from config.constants import COL_PROMPT, COL_CHOSEN, COL_REJECTED

def benchmark():
    print("Generating 100,000 rows of dummy DPO data...")
    data = {
        COL_PROMPT: ["Prompt " + str(i) for i in range(100000)],
        COL_CHOSEN: ["Chosen " + str(i) for i in range(100000)],
        COL_REJECTED: ["Rejected " + str(i) for i in range(100000)]
    }
    # Add some empty rows
    data[COL_PROMPT][10] = ""
    data[COL_CHOSEN][20] = ""
    # Add some duplicates
    data[COL_PROMPT][100:1100] = ["Dup Prompt"] * 1000
    data[COL_CHOSEN][100:1100] = ["Dup Chosen"] * 1000
    data[COL_REJECTED][100:1100] = ["Dup Rejected"] * 1000

    ds = Dataset.from_dict(data)

    print(f"Running validate_and_clean_dataset (with is_dpo=True) on {len(ds)} rows...")
    start_time = time.time()
    cleaned_ds, issues = validate_and_clean_dataset(ds, is_dpo=True)
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"DPO processing and deduplication time: {elapsed:.4f} seconds")
    print(f"Issues: {issues}")
    print(f"Cleaned dataset size: {len(cleaned_ds)}")

    # Assert correctness
    assert len(cleaned_ds) == 98999, f"Expected 98999 rows after removing 2 empty and 999 duplicate rows, but got {len(cleaned_ds)}"
    print("Verification passed successfully!")

if __name__ == "__main__":
    benchmark()
