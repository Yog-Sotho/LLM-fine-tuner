
import time
import pandas as pd
from datasets import Dataset
import os
import sys

# Add current directory to path so we can import data.preprocessing
sys.path.append(os.getcwd())

from data.preprocessing import validate_and_clean_dataset
from config.constants import COL_INSTRUCTION, COL_OUTPUT

def benchmark():
    print("Generating 100,000 rows of dummy data...")
    data = {
        COL_INSTRUCTION: ["Instruction " + str(i) for i in range(100000)],
        COL_OUTPUT: ["Output " + str(i) for i in range(100000)]
    }
    # Add some empty rows
    data[COL_INSTRUCTION][10] = ""
    data[COL_OUTPUT][20] = ""
    # Add some duplicates
    data[COL_INSTRUCTION][100:200] = ["Dup"] * 100
    data[COL_OUTPUT][100:200] = ["Dup"] * 100

    ds = Dataset.from_dict(data)

    print(f"Running validate_and_clean_dataset on {len(ds)} rows...")
    start_time = time.time()
    cleaned_ds, issues = validate_and_clean_dataset(ds)
    end_time = time.time()

    print(f"Original time: {end_time - start_time:.4f} seconds")
    print(f"Issues: {issues}")
    print(f"Cleaned dataset size: {len(cleaned_ds)}")

if __name__ == "__main__":
    benchmark()
