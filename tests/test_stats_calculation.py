
import numpy as np
import pandas as pd
from datasets import Dataset
import pytest
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from config.constants import (
    COL_PROMPT, COL_CHOSEN, COL_REJECTED,
    COL_TEXT, COL_INSTRUCTION, COL_OUTPUT
)
from data.preprocessing import get_dataset_stats

def original_stats_logic(ds):
    """Old logic for comparison."""
    try:
        if COL_PROMPT in ds.column_names and COL_CHOSEN in ds.column_names:
            _lengths = [
                len(str(p)) + len(str(c)) + len(str(r))
                for p, c, r in zip(ds[COL_PROMPT], ds[COL_CHOSEN], ds[COL_REJECTED])
            ]
        elif COL_TEXT in ds.column_names:
            _lengths = [len(str(t)) for t in ds[COL_TEXT]]
        elif COL_INSTRUCTION in ds.column_names and COL_OUTPUT in ds.column_names:
            _lengths = [
                len(str(i)) + len(str(o))
                for i, o in zip(ds[COL_INSTRUCTION], ds[COL_OUTPUT])
            ]
        else:
            first_col = ds.column_names[0] if ds.column_names else None
            _lengths = [len(str(v)) for v in ds[first_col]] if first_col else []
    except Exception:
        _lengths = [100] * len(ds)
    return float(np.mean(_lengths)) if _lengths else 0.0

def test_stats_parity():
    # 1. DPO Data
    data_dpo = {
        COL_PROMPT: ["p1", "p22"],
        COL_CHOSEN: ["c111", "c2"],
        COL_REJECTED: ["r1", "r222"]
    }
    ds_dpo = Dataset.from_dict(data_dpo)

    # 2. SFT Instruction Data
    data_sft_inst = {
        COL_INSTRUCTION: ["i1", "i22"],
        COL_OUTPUT: ["o111", "o2"]
    }
    ds_sft_inst = Dataset.from_dict(data_sft_inst)

    # 3. SFT Text Data
    data_sft_text = {
        COL_TEXT: ["t1", "t222"]
    }
    ds_sft_text = Dataset.from_dict(data_sft_text)

    datasets = [ds_dpo, ds_sft_inst, ds_sft_text]

    for ds in datasets:
        # Expected
        expected_avg = original_stats_logic(ds)

        # Actual
        actual_stats = get_dataset_stats(ds)
        actual_avg = actual_stats["avg_length"]

        assert actual_avg == pytest.approx(expected_avg)
        assert actual_stats["num_examples"] == len(ds)

if __name__ == "__main__":
    test_stats_parity()
    print("Stats parity test passed!")
