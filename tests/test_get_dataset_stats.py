import pytest
import pandas as pd
from datasets import Dataset
from data.preprocessing import get_dataset_stats
from config.constants import COL_TEXT, COL_INSTRUCTION, COL_OUTPUT, COL_PROMPT, COL_CHOSEN, COL_REJECTED

def test_get_dataset_stats_sft_text():
    data = {COL_TEXT: ["Hello", "World", "Testing"]}
    ds = Dataset.from_dict(data)
    # Average length: (5 + 5 + 7) / 3 = 17 / 3 = 5.666...
    stats = get_dataset_stats(ds, is_dpo=False)
    assert stats["num_examples"] == 3
    assert pytest.approx(stats["avg_length"], 0.001) == 5.666

def test_get_dataset_stats_sft_inst():
    data = {
        COL_INSTRUCTION: ["Inst1", "Inst2"],
        COL_OUTPUT: ["Out1", "Out2"]
    }
    ds = Dataset.from_dict(data)
    # Avg length: (5+4 + 5+4) / 2 = 18 / 2 = 9.0
    stats = get_dataset_stats(ds, is_dpo=False)
    assert stats["num_examples"] == 2
    assert stats["avg_length"] == 9.0

def test_get_dataset_stats_dpo():
    data = {
        COL_PROMPT: ["P1"],
        COL_CHOSEN: ["C1"],
        COL_REJECTED: ["R1"]
    }
    ds = Dataset.from_dict(data)
    # Avg length: (2 + 2 + 2) / 1 = 6.0
    stats = get_dataset_stats(ds, is_dpo=True)
    assert stats["num_examples"] == 1
    assert stats["avg_length"] == 6.0

def test_get_dataset_stats_empty():
    ds = Dataset.from_dict({COL_TEXT: []})
    stats = get_dataset_stats(ds)
    assert stats["num_examples"] == 0
    assert stats["avg_length"] == 0.0

def test_get_dataset_stats_fallback():
    # Dataset with unknown columns
    data = {"unknown": ["abc", "defg"]}
    ds = Dataset.from_dict(data)
    # Fallback uses first column
    stats = get_dataset_stats(ds)
    assert stats["num_examples"] == 2
    assert stats["avg_length"] == 3.5
