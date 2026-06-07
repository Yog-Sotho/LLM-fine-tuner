
import pytest
import pandas as pd
from datasets import Dataset
from data.preprocessing import get_dataset_stats
from config.constants import COL_INSTRUCTION, COL_OUTPUT, COL_PROMPT, COL_CHOSEN, COL_REJECTED, COL_TEXT

def test_get_dataset_stats_sft_instruction():
    data = {
        COL_INSTRUCTION: ["Inst 1", "Inst 2", "Inst 3"],
        COL_OUTPUT: ["Out 1", "Out 2", "Out 3"]
    }
    ds = Dataset.from_dict(data)
    stats = get_dataset_stats(ds)

    assert stats["num_examples"] == 3
    # "Inst 1" (6) + "Out 1" (5) = 11
    # Average should be 11.0
    assert stats["avg_length"] == 11.0

def test_get_dataset_stats_sft_text():
    data = {
        COL_TEXT: ["Hello world", "Test"]
    }
    ds = Dataset.from_dict(data)
    stats = get_dataset_stats(ds)

    assert stats["num_examples"] == 2
    # "Hello world" (11) + "Test" (4) = 15. Average = 7.5
    assert stats["avg_length"] == 7.5

def test_get_dataset_stats_dpo():
    data = {
        COL_PROMPT: ["P1", "P2"],
        COL_CHOSEN: ["C1", "C2"],
        COL_REJECTED: ["R1", "R2"]
    }
    ds = Dataset.from_dict(data)
    stats = get_dataset_stats(ds)

    assert stats["num_examples"] == 2
    # P1(2) + C1(2) + R1(2) = 6. Average = 6.0
    assert stats["avg_length"] == 6.0

def test_get_dataset_stats_empty():
    ds = Dataset.from_dict({})
    stats = get_dataset_stats(ds)
    assert stats["num_examples"] == 0
    assert stats["avg_length"] == 0.0

def test_get_dataset_stats_unknown_cols():
    data = {"unknown": ["val1", "val2"]}
    ds = Dataset.from_dict(data)
    stats = get_dataset_stats(ds)

    assert stats["num_examples"] == 2
    # "val1" (4) + "val2" (4) = 8. Average = 4.0
    assert stats["avg_length"] == 4.0
