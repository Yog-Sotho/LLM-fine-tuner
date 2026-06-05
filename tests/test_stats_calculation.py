
import pytest
import pandas as pd
from datasets import Dataset
from data.preprocessing import get_dataset_stats
from config.constants import COL_INSTRUCTION, COL_OUTPUT, COL_TEXT, COL_PROMPT, COL_CHOSEN, COL_REJECTED

def test_get_dataset_stats_sft_instruction():
    data = {
        COL_INSTRUCTION: ["I1", "I22"],
        COL_OUTPUT: ["O1", "O22"]
    }
    ds = Dataset.from_dict(data)
    stats = get_dataset_stats(ds)
    assert stats["num_examples"] == 2
    # "I1" + "O1" = 4, "I22" + "O22" = 6. Avg = 5.0
    assert stats["avg_length"] == 5.0

def test_get_dataset_stats_sft_text():
    data = {
        COL_TEXT: ["T1", "T222"]
    }
    ds = Dataset.from_dict(data)
    stats = get_dataset_stats(ds)
    assert stats["num_examples"] == 2
    # "T1" = 2, "T222" = 4. Avg = 3.0
    assert stats["avg_length"] == 3.0

def test_get_dataset_stats_dpo():
    data = {
        COL_PROMPT: ["P1"],
        COL_CHOSEN: ["C1"],
        COL_REJECTED: ["R1"]
    }
    ds = Dataset.from_dict(data)
    stats = get_dataset_stats(ds)
    assert stats["num_examples"] == 1
    # "P1" + "C1" + "R1" = 6. Avg = 6.0
    assert stats["avg_length"] == 6.0

def test_get_dataset_stats_empty():
    ds = Dataset.from_dict({COL_TEXT: []})
    stats = get_dataset_stats(ds)
    assert stats["num_examples"] == 0
    assert stats["avg_length"] == 0.0

def test_get_dataset_stats_unknown_column():
    data = {"unknown": ["12345"]}
    ds = Dataset.from_dict(data)
    stats = get_dataset_stats(ds)
    assert stats["num_examples"] == 1
    assert stats["avg_length"] == 5.0
