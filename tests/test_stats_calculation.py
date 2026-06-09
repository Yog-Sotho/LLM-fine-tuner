import datasets
import pytest
from data.preprocessing import get_dataset_stats
from config.constants import COL_INSTRUCTION, COL_OUTPUT, COL_TEXT, COL_PROMPT, COL_CHOSEN, COL_REJECTED

def test_get_dataset_stats_sft_text():
    ds = datasets.Dataset.from_list([
        {COL_TEXT: "hello"},
        {COL_TEXT: "world!"}, # 5 + 6 = 11, avg = 5.5
    ])
    stats = get_dataset_stats(ds, is_dpo=False)
    assert stats["num_examples"] == 2
    assert stats["avg_length"] == 5.5

def test_get_dataset_stats_sft_instruction():
    ds = datasets.Dataset.from_list([
        {COL_INSTRUCTION: "hi", COL_OUTPUT: "there"}, # 2 + 5 = 7
        {COL_INSTRUCTION: "abc", COL_OUTPUT: "defg"}, # 3 + 4 = 7
    ])
    stats = get_dataset_stats(ds, is_dpo=False)
    assert stats["num_examples"] == 2
    assert stats["avg_length"] == 7.0

def test_get_dataset_stats_dpo():
    ds = datasets.Dataset.from_list([
        {COL_PROMPT: "P", COL_CHOSEN: "C", COL_REJECTED: "R"}, # 1+1+1 = 3
        {COL_PROMPT: "PP", COL_CHOSEN: "CC", COL_REJECTED: "RR"}, # 2+2+2 = 6
    ])
    stats = get_dataset_stats(ds, is_dpo=True)
    assert stats["num_examples"] == 2
    assert stats["avg_length"] == 4.5

def test_get_dataset_stats_empty():
    ds = datasets.Dataset.from_list([])
    stats = get_dataset_stats(ds)
    assert stats["num_examples"] == 0
    assert stats["avg_length"] == 0.0

def test_get_dataset_stats_fallback():
    # Dataset with unknown columns
    ds = datasets.Dataset.from_list([
        {"unknown": "12345"},
        {"unknown": "1234567890"},
    ])
    stats = get_dataset_stats(ds)
    assert stats["num_examples"] == 2
    assert stats["avg_length"] == 7.5
