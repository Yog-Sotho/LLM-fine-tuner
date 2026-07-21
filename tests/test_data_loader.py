"""
tests/test_data_loader.py
==========================
Unit tests for data/loader.py.

Covers:
  - detect_file_type returns correct type strings
  - load_dataset_from_file works for CSV and JSONL
  - Column mapping renames columns correctly
  - DPO datasets load expected columns
  - ZIP extraction raises on path traversal (safe_extract_zip)
  - Unknown extensions return None from detect_file_type
"""

import io
import json
import os
import tempfile
import zipfile

import pandas as pd
import pytest

from data.loader import detect_file_type, load_dataset_from_file, safe_extract_zip
from config.constants import COL_INSTRUCTION, COL_OUTPUT, COL_TEXT, COL_PROMPT, COL_CHOSEN, COL_REJECTED


# ── Helpers ────────────────────────────────────────────────────────────────

class DummyFile:
    def __init__(self, name):
        self.name = name


def _write_csv(tmp_dir, rows: list[dict], fname="data.csv") -> str:
    path = os.path.join(tmp_dir, fname)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_jsonl(tmp_dir, rows: list[dict], fname="data.jsonl") -> str:
    path = os.path.join(tmp_dir, fname)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


# ── detect_file_type ───────────────────────────────────────────────────────

def test_detect_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        assert detect_file_type(DummyFile(path)) == "csv"
    finally:
        os.unlink(path)


def test_detect_jsonl():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        assert detect_file_type(DummyFile(path)) == "jsonl"
    finally:
        os.unlink(path)


def test_detect_txt():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        path = f.name
    try:
        assert detect_file_type(DummyFile(path)) == "txt"
    finally:
        os.unlink(path)


def test_detect_unknown_returns_none():
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        path = f.name
    try:
        assert detect_file_type(DummyFile(path)) is None
    finally:
        os.unlink(path)


# ── load_dataset_from_file — SFT ───────────────────────────────────────────

def test_load_csv_sft_standard_columns():
    with tempfile.TemporaryDirectory() as d:
        rows = [{"instruction": "Q1", "output": "A1"}, {"instruction": "Q2", "output": "A2"}]
        path = _write_csv(d, rows)
        ds = load_dataset_from_file(DummyFile(path), "csv")
        assert len(ds) == 2
        assert COL_INSTRUCTION in ds.column_names
        assert COL_OUTPUT in ds.column_names


def test_load_jsonl_sft_text_column():
    with tempfile.TemporaryDirectory() as d:
        rows = [{"text": "hello world"}, {"text": "foo bar"}]
        path = _write_jsonl(d, rows)
        ds = load_dataset_from_file(DummyFile(path), "jsonl")
        assert len(ds) == 2
        assert COL_TEXT in ds.column_names


def test_load_csv_with_column_mapping():
    with tempfile.TemporaryDirectory() as d:
        rows = [{"q": "What?", "a": "Answer"}, {"q": "How?", "a": "Fine"}]
        path = _write_csv(d, rows)
        col_map = {"q": COL_INSTRUCTION, "a": COL_OUTPUT}
        ds = load_dataset_from_file(DummyFile(path), "csv", column_mapping=col_map)
        assert COL_INSTRUCTION in ds.column_names
        assert COL_OUTPUT in ds.column_names


# ── load_dataset_from_file — DPO ───────────────────────────────────────────

def test_load_csv_dpo_columns():
    with tempfile.TemporaryDirectory() as d:
        rows = [
            {"prompt": "P", "chosen": "C", "rejected": "R"},
            {"prompt": "P2", "chosen": "C2", "rejected": "R2"},
        ]
        path = _write_csv(d, rows)
        ds = load_dataset_from_file(DummyFile(path), "csv", is_dpo=True)
        assert COL_PROMPT in ds.column_names
        assert COL_CHOSEN in ds.column_names
        assert COL_REJECTED in ds.column_names


# ── safe_extract_zip ───────────────────────────────────────────────────────

def test_safe_extract_zip_normal():
    with tempfile.TemporaryDirectory() as d:
        zip_path = os.path.join(d, "model.zip")
        extract_dir = os.path.join(d, "out")
        os.makedirs(extract_dir)
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("adapter_config.json", '{"base_model_name_or_path": "gpt2"}')
        safe_extract_zip(zip_path, extract_dir)
        assert os.path.isfile(os.path.join(extract_dir, "adapter_config.json"))


def test_safe_extract_zip_relative_path_traversal_blocked():
    """Ensure safe_extract_zip raises on relative path-traversal (../…) entries."""
    with tempfile.TemporaryDirectory() as d:
        zip_path = os.path.join(d, "evil_relative.zip")
        extract_dir = os.path.join(d, "out")
        os.makedirs(extract_dir)
        with zipfile.ZipFile(zip_path, "w") as zf:
            info = zipfile.ZipInfo("../../../etc/passwd")
            zf.writestr(info, "root:x:0:0:root:/root:/bin/bash")
        with pytest.raises(Exception):
            safe_extract_zip(zip_path, extract_dir)


def test_safe_extract_zip_absolute_path_blocked():
    """L8 FIX: safe_extract_zip must block absolute paths like /etc/passwd.

    The original implementation only checked for '../' prefix, leaving
    absolute paths entirely unblocked (proven exploitable in audit C2).
    The realpath-based containment check blocks both vectors.
    """
    with tempfile.TemporaryDirectory() as d:
        zip_path = os.path.join(d, "evil_absolute.zip")
        extract_dir = os.path.join(d, "out")
        os.makedirs(extract_dir)
        with zipfile.ZipFile(zip_path, "w") as zf:
            # Absolute path entry — would land at /tmp/evil on extraction
            info = zipfile.ZipInfo("/tmp/evil_payload.txt")
            zf.writestr(info, "pwned")
        with pytest.raises(Exception, match="Path traversal"):
            safe_extract_zip(zip_path, extract_dir)


def test_load_dataset_from_file_path_traversal_blocked():
    """Verify that load_dataset_from_file blocks path traversal and null bytes in filenames."""
    # Test relative path traversal (..)
    with pytest.raises(RuntimeError) as exc_info:
        load_dataset_from_file(DummyFile("../secret.csv"), "csv")
    assert "❌ Path traversal attempt detected." in str(exc_info.value)

    # Test backslash traversal (\)
    with pytest.raises(RuntimeError) as exc_info:
        load_dataset_from_file(DummyFile("data\\..\\secret.csv"), "csv")
    assert "❌ Path traversal attempt detected." in str(exc_info.value)

    # Test null byte injection (\0)
    with pytest.raises(RuntimeError) as exc_info:
        load_dataset_from_file(DummyFile("data\0secret.csv"), "csv")
    assert "❌ Path traversal attempt detected." in str(exc_info.value)
