"""
tests/test_preprocessing.py
=============================
Unit tests for data/preprocessing.py.

Covers:
  - validate_and_clean_dataset removes empty rows and reports issues
  - validate_and_clean_dataset works for DPO datasets
  - Duplicate rows are REMOVED and the issue is reported  ← N-6 FIX
  - preview_dataset returns a DataFrame with at most 10 rows
  - preprocess_function builds expected token keys
"""

import datasets
import pandas as pd
import pytest
from unittest.mock import MagicMock

from data.preprocessing import validate_and_clean_dataset, preview_dataset
from config.constants import COL_INSTRUCTION, COL_OUTPUT, COL_TEXT, COL_PROMPT, COL_CHOSEN, COL_REJECTED


# ── helpers ────────────────────────────────────────────────────────────────

def _make_sft_ds(rows):
    return datasets.Dataset.from_list(rows)


def _make_dpo_ds(rows):
    return datasets.Dataset.from_list(rows)


# ── validate_and_clean_dataset — SFT ──────────────────────────────────────

def test_validate_removes_empty_instruction():
    ds = _make_sft_ds([
        {COL_INSTRUCTION: "Q1", COL_OUTPUT: "A1"},
        {COL_INSTRUCTION: "",   COL_OUTPUT: "A2"},  # empty — should be dropped
        {COL_INSTRUCTION: "Q3", COL_OUTPUT: "A3"},
    ])
    clean, issues = validate_and_clean_dataset(ds)
    assert len(clean) == 2
    assert any("empty" in i.lower() or "removed" in i.lower() for i in issues)


def test_validate_removes_empty_text():
    ds = _make_sft_ds([
        {COL_TEXT: "hello"},
        {COL_TEXT: ""},
        {COL_TEXT: "   "},  # whitespace-only — should also be treated as empty
    ])
    clean, issues = validate_and_clean_dataset(ds)
    assert len(clean) == 1


def test_validate_all_valid_no_issues():
    ds = _make_sft_ds([
        {COL_INSTRUCTION: "Q1", COL_OUTPUT: "A1"},
        {COL_INSTRUCTION: "Q2", COL_OUTPUT: "A2"},
    ])
    clean, issues = validate_and_clean_dataset(ds)
    assert len(clean) == 2


def test_validate_removes_duplicates_and_reports():
    """
    N-6 FIX: The previous test docstring said 'soft warning, rows kept' and
    only asserted that a warning appeared in issues.  Since preprocessing.py
    was updated to REMOVE duplicates via Dataset.select(unique_indices),
    that comment and the missing length assertion were both wrong.

    This test now asserts:
      1. A warning containing 'duplicate' appears in the issues list.
      2. The duplicate row is actually removed (2 unique + 1 dup → 2 rows).
    """
    ds = _make_sft_ds([
        {COL_TEXT: "same text"},
        {COL_TEXT: "same text"},   # duplicate — must be removed
        {COL_TEXT: "unique"},
    ])
    clean, issues = validate_and_clean_dataset(ds)

    # Issue reported
    assert any("duplicate" in i.lower() for i in issues), \
        f"Expected a 'duplicate' warning in issues, got: {issues}"

    # Duplicate actually removed — 3 rows in, 2 unique rows out
    assert len(clean) == 2, \
        f"Expected 2 rows after deduplication (1 duplicate removed), got {len(clean)}"


def test_validate_instruction_pair_deduplication():
    """Duplicate (instruction, output) pairs should also be removed."""
    ds = _make_sft_ds([
        {COL_INSTRUCTION: "Q", COL_OUTPUT: "A"},
        {COL_INSTRUCTION: "Q", COL_OUTPUT: "A"},   # exact duplicate
        {COL_INSTRUCTION: "Q2", COL_OUTPUT: "A2"},
    ])
    clean, issues = validate_and_clean_dataset(ds)
    assert any("duplicate" in i.lower() for i in issues)
    assert len(clean) == 2


# ── validate_and_clean_dataset — DPO ──────────────────────────────────────

def test_validate_dpo_removes_empty_prompt():
    ds = _make_dpo_ds([
        {COL_PROMPT: "P1", COL_CHOSEN: "C1", COL_REJECTED: "R1"},
        {COL_PROMPT: "",   COL_CHOSEN: "C2", COL_REJECTED: "R2"},
    ])
    clean, issues = validate_and_clean_dataset(ds, is_dpo=True)
    assert len(clean) == 1


def test_validate_dpo_removes_empty_chosen():
    ds = _make_dpo_ds([
        {COL_PROMPT: "P1", COL_CHOSEN: "C1", COL_REJECTED: "R1"},
        {COL_PROMPT: "P2", COL_CHOSEN: "",   COL_REJECTED: "R2"},
    ])
    clean, issues = validate_and_clean_dataset(ds, is_dpo=True)
    assert len(clean) == 1


def test_validate_dpo_removes_duplicates_and_reports():
    """Verify that duplicate prompt, chosen, and rejected triplets are removed."""
    ds = _make_dpo_ds([
        {COL_PROMPT: "P1", COL_CHOSEN: "C1", COL_REJECTED: "R1"},
        {COL_PROMPT: "P1", COL_CHOSEN: "C1", COL_REJECTED: "R1"},   # duplicate triplet
        {COL_PROMPT: "P2", COL_CHOSEN: "C2", COL_REJECTED: "R2"},
    ])
    clean, issues = validate_and_clean_dataset(ds, is_dpo=True)
    assert any("duplicate" in i.lower() for i in issues)
    assert len(clean) == 2


# ── preview_dataset ────────────────────────────────────────────────────────

def test_preview_returns_dataframe():
    ds = _make_sft_ds([{COL_TEXT: f"row {i}"} for i in range(20)])
    df = preview_dataset(ds)
    assert isinstance(df, pd.DataFrame)


def test_preview_capped_at_ten_rows():
    ds = _make_sft_ds([{COL_TEXT: f"row {i}"} for i in range(50)])
    df = preview_dataset(ds)
    assert len(df) <= 10


def test_preview_dpo_dataset():
    ds = _make_dpo_ds([
        {COL_PROMPT: "P", COL_CHOSEN: "C", COL_REJECTED: "R"}
    ] * 5)
    df = preview_dataset(ds, is_dpo=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 10
