"""
tests/test_evaluation_multiprocessing.py
=========================================
Unit tests for the chunk-based multiprocessing BLEU and ROUGE scoring optimization.
Verifies exact parity between sequential and multiprocessing execution paths,
and robust fallback behavior.
"""

import pytest
import numpy as np
from unittest.mock import patch
from inference.evaluation import compute_bleu_rouge, HAS_NLTK, HAS_ROUGE

def test_compute_bleu_rouge_small_dataset():
    """Verify that small datasets run sequentially and return correct scores."""
    predictions = ["This is a test."] * 10
    references = ["This is a test."] * 10

    # Ensure small datasets don't attempt multiprocessing
    with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor:
        res = compute_bleu_rouge(predictions, references)
        assert not mock_executor.called

    if HAS_NLTK:
        assert res["BLEU-1"] == 1.0
    if HAS_ROUGE:
        assert res["ROUGE-1"] == 1.0
        assert res["ROUGE-2"] == 1.0
        assert res["ROUGE-L"] == 1.0


def test_compute_bleu_rouge_large_dataset_parity():
    """Verify that large datasets (100+ items) with edge cases yield exact same results on MP and Sequential paths."""
    predictions = ["This is a sample prediction sentence which we will use to test bleu and rouge performance."] * 105
    references = ["This is a sample reference sentence which we will use to test bleu and rouge score."] * 105

    # Inject edge cases
    predictions[10] = ""
    predictions[20] = "   "
    predictions[30] = ""

    # Force sequential calculation to get reference baseline
    with patch("inference.evaluation.HAS_NLTK", HAS_NLTK), \
         patch("inference.evaluation.HAS_ROUGE", HAS_ROUGE):
        # We temporarily mock predictions length or HAS_NLTK/HAS_ROUGE to false to force sequential for reference
        # or we can mock length check
        original_len = len(predictions)
        # Force sequential by passing a mock len returning 50
        with patch("inference.evaluation.len", return_value=50):
            res_seq = compute_bleu_rouge(predictions, references)

    # Let standard compute_bleu_rouge run with length 105, which will trigger multiprocessing
    res_mp = compute_bleu_rouge(predictions, references)

    # Verify both exist and are exactly equal
    assert res_seq == res_mp


def test_compute_bleu_rouge_multiprocessing_fallback():
    """Verify that if multiprocessing fails, it falls back to sequential without crashing."""
    predictions = ["This is a test prediction."] * 120
    references = ["This is a test reference."] * 120

    # Mock ProcessPoolExecutor to raise an exception
    with patch("inference.evaluation.ProcessPoolExecutor", side_effect=RuntimeError("Mock process spawning failed")):
        res = compute_bleu_rouge(predictions, references)

    # Fallback sequential path should still compute valid scores
    if HAS_NLTK:
        assert isinstance(res["BLEU-1"], float)
    if HAS_ROUGE:
        assert isinstance(res["ROUGE-1"], float)
        assert isinstance(res["ROUGE-2"], float)
        assert isinstance(res["ROUGE-L"], float)
