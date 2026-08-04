import pytest
from unittest.mock import patch, MagicMock
from inference.evaluation import compute_bleu_rouge

def test_compute_bleu_rouge_parity():
    """Verify that parallel and sequential paths return identical results."""
    predictions = [f"This is prediction {i} with some dummy words." for i in range(120)]
    references = [f"This is reference {i} with some other dummy words." for i in range(120)]

    # 1. Force sequential path
    with patch("os.cpu_count", return_value=1):
        res_seq = compute_bleu_rouge(predictions, references)

    # 2. Force parallel path (ensure os.cpu_count > 1)
    with patch("os.cpu_count", return_value=4):
        res_para = compute_bleu_rouge(predictions, references)

    assert res_seq == res_para
    assert "BLEU-1" in res_seq
    assert "ROUGE-1" in res_seq
    assert "ROUGE-2" in res_seq
    assert "ROUGE-L" in res_seq


def test_compute_bleu_rouge_small_dataset_is_sequential():
    """Verify that datasets with < 100 items do not trigger multiprocessing."""
    predictions = [f"This is prediction {i}" for i in range(50)]
    references = [f"This is reference {i}" for i in range(50)]

    # Patch ProcessPoolExecutor to detect if it gets called
    with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor:
        with patch("os.cpu_count", return_value=4):
            res = compute_bleu_rouge(predictions, references)

        # ProcessPoolExecutor should never be instantiated/used
        mock_executor.assert_not_called()
        assert "BLEU-1" in res


def test_compute_bleu_rouge_fallback_on_exception():
    """Verify that compute_bleu_rouge falls back to sequential path if multiprocessing raises an error."""
    predictions = [f"This is prediction {i}" for i in range(120)]
    references = [f"This is reference {i}" for i in range(120)]

    # Mock multiprocessing.get_context to raise an error
    with patch("multiprocessing.get_context", side_effect=RuntimeError("Process spawning disabled for testing")):
        with patch("os.cpu_count", return_value=4):
            # Should not raise exception, should succeed via fallback sequential path
            res = compute_bleu_rouge(predictions, references)

        assert "BLEU-1" in res
        assert "ROUGE-1" in res
