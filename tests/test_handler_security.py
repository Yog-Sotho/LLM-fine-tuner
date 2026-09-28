from unittest.mock import MagicMock, patch

import pytest

from export.gguf import on_export_gguf
from inference.vllm_runner import on_vllm_generate
from training.sft import train_model


def test_on_export_gguf_path_traversal():
    # Test with traversal pattern
    status, file_path = on_export_gguf("../unsafe_path", "q6_k")
    assert "❌ Path traversal attempt detected." in status
    assert file_path is None


def test_on_export_gguf_whitespace_stripping():
    # Test with whitespace that should be stripped
    # Use a non-existent directory to trigger the "No trained model found" error
    # but after stripping and path traversal check.
    with patch("os.path.isdir", return_value=False):
        status, file_path = on_export_gguf("  non_existent_dir  ", " q6_k ")
        assert "❌ No trained model found." in status


def test_on_vllm_generate_path_traversal():
    # Test with traversal pattern
    status = on_vllm_generate("../unsafe_path", "prompt", "none", 512, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status


def test_on_vllm_generate_whitespace_stripping():
    # Test with whitespace that should be stripped
    with patch("inference.vllm_runner.HAS_VLLM", True):
        with patch("os.path.isdir", return_value=False):
            status = on_vllm_generate("  non_existent_dir  ", "prompt", "none", 512, 0.7, 0.9)
            assert "❌ No trained model path found." in status


# Formerly in the uncollected tests/verify_*.py scripts.


@pytest.mark.parametrize("quant", ["q6_k/../../etc/passwd", "q6_k\\..\\..\\etc", "sub/dir"])
def test_on_export_gguf_rejects_paths_in_quantization(quant):
    status, file_path = on_export_gguf("./ok", quant)
    assert "❌ Path traversal attempt detected." in status and file_path is None


@pytest.mark.parametrize("quant", ["awq/../traversal", "awq\\..\\traversal", "illegal/slash"])
def test_on_vllm_generate_rejects_paths_in_quantization(quant):
    status = on_vllm_generate("./ok", "prompt", quant, 128, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status


def test_train_model_rejects_model_name_traversal():
    with pytest.raises(ValueError, match="Path traversal attempt detected"):
        train_model(
            "../unsafe", MagicMock(), "./ok", {}, "cpu", "LoRA", True, 8, 16, 30, 512, 2,
            20, 16, False, 3, "cosine", True, False, False, "test",
        )  # fmt: skip
