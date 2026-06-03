
import pytest
from unittest.mock import MagicMock, patch
from export.gguf import on_export_gguf
from inference.vllm_runner import on_vllm_generate

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

def test_on_export_gguf_quant_traversal():
    # Test on_export_gguf with malicious quantization strings
    with patch("os.path.isdir", return_value=True):
        status, file_path = on_export_gguf("ok_dir", "../unsafe")
        assert "❌ Path traversal attempt detected." in status
        assert file_path is None

        status, file_path = on_export_gguf("ok_dir", "ok/unsafe")
        assert "❌ Path traversal attempt detected." in status
        assert file_path is None

def test_on_vllm_generate_quant_traversal():
    # Test on_vllm_generate with malicious vllm_quant strings
    status = on_vllm_generate("ok_dir", "prompt", "../unsafe", 512, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status

    status = on_vllm_generate("ok_dir", "prompt", "ok\\unsafe", 512, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status
