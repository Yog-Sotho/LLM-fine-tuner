
import pytest
from unittest.mock import MagicMock, patch
from export.gguf import on_export_gguf
from inference.vllm_runner import on_vllm_generate

def test_on_export_gguf_path_traversal():
    # Test with traversal pattern in model_path
    status, file_path = on_export_gguf("../unsafe_path", "q6_k")
    assert "❌ Path traversal attempt detected." in status
    assert file_path is None

def test_on_export_gguf_quant_traversal():
    # Sentinel: Verify hardening of quantization parameter
    with patch("os.path.isdir", return_value=True):
        status, file_path = on_export_gguf("valid_path", "../q6_k")
        assert "❌ Path traversal attempt detected." in status

def test_on_export_gguf_quant_slash():
    # Sentinel: Verify blocking of forward slashes in quantization
    with patch("os.path.isdir", return_value=True):
        status, file_path = on_export_gguf("valid_path", "sub/q6_k")
        assert "❌ Path traversal attempt detected." in status

def test_on_export_gguf_whitespace_stripping():
    # Test with whitespace that should be stripped
    with patch("os.path.isdir", return_value=False):
        status, file_path = on_export_gguf("  non_existent_dir  ", " q6_k ")
        assert "❌ No trained model found." in status

def test_on_vllm_generate_path_traversal():
    # Test with traversal pattern in model_path
    status = on_vllm_generate("../unsafe_path", "prompt", "none", 512, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status

def test_on_vllm_generate_quant_traversal():
    # Sentinel: Verify hardening of vllm_quant parameter
    with patch("os.path.isdir", return_value=True):
        status = on_vllm_generate("valid_path", "prompt", "../none", 512, 0.7, 0.9)
        assert "❌ Path traversal attempt detected." in status

def test_on_vllm_generate_whitespace_stripping():
    # Test with whitespace that should be stripped
    with patch("inference.vllm_runner.HAS_VLLM", True):
        with patch("os.path.isdir", return_value=False):
            status = on_vllm_generate("  non_existent_dir  ", "prompt", "none", 512, 0.7, 0.9)
            assert "❌ No trained model path found." in status
