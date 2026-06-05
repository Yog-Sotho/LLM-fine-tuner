
import pytest
from unittest.mock import MagicMock, patch
from export.gguf import on_export_gguf
from inference.vllm_runner import on_vllm_generate

def test_on_export_gguf_quant_path_traversal():
    """Test that path traversal in GGUF quantization string is blocked."""
    # Test with traversal pattern
    status, file_path = on_export_gguf("./ok_model", "../unsafe_quant")
    assert "❌ Path traversal attempt detected." in status
    assert file_path is None

    # Test with forward slash (directory escape)
    status, file_path = on_export_gguf("./ok_model", "q4_k/../../etc")
    assert "❌ Path traversal attempt detected." in status
    assert file_path is None

def test_on_export_gguf_quant_whitespace_stripping():
    """Test that whitespace in GGUF quantization string is stripped."""
    with patch("export.gguf.validate_path_traversal", return_value=None):
        with patch("os.path.isdir", return_value=False):
             # If it strips correctly, it should proceed to the next check which is model_path existence
             status, file_path = on_export_gguf("model", "  q6_k  ")
             assert "❌ No trained model found." in status

def test_on_vllm_generate_quant_path_traversal():
    """Test that path traversal in vLLM quantization string is blocked."""
    # Test with traversal pattern
    status = on_vllm_generate("./ok_model", "prompt", "../unsafe_quant", 512, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status

    # Test with forward slash
    status = on_vllm_generate("./ok_model", "prompt", "bnb/../../etc", 512, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status

def test_on_vllm_generate_quant_whitespace_stripping():
    """Test that whitespace in vLLM quantization string is stripped."""
    with patch("inference.vllm_runner.validate_path_traversal", return_value=None):
        with patch("inference.vllm_runner.HAS_VLLM", True):
            with patch("os.path.isdir", return_value=False):
                status = on_vllm_generate("model", "prompt", "  bnb  ", 512, 0.7, 0.9)
                assert "❌ No trained model path found." in status
