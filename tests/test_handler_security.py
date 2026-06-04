
import pytest
from unittest.mock import MagicMock, patch
from export.gguf import on_export_gguf
from inference.vllm_runner import on_vllm_generate
from export.utils import validate_hf_token

def test_validate_hf_token():
    # Test valid token
    assert validate_hf_token("hf_123456789012345678901234567890123456") is None

    # Test missing token
    assert "required" in validate_hf_token(None)
    assert "required" in validate_hf_token("")

    # Test invalid prefix
    assert "start with 'hf_'" in validate_hf_token("gh_123456789012345678901234567890123456")

    # Test too short
    assert "at least 36 characters" in validate_hf_token("hf_short")

    # Test whitespace stripping
    assert validate_hf_token("  hf_123456789012345678901234567890123456  ") is None

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

def test_on_export_gguf_quantization_traversal():
    # Test with malicious quantization string
    status, file_path = on_export_gguf("./ok_dir", "../unsafe_quant")
    assert "❌ Path traversal attempt detected." in status
    assert file_path is None

    status, file_path = on_export_gguf("./ok_dir", "sub/dir")
    assert "❌ Invalid quantization format." in status
    assert file_path is None

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

def test_on_vllm_generate_quantization_traversal():
    # Test with malicious quantization string
    status = on_vllm_generate("./ok_dir", "prompt", "../unsafe_quant", 512, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status
