
import pytest
from export.gguf import on_export_gguf

def test_on_export_gguf_path_traversal():
    # Test path traversal in model_path
    result, file_path = on_export_gguf("../unsafe_path", "q6_k")
    assert "❌ Path traversal attempt detected." in result
    assert file_path is None

    result, file_path = on_export_gguf("safe/path/..", "q6_k")
    assert "❌ Path traversal attempt detected." in result
    assert file_path is None

    result, file_path = on_export_gguf("C:\\Windows", "q6_k")
    assert "❌ Path traversal attempt detected." in result
    assert file_path is None

    # Test path traversal in quantization
    result, file_path = on_export_gguf("./valid_model", "q6_k/../../etc/passwd")
    assert "❌ Path traversal attempt detected." in result
    assert file_path is None

def test_on_export_gguf_whitespace_stripping():
    # Test that whitespace is stripped before validation
    # If it wasn't stripped, validate_path_traversal might still catch it if it contains '..'
    # but we want to ensure it handles it cleanly.
    # We use a path that is "safe" but non-existent to see if it passes the security check
    # and fails at the directory check.
    result, file_path = on_export_gguf("  non_existent_model  ", "q6_k")
    assert "❌ No trained model found." in result
    assert file_path is None
