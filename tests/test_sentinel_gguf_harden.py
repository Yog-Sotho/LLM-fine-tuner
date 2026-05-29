
import pytest
from export.gguf import on_export_gguf

def test_gguf_export_path_traversal():
    """Verify that on_export_gguf blocks path traversal attempts."""
    # Test with ".."
    status, path = on_export_gguf("../unsafe_model", "q6_k")
    assert "❌ Path traversal attempt detected." in status
    assert path is None

    # Test with backslash
    status, path = on_export_gguf("C:\\Windows\\System32", "q6_k")
    assert "❌ Path traversal attempt detected." in status
    assert path is None

    # Test with leading/trailing whitespace containing traversal
    status, path = on_export_gguf("  ../secret  ", "q6_k")
    assert "❌ Path traversal attempt detected." in status
    assert path is None

def test_gguf_export_invalid_dir():
    """Verify that on_export_gguf returns an error for non-existent directories (but safe paths)."""
    status, path = on_export_gguf("/tmp/non_existent_model_xyz_123", "q6_k")
    assert "❌ No trained model found." in status
    assert path is None
