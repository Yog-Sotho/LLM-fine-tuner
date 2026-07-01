
import pytest
from core.state import validate_path_traversal, validate_identifier

def test_validate_path_traversal_null_byte():
    """Verify that null bytes are blocked in paths."""
    assert validate_path_traversal("/safe/path\0") == "❌ Path traversal attempt detected."
    assert validate_path_traversal("\0/unsafe") == "❌ Path traversal attempt detected."
    assert validate_path_traversal("safe/path") is None

def test_validate_identifier_robust():
    """Verify that validate_identifier blocks all forbidden characters."""
    # Forbidden characters
    assert validate_identifier("v1.0/") == "❌ Path traversal attempt detected."
    assert validate_identifier("v1.0\\") == "❌ Path traversal attempt detected."
    assert validate_identifier("v1.0/..") == "❌ Path traversal attempt detected."
    assert validate_identifier("v1.0\0") == "❌ Path traversal attempt detected."

    # Safe identifiers
    assert validate_identifier("v1.0.0-final") is None
    assert validate_identifier("q4_k_m") is None
    assert validate_identifier("my-model-v1") is None

def test_validate_path_traversal_standard():
    """Verify standard path traversal patterns are still blocked."""
    assert validate_path_traversal("../etc/passwd") == "❌ Path traversal attempt detected."
    assert validate_path_traversal("C:\\Windows") == "❌ Path traversal attempt detected."
    assert validate_path_traversal("normal/path") is None
