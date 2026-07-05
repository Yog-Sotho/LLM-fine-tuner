import pytest
from core.state import validate_path_traversal, validate_identifier

def test_validate_path_traversal_null_byte():
    assert validate_path_traversal("path\0with\0null") == "❌ Path traversal attempt detected."
    assert validate_path_traversal("safe/path") is None

def test_validate_identifier_robust():
    # Should block separators
    assert validate_identifier("v1/0") == "❌ Path traversal attempt detected."
    assert validate_identifier("v1\\0") == "❌ Path traversal attempt detected."
    assert validate_identifier("v1..") == "❌ Path traversal attempt detected."

    # Should block null bytes
    assert validate_identifier("v1\0tag") == "❌ Path traversal attempt detected."

    # Should allow safe identifiers
    assert validate_identifier("v1.0.0") is None
    assert validate_identifier("q4_k_m") is None
    assert validate_identifier("my-model-v2") is None
    assert validate_identifier("") is None
    assert validate_identifier(None) is None
