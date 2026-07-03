
import pytest
from core.state import validate_path_traversal, validate_identifier

def test_validate_path_traversal_hardened():
    # Standard traversal
    assert validate_path_traversal("../unsafe") == "❌ Path traversal attempt detected."
    assert validate_path_traversal("..\\unsafe") == "❌ Path traversal attempt detected."

    # Null byte injection
    assert validate_path_traversal("/tmp/model\0.py") == "❌ Path traversal attempt detected."

    # Safe paths
    assert validate_path_traversal("/tmp/model") is None
    assert validate_path_traversal("model_dir") is None
    assert validate_path_traversal(None) is None
    assert validate_path_traversal("") is None

def test_validate_identifier_hardened():
    # Directory separators
    assert validate_identifier("sub/dir") == "❌ Path traversal attempt detected."
    assert validate_identifier("sub\\dir") == "❌ Path traversal attempt detected."

    # Traversal
    assert validate_identifier("q6_k/../traversal") == "❌ Path traversal attempt detected."

    # Null byte injection
    assert validate_identifier("v1.0\0") == "❌ Path traversal attempt detected."

    # Safe identifiers
    assert validate_identifier("q6_k") is None
    assert validate_identifier("v1.0.0") is None
    assert validate_identifier("model-name") is None
    assert validate_identifier(None) is None
    assert validate_identifier("") is None
