
import os
import sys

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.state import validate_path_traversal, validate_identifier

def test_validate_path_traversal_robust():
    # Standard traversal
    assert validate_path_traversal("../etc/passwd") == "❌ Path traversal attempt detected."
    assert validate_path_traversal("..\\windows\\win.ini") == "❌ Path traversal attempt detected."

    # Null byte injection
    assert validate_path_traversal("safe/path\0/unsafe") == "❌ Path traversal attempt detected."
    assert validate_path_traversal("model.bin\0.txt") == "❌ Path traversal attempt detected."

    # Safe paths
    assert validate_path_traversal("safe/path/to/model") is None
    assert validate_path_traversal("model_v1.bin") is None
    assert validate_path_traversal(None) is None
    assert validate_path_traversal("") is None

def test_validate_identifier_robust():
    # Path separators
    assert validate_identifier("sub/dir") == "❌ Path traversal attempt detected."
    assert validate_identifier("sub\\dir") == "❌ Path traversal attempt detected."

    # Traversal
    assert validate_identifier("..") == "❌ Path traversal attempt detected."
    assert validate_identifier("ok/../traversal") == "❌ Path traversal attempt detected."

    # Null bytes
    assert validate_identifier("id\0withnull") == "❌ Path traversal attempt detected."

    # Safe identifiers
    assert validate_identifier("v1.0.0") is None
    assert validate_identifier("q4_k_m") is None
    assert validate_identifier("my-model-v2") is None
    assert validate_identifier(None) is None
    assert validate_identifier("") is None

if __name__ == "__main__":
    test_validate_path_traversal_robust()
    test_validate_identifier_robust()
    print("✅ test_security_guards_robust passed.")
