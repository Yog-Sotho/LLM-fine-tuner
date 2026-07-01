
import os
import sys

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.state import validate_path_traversal, validate_identifier

def test_validate_path_traversal_null_byte():
    print("Testing validate_path_traversal for null bytes...")
    assert validate_path_traversal("normal/path") is None
    assert validate_path_traversal("path/with\0/null") == "❌ Path traversal attempt detected."
    print("✅ validate_path_traversal null byte check passed.")

def test_validate_identifier_slashes():
    print("Testing validate_identifier for slashes and traversals...")
    assert validate_identifier("safe_id") is None
    assert validate_identifier("sub/dir") == "❌ Path traversal attempt detected."
    assert validate_identifier("..\traversal") == "❌ Path traversal attempt detected."
    assert validate_identifier("back\\slash") == "❌ Path traversal attempt detected."
    assert validate_identifier("null\0byte") == "❌ Path traversal attempt detected."
    print("✅ validate_identifier checks passed.")

if __name__ == "__main__":
    test_validate_path_traversal_null_byte()
    test_validate_identifier_slashes()
