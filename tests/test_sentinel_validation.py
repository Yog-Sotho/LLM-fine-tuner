import pytest
from core.state import validate_path_traversal

def test_validate_path_traversal_safe():
    assert validate_path_traversal("standard/path/to/model") is None
    assert validate_path_traversal("my-model-v1.0") is None
    assert validate_path_traversal("./local_dir") is None
    assert validate_path_traversal("username/repo") is None

def test_validate_path_traversal_unsafe_dotdot():
    err = validate_path_traversal("../../etc/passwd")
    assert err is not None
    assert "❌" in err
    assert ".." in err

def test_validate_path_traversal_unsafe_backslash():
    err = validate_path_traversal("C:\\Windows\\System32")
    assert err is not None
    assert "❌" in err
    assert "\\" in err

def test_validate_path_traversal_empty():
    assert validate_path_traversal("") is None
    assert validate_path_traversal(None) is None
