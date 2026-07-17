
import pytest
from unittest.mock import MagicMock, patch
import os

from export.hub import push_to_hub
from export.registry import on_registry_upload, on_registry_list

def test_push_to_hub_token_security():
    # Test with null byte in token
    with patch("os.path.isdir", return_value=True):
        result = push_to_hub("./model", "user/repo", "hf_valid_token\0")
        assert "❌ Path traversal attempt detected." in result

    # Test with traversal sequence in token
    with patch("os.path.isdir", return_value=True):
        result = push_to_hub("./model", "user/repo", "hf_valid_token..")
        assert "❌ Path traversal attempt detected." in result

    # Test with backslash in token
    with patch("os.path.isdir", return_value=True):
        result = push_to_hub("./model", "user/repo", "hf_valid\\token")
        assert "❌ Path traversal attempt detected." in result

def test_on_registry_upload_token_security():
    # Test with null byte in token
    with patch("os.path.isdir", return_value=True):
        result = on_registry_upload("./model", "user/repo", "hf_valid_token\0", "v1", "notes")
        assert "❌ Path traversal attempt detected." in result

    # Test with traversal sequence in token
    with patch("os.path.isdir", return_value=True):
        result = on_registry_upload("./model", "user/repo", "hf_valid_token..", "v1", "notes")
        assert "❌ Path traversal attempt detected." in result

def test_on_registry_list_token_security():
    # Test with null byte in token
    result = on_registry_list("user/repo", "hf_valid_token\0")
    assert "❌ Path traversal attempt detected." in result

    # Test with traversal sequence in token
    result = on_registry_list("user/repo", "hf_valid_token..")
    assert "❌ Path traversal attempt detected." in result
