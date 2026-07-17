import pytest
from unittest.mock import MagicMock, patch
from export.hub import push_to_hub
from export.registry import on_registry_upload, on_registry_list

def test_push_to_hub_token_security():
    # Test with traversal token
    status = push_to_hub("./dummy", "username/model", "../unsafe_token")
    assert "❌ Path traversal attempt detected." in status

    # Test with null-byte token
    status = push_to_hub("./dummy", "username/model", "safe_prefix\0unsafe")
    assert "❌ Path traversal attempt detected." in status

    # Test with backslash token
    status = push_to_hub("./dummy", "username/model", "safe\\prefix")
    assert "❌ Path traversal attempt detected." in status


def test_on_registry_upload_token_security():
    # Test with traversal token
    status = on_registry_upload("./dummy", "username/model", "../unsafe_token", "1.0", "notes")
    assert "❌ Path traversal attempt detected." in status

    # Test with null-byte token
    status = on_registry_upload("./dummy", "username/model", "safe_prefix\0unsafe", "1.0", "notes")
    assert "❌ Path traversal attempt detected." in status

    # Test with backslash token
    status = on_registry_upload("./dummy", "username/model", "safe\\prefix", "1.0", "notes")
    assert "❌ Path traversal attempt detected." in status


def test_on_registry_list_token_security():
    # Test with traversal token
    status = on_registry_list("username/model", "../unsafe_token")
    assert "❌ Path traversal attempt detected." in status

    # Test with null-byte token
    status = on_registry_list("username/model", "safe_prefix\0unsafe")
    assert "❌ Path traversal attempt detected." in status

    # Test with backslash token
    status = on_registry_list("username/model", "safe\\prefix")
    assert "❌ Path traversal attempt detected." in status
