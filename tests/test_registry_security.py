
import pytest
import os
from unittest.mock import MagicMock, patch
from export.registry import on_registry_upload

def test_on_registry_upload_security():
    """Verify security guards in on_registry_upload."""
    # 1. Test traversal in model_path
    status = on_registry_upload("../unsafe", "user/repo", "hf_abcdefghijklmnopqrstuvwxyz0123456789", "v1", "")
    assert "❌ Path traversal attempt detected." in status

    # 2. Test traversal in repo_id
    status = on_registry_upload("./ok", "user/repo/../unsafe", "hf_abcdefghijklmnopqrstuvwxyz0123456789", "v1", "")
    assert "❌ Path traversal attempt detected." in status

    # 3. Test illegal version tag (with slash)
    status = on_registry_upload("./ok", "user/repo", "hf_abcdefghijklmnopqrstuvwxyz0123456789", "v1/traversal", "")
    assert "❌ Path traversal attempt detected." in status

    # 4. Test illegal version tag (with null byte)
    status = on_registry_upload("./ok", "user/repo", "hf_abcdefghijklmnopqrstuvwxyz0123456789", "v1\0", "")
    assert "❌ Path traversal attempt detected." in status

    # 5. Test valid-ish inputs (reaching next check)
    with patch("os.path.isdir", return_value=False):
        status = on_registry_upload("  /tmp/model  ", "  user/repo  ", "hf_abcdefghijklmnopqrstuvwxyz0123456789", " v1 ", "")
        assert "❌ No trained model found." in status
