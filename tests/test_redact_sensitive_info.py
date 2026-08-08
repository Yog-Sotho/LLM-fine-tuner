import os

from core.state import redact_sensitive_info


def test_redact_sensitive_info_regex():
    # Test typical HF token pattern (hf_ + 30+ chars)
    token = "hf_abcdefghijklmnopqrstuvwxyz0123456789"
    error_msg = f"Failed to upload model because of invalid token {token}."
    redacted = redact_sensitive_info(error_msg)
    assert "[REDACTED]" in redacted
    assert token not in redacted
    assert redacted == "Failed to upload model because of invalid token [REDACTED]."

def test_redact_sensitive_info_multiple_tokens():
    token1 = "hf_abcdefghijklmnopqrstuvwxyz012345"
    token2 = "hf_12345678901234567890123456789012"
    error_msg = f"Token 1: {token1}, Token 2: {token2}"
    redacted = redact_sensitive_info(error_msg)
    assert "[REDACTED]" in redacted
    assert token1 not in redacted
    assert token2 not in redacted
    assert redacted == "Token 1: [REDACTED], Token 2: [REDACTED]"

def test_redact_sensitive_info_env_vars():
    # Set HF_TOKEN in environment
    os.environ["HF_TOKEN"] = "my_super_secret_env_token_value_123"

    error_msg = "Error: Authentication failed with token my_super_secret_env_token_value_123."
    redacted = redact_sensitive_info(error_msg)
    assert "[REDACTED]" in redacted
    assert "my_super_secret_env_token_value_123" not in redacted

    # Cleanup
    del os.environ["HF_TOKEN"]

def test_redact_sensitive_info_none():
    assert redact_sensitive_info(None) == ""
    assert redact_sensitive_info("") == ""
