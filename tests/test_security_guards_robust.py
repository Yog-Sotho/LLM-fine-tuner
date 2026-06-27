
import os
import sys
import pytest
from unittest.mock import MagicMock

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.state import validate_path_traversal, validate_identifier

def test_validate_path_traversal_null_byte():
    assert validate_path_traversal("normal/path") is None
    assert validate_path_traversal("path/with/../traversal") == "❌ Path traversal attempt detected."
    assert validate_path_traversal(r"path\with\backslash") == "❌ Path traversal attempt detected."
    # Test null byte injection
    assert validate_path_traversal("path/with/\0/nullbyte") == "❌ Path traversal attempt detected."

def test_validate_identifier_robust():
    assert validate_identifier("v1.0") is None
    assert validate_identifier("q6_k") is None

    # Test directory separators
    assert validate_identifier("v1/0") == "❌ Path traversal attempt detected."
    assert validate_identifier(r"v1\0") == "❌ Path traversal attempt detected."
    assert validate_identifier("v1..0") == "❌ Path traversal attempt detected."

    # Test null byte injection
    assert validate_identifier("v1\0_0") == "❌ Path traversal attempt detected."

def test_handler_harden_registry():
    # Mock dependencies for on_registry_upload
    import export.registry
    original_api = export.registry.ModelRegistry
    export.registry.ModelRegistry = MagicMock()

    from export.registry import on_registry_upload

    # Test invalid version identifier (forward slash)
    res = on_registry_upload("path", "user/repo", "hf_123456789012345678901234567890123456", "v1/0", "notes")
    assert res == "❌ Path traversal attempt detected."

    # Test null byte in version
    res = on_registry_upload("path", "user/repo", "hf_123456789012345678901234567890123456", "v1\0", "notes")
    assert res == "❌ Path traversal attempt detected."

    export.registry.ModelRegistry = original_api

def test_handler_harden_gguf():
    from export.gguf import on_export_gguf

    # Test invalid quantization identifier (forward slash)
    res, _ = on_export_gguf("path", "q4/k")
    assert res == "❌ Path traversal attempt detected."

    # Test null byte in quantization
    res, _ = on_export_gguf("path", "q4\0k")
    assert res == "❌ Path traversal attempt detected."

def test_sft_hardening():
    from training.sft import train_model

    # Mock everything inside train_model to avoid heavy imports
    with pytest.raises(RuntimeError) as excinfo:
        train_model(
            model_name="../unsafe",
            dataset=MagicMock(),
            output_dir="./output",
            hyperparams={},
            device="cpu",
            peft_method="LoRA",
            use_lora=True,
            lora_rank=8,
            lora_alpha=16,
            prefix_tuning_num_virtual_tokens=30,
            prefix_tuning_token_dim=512,
            prefix_tuning_num_layers=2,
            prompt_tuning_num_virtual_tokens=20,
            adapter_reduction_factor=16,
            resume_from_checkpoint=False,
            early_stop=0,
            lr_scheduler_type="linear",
            gradient_checkpointing=False,
            use_unsloth=False,
            use_chat_template=False,
            system_prompt="",
            progress=None
        )
    assert "Path traversal attempt detected" in str(excinfo.value)
