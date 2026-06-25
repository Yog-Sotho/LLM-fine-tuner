import pytest
from core.state import validate_path_traversal, validate_identifier
from training.sft import train_model
from unittest.mock import MagicMock, patch

def test_validate_path_traversal_null_byte():
    assert "❌" in validate_path_traversal("test\0")
    assert "❌" in validate_path_traversal("../test")
    assert validate_path_traversal("test/safe") is None

def test_validate_identifier_slash_and_null():
    assert "❌" in validate_identifier("test/")
    assert "❌" in validate_identifier("test\0")
    assert "❌" in validate_identifier("test\\")
    assert "❌" in validate_identifier("test..")
    assert validate_identifier("safe_id-123") is None

def test_train_model_hardening():
    # Mock dependencies of train_model to avoid heavy loading
    with patch("training.sft.AutoTokenizer"), \
         patch("training.sft.AutoModelForCausalLM"), \
         patch("training.sft.app_state"):

        # Test model_name traversal
        with pytest.raises(ValueError, match="Path traversal attempt detected"):
            train_model(model_name="../unsafe", dataset=MagicMock(), output_dir="safe", hyperparams={}, device="cpu", peft_method="LoRA", use_lora=True, lora_rank=8, lora_alpha=16, prefix_tuning_num_virtual_tokens=10, prefix_tuning_token_dim=10, prefix_tuning_num_layers=1, prompt_tuning_num_virtual_tokens=10, adapter_reduction_factor=1, resume_from_checkpoint=False, early_stop=0, lr_scheduler_type="linear", gradient_checkpointing=False, use_unsloth=False, use_chat_template=False, system_prompt="")

        # Test output_dir traversal
        with pytest.raises(ValueError, match="Path traversal attempt detected"):
            train_model(model_name="safe", dataset=MagicMock(), output_dir="unsafe/..", hyperparams={}, device="cpu", peft_method="LoRA", use_lora=True, lora_rank=8, lora_alpha=16, prefix_tuning_num_virtual_tokens=10, prefix_tuning_token_dim=10, prefix_tuning_num_layers=1, prompt_tuning_num_virtual_tokens=10, adapter_reduction_factor=1, resume_from_checkpoint=False, early_stop=0, lr_scheduler_type="linear", gradient_checkpointing=False, use_unsloth=False, use_chat_template=False, system_prompt="")

if __name__ == "__main__":
    pytest.main([__file__])
