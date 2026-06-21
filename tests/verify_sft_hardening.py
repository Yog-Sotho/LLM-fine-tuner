
import sys
from unittest.mock import MagicMock

# Mock heavy ML dependencies before importing training.sft
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["trl"] = MagicMock()
sys.modules["unsloth"] = MagicMock()
sys.modules["bitsandbytes"] = MagicMock()
sys.modules["gradio"] = MagicMock()
sys.modules["pandas"] = MagicMock()
sys.modules["datasets"] = MagicMock()

from core.state import validate_path_traversal, validate_identifier
from training.sft import train_model, load_qlora_model_v27

def test_validate_path_traversal():
    assert validate_path_traversal("ok") is None
    assert validate_path_traversal("../unsafe") is not None
    assert validate_path_traversal("ok\\unsafe") is not None
    assert validate_path_traversal("ok\x00unsafe") is not None

def test_validate_identifier():
    assert validate_identifier("ok") is None
    assert validate_identifier("../unsafe") is not None
    assert validate_identifier("ok/unsafe") is not None
    assert validate_identifier("ok\\unsafe") is not None

def test_train_model_path_traversal():
    # Test malicious model_name
    try:
        train_model(
            model_name="../unsafe",
            dataset=MagicMock(),
            output_dir="./ok",
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
            early_stop=3,
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            use_unsloth=False,
            use_chat_template=False,
            system_prompt="test"
        )
        raise AssertionError("Should have raised ValueError for malicious model_name")
    except ValueError as e:
        assert "❌ Path traversal attempt detected." in str(e)

    # Test malicious output_dir
    try:
        train_model(
            model_name="ok",
            dataset=MagicMock(),
            output_dir="../unsafe",
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
            early_stop=3,
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            use_unsloth=False,
            use_chat_template=False,
            system_prompt="test"
        )
        raise AssertionError("Should have raised ValueError for malicious output_dir")
    except ValueError as e:
        assert "❌ Path traversal attempt detected." in str(e)

def test_load_qlora_model_path_traversal():
    try:
        load_qlora_model_v27(model_name="..\\unsafe")
        raise AssertionError("Should have raised ValueError for malicious model_name")
    except ValueError as e:
        assert "❌ Path traversal attempt detected." in str(e)

def test_whitespace_stripping():
    # If it strips whitespace, it should still trigger the traversal check if the path is malicious
    try:
        load_qlora_model_v27(model_name="  ../unsafe  ")
        raise AssertionError("Should have raised ValueError for malicious model_name")
    except ValueError as e:
        assert "❌ Path traversal attempt detected." in str(e)

if __name__ == "__main__":
    # Manually run tests if pytest is not available as a module
    try:
        test_validate_path_traversal()
        print("✅ test_validate_path_traversal passed")
        test_validate_identifier()
        print("✅ test_validate_identifier passed")
        test_train_model_path_traversal()
        print("✅ test_train_model_path_traversal passed")
        test_load_qlora_model_path_traversal()
        print("✅ test_load_qlora_model_path_traversal passed")
        test_whitespace_stripping()
        print("✅ test_whitespace_stripping passed")
        print("\nAll security hardening tests passed!")
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        sys.exit(1)
