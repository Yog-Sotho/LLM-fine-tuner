
import os
import sys
from unittest.mock import MagicMock, patch

# Add current directory to path
sys.path.append(os.getcwd())

# Mock machine learning libraries that might not be installed
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['vllm'] = MagicMock()
sys.modules['unsloth'] = MagicMock()
sys.modules['gradio'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['PyPDF2'] = MagicMock()
sys.modules['trl'] = MagicMock()

from training.sft import train_model, load_qlora_model_v27

def test_train_model_hardening():
    # Test traversal in model_name
    try:
        train_model(
            model_name="../unsafe",
            dataset=MagicMock(),
            output_dir="./ok",
            hyperparams={}, device="cpu", peft_method="LoRA",
            use_lora=True, lora_rank=8, lora_alpha=16,
            prefix_tuning_num_virtual_tokens=30, prefix_tuning_token_dim=512,
            prefix_tuning_num_layers=2, prompt_tuning_num_virtual_tokens=20,
            adapter_reduction_factor=16, resume_from_checkpoint=False,
            early_stop=3, lr_scheduler_type="cosine",
            gradient_checkpointing=True, use_unsloth=False,
            use_chat_template=False, system_prompt="test"
        )
        assert False, "Should have raised ValueError for model_name traversal"
    except ValueError as e:
        assert "❌ Path traversal attempt detected." in str(e)

def test_load_qlora_model_v27_hardening():
    # Test traversal in model_name
    try:
        load_qlora_model_v27(model_name="../unsafe")
        assert False, "Should have raised ValueError for model_name traversal"
    except ValueError as e:
        assert "❌ Path traversal attempt detected." in str(e)

if __name__ == "__main__":
    test_train_model_hardening()
    test_load_qlora_model_v27_hardening()
    print("SFT hardening verification tests passed!")
