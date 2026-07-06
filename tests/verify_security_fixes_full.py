
import os
import sys
from unittest.mock import MagicMock, patch

# Mock machine learning libraries
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
sys.modules['openpyxl'] = MagicMock()

from core.state import validate_path_traversal, validate_identifier
from training.sft import train_model
from export.registry import on_registry_upload
from export.gguf import on_export_gguf
from inference.vllm_runner import on_vllm_generate

def test_core_guards():
    print("Testing core guards...")
    # Path traversal null byte
    assert validate_path_traversal("path\0") == "❌ Path traversal attempt detected."
    # Identifier slash
    assert validate_identifier("v1/0") == "❌ Path traversal attempt detected."
    # Identifier null byte
    assert validate_identifier("tag\0") == "❌ Path traversal attempt detected."
    print("✅ Core guards tests passed!")

def test_train_model_hardening():
    print("Testing train_model hardening...")
    try:
        train_model(model_name="../unsafe", dataset=None, output_dir="./ok", hyperparams={}, device="cpu", peft_method="LoRA", use_lora=True, lora_rank=8, lora_alpha=16, prefix_tuning_num_virtual_tokens=30, prefix_tuning_token_dim=512, prefix_tuning_num_layers=2, prompt_tuning_num_virtual_tokens=20, adapter_reduction_factor=16, resume_from_checkpoint=False, early_stop=3, lr_scheduler_type="cosine", gradient_checkpointing=True, use_unsloth=False, use_chat_template=False, system_prompt="test", training_mode="sft")
        assert False
    except RuntimeError as e:
        assert "❌ Path traversal attempt detected." in str(e)
    print("✅ train_model tests passed!")

def test_registry_hardening():
    print("Testing registry hardening...")
    status = on_registry_upload("./ok", "user/repo", "hf_123456789012345678901234567890123456", "v1/0", "test")
    assert "❌ Path traversal attempt detected." in status
    print("✅ Registry tests passed!")

def test_gguf_vllm_hardening():
    print("Testing GGUF/vLLM hardening...")
    status, _ = on_export_gguf("./ok", "q4/0")
    assert "❌ Path traversal attempt detected." in status
    status = on_vllm_generate("./ok", "prompt", "bnb/../unsafe", 128, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status
    print("✅ GGUF/vLLM tests passed!")

if __name__ == "__main__":
    try:
        test_core_guards()
        test_train_model_hardening()
        test_registry_hardening()
        test_gguf_vllm_hardening()
        print("\n🎉 All comprehensive security tests passed!")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        sys.exit(1)
