
import os
import sys
from unittest.mock import MagicMock, patch

# Add current directory to path
sys.path.append(os.getcwd())

# Mock machine learning libraries
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['vllm'] = MagicMock()
sys.modules['unsloth'] = MagicMock()
sys.modules['gradio'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['trl'] = MagicMock()

from training.sft import train_model
from training.reward import train_reward_model_v27
from training.ppo import run_ppo_v27
from training.orpo import train_orpo_v27
from inference.vllm_runner import on_merge_adapter_click, on_vllm_generate
from export.gguf import on_export_gguf

def test_security_hardening():
    print("Running comprehensive security hardening tests...")

    # --- training/sft.py ---
    print("Testing training/sft.py...")
    try:
        train_model(model_name="../unsafe", dataset=None, output_dir="./ok", hyperparams={}, device="cpu", peft_method="Auto", use_lora=True, lora_rank=8, lora_alpha=16, prefix_tuning_num_virtual_tokens=30, prefix_tuning_token_dim=512, prefix_tuning_num_layers=2, prompt_tuning_num_virtual_tokens=20, adapter_reduction_factor=16, resume_from_checkpoint=False, early_stop=3, lr_scheduler_type="cosine", gradient_checkpointing=True, use_unsloth=False, use_chat_template=False, system_prompt="Prompt")
        assert False, "train_model should have blocked path traversal"
    except ValueError as e:
        assert "❌ Path traversal attempt detected." in str(e)

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
        mock_tok.side_effect = Exception("Validation Passed")
        try:
             train_model(model_name=" gpt2 ", dataset=None, output_dir=" ./ok ", hyperparams={}, device="cpu", peft_method="Auto", use_lora=True, lora_rank=8, lora_alpha=16, prefix_tuning_num_virtual_tokens=30, prefix_tuning_token_dim=512, prefix_tuning_num_layers=2, prompt_tuning_num_virtual_tokens=20, adapter_reduction_factor=16, resume_from_checkpoint=False, early_stop=3, lr_scheduler_type="cosine", gradient_checkpointing=True, use_unsloth=False, use_chat_template=False, system_prompt="Prompt")
        except RuntimeError as e:
            assert "Validation Passed" in str(e)
    print("✅ training/sft.py passed")

    # --- training/reward.py ---
    print("Testing training/reward.py...")
    res = train_reward_model_v27(model_name="../unsafe", reward_file=None, output_dir="./ok")
    assert "❌ Path traversal attempt detected." in res
    print("✅ training/reward.py passed")

    # --- training/ppo.py ---
    print("Testing training/ppo.py...")
    res = run_ppo_v27(policy_model_name="../unsafe", reward_model_path="./ok", ppo_file=None, output_dir="./ok")
    assert "❌ Path traversal attempt detected." in res
    print("✅ training/ppo.py passed")

    # --- training/orpo.py ---
    print("Testing training/orpo.py...")
    res = train_orpo_v27(model_name="../unsafe", orpo_file=None, output_dir="./ok")
    assert "❌ Path traversal attempt detected." in res
    print("✅ training/orpo.py passed")

    # --- inference/vllm_runner.py ---
    print("Testing inference/vllm_runner.py...")
    res, _ = on_merge_adapter_click(base_model_name="../unsafe", adapter_path="./ok", model_path_state="./ok")
    assert "❌ Path traversal attempt detected." in res

    res = on_vllm_generate(model_path_state="../unsafe", vllm_prompt="p", vllm_quant="none", vllm_max_tokens=10, vllm_temp=0.7, vllm_top_p=0.9)
    assert "❌ Path traversal attempt detected." in res
    print("✅ inference/vllm_runner.py passed")

    # --- export/gguf.py ---
    print("Testing export/gguf.py...")
    res, _ = on_export_gguf(model_path="../unsafe", quantization="q4_k")
    assert "❌ Path traversal attempt detected." in res
    print("✅ export/gguf.py passed")

    print("\n🎉 All security hardening tests passed!")

if __name__ == "__main__":
    test_security_hardening()
